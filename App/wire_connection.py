"""WireConnection - A data flow connection between two ports."""

import uuid
import numpy as np


class WireConnection:
    """Represents a wire from an output port to an input port on the canvas.
    Renders as a Bezier curve with optional draggable waypoints.
    """

    def __init__(self, source_port=None, dest_port=None):
        self.id = f"wire_{uuid.uuid4()}"
        self.source_port = source_port
        self.dest_port = dest_port
        self.waypoints = []  # List of (x, y) tuples
        self.color = (0.5, 0.5, 0.5)
        self.is_selected = False
        self.route_mode = "auto"   # "auto" (router-managed) | "manual" (user)

        # Register connection on ports
        if self.source_port is not None:
            self.source_port.add_connection(self)
        if self.dest_port is not None:
            self.dest_port.add_connection(self)

    def disconnect(self):
        """Remove this wire from its ports."""
        if self.source_port is not None:
            self.source_port.remove_connection(self)
        if self.dest_port is not None:
            self.dest_port.remove_connection(self)

    def contains_point(self, point, tolerance=5):
        """Hit-test: is the point near this wire?"""
        if self.source_port is None or self.dest_port is None:
            return False

        src_pos = self.source_port.position
        dst_pos = self.dest_port.position

        if src_pos == (0, 0) or dst_pos == (0, 0):
            return False

        if not self.waypoints:
            pts = WireConnection.compute_bezier_points(src_pos, dst_pos)
        else:
            pts = WireConnection.compute_spline_points(src_pos, dst_pos, self.waypoints)

        for i in range(len(pts) - 1):
            d = WireConnection.point_to_segment_dist(point, pts[i], pts[i + 1])
            if d <= tolerance:
                return True
        return False

    def find_waypoint_at_point(self, point, tolerance=8):
        """Return index of waypoint near point, or -1 if none."""
        for k, wp in enumerate(self.waypoints):
            d = np.sqrt((point[0] - wp[0]) ** 2 + (point[1] - wp[1]) ** 2)
            if d <= tolerance:
                return k
        return -1

    def insert_waypoint(self, point):
        """Insert a new waypoint at the correct position along the wire."""
        if not self.waypoints:
            self.waypoints.append(point)
            return 0

        src_pos = self.source_port.position
        dst_pos = self.dest_port.position
        all_pts = [src_pos] + self.waypoints + [dst_pos]

        best_dist = float('inf')
        best_seg = 0

        for s in range(len(all_pts) - 1):
            seg_pts = WireConnection.compute_bezier_points(all_pts[s], all_pts[s + 1])
            for j in range(len(seg_pts) - 1):
                d = WireConnection.point_to_segment_dist(point, seg_pts[j], seg_pts[j + 1])
                if d < best_dist:
                    best_dist = d
                    best_seg = s

        idx = best_seg
        self.waypoints.insert(idx, point)
        return idx

    def remove_waypoint(self, wp_idx):
        """Remove the waypoint at index wp_idx."""
        if 0 <= wp_idx < len(self.waypoints):
            self.waypoints.pop(wp_idx)

    def get_path_points(self):
        """Get the curve points for rendering."""
        if self.source_port is None or self.dest_port is None:
            return []
        src_pos = self.source_port.position
        dst_pos = self.dest_port.position
        if src_pos == (0, 0) or dst_pos == (0, 0):
            return []
        if not self.waypoints:
            return WireConnection.compute_bezier_points(src_pos, dst_pos)
        else:
            return WireConnection.compute_spline_points(src_pos, dst_pos, self.waypoints)

    def to_struct(self):
        """Serialize for JSON export."""
        s = {"id": self.id}
        if self.source_port and self.source_port.parent_block:
            s["sourceBlock"] = self.source_port.parent_block.id
            s["sourcePort"] = self.source_port.name
        if self.dest_port and self.dest_port.parent_block:
            s["destBlock"] = self.dest_port.parent_block.id
            s["destPort"] = self.dest_port.name
        s["waypoints"] = self.waypoints
        s["routeMode"] = self.route_mode
        return s

    @staticmethod
    def compute_bezier_points(p0, p3, n_pts=30):
        """Compute cubic Bezier curve from p0 to p3.

        Control points are placed purely horizontally so the wire
        exits perpendicular (rightward) from the source port and
        enters perpendicular (leftward) to the destination port.
        """
        p0 = np.array(p0, dtype=float)
        p3 = np.array(p3, dtype=float)

        dx = abs(p3[0] - p0[0])
        dist = np.sqrt((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2)

        # Horizontal offset for control points
        h_offset = max(30, min(dist * 0.4, max(50, dx * 0.5)))

        # Allow up to 45 degrees from horizontal: v_offset clamped to h_offset
        dy = p3[1] - p0[1]
        v_offset = np.clip(dy * 0.25, -h_offset, h_offset)

        p1 = np.array([p0[0] + h_offset, p0[1] + v_offset])
        p2 = np.array([p3[0] - h_offset, p3[1] - v_offset])

        t = np.linspace(0, 1, n_pts).reshape(-1, 1)
        pts = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 +
               3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)

        return [(pts[i, 0], pts[i, 1]) for i in range(n_pts)]

    @staticmethod
    def compute_spline_points(src_pos, dst_pos, waypoints, pts_per_seg=20):
        """Compute smooth Catmull-Rom spline through [src, waypoints, dst]."""
        knots = [np.array(src_pos)] + [np.array(w) for w in waypoints] + [np.array(dst_pos)]
        n_knots = len(knots)

        h_off = 30
        virtual_start = np.array(src_pos) + np.array([-h_off, 0])
        virtual_end = np.array(dst_pos) + np.array([h_off, 0])

        ext_knots = [virtual_start] + knots + [virtual_end]

        all_pts = []
        for seg in range(n_knots - 1):
            P0 = ext_knots[seg]
            P1 = ext_knots[seg + 1]
            P2 = ext_knots[seg + 2]
            P3 = ext_knots[seg + 3]

            m1 = 0.5 * (P2 - P0)
            m2 = 0.5 * (P3 - P1)

            B0 = P1
            B1 = P1 + m1 / 3
            B2 = P2 - m2 / 3
            B3 = P2

            t = np.linspace(0, 1, pts_per_seg).reshape(-1, 1)
            seg_pts = ((1 - t) ** 3 * B0 + 3 * (1 - t) ** 2 * t * B1 +
                       3 * (1 - t) * t ** 2 * B2 + t ** 3 * B3)

            start_idx = 0 if not all_pts else 1
            for i in range(start_idx, pts_per_seg):
                all_pts.append((seg_pts[i, 0], seg_pts[i, 1]))

        return all_pts

    @staticmethod
    def point_to_segment_dist(p, a, b):
        """Distance from point p to line segment a-b."""
        p = np.array(p, dtype=float)
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        ab = b - a
        ap = p - a
        ab_dot = np.dot(ab, ab)
        if ab_dot == 0:
            return np.linalg.norm(p - a)
        t = np.clip(np.dot(ap, ab) / ab_dot, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
