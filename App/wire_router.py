"""Pure geometry for auto-routing wires around blocks (no Qt).

Coordinates are canvas coords (Y-up), the same space as block.position,
port.position and wire.waypoints. A rect is (x_min, x_max, y_min, y_max)
with y_max = top edge, y_min = bottom edge.
"""

MARGIN = 14
ANGLE_LIMIT_DEG = 30
MAX_WAYPOINTS = 8


def _expand(rect, m):
    x0, x1, y0, y1 = rect
    return (x0 - m, x1 + m, y0 - m, y1 + m)


def _inside_any(p, rects):
    px, py = p
    for x0, x1, y0, y1 in rects:
        if x0 <= px <= x1 and y0 <= py <= y1:
            return True
    return False


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d):
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def segment_intersects_rect(a, b, rect):
    """True if segment a-b intersects the axis-aligned rect."""
    x0, x1, y0, y1 = rect
    ax, ay = a
    bx, by = b
    if ax < x0 and bx < x0: return False
    if ax > x1 and bx > x1: return False
    if ay < y0 and by < y0: return False
    if ay > y1 and by > y1: return False
    if x0 <= ax <= x1 and y0 <= ay <= y1: return True
    if x0 <= bx <= x1 and y0 <= by <= y1: return True
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for i in range(4):
        if _segments_cross(a, b, corners[i], corners[(i + 1) % 4]):
            return True
    return False


from wire_connection import WireConnection


def _first_hit(poly, rects):
    """Return (segment_index, rect) of the first polyline segment that hits an
    obstacle, scanning from the source end. None if the path is clear."""
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        for rect in rects:
            if segment_intersects_rect(a, b, rect):
                return (i, rect)
    return None


def _detour(a, b, rect):
    """Two waypoints routing the a->b segment over or under `rect` (expanded),
    whichever needs less vertical deviation. Ordered along travel direction."""
    x0, x1, y0, y1 = rect
    cx = (x0 + x1) / 2.0
    ax, ay = a
    bx, by = b
    if bx != ax:
        t = (cx - ax) / (bx - ax)
        seg_y = ay + t * (by - ay)
    else:
        seg_y = (ay + by) / 2.0
    over = (y1 - seg_y) <= (seg_y - y0)   # closer to top edge -> go over
    edge_y = (y1 + 1) if over else (y0 - 1)   # push just outside the rect edge
    left = (x0 - 1, edge_y)
    right = (x1 + 1, edge_y)
    return [left, right] if bx >= ax else [right, left]


def spline_clearance_ok(src, dst, waypoints, obstacles, margin=0):
    """True if the rendered Catmull-Rom spline keeps every sample outside all
    obstacle rects (optionally expanded by `margin`)."""
    pts = WireConnection.compute_spline_points(src, dst, waypoints) \
        if waypoints else WireConnection.compute_bezier_points(src, dst)
    rects = [_expand(r, margin) for r in obstacles] if margin else list(obstacles)
    return not any(_inside_any(p, rects) for p in pts)


def _vert_crossing_y(a, b, edge_x, ymin, ymax):
    """Y where segment a-b crosses the vertical line x=edge_x within [ymin,ymax],
    or None."""
    ax, ay = a
    bx, by = b
    if (ax - edge_x) * (bx - edge_x) > 0:
        return None            # both endpoints on the same side
    if ax == bx:
        return None
    t = (edge_x - ax) / (bx - ax)
    if t < 0 or t > 1:
        return None
    cy = ay + t * (by - ay)
    return cy if ymin <= cy <= ymax else None


def _clamp_boundary_crossings(src, dst, waypoints, annotations, block_rects,
                              margin, angle_limit_deg):
    """Where the path crosses an annotation's left/right edge too steeply, add a
    short waypoint just outside that edge so the wire crosses ~horizontally.
    Skip the clamp if its waypoint would land inside a block rect."""
    import math
    limit = math.tan(math.radians(angle_limit_deg))
    result = list(waypoints)
    for ax0, ax1, ay0, ay1 in annotations:
        for edge_x, outside_x in ((ax0, ax0 - margin), (ax1, ax1 + margin)):
            poly = [src] + result + [dst]
            for i in range(len(poly) - 1):
                a, b = poly[i], poly[i + 1]
                cy = _vert_crossing_y(a, b, edge_x, ay0, ay1)
                if cy is None:
                    continue
                dx = b[0] - a[0]
                if dx == 0 or abs((b[1] - a[1]) / dx) <= limit:
                    continue                       # vertical or already shallow
                wp = (outside_x, cy)
                if _inside_any(wp, block_rects):
                    continue                       # block avoidance wins
                result[i:i] = [wp]
                break                  # at most one clamp per annotation edge
    return result


def route_wire(src, dst, obstacles, annotations=(), *,
               margin=MARGIN, angle_limit_deg=ANGLE_LIMIT_DEG,
               max_waypoints=MAX_WAYPOINTS):
    """Compute waypoints routing src->dst around `obstacles` (raw block rects,
    excluding the wire's own endpoints). Returns [] if already clear."""
    expanded = [_expand(r, margin) for r in obstacles]
    # Drop any obstacle whose expanded rect engulfs an endpoint: a block hugging
    # the source/dest port can't be cleanly routed around (waypoints would land
    # behind the endpoint and zig-zag to the cap). Accept the short crossing.
    expanded = [r for r in expanded
                if not _inside_any(src, [r]) and not _inside_any(dst, [r])]
    waypoints = []
    for _ in range(max_waypoints):
        poly = [src] + waypoints + [dst]
        hit = _first_hit(poly, expanded)
        if hit is None:
            break
        seg_i, rect = hit
        pair = _detour(poly[seg_i], poly[seg_i + 1], rect)
        waypoints[seg_i:seg_i] = pair        # insert into the gap after poly[seg_i]
        if len(waypoints) >= max_waypoints:
            break
    return _clamp_boundary_crossings(
        src, dst, waypoints, annotations, expanded, margin, angle_limit_deg)
