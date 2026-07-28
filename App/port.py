"""Port - Input or output connection point on a BlockNode."""

import uuid
import math


#: Framework port names that keep the default lowercase label styling even on
#: dynamic blocks. Everything else on a dynamic block is a user/variable name
#: whose case is meaningful and shown verbatim.
FRAMEWORK_PORT_NAMES = {"Run", "addInput", "addOutput", "Pause", "Choice"}


class Port:
    """Input or output connection point on a BlockNode.

    Ports are the endpoints for WireConnections. Input ports appear on
    the left edge of a block, output ports on the right edge.
    """

    # Color palette for ports
    PALETTE = [
        (0.20, 0.60, 0.90),  # Blue
        (0.90, 0.55, 0.20),  # Orange
        (0.35, 0.75, 0.35),  # Green
        (0.90, 0.25, 0.55),  # Pink
        (0.55, 0.40, 0.85),  # Purple
        (0.20, 0.80, 0.75),  # Teal
        (0.85, 0.75, 0.20),  # Gold
        (0.60, 0.30, 0.30),  # Brown
    ]

    def __init__(self, name="", direction="input", type_="any",
                 required=True, description=""):
        self.name = name
        self.display_name = name if name else ""
        self.preserve_case = False  # render label verbatim (skip lowercasing)
        self.type = type_
        self.direction = direction
        self.required = required
        self.description = description
        self.parent_block = None
        self.connections = []  # WireConnection objects
        self.index = 1
        self.has_data = False

    @property
    def is_connected(self):
        return len(self.connections) > 0

    @property
    def position(self):
        """Compute port position based on parent block position and index."""
        if self.parent_block is None:
            return (0, 0)

        blk = self.parent_block
        bx, by = blk.position
        bw, bh = blk.size

        if getattr(blk, 'is_compact', False):
            y_pos = by - bh / 2
        elif getattr(blk, 'center_ports', False):
            # Center ports vertically within the block
            ports = blk.input_ports if self.direction == "input" else blk.output_ports
            n = len(ports)
            total_h = (n - 1) * 20
            start_y = by - bh / 2 + total_h / 2
            vis_idx = self._visible_index()
            y_pos = start_y - (vis_idx - 1) * 20
        else:
            header_h = 22
            port_spacing = 20
            port_area_top = by - header_h
            # Use visible index when collapsed
            vis_idx = self._visible_index()
            y_pos = port_area_top - 10 - (vis_idx - 1) * port_spacing

        if self.direction == "input":
            x_pos = bx
        else:
            x_pos = bx + bw

        return (x_pos, y_pos)

    def _visible_index(self):
        """Return 1-based index among visible ports (all ports if expanded)."""
        blk = self.parent_block
        if blk is None or not getattr(blk, 'is_collapsed', False):
            return self.index
        visible = blk.get_visible_ports(self.direction)
        for i, p in enumerate(visible):
            if p is self:
                return i + 1
        return self.index  # fallback

    def add_connection(self, wire):
        self.connections.append(wire)

    def remove_connection(self, wire):
        self.connections = [w for w in self.connections if w is not wire]

    def clear_connections(self):
        self.connections = []

    def is_compatible_with(self, other):
        """Check if this port can connect to another port."""
        if other is None:
            return False
        if self.direction == other.direction:
            return False
        if self.type == "any" or other.type == "any":
            return True
        return self.type == other.type

    def contains_point(self, point):
        """Hit-test: is the given (x, y) point within this port's circle?"""
        pos = self.position
        radius = 10
        dist = math.sqrt((point[0] - pos[0]) ** 2 + (point[1] - pos[1]) ** 2)
        return dist <= radius

    def get_color(self):
        """Get the color for this port based on index and connections."""
        palette = self.PALETTE

        if self.direction == "input" and self.is_connected:
            wire = self.connections[0]
            if wire.source_port is not None:
                src_idx = (wire.source_port.index - 1) % len(palette)
                return palette[src_idx]

        idx = (self.index - 1) % len(palette)
        return palette[idx]

    def get_face_color(self):
        """Get fill color based on connection and data state."""
        color = self.get_color()
        if not self.is_connected:
            return (1, 1, 1)  # Hollow
        elif self.has_data:
            return color  # Colored
        else:
            return (0.7, 0.7, 0.7)  # Gray

    def get_edge_color(self):
        """Get edge color based on connection and data state."""
        color = self.get_color()
        if not self.is_connected:
            return (0.35, 0.35, 0.35)
        elif self.has_data:
            return tuple(c * 0.5 for c in color)
        else:
            return (0.5, 0.5, 0.5)
