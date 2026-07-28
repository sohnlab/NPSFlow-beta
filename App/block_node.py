"""BlockNode - A processing block on the workflow canvas."""

import uuid
from port import Port
from theme import theme


class BlockNode:
    """Represents a single node in the visual pipeline.
    Each block wraps a processing function and manages its input/output ports,
    execution status, and visual rendering.
    """

    def __init__(self, definition_name="", position=(0, 0)):
        self.id = f"blk_{uuid.uuid4()}"
        self.block_id = None  # short numeric ID for debugging (e.g. 1, 2, 3)
        self.definition_name = definition_name
        self.display_name = "Block"
        self.category = "Utility"
        self.color = (0.3, 0.6, 0.9)
        # Optional custom gradient end-color (super blocks only). When None,
        # the renderer derives the gradient endpoint from `color` via the
        # complementary shift; when set, that explicit color is used.
        self.gradient_end = None
        self.position = tuple(position)
        self.size = theme.sizes.block_default
        self.default_size = self.size  # populated from defn; lower bound for resize
        self.status = "pending"
        self.previous_status = "pending"
        self.error_message = ""
        self.input_ports = []
        self.output_ports = []
        self.parameters = {}
        self.output_data = {}
        # Match the canvas base defaults (WorkflowCanvas.base_font_size /
        # base_port_font_size) so a block renders consistently even before a
        # creation path applies the canvas's current base style.
        self.font_size = 11
        self.port_font_size = 9
        self.marker_scale = 1.0
        self.node_shape = "rectangular"
        self.is_interactive = False
        self.is_visualization_only = False
        self.is_compact = False
        self.hide_port_labels = False
        self.show_port_labels = False
        self.is_reroute = False
        self.is_custom_name = False
        self.is_selected = False
        self.is_collapsed = False
        self._expanded_size = None  # stored size before collapse

    def init_from_definition(self, defn):
        """Initialize block properties from a block definition dict."""
        self.definition_name = defn["name"]
        self.display_name = defn["displayName"]
        self.category = defn["category"]
        self.color = tuple(defn["color"])

        if defn.get("isInteractive"):
            self.is_interactive = True
        if defn.get("isVisualizationOnly"):
            self.is_visualization_only = True
        if defn.get("isCompact"):
            self.is_compact = True
        if defn.get("hidePortLabels"):
            self.hide_port_labels = True
        if defn.get("showPortLabels"):
            self.show_port_labels = True
        if defn.get("isReroute"):
            self.is_reroute = True

        # Create input ports
        self.input_ports = []
        for i, inp in enumerate(defn.get("inputs", [])):
            p = Port(
                name=inp["name"],
                direction="input",
                type_=inp.get("type", "any"),
                required=inp.get("required", True),
                description=inp.get("description", ""),
            )
            if inp.get("displayName"):
                p.display_name = inp["displayName"]
            p.preserve_case = inp.get("preserveCase", False)
            p.parent_block = self
            p.index = i + 1
            self.input_ports.append(p)

        # Create output ports
        self.output_ports = []
        for i, out in enumerate(defn.get("outputs", [])):
            req = out.get("required", True)
            p = Port(
                name=out["name"],
                direction="output",
                type_=out.get("type", "any"),
                required=req,
                description=out.get("description", ""),
            )
            if out.get("displayName"):
                p.display_name = out["displayName"]
            p.preserve_case = out.get("preserveCase", False)
            p.parent_block = self
            p.index = i + 1
            self.output_ports.append(p)

        # Initialize default parameters
        if "defaultParameters" in defn:
            for key, val in defn["defaultParameters"].items():
                if key not in self.parameters:
                    self.parameters[key] = val

        # Toggle blocks: initialize OutputData from current value
        if defn.get("isToggle"):
            if "value" not in self.parameters:
                self.parameters["value"] = False
            self.output_data = {"value": self.parameters["value"]}
            for p in self.output_ports:
                p.has_data = True

        # Auto-size block
        if defn.get("defaultSize"):
            self.size = tuple(defn["defaultSize"])
        elif self.is_compact:
            self.size = (90, 20)
        else:
            max_ports = max(len(self.input_ports), len(self.output_ports))
            header_h = 22
            min_height = header_h + 35
            port_height = max(min_height, header_h + 10 + max_ports * 20)
            self.size = (160, port_height)
        # Default size doubles as the user-resize floor.
        self.default_size = self.size

    def contains_point(self, point):
        """Hit-test: is (x, y) inside this block's body rectangle?"""
        x, y = self.position
        w, h = self.size
        return (point[0] >= x and point[0] <= x + w and
                point[1] <= y and point[1] >= y - h)

    def is_on_resize_handle(self, point):
        """Check if point is near the bottom-right corner."""
        x, y = self.position
        w, h = self.size
        corner_x = x + w
        corner_y = y - h
        handle_size = 10
        return (point[0] >= corner_x - handle_size and point[0] <= corner_x + 2 and
                point[1] >= corner_y - 2 and point[1] <= corner_y + handle_size)

    def find_port_at_point(self, point):
        """Find a port at the given canvas coordinate (only visible ports)."""
        for p in self.get_visible_ports("input"):
            if p.contains_point(point):
                return p
        for p in self.get_visible_ports("output"):
            if p.contains_point(point):
                return p
        return None

    def get_input_port(self, name):
        """Get input port by name."""
        for p in self.input_ports:
            if p.name == name:
                return p
        return None

    def get_output_port(self, name):
        """Get output port by name."""
        for p in self.output_ports:
            if p.name == name:
                return p
        return None

    def set_selected(self, tf):
        self.is_selected = tf

    def toggle_collapsed(self):
        """Toggle collapsed state. When collapsed, only connected ports are visible."""
        if self.is_collapsed:
            # Expand: recalculate size for all ports
            self.is_collapsed = False
            self._expanded_size = None
            self.resize_to_fit_ports()
        else:
            # Collapse: resize to fit connected ports only
            self.is_collapsed = True
            self._resize_for_visible_ports()

    def resize_to_fit_ports(self):
        """Resize height to fit ports; collapsed blocks fit visible ports only."""
        if self.is_collapsed:
            self._resize_for_visible_ports()
            return
        max_ports = max(len(self.input_ports), len(self.output_ports))
        header_h = 22
        new_height = max(header_h + 35, header_h + 10 + max_ports * 20)
        self.size = (self.size[0], new_height)

    def _resize_for_visible_ports(self):
        """Resize block height to fit only visible (connected) ports."""
        visible_in = [p for p in self.input_ports if p.is_connected]
        visible_out = [p for p in self.output_ports if p.is_connected]
        max_visible = max(len(visible_in), len(visible_out), 1)
        header_h = 22
        new_height = max(header_h + 35, header_h + 10 + max_visible * 20)
        self.size = (self.size[0], new_height)

    def get_visible_ports(self, direction):
        """Return list of ports visible in current state."""
        ports = self.input_ports if direction == "input" else self.output_ports
        if not self.is_collapsed:
            return list(ports)
        return [p for p in ports if p.is_connected]

    def port_defs(self, direction):
        """Serializable defs for 'input' or 'output' ports (persisted in parameters)."""
        ports = self.input_ports if direction == "input" else self.output_ports
        return [
            {"name": p.name, "type": p.type, "required": p.required,
             "description": p.description, "displayName": p.display_name}
            for p in ports
        ]

    def to_struct(self):
        """Serialize block to dict for JSON export."""
        # Always save port order so rearranged ports persist across save/load
        self.parameters["inputPortDefs"] = self.port_defs("input")
        self.parameters["outputPortDefs"] = self.port_defs("output")

        s = {
            "id": self.id,
            "definition": self.definition_name,
            "displayName": self.display_name,
            "position": list(self.position),
            "size": list(self.size),
            "color": list(self.color),
            "parameters": self.parameters,
        }
        if self.block_id is not None:
            s["blockId"] = self.block_id
        if self.is_custom_name:
            s["isCustomName"] = True
        if self.is_collapsed:
            s["isCollapsed"] = True
        if self.gradient_end is not None:
            s["gradientEnd"] = list(self.gradient_end)
        return s
