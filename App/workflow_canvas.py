"""WorkflowCanvas - Visual canvas for the node-based workflow editor.

Uses PySide6 QGraphicsView/QGraphicsScene for rendering blocks, wires,
and handling mouse interaction (drag, connect, zoom, pan).
"""

import copy
import uuid
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsItem,
    QMenu,
    QInputDialog,
    QWidget,
    QPushButton,
    QColorDialog,
    QPinchGesture,
)
from PySide6.QtCore import Qt, QRectF, QPointF, QEvent
from PySide6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QPainterPath,
    QFont,
    QPainter,
    QLinearGradient,
)

import sys

from block_node import BlockNode
from wire_connection import WireConnection
import wire_router
from port import Port, FRAMEWORK_PORT_NAMES
from theme import theme, get_theme_name

# Cross-platform font families
MONO_FONT = "Menlo" if sys.platform == "darwin" else "Consolas"
SANS_FONT = "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"


def rgb_to_qcolor(rgb, alpha=255):
    """Convert (r, g, b) float tuple to QColor."""
    return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)


def _hex_to_rgb01(s):
    s = s.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _mix(a, b, t):
    """Linear interpolate two RGB tuples; t=0 → a, t=1 → b."""
    return tuple(x * (1 - t) + y * t for x, y in zip(a, b))


def themed_body_color(block_rgb):
    """Body fill for a block, theme-aware. Returns QColor.

    Light: pastel wash toward white (preserves v1 look).
    Dark:  muted tint over the canvas background — keeps block identity
           color but reads as a dark surface.
    """
    if get_theme_name() == "dark":
        return rgb_to_qcolor(
            _mix(block_rgb, _hex_to_rgb01(theme.palette.canvas_bg), 0.78)
        )
    return rgb_to_qcolor(_mix(block_rgb, (1.0, 1.0, 1.0), 0.85))


def themed_edge_color(block_rgb):
    """Border colour for a block, theme-aware. Returns QColor.

    Light: darken toward black (v1).
    Dark:  brighten toward white so the block outline reads against the
           muted body and dark canvas.
    """
    if get_theme_name() == "dark":
        return rgb_to_qcolor(_mix(block_rgb, (1.0, 1.0, 1.0), 0.30))
    return rgb_to_qcolor(tuple(c * 0.7 for c in block_rgb))


def themed_outline_color(rgb):
    """Generic outline color (used for status indicators), theme-aware."""
    if get_theme_name() == "dark":
        return rgb_to_qcolor(_mix(rgb, (1.0, 1.0, 1.0), 0.45))
    return rgb_to_qcolor(tuple(c * 0.7 for c in rgb))


class AnnotationRect:
    """Data model for a canvas annotation rectangle."""

    def __init__(self, position=(0, 0), size=(200, 150)):
        self.id = f"ann_{uuid.uuid4()}"
        self.position = tuple(position)  # canvas coords (top-left, Y-up)
        self.size = tuple(size)  # (width, height)
        self.text = ""
        self.outline_color = (0.5, 0.5, 0.5)  # RGB [0-1]
        self.fill_color = (0.85, 0.9, 0.95, 0.25)  # RGBA [0-1]
        self.is_selected = False
        self.is_locked = False

    def contains_point(self, canvas_pt):
        """Check if a canvas-coordinate point is inside this annotation."""
        x, y = self.position
        w, h = self.size
        return x <= canvas_pt[0] <= x + w and y - h <= canvas_pt[1] <= y

    def corner_handle_at(self, canvas_pt, handle_size=10):
        """Return corner id ('tl','tr','bl','br') if near a corner, else None."""
        x, y = self.position
        w, h = self.size
        corners = {
            "tl": (x, y),
            "tr": (x + w, y),
            "bl": (x, y - h),
            "br": (x + w, y - h),
        }
        for name, (cx, cy) in corners.items():
            if (
                abs(canvas_pt[0] - cx) < handle_size
                and abs(canvas_pt[1] - cy) < handle_size
            ):
                return name
        return None

    def move_handle_at(self, canvas_pt, handle_w=20, handle_h=8):
        """Return True if point is on the top-center move handle."""
        x, y = self.position
        w, h = self.size
        cx = x + w / 2
        cy = y  # top edge in canvas coords (Y-up)
        return abs(canvas_pt[0] - cx) < handle_w and abs(canvas_pt[1] - cy) < handle_h

    def lock_icon_at(self, canvas_pt, icon_size=14):
        """Return True if point is on the lock icon (top-right corner)."""
        x, y = self.position
        w, _h = self.size
        # Icon is at top-right, offset inward by a small margin (must match
        # _draw_lock_icon).
        ix = x + w - icon_size - 10
        iy = y - 10  # top edge in canvas coords (Y-up), icon hangs below
        return (
            ix <= canvas_pt[0] <= ix + icon_size
            and iy - icon_size <= canvas_pt[1] <= iy
        )

    def contains_block(self, block):
        """Check if a block is fully inside this annotation."""
        x, y = self.position
        w, h = self.size
        bx, by = block.position
        bw, bh = block.size
        return bx >= x and bx + bw <= x + w and by - bh >= y - h and by <= y

    def contains_annotation(self, other):
        """Check if another annotation is fully inside this one."""
        if other is self:
            return False
        x, y = self.position
        w, h = self.size
        ox, oy = other.position
        ow, oh = other.size
        return ox >= x and ox + ow <= x + w and oy - oh >= y - h and oy <= y

    def to_struct(self):
        """Serialize to dict."""
        return {
            "id": self.id,
            "position": list(self.position),
            "size": list(self.size),
            "text": self.text,
            "outlineColor": list(self.outline_color),
            "fillColor": list(self.fill_color),
            "isLocked": self.is_locked,
        }

    @classmethod
    def from_struct(cls, data):
        """Deserialize from dict."""
        ann = cls(
            tuple(data.get("position", [0, 0])), tuple(data.get("size", [200, 150]))
        )
        ann.id = data.get("id", ann.id)
        ann.text = data.get("text", "")
        if "outlineColor" in data:
            ann.outline_color = tuple(data["outlineColor"])
        if "fillColor" in data:
            ann.fill_color = tuple(data["fillColor"])
        ann.is_locked = data.get("isLocked", False)
        return ann


class TextLabel:
    """Data model for a canvas text label."""

    # Padding used when wrapping inside a parent annotation
    ANNOTATION_PADDING = 10

    def __init__(self, position=(0, 0), text="Label"):
        self.id = f"txt_{uuid.uuid4()}"
        self.position = tuple(position)  # canvas coords (Y-up)
        self.text = text
        self.font_size = 12
        self.color = (0.2, 0.2, 0.2)  # RGB [0-1]
        self.is_selected = False
        self.parent_annotation_id = None  # when snapped inside an annotation
        self.relative_position = None  # (rx, ry) offset from parent ann top-left
        self.width_limit = None  # manual wrap width (px), used when unparented
        # Cached rendered bounds (set by TextLabelGraphicsItem)
        self._rendered_w = 0
        self._rendered_h = 0

    def contains_point(self, canvas_pt, margin=15):
        """Hit-test using cached rendered bounds when available."""
        x, y = self.position
        if self._rendered_w > 0 and self._rendered_h > 0:
            w, h = self._rendered_w, self._rendered_h
        else:
            # Fallback estimate
            w = max(40, len(self.text) * self.font_size * 0.6)
            h = self.font_size * 1.5
        return x - 5 <= canvas_pt[0] <= x + w + 2 and y - h <= canvas_pt[1] <= y + 5

    def to_struct(self):
        d = {
            "id": self.id,
            "position": list(self.position),
            "text": self.text,
            "fontSize": self.font_size,
            "color": list(self.color),
        }
        if self.parent_annotation_id:
            d["parentAnnotationId"] = self.parent_annotation_id
            if self.relative_position is not None:
                d["relativePosition"] = list(self.relative_position)
        if self.width_limit is not None:
            d["widthLimit"] = self.width_limit
        return d

    @classmethod
    def from_struct(cls, data):
        lbl = cls(tuple(data.get("position", [0, 0])), data.get("text", "Label"))
        lbl.id = data.get("id", lbl.id)
        lbl.font_size = data.get("fontSize", 12)
        if "color" in data:
            lbl.color = tuple(data["color"])
        lbl.parent_annotation_id = data.get("parentAnnotationId")
        rp = data.get("relativePosition")
        if rp is not None:
            lbl.relative_position = tuple(rp)
        wl = data.get("widthLimit")
        if wl is not None:
            lbl.width_limit = float(wl)
        return lbl


class TextLabelGraphicsItem(QGraphicsTextItem):
    """Graphics item for a TextLabel on the canvas."""

    def __init__(self, label, canvas):
        super().__init__()
        self.label = label
        self.canvas = canvas
        self.setZValue(5)  # above blocks
        self.update_graphics()

    def update_graphics(self):
        lbl = self.label
        self.setPlainText(lbl.text)
        self.setPos(lbl.position[0], -lbl.position[1])
        # Dark mode: force white so labels remain readable regardless of the
        # user-stored color (which is typically chosen for a light background).
        if get_theme_name() == "dark":
            color = QColor(255, 255, 255)
        else:
            color = QColor(
                int(lbl.color[0] * 255),
                int(lbl.color[1] * 255),
                int(lbl.color[2] * 255),
            )
        self.setDefaultTextColor(color)
        font = QFont(SANS_FONT)
        font.setPixelSize(lbl.font_size)
        if lbl.is_selected:
            font.setUnderline(True)
        self.setFont(font)

        # Determine effective wrap width
        wrap_w = -1.0  # -1 disables wrapping in QGraphicsTextItem
        parent_ann = None
        if lbl.parent_annotation_id:
            parent_ann = next(
                (
                    a
                    for a in self.canvas.annotations
                    if a.id == lbl.parent_annotation_id
                ),
                None,
            )
        if parent_ann is not None:
            pad = TextLabel.ANNOTATION_PADDING
            wrap_w = max(20.0, parent_ann.size[0] - 2 * pad)
        elif lbl.width_limit is not None and lbl.width_limit > 0:
            wrap_w = float(lbl.width_limit)
        self.setTextWidth(wrap_w)

        # Cache rendered bounds for hit-testing
        br = self.boundingRect()
        lbl._rendered_w = br.width()
        lbl._rendered_h = br.height()


def _block_shape_radius(shape, w, h):
    """Corner radius for a block of the given shape."""
    if shape == "rounded":
        return min(10.0, w / 2, h / 2)
    if shape == "soft":
        return min(4.0, w / 2, h / 2)
    if shape == "pill":
        return min(w, h) / 2
    return 0.0  # rectangular


def _make_rounded_rect_path(x, y, w, h, radius):
    """Build a QPainterPath for a rectangle, optionally with rounded corners."""
    path = QPainterPath()
    if radius <= 0:
        path.addRect(x, y, w, h)
    else:
        path.addRoundedRect(x, y, w, h, radius, radius)
    return path


def _make_top_rounded_path(x, y, w, h, radius):
    """Rectangle with rounded top corners and a flat bottom edge."""
    path = QPainterPath()
    if radius <= 0:
        path.addRect(x, y, w, h)
        return path
    r = min(radius, w / 2, h)
    path.moveTo(x, y + h)
    path.lineTo(x, y + r)
    path.quadTo(x, y, x + r, y)
    path.lineTo(x + w - r, y)
    path.quadTo(x + w, y, x + w, y + r)
    path.lineTo(x + w, y + h)
    path.closeSubpath()
    return path


class BlockGraphicsItem(QGraphicsRectItem):
    """Graphics item representing a BlockNode on the canvas."""

    def __init__(self, block, canvas):
        super().__init__()
        self.block = block
        self.canvas = canvas
        self.port_items = {}  # port -> QGraphicsEllipseItem
        self.label_items = {}  # port -> QGraphicsTextItem
        # PySide6 garbage-collects child QGraphicsTextItem (a QGraphicsObject)
        # when its Python wrapper goes out of scope, even with a Qt parent.
        # Keep titles/displayText alive via this list.
        self._owned_text_items = []
        self.setAcceptHoverEvents(True)
        self.update_graphics()

    def contextMenuEvent(self, event):
        # Prevent macOS system context menu on text items
        event.accept()

    def apply_shape(self, shape):
        """Fast-path body+header repath when only the corner style changed.

        Falls back to a full rebuild for block kinds whose body isn't a
        QGraphicsPathItem (reroute, boundary).
        """
        body = getattr(self, "_body_path_item", None)
        if body is None:
            self.update_graphics()
            return
        block = self.block
        w, h = block.size
        radius = _block_shape_radius(shape, w, h)
        try:
            body.setPath(_make_rounded_rect_path(0, 0, w, h, radius))
            header = getattr(self, "_header_path_item", None)
            if header is not None:
                header_h = 22
                header.setPath(_make_top_rounded_path(0, 0, w, header_h, radius))
        except RuntimeError:
            # Underlying Qt object was deleted — fall back to a full rebuild.
            self.update_graphics()

    def update_graphics(self):
        """Rebuild all graphics for this block."""
        # Remove children
        for child in self.childItems():
            self.scene().removeItem(child) if self.scene() else None

        self.port_items.clear()
        self.label_items.clear()
        # Body / header path items are reattached below; clear stale refs so
        # apply_shape() doesn't try to update freed Qt objects.
        self._body_path_item = None
        self._header_path_item = None

        block = self.block
        w, h = block.size
        header_h = 22

        # Position: flip Y for scene coordinates (canvas Y-up -> scene Y-down)
        self.setPos(block.position[0], -block.position[1])
        self.setRect(0, 0, w, h)

        color = rgb_to_qcolor(block.color)
        body_color = themed_body_color(block.color)
        edge_color = themed_edge_color(block.color)

        # --- Reroute node: small diamond ---
        if block.is_reroute:
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.setPen(QPen(Qt.PenStyle.NoPen))
            s = max(w, h)
            cx, cy = w / 2, h / 2
            diamond = QPainterPath()
            diamond.moveTo(cx, cy - s / 2)
            diamond.lineTo(cx + s / 2, cy)
            diamond.lineTo(cx, cy + s / 2)
            diamond.lineTo(cx - s / 2, cy)
            diamond.closeSubpath()
            dot = QGraphicsPathItem(diamond, self)
            dot.setBrush(QBrush(color))
            dot.setPen(QPen(edge_color, 1.2))
            # Selection highlight
            if block.is_selected:
                pad = 4
                sel_diamond = QPainterPath()
                sel_diamond.moveTo(cx, cy - s / 2 - pad)
                sel_diamond.lineTo(cx + s / 2 + pad, cy)
                sel_diamond.lineTo(cx, cy + s / 2 + pad)
                sel_diamond.lineTo(cx - s / 2 - pad, cy)
                sel_diamond.closeSubpath()
                sel = QGraphicsPathItem(sel_diamond, self)
                sel.setPen(QPen(QColor(51, 128, 255), 2))
                sel.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                sel.setZValue(-1)
            return  # No ports, status, or labels for reroute

        # --- Boundary blocks: full-color rounded body, no header ---
        is_boundary = block.id in ("_sb_inputs", "_sb_outputs")
        if is_boundary:
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.setPen(QPen(Qt.PenStyle.NoPen))
            body = QGraphicsPathItem(self)
            path = QPainterPath()
            path.addRoundedRect(0, 0, w, h, 8, 8)
            body.setPath(path)
            body.setBrush(QBrush(color))
            body.setPen(QPen(edge_color, 2.0))
            title = QGraphicsTextItem(block.display_name, self)
            self._owned_text_items.append(title)
            title.setDefaultTextColor(QColor(255, 255, 255))
            font = QFont(SANS_FONT)
            font.setPixelSize(12)
            font.setBold(True)
            title.setFont(font)
            title.setPos(w / 2 - title.boundingRect().width() / 2, 4)
        elif block.is_compact:
            shape = getattr(block, "node_shape", "rectangular")
            radius = _block_shape_radius(shape, w, h)
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.setPen(QPen(Qt.PenStyle.NoPen))
            body_item = QGraphicsPathItem(
                _make_rounded_rect_path(0, 0, w, h, radius), self
            )
            body_item.setBrush(QBrush(color))
            body_item.setPen(QPen(edge_color, 1.2))
            self._body_path_item = body_item
            self._header_path_item = None

            # Title (skip if displayText will be shown instead)
            _dt = block.parameters.get("displayText", "")
            if not _dt and block.definition_name:
                try:
                    _defn = self._canvas.registry.get(block.definition_name)
                    _dt = _defn.get("defaultParameters", {}).get("displayText", "")
                except Exception:
                    pass
            show_title = not _dt
            if show_title:
                title = QGraphicsTextItem(block.display_name, self)
                self._owned_text_items.append(title)
                title.setDefaultTextColor(QColor(255, 255, 255))
                font = QFont(SANS_FONT)
                font.setPixelSize(block.font_size)
                font.setBold(True)
                title.setFont(font)
                title.setPos(
                    w / 2 - title.boundingRect().width() / 2,
                    h / 2 - title.boundingRect().height() / 2,
                )
        else:
            # Body
            is_sub = block.definition_name.startswith("SubPipeline_")
            if not is_sub:
                try:
                    _d = self._canvas.registry.get(block.definition_name)
                    is_sub = _d.get("isSubPipeline", False)
                except Exception:
                    pass

            shape = getattr(block, "node_shape", "rectangular")
            radius = _block_shape_radius(shape, w, h)
            # Make self transparent — body is drawn as a child path so corner
            # rounding can vary by shape.
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.setPen(QPen(Qt.PenStyle.NoPen))

            body_item = QGraphicsPathItem(
                _make_rounded_rect_path(0, 0, w, h, radius), self
            )
            body_item.setPen(QPen(edge_color, 2.0 if is_sub else 1.2))
            self._body_path_item = body_item

            # Header path with rounded top corners only
            header = QGraphicsPathItem(
                _make_top_rounded_path(0, 0, w, header_h, radius), self
            )
            self._header_path_item = header
            if is_sub:
                # Linear gradient for super blocks. Honour an explicit
                # `block.gradient_end` (set by the Custom… picker); otherwise
                # derive the endpoint with a vibrant complementary shift.
                grad = QLinearGradient(0, 0, w, 0)
                r, g, b = block.color
                gend = getattr(block, "gradient_end", None)
                if gend is not None:
                    r2, g2, b2 = gend
                else:
                    r2 = min(0.95, max(0.1, 1.0 - r * 0.5))
                    g2 = min(0.95, max(0.1, g * 0.4 + 0.3))
                    b2 = min(0.95, max(0.1, b * 0.5 + 0.4))
                grad.setColorAt(0.0, rgb_to_qcolor((r, g, b)))
                grad.setColorAt(1.0, rgb_to_qcolor((r2, g2, b2)))
                header.setBrush(QBrush(grad))
                # Gradient body too — route both endpoints through the same
                # theme-aware mixer used for regular blocks so dark mode gets
                # a muted-dark surface tinted with the category color.
                body_grad = QLinearGradient(0, 0, w, 0)
                body_grad.setColorAt(0.0, themed_body_color((r, g, b)))
                body_grad.setColorAt(1.0, themed_body_color((r2, g2, b2)))
                body_item.setBrush(QBrush(body_grad))
            else:
                header.setBrush(QBrush(color))
                body_item.setBrush(QBrush(body_color))
            header.setPen(QPen(edge_color, 2.0 if is_sub else 1.2))

            # Inner dashed border for sub-pipeline blocks — uses the same
            # rounded-rect path as the solid outline so the two are
            # concentric, and sits with just a small gap inside the solid
            # frame so it doesn't crop into the title text.
            if is_sub:
                inset = 3
                inner_radius = max(0.0, radius - inset)
                inner = QGraphicsPathItem(
                    _make_rounded_rect_path(
                        inset, inset, w - 2 * inset, h - 2 * inset, inner_radius
                    ),
                    self,
                )
                inner.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                inner.setPen(QPen(edge_color, 0.8, Qt.PenStyle.DashLine))

            # Title
            title = QGraphicsTextItem(block.display_name, self)
            self._owned_text_items.append(title)
            title.setDefaultTextColor(QColor(255, 255, 255))
            font = QFont(SANS_FONT)
            font.setPixelSize(block.font_size)
            font.setBold(True)
            title.setFont(font)
            title.setPos(
                w / 2 - title.boundingRect().width() / 2,
                header_h / 2 - title.boundingRect().height() / 2,
            )

        # Status indicator (skip for boundary blocks).
        # For compact blocks with no input ports (Start, Constant, etc.), the
        # output dot sits at the right edge and the title is centered — place
        # the status indicator at the left so it doesn't fight the output port.
        if not is_boundary:
            s_color = theme.status.get(block.status)
            s_qcolor = rgb_to_qcolor(s_color)
            radius = 4
            if block.is_compact:
                status_y = h / 2
                if not block.input_ports:
                    status_x = 5
                else:
                    status_x = w - radius * 2 - 5
            else:
                status_y = header_h / 2
                status_x = 5
            status = QGraphicsEllipseItem(
                status_x, status_y - radius, radius * 2, radius * 2, self
            )
            status.setBrush(QBrush(s_qcolor))
            status.setPen(QPen(themed_outline_color(s_color), 1))

        # Draw ports (only visible ones when collapsed)
        port_header = 0 if is_boundary else header_h
        visible_in = block.get_visible_ports("input")
        visible_out = block.get_visible_ports("output")
        for port in visible_in:
            self._draw_port(port, w, h, port_header)
        for port in visible_out:
            self._draw_port(port, w, h, port_header)

        # Collapse/expand indicator (small triangle in header)
        if not block.is_compact:
            tri_size = 5
            tri_x = w - 14
            tri_y = header_h / 2
            tri_path = QPainterPath()
            if block.is_collapsed:
                # Right-pointing triangle (collapsed)
                tri_path.moveTo(tri_x, tri_y - tri_size)
                tri_path.lineTo(tri_x + tri_size, tri_y)
                tri_path.lineTo(tri_x, tri_y + tri_size)
            else:
                # Down-pointing triangle (expanded)
                tri_path.moveTo(tri_x - tri_size / 2, tri_y - tri_size / 2)
                tri_path.lineTo(tri_x + tri_size / 2, tri_y - tri_size / 2)
                tri_path.lineTo(tri_x, tri_y + tri_size / 2)
            tri_path.closeSubpath()
            tri_item = QGraphicsPathItem(tri_path, self)
            tri_item.setBrush(QBrush(QColor(255, 255, 255, 180)))
            tri_item.setPen(QPen(Qt.PenStyle.NoPen))

        # Selection highlight
        if block.is_selected:
            pad = 3
            sel = QGraphicsRectItem(-pad, -pad, w + 2 * pad, h + 2 * pad, self)
            sel.setPen(QPen(QColor(51, 128, 255), 2.5))
            sel.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            sel.setZValue(-1)

        # Running highlight (green frame)
        if block.status == "running":
            pad = 4
            run_rect = QGraphicsRectItem(-pad, -pad, w + 2 * pad, h + 2 * pad, self)
            run_rect.setPen(QPen(QColor(0, 200, 0), 3))
            run_rect.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            run_rect.setZValue(-1)

        # Error highlight + hover tooltip with the error message
        if block.status == "error":
            pad = 4
            err = QGraphicsRectItem(-pad, -pad, w + 2 * pad, h + 2 * pad, self)
            err.setPen(QPen(QColor(230, 25, 25), 3))
            err.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            err.setZValue(-1)
            msg = (block.error_message or "Block failed").strip()
            # Qt wraps tooltips automatically when wrapped in <p>; preserve
            # newlines for tracebacks.
            self.setToolTip(
                f"<p style='white-space:pre-wrap; max-width:480px'>"
                f"<b>{block.display_name}</b><br>{msg}</p>"
            )
        else:
            self.setToolTip("")

        # Display text (hide placeholder when outputs already exist)
        display_text = block.parameters.get("displayText", "")
        if not display_text and block.definition_name:
            # Fall back to definition's default displayText
            try:
                defn = self._canvas.registry.get(block.definition_name)
                display_text = defn.get("defaultParameters", {}).get("displayText", "")
            except Exception:
                pass
        if display_text == "Edit to add output" and block.output_ports:
            display_text = ""
        if display_text:
            dt = QGraphicsTextItem(display_text, self)
            self._owned_text_items.append(dt)
            if block.is_compact:
                # On compact blocks, displayText replaces the title
                dt.setDefaultTextColor(QColor(255, 255, 255))
                dt_font = QFont(SANS_FONT)
                dt_font.setPixelSize(block.font_size)
                dt_font.setBold(True)
                dt.setFont(dt_font)
                dt.setPos(
                    w / 2 - dt.boundingRect().width() / 2,
                    h / 2 - dt.boundingRect().height() / 2,
                )
            else:
                dt.setDefaultTextColor(QColor(theme.palette.text))
                dt_font = QFont(MONO_FONT)
                dt_font.setPixelSize(block.port_font_size)
                dt.setFont(dt_font)
                body_top = header_h
                body_bot = h
                dt.setPos(
                    w / 2 - dt.boundingRect().width() / 2,
                    (body_top + body_bot) / 2 - dt.boundingRect().height() / 2,
                )

        # Resize grip (bottom-right corner triangle)
        if not block.is_compact:
            grip_size = 8
            grip_path = QPainterPath()
            grip_path.moveTo(w, h)
            grip_path.lineTo(w - grip_size, h)
            grip_path.lineTo(w, h - grip_size)
            grip_path.closeSubpath()
            grip = QGraphicsPathItem(grip_path, self)
            grip_color = QColor(
                int(block.color[0] * 0.5 * 255 + 0.5 * 128),
                int(block.color[1] * 0.5 * 255 + 0.5 * 128),
                int(block.color[2] * 0.5 * 255 + 0.5 * 128),
                160,
            )
            grip.setBrush(QBrush(grip_color))
            grip.setPen(QPen(Qt.PenStyle.NoPen))

        # In-canvas plot for PlotDisplay blocks
        plot_data = block.parameters.get("plotData")
        if block.status == "done" and plot_data is not None and len(plot_data) > 1:
            pad = 6
            p_left = pad
            p_right = w - pad
            p_top = header_h + pad
            p_bot = h - pad
            if p_right > p_left and p_bot > p_top:
                import numpy as np

                data = np.asarray(plot_data, dtype=float).ravel()
                n = len(data)
                y_min, y_max = float(np.min(data)), float(np.max(data))
                if y_max == y_min:
                    y_max = y_min + 1
                # Map data to block body coordinates
                xs = np.linspace(p_left, p_right, n)
                ys = p_bot - (data - y_min) / (y_max - y_min) * (p_bot - p_top)
                # Draw as a polyline path
                path = QPainterPath()
                path.moveTo(float(xs[0]), float(ys[0]))
                for j in range(1, n):
                    path.lineTo(float(xs[j]), float(ys[j]))
                plot_line = QGraphicsPathItem(path, self)
                plot_line.setPen(QPen(QColor(0, 114, 189), 1.2))
                plot_line.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False
                )

        # Disable text interaction on all child text items to prevent
        # macOS system context menu (Copy/Select All) from appearing
        for child in self.childItems():
            if isinstance(child, QGraphicsTextItem):
                child.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

    def _draw_port(self, port, block_w, block_h, header_h):
        """Draw a single port circle and label."""
        # Calculate position relative to block graphics item
        pos = port.position
        bx, by = self.block.position

        px = pos[0] - bx
        py = by - pos[1]  # Convert from canvas coords to item coords

        radius = 3
        face_color = rgb_to_qcolor(port.get_face_color())
        edge_color = rgb_to_qcolor(port.get_edge_color())

        circle = QGraphicsEllipseItem(
            px - radius, py - radius, radius * 2, radius * 2, self
        )
        circle.setBrush(QBrush(face_color))
        circle.setPen(QPen(edge_color, 1.2))
        if port.description:
            circle.setToolTip(port.description)
        self.port_items[id(port)] = circle

        # Label (skip for compact, hidePortLabels, or when displayText is shown)
        has_display_text = bool(self.block.parameters.get("displayText", ""))
        show_labels = self.block.show_port_labels
        is_boundary = self.block.id in ("_sb_inputs", "_sb_outputs")
        if (
            not self.block.is_compact
            and not self.block.hide_port_labels
            and (show_labels or not has_display_text)
        ):
            port_label = (
                port.display_name
                if getattr(port, "preserve_case", False)
                else port.display_name.lower()
            )
            label = QGraphicsTextItem(port_label, self)
            # In dark mode, brighten the muted palette tone so port labels
            # read clearly against block bodies that are themselves muted.
            if is_boundary:
                label_color = QColor(255, 255, 255)
            elif get_theme_name() == "dark":
                label_color = QColor("#d8dbe0")
            else:
                label_color = QColor(theme.palette.text_muted)
            label.setDefaultTextColor(label_color)
            lbl_font = QFont(SANS_FONT)
            lbl_font.setPixelSize(self.block.port_font_size)
            label.setFont(lbl_font)
            if port.direction == "input":
                label.setPos(px + radius + 4, py - label.boundingRect().height() / 2)
            else:
                label.setPos(
                    px - radius - 4 - label.boundingRect().width(),
                    py - label.boundingRect().height() / 2,
                )
            if port.description:
                label.setToolTip(port.description)
            self.label_items[id(port)] = label

    def hoverMoveEvent(self, event):
        """Change cursor to resize when hovering over bottom-right corner."""
        # Convert item-local pos to canvas coords
        bx, by = self.block.position
        item_pos = event.pos()
        canvas_pt = (bx + item_pos.x(), by - item_pos.y())
        if self.block.is_on_resize_handle(canvas_pt):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()

    def hoverLeaveEvent(self, event):
        self.unsetCursor()


class WireGraphicsItem(QGraphicsPathItem):
    """Graphics item representing a WireConnection."""

    def __init__(self, wire, canvas):
        super().__init__()
        self.wire = wire
        self.canvas = canvas
        self._handle_items = []
        self.setZValue(-10)
        self.update_path()

    def update_path(self):
        """Recompute the wire path."""
        pts = self.wire.get_path_points()

        if not pts:
            return

        path = QPainterPath()
        path.moveTo(pts[0][0], -pts[0][1])  # Flip Y for scene coords
        for pt in pts[1:]:
            path.lineTo(pt[0], -pt[1])
        self.setPath(path)

        is_hovered = (
            hasattr(self.canvas, "_hover_wire") and self.canvas._hover_wire is self.wire
        )
        # Grey the wire until its source port carries data (i.e. the upstream
        # block has run); restore the wire color once it does. Same scheme as
        # the port circles, which render grey when connected-but-no-data and
        # colored after data arrives. The engine clears has_data on a new run,
        # so wires grey out again on re-run.
        src = self.wire.source_port
        has_data = src is not None and getattr(src, "has_data", False)
        base = (
            rgb_to_qcolor(self.wire.color)
            if has_data
            else rgb_to_qcolor((0.6, 0.6, 0.6))
        )
        if self.wire.is_selected:
            color = QColor(51, 128, 255)
            width = 2.5
        elif is_hovered:
            color = base.lighter(130)
            width = 2.5
        else:
            color = base
            width = 1.5

        self.setPen(QPen(color, width))

        # Waypoint handles — shown only while this wire is selected or hovered.
        for h in self._handle_items:
            sc = h.scene()
            if sc is not None:
                sc.removeItem(h)
        self._handle_items = []
        show = self.wire.is_selected or is_hovered
        if show and self.wire.waypoints:
            for wx, wy in self.wire.waypoints:
                r = 4
                h = QGraphicsEllipseItem(wx - r, -wy - r, 2 * r, 2 * r, self)
                h.setBrush(QBrush(QColor(51, 128, 255)))
                h.setPen(QPen(QColor(255, 255, 255), 1))
                h.setZValue(1)
                self._handle_items.append(h)


class AnnotationGraphicsItem(QGraphicsRectItem):
    """Graphics item representing an AnnotationRect on the canvas."""

    def __init__(self, annotation, canvas):
        super().__init__()
        self.annotation = annotation
        self.canvas = canvas
        self.setZValue(-50)  # behind wires (-10) and blocks (0), above grid (-100)
        # Anchor for PySide6-GC-prone QGraphicsTextItem children.
        self._owned_text_items = []
        self.update_graphics()

    def paint(self, painter, option, widget=None):
        """Draw the annotation box, applying the current block style's corner
        radius so annotations track the Settings → Block Style choice.
        Annotations are big containers, so the radius is roughly 2× the block
        radius — same look proportionally."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        shape = getattr(self.canvas, "base_node_shape", "rectangular")
        r = self.rect()
        if shape == "rounded":
            radius = min(22.0, r.width() / 2, r.height() / 2)
        elif shape == "soft":
            radius = min(10.0, r.width() / 2, r.height() / 2)
        else:
            radius = 0.0
        if radius > 0:
            painter.drawRoundedRect(r, radius, radius)
        else:
            painter.drawRect(r)

    def update_graphics(self):
        """Rebuild all graphics for this annotation."""
        # Remove children
        for child in self.childItems():
            if self.scene():
                self.scene().removeItem(child)
        self._owned_text_items.clear()

        ann = self.annotation
        w, h = ann.size

        # Position: flip Y for scene coordinates (canvas Y-up -> scene Y-down)
        self.setPos(ann.position[0], -ann.position[1])
        self.setRect(0, 0, w, h)

        is_dark = get_theme_name() == "dark"

        # Pen: dashed outline. In dark mode, brighten the user-picked outline
        # so it reads against the dark canvas.
        outline_qcolor = QColor(
            int(ann.outline_color[0] * 255),
            int(ann.outline_color[1] * 255),
            int(ann.outline_color[2] * 255),
        )
        if is_dark:
            outline_qcolor = outline_qcolor.lighter(140)
        pen = QPen(outline_qcolor, 1.5, Qt.PenStyle.DashLine)
        self.setPen(pen)

        # Brush: fill with alpha. Make annotations noticeably more
        # transparent in dark mode so the contained blocks read clearly
        # against the dark canvas.
        fc = ann.fill_color
        base_alpha = int(fc[3] * 255) if len(fc) > 3 else 64
        fill_alpha = max(0, int(base_alpha * 0.4)) if is_dark else base_alpha
        fill_qcolor = QColor(
            int(fc[0] * 255), int(fc[1] * 255), int(fc[2] * 255), fill_alpha
        )
        self.setBrush(QBrush(fill_qcolor))

        # Text label — white in dark mode (readable regardless of user color);
        # darker tint of outline in light mode (preserves v1 look).
        if ann.text:
            text_item = QGraphicsTextItem(ann.text, self)
            self._owned_text_items.append(text_item)
            text_color = (
                QColor(255, 255, 255) if is_dark else outline_qcolor.darker(130)
            )
            text_item.setDefaultTextColor(text_color)
            font = QFont(SANS_FONT)
            font.setPixelSize(11)
            text_item.setFont(font)
            text_item.setPos(5, 3)

        # Lock icon (top-right corner)
        self._draw_lock_icon(w, ann.is_locked, outline_qcolor)

        # Selection highlight
        if ann.is_selected:
            sel_color = QColor(51, 128, 255)
            pad = 3
            sel = QGraphicsRectItem(-pad, -pad, w + 2 * pad, h + 2 * pad, self)
            sel.setPen(QPen(sel_color, 2.0))
            sel.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            sel.setZValue(-1)

            # Corner handles (6x6 filled blue squares)
            handle_size = 6
            hs = handle_size / 2
            corners = [(0, 0), (w, 0), (0, h), (w, h)]
            for cx, cy in corners:
                handle = QGraphicsRectItem(
                    cx - hs, cy - hs, handle_size, handle_size, self
                )
                handle.setBrush(QBrush(sel_color))
                handle.setPen(QPen(Qt.PenStyle.NoPen))
                handle.setZValue(1)

            # Top-center move handle (rounded pill shape)
            mh_w, mh_h = 30, 6
            move_handle = QGraphicsRectItem(
                w / 2 - mh_w / 2, -mh_h / 2, mh_w, mh_h, self
            )
            move_handle.setBrush(QBrush(sel_color))
            move_handle.setPen(QPen(Qt.PenStyle.NoPen))
            move_handle.setZValue(1)

    def _draw_lock_icon(self, box_w, is_locked, outline_color):
        """Draw a small lock/unlock icon at the top-right of the annotation."""
        icon_size = 14
        margin = 10
        ix = box_w - icon_size - margin  # x in local item coords
        iy = margin  # y in local item coords

        if get_theme_name() == "dark":
            color = QColor(255, 255, 255)
        else:
            color = outline_color.darker(120) if is_locked else QColor(outline_color)
        color.setAlpha(200 if is_locked else 120)

        # Lock body (rounded rect)
        body_x = ix + 2
        body_y = iy + 6
        body_w = icon_size - 4
        body_h = icon_size - 7
        body = QGraphicsRectItem(body_x, body_y, body_w, body_h, self)
        body.setBrush(QBrush(color) if is_locked else QBrush(Qt.BrushStyle.NoBrush))
        body.setPen(QPen(color, 1.2))
        body.setZValue(2)

        # Shackle (arc)
        path = QPainterPath()
        shackle_w = body_w - 2
        shackle_h = 5
        cx = ix + icon_size / 2
        if is_locked:
            # Closed shackle
            path.moveTo(cx - shackle_w / 2, body_y)
            path.lineTo(cx - shackle_w / 2, body_y - shackle_h)
            path.cubicTo(
                cx - shackle_w / 2,
                body_y - shackle_h - 3,
                cx + shackle_w / 2,
                body_y - shackle_h - 3,
                cx + shackle_w / 2,
                body_y - shackle_h,
            )
            path.lineTo(cx + shackle_w / 2, body_y)
        else:
            # Open shackle (right side lifted)
            path.moveTo(cx - shackle_w / 2, body_y)
            path.lineTo(cx - shackle_w / 2, body_y - shackle_h)
            path.cubicTo(
                cx - shackle_w / 2,
                body_y - shackle_h - 3,
                cx + shackle_w / 2,
                body_y - shackle_h - 3,
                cx + shackle_w / 2,
                body_y - shackle_h,
            )
            path.lineTo(cx + shackle_w / 2, body_y - 3)
        shackle = QGraphicsPathItem(path, self)
        shackle.setPen(QPen(color, 1.5))
        shackle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        shackle.setZValue(2)


class _MiniMap(QWidget):
    """Overlay minimap showing a bird's-eye view of all blocks on the canvas."""

    def __init__(self, parent_canvas):
        super().__init__(parent_canvas)
        self._canvas = parent_canvas
        self._size = (180, 120)
        self._margin = 8
        self.setFixedSize(*self._size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dragging = False
        self.raise_()
        self.show()

    def reposition(self):
        """Place at top-right of the parent canvas."""
        pw = self._canvas.width()
        self.move(pw - self._size[0] - self._margin, self._margin)
        self.raise_()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background — theme-aware so the minimap reads on both light/dark.
        if get_theme_name() == "dark":
            bg = QColor(theme.palette.panel_bg)
            bg.setAlpha(220)
            border = QColor(theme.palette.border)
        else:
            bg = QColor(255, 255, 255, 210)
            border = QColor(180, 180, 180)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border, 1))
        p.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 4, 4)

        blocks = self._canvas.blocks
        if not blocks:
            p.end()
            return

        # Compute bounding rect of all blocks in scene coords
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for b in blocks:
            bx, by = b.position[0], -b.position[1]  # scene coords (Y flipped)
            bw, bh = b.size
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            max_x = max(max_x, bx + bw)
            max_y = max(max_y, by + bh)

        # Add padding
        pad = 50
        min_x -= pad
        min_y -= pad
        max_x += pad
        max_y += pad
        content_w = max_x - min_x
        content_h = max_y - min_y
        if content_w <= 0 or content_h <= 0:
            p.end()
            return

        # Map area inside minimap widget
        inset = 6
        map_w = self.width() - 2 * inset
        map_h = self.height() - 2 * inset
        scale = min(map_w / content_w, map_h / content_h)

        # Centering offset
        drawn_w = content_w * scale
        drawn_h = content_h * scale
        off_x = inset + (map_w - drawn_w) / 2
        off_y = inset + (map_h - drawn_h) / 2

        def to_mini(sx, sy):
            return (off_x + (sx - min_x) * scale, off_y + (sy - min_y) * scale)

        # Draw blocks
        for b in blocks:
            bx, by = b.position[0], -b.position[1]
            bw, bh = b.size
            mx, my = to_mini(bx, by)
            mw = max(bw * scale, 2)
            mh = max(bh * scale, 2)
            color = rgb_to_qcolor(b.color)
            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(mx, my, mw, mh))

        # Draw viewport rect — use the active theme's accent so it reads
        # against either light or dark minimap backgrounds.
        vp_rect = self._canvas.mapToScene(self._canvas.viewport().rect()).boundingRect()
        vx, vy = to_mini(vp_rect.left(), vp_rect.top())
        vw = vp_rect.width() * scale
        vh = vp_rect.height() * scale
        accent = QColor(theme.palette.accent)
        vp_fill = QColor(accent)
        vp_fill.setAlpha(60)
        vp_edge = QColor(accent)
        vp_edge.setAlpha(180)
        p.setBrush(QBrush(vp_fill))
        p.setPen(QPen(vp_edge, 1.5))
        p.drawRect(QRectF(vx, vy, vw, vh))

        # Store mapping for mouse interaction
        self._map_min_x = min_x
        self._map_min_y = min_y
        self._map_scale = scale
        self._map_off_x = off_x
        self._map_off_y = off_y

        p.end()

    def _mini_to_scene(self, mx, my):
        """Convert minimap widget coords to scene coords."""
        if not hasattr(self, "_map_scale") or self._map_scale <= 0:
            return QPointF(0, 0)
        sx = self._map_min_x + (mx - self._map_off_x) / self._map_scale
        sy = self._map_min_y + (my - self._map_off_y) / self._map_scale
        return QPointF(sx, sy)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            scene_pt = self._mini_to_scene(event.position().x(), event.position().y())
            self._canvas.centerOn(scene_pt)
            self._canvas._minimap.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            scene_pt = self._mini_to_scene(event.position().x(), event.position().y())
            self._canvas.centerOn(scene_pt)
            self._canvas._minimap.update()

    def mouseReleaseEvent(self, event):
        self._dragging = False


class WorkflowCanvas(QGraphicsView):
    """Visual canvas for the node-based workflow editor.
    Manages block rendering, mouse interaction, and visual state.
    """

    def __init__(self, registry, parent=None):
        super().__init__(parent)

        self.registry = registry
        self.blocks = []
        self.wires = []
        self._next_block_id = 1
        self._debug_locked_block = None  # block locked during debug pause
        self.selected_block = None
        self.selected_blocks = []
        self.selected_wire = None
        self._is_dragging_waypoint = False
        self._drag_wire = None
        self._drag_wp_idx = -1
        # When False (default), new wires draw straight (no auto-route around
        # blocks); set from the "Wire Routing" app setting.
        self.auto_route_on_create = False

        self.base_font_size = 11
        self.base_port_font_size = 9
        self.base_annotation_font_size = 12
        self.base_node_shape = "rectangular"  # block style; annotations follow it
        self.grid_spacing = 20
        self.grid_pattern = "grid"  # "grid", "dots", "none"
        self._grid_items = []

        # Theme colours — read from active theme so dark mode applies at startup.
        self._canvas_color = QColor(theme.palette.canvas_bg)
        self._grid_color = QColor(theme.palette.canvas_grid)

        # Interaction state
        self._is_dragging = False
        self._drag_block = None
        self._drag_offset = (0, 0)
        self._is_drawing_wire = False
        self._replacing_wire = False
        self._wire_start_port = None
        self._preview_wire_item = None
        self._is_panning = False
        self._pan_start = QPointF()
        self._is_resizing = False
        self._resize_block = None
        self._hover_block = None
        self._hover_port = None
        self._hover_wire = None

        # Undo/redo
        self._undo_stack = []
        self._redo_stack = []
        self._max_undo = 50

        # Graphics items
        self._block_items = {}  # block.id -> BlockGraphicsItem
        self._wire_items = {}  # wire.id -> WireGraphicsItem
        self._wire_label_items = []  # Floating labels for off-screen wire endpoints

        # Annotation state
        self.annotations = []
        self._annotation_items = {}  # ann.id -> AnnotationGraphicsItem
        self.selected_annotation = None
        self._annotation_mode = False
        self._is_drawing_annotation = False
        self._annotation_draw_start = None
        self._preview_annotation = None
        self._is_dragging_annotation = False
        self._drag_annotation = None
        self._drag_ann_offset = (0, 0)
        self._drag_ann_blocks = []  # blocks to move with annotation
        self._drag_ann_children = []  # child annotations to move with parent
        self._is_resizing_annotation = False
        self._resize_annotation = None
        self._resize_corner = None
        self._resize_ann_fixed = None  # fixed opposite corner

        # Mouse tracking
        self._last_mouse_pos = (0, 0)
        self._last_context_pos = None

        # Callbacks
        self.block_double_click_callback = None
        self.block_edit_callback = None
        self.block_context_menu_callback = None
        self.canvas_context_menu_callback = None
        self.wire_change_callback = None
        self.delete_callback = None
        self.drop_block_callback = None
        self.state_changed_callback = None

        self._setup_scene()
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        # Minimap overlay
        self._minimap = _MiniMap(self)
        self._minimap.reposition()

        # Overlay button styles (rebuilt by _rebuild_overlay_styles()).
        self._ann_btn_style_normal = ""
        self._ann_btn_style_active = ""
        self._btn_fixed_width = 80
        self._rebuild_overlay_styles()
        # Annotate button — toggles sub-buttons
        self._annotation_btn = QPushButton("Annotate", self)
        self._annotation_btn.setFixedWidth(self._btn_fixed_width)
        self._annotation_btn.setStyleSheet(self._ann_btn_style_normal)
        self._annotation_btn.clicked.connect(self._toggle_annotate_popup)
        self._annotation_btn.show()

        # Sub-buttons for annotation modes (hidden by default)
        self._ann_box_btn = QPushButton("Box", self)
        self._ann_box_btn.setFixedWidth(self._btn_fixed_width)
        self._ann_box_btn.setStyleSheet(self._ann_btn_style_normal)
        self._ann_box_btn.setCheckable(True)
        self._ann_box_btn.clicked.connect(self._toggle_annotation_mode)
        self._ann_box_btn.hide()

        self._ann_text_btn = QPushButton("Text", self)
        self._ann_text_btn.setFixedWidth(self._btn_fixed_width)
        self._ann_text_btn.setStyleSheet(self._ann_btn_style_normal)
        self._ann_text_btn.setCheckable(True)
        self._ann_text_btn.clicked.connect(self._toggle_text_mode)
        self._ann_text_btn.hide()

        self._annotate_popup_visible = False

        # Select button (box-select mode)
        self._select_btn = QPushButton("Select", self)
        self._select_btn.setFixedWidth(self._btn_fixed_width)
        self._select_btn.setStyleSheet(self._ann_btn_style_normal)
        self._select_btn.setCheckable(True)
        self._select_btn.clicked.connect(self._toggle_select_mode)
        self._select_btn.show()
        self._select_mode = False
        self._is_box_selecting = False
        self._box_select_start = None
        self._preview_box_select = None
        self._multi_drag_blocks = []
        self._multi_drag_annotations = []
        self._is_dragging_multi = False
        self._multi_drag_offset = (0, 0)

        # Text label state
        self._text_mode = False
        self.text_labels = []
        self._text_label_items = {}
        self.selected_text_label = None
        self._is_dragging_text_label = False
        self._drag_text_label = None
        self._drag_text_offset = (0, 0)

        self._reposition_annotation_btn()

        # Floating zoom controls (top-right)
        self._zoom_btns = []
        for label, tooltip, handler in (
            ("+", "Zoom in", self._zoom_in),
            ("−", "Zoom out", self._zoom_out),
            ("1×", "Reset zoom to 100%", self._zoom_actual),
            ("⌂", "Home — center on Start", self._zoom_home),
        ):
            btn = QPushButton(label, self)
            btn.setToolTip(tooltip)
            btn.setFixedSize(28, 28)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(self._zoom_btn_style())
            btn.clicked.connect(handler)
            btn.show()
            self._zoom_btns.append(btn)
        self._reposition_zoom_btns()

        # Touchpad / trackpad gestures.
        # macOS delivers pinch as `QNativeGestureEvent` (no grab needed);
        # Windows / Linux route pinch through Qt's gesture framework.
        self.grabGesture(Qt.GestureType.PinchGesture)
        # Track an in-flight pinch's starting scale so we can apply the
        # gesture's totalScaleFactor() relative to it (clean Windows path).
        self._pinch_start_scale = None

    def _zoom_btn_style(self):
        p = theme.palette
        return (
            f"QPushButton {{ background: {p.button_bg}; color: {p.button_text};"
            f" border: 1px solid {p.border}; border-radius: 4px;"
            f" font-size: 14px; padding: 0; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover};"
            f" border-color: {p.accent}; }}"
        )

    def _reposition_zoom_btns(self):
        """Stack zoom buttons vertically at the bottom-right of the canvas."""
        if not getattr(self, "_zoom_btns", None):
            return
        margin = 12
        btn_h = self._zoom_btns[0].height()
        gap = 4
        total_h = len(self._zoom_btns) * btn_h + (len(self._zoom_btns) - 1) * gap
        x = self.width() - self._zoom_btns[0].width() - margin
        y = self.height() - total_h - margin
        for btn in self._zoom_btns:
            btn.move(x, y)
            y += btn_h + gap

    def _zoom_clamp(self, factor):
        cur = self.transform().m11()
        new = cur * factor
        if new < 1 / 3.0 or new > 3.0:
            return
        self.scale(factor, factor)
        if hasattr(self, "_minimap"):
            self._minimap.update()

    def _zoom_in(self):
        self._zoom_clamp(1.15)

    def _zoom_out(self):
        self._zoom_clamp(1 / 1.15)

    def _zoom_actual(self):
        """Reset zoom to 100% while keeping the same scene point at the
        viewport center. Uses AnchorViewCenter on scale() so the centering
        is exact and does not drift across repeated clicks."""
        cur = self.transform().m11()
        if cur <= 0:
            return
        factor = 1.0 / cur
        prev_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(factor, factor)
        self.setTransformationAnchor(prev_anchor)
        if hasattr(self, "_minimap"):
            self._minimap.update()

    def _zoom_home(self):
        """Pan so the Start block sits ~25% from the left edge, preserving zoom."""
        target = None
        for block in self.blocks:
            if block.definition_name == "StartBlock":
                target = block
                break
            if self.registry and self.registry.has(block.definition_name):
                try:
                    defn = self.registry.get(block.definition_name)
                    if defn.get("isStart"):
                        target = block
                        break
                except Exception:
                    pass
        if target is None and self.blocks:
            target = self.blocks[0]
        if target is None:
            self.centerOn(0, 0)
            return

        x, y = target.position
        w, h = target.size
        bx = x + w / 2
        by = -(y - h / 2)

        vw = self.viewport().width()
        scale = self.transform().m11() if self.transform().m11() != 0 else 1.0
        # Center is offset to the right so the block lands at 1/4 of the
        # viewport width from the left edge.
        offset_x = (vw * 0.25) / scale
        self.centerOn(bx + offset_x, by)
        if hasattr(self, "_minimap"):
            self._minimap.update()

    def _toggle_annotate_popup(self):
        """Show/hide the Box and Text sub-buttons above Annotate."""
        self._annotate_popup_visible = not self._annotate_popup_visible
        if self._annotate_popup_visible:
            self._annotation_btn.setStyleSheet(self._ann_btn_style_active)
            self._ann_box_btn.show()
            self._ann_text_btn.show()
        else:
            self._annotation_btn.setStyleSheet(self._ann_btn_style_normal)
            self._ann_box_btn.hide()
            self._ann_text_btn.hide()
            # Deactivate both modes
            self._annotation_mode = False
            self._text_mode = False
            self._ann_box_btn.setChecked(False)
            self._ann_text_btn.setChecked(False)
            self._ann_box_btn.setStyleSheet(self._ann_btn_style_normal)
            self._ann_text_btn.setStyleSheet(self._ann_btn_style_normal)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._reposition_annotation_btn()

    def _toggle_annotation_mode(self):
        """Toggle annotation box drawing mode."""
        self._annotation_mode = self._ann_box_btn.isChecked()
        if self._annotation_mode:
            self._ann_box_btn.setStyleSheet(self._ann_btn_style_active)
            self.setCursor(Qt.CursorShape.CrossCursor)
            # Deactivate other modes
            self._text_mode = False
            self._ann_text_btn.setChecked(False)
            self._ann_text_btn.setStyleSheet(self._ann_btn_style_normal)
            self._select_mode = False
            self._select_btn.setChecked(False)
            self._select_btn.setStyleSheet(self._ann_btn_style_normal)
        else:
            self._ann_box_btn.setStyleSheet(self._ann_btn_style_normal)
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _toggle_text_mode(self):
        """Toggle text label placement mode."""
        self._text_mode = self._ann_text_btn.isChecked()
        if self._text_mode:
            self._ann_text_btn.setStyleSheet(self._ann_btn_style_active)
            self.setCursor(Qt.CursorShape.IBeamCursor)
            # Deactivate other modes
            self._annotation_mode = False
            self._ann_box_btn.setChecked(False)
            self._ann_box_btn.setStyleSheet(self._ann_btn_style_normal)
            self._select_mode = False
            self._select_btn.setChecked(False)
            self._select_btn.setStyleSheet(self._ann_btn_style_normal)
        else:
            self._ann_text_btn.setStyleSheet(self._ann_btn_style_normal)
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _toggle_select_mode(self):
        """Toggle box-select mode."""
        self._select_mode = self._select_btn.isChecked()
        if self._select_mode:
            self._select_btn.setStyleSheet(self._ann_btn_style_active)
            self.setCursor(Qt.CursorShape.CrossCursor)
            # Deactivate annotation mode if active
            self._annotation_mode = False
            self._annotation_btn.setChecked(False)
            self._annotation_btn.setStyleSheet(self._ann_btn_style_normal)
        else:
            self._select_btn.setStyleSheet(self._ann_btn_style_normal)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._deselect_multi()

    def _deselect_multi(self):
        """Clear multi-block and multi-annotation selection."""
        for b in self._multi_drag_blocks:
            b.set_selected(False)
            self.update_block_graphics(b)
        self._multi_drag_blocks = []
        for ann in self._multi_drag_annotations:
            ann.is_selected = False
            self._update_annotation_graphics(ann)
        self._multi_drag_annotations = []

    def _reposition_annotation_btn(self):
        """Place the Annotate, Select, and sub-buttons at bottom-center."""
        gap = 8
        bw = self._btn_fixed_width
        bh = self._annotation_btn.sizeHint().height()
        total_w = bw + gap + bw
        x = (self.width() - total_w) // 2
        y = self.height() - bh - 12
        self._annotation_btn.move(x, y)
        self._select_btn.move(x + bw + gap, y)
        # Sub-buttons above Annotate
        if self._annotate_popup_visible:
            sub_y = y - bh - 4
            self._ann_box_btn.move(x, sub_y)
            self._ann_text_btn.move(x, sub_y - bh - 4)

    def _setup_scene(self):
        """Initialize the graphics scene."""
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(-5000, -5000, 10000, 10000)
        self.setScene(self._scene)
        self._scene_margin = 2000  # margin around content bounds

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # AnchorViewCenter on resize keeps the user's current view stable when
        # the viewport changes size (e.g. a dock appears). AnchorUnderMouse
        # here would re-anchor on the cursor and pan the canvas off-screen.
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Draw grid
        self._draw_grid()

    def _content_bounds(self):
        """Return (min_x, min_y, max_x, max_y) in scene coords for all content."""
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for block in self.blocks:
            bx, by = block.position
            bw, bh = block.size
            # Scene coords: x same, y flipped
            sx = bx
            sy_top = -by
            sy_bot = -(by - bh)
            min_x = min(min_x, sx)
            max_x = max(max_x, sx + bw)
            min_y = min(min_y, sy_top)
            max_y = max(max_y, sy_bot)
        for ann in self.annotations:
            ax, ay = ann.position
            aw, ah = ann.size
            sx = ax
            sy_top = -ay
            sy_bot = -(ay - ah)
            min_x = min(min_x, sx)
            max_x = max(max_x, sx + aw)
            min_y = min(min_y, sy_top)
            max_y = max(max_y, sy_bot)
        if min_x == float("inf"):
            return (-2000, -2000, 2000, 2000)
        return (min_x, min_y, max_x, max_y)

    def _update_scene_rect(self):
        """Expand (never shrink) the scene rect to fit all content with margin."""
        cx1, cy1, cx2, cy2 = self._content_bounds()
        m = self._scene_margin
        need = QRectF(cx1 - m, cy1 - m, (cx2 - cx1) + 2 * m, (cy2 - cy1) + 2 * m)
        current = self._scene.sceneRect()
        if not current.contains(need):
            united = current.united(need)
            self._scene.setSceneRect(united)
            self._draw_grid()

    def _draw_grid(self):
        """Refresh background grid.

        The grid used to be materialised as up to ~250 000 QGraphicsEllipseItem
        (dots) / a few thousand QGraphicsLineItem (grid) — created up-front
        across the whole 10k×10k scene. Switching pattern or size then meant
        deleting and rebuilding all of them, which was the dominant cost of
        the live Settings panel. We now paint the grid lazily inside
        drawBackground(), so this method just clears any stale items and
        triggers a viewport repaint.
        """
        for item in self._grid_items:
            self._scene.removeItem(item)
        self._grid_items.clear()
        self._scene.setBackgroundBrush(QBrush(self._canvas_color))
        if self.viewport() is not None:
            self.viewport().update()

    def drawBackground(self, painter, rect):
        """Paint the grid/dots covering just the exposed rect."""
        super().drawBackground(painter, rect)
        if self.grid_pattern == "none":
            return
        spacing = self.grid_spacing
        if spacing <= 0:
            return

        # Align to the grid so lines stay put while the view scrolls.
        x0 = int(rect.left() // spacing) * spacing
        x1 = int(rect.right() // spacing + 1) * spacing
        y0 = int(rect.top() // spacing) * spacing
        y1 = int(rect.bottom() // spacing + 1) * spacing

        if self.grid_pattern == "dots":
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._grid_color))
            r = 1.0
            x = x0
            while x <= x1:
                y = y0
                while y <= y1:
                    painter.drawEllipse(QPointF(x, y), r, r)
                    y += spacing
                x += spacing
            painter.restore()
        else:  # "grid"
            painter.save()
            painter.setPen(QPen(self._grid_color, 0))
            x = x0
            while x <= x1:
                painter.drawLine(x, y0, x, y1)
                x += spacing
            y = y0
            while y <= y1:
                painter.drawLine(x0, y, x1, y)
                y += spacing
            painter.restore()

    def set_theme(self, canvas_color, grid_color):
        """Set canvas and grid colours and redraw grid."""
        self._canvas_color = canvas_color
        self._grid_color = grid_color
        self._draw_grid()

    def apply_theme(self):
        """Re-read colours from the active theme and redraw."""
        self.set_theme(
            QColor(theme.palette.canvas_bg),
            QColor(theme.palette.canvas_grid),
        )
        self._rebuild_overlay_styles()
        self._reapply_overlay_styles()
        # Block items were built with explicit theme-derived colors; redraw
        # so they pick up the new body/edge/status palette.
        self.redraw()
        # Minimap reads theme tokens in its paintEvent; force a repaint.
        if hasattr(self, "_minimap") and self._minimap is not None:
            self._minimap.update()

    def _rebuild_overlay_styles(self):
        """Build overlay-button QSS strings from the active theme."""
        p = theme.palette
        self._ann_btn_style_normal = (
            f"QPushButton {{ background: {p.button_bg}; color: {p.button_text};"
            f" border: 1px solid {p.border}; border-radius: 4px;"
            f" padding: 4px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover};"
            f" border-color: {p.accent}; }}"
            f"QPushButton:disabled {{ background: {p.alt_base_bg};"
            f" color: {p.text_muted}; border-color: {p.border}; }}"
        )
        self._ann_btn_style_active = (
            f"QPushButton {{ background: {p.accent}; color: {p.accent_text};"
            f" border: 1px solid {p.accent}; border-radius: 4px;"
            f" padding: 4px 12px; font-size: 12px; }}"
        )

    def _reapply_overlay_styles(self):
        """Re-apply the current normal/active style to every overlay button."""
        for btn_name in (
            "_annotation_btn",
            "_ann_box_btn",
            "_ann_text_btn",
            "_select_btn",
        ):
            btn = getattr(self, btn_name, None)
            if btn is None:
                continue
            active = btn.isCheckable() and btn.isChecked()
            btn.setStyleSheet(
                self._ann_btn_style_active if active else self._ann_btn_style_normal
            )
        zoom_style = self._zoom_btn_style()
        for btn in getattr(self, "_zoom_btns", []):
            btn.setStyleSheet(zoom_style)

    def set_grid_settings(self, pattern, spacing):
        """Update grid pattern and spacing, then redraw."""
        self.grid_pattern = pattern
        self.grid_spacing = spacing
        self._draw_grid()

    # --- Drag-and-drop from palette ---

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-npsview-block"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-npsview-block"):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-npsview-block"):
            def_name = bytes(
                event.mimeData().data("application/x-npsview-block")
            ).decode("utf-8")
            scene_pos = self.mapToScene(event.position().toPoint())
            canvas_pt = (scene_pos.x(), -scene_pos.y())
            if self.drop_block_callback:
                self.drop_block_callback(def_name, canvas_pt)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # --- Block management ---

    def add_block(self, def_name, position):
        """Add a new block to the canvas."""
        self.push_undo()
        block = self.registry.create_block(def_name, position)
        block.block_id = self._next_block_id
        self._next_block_id += 1
        # Carry the canvas's current base style so a freshly dropped block
        # matches blocks restored from a saved workflow (which get these in the
        # load path); otherwise it keeps BlockNode's smaller defaults.
        block.font_size = self.base_font_size
        block.port_font_size = self.base_port_font_size
        block.node_shape = self.base_node_shape
        self.blocks.append(block)

        item = BlockGraphicsItem(block, self)
        self._scene.addItem(item)
        self._block_items[block.id] = item

        self._update_scene_rect()
        self._minimap.update()
        return block

    def remove_block(self, block):
        """Remove a block and its wires from the canvas."""
        self.push_undo()
        self._remove_block_no_undo(block)

    def _remove_block_no_undo(self, block):
        """Remove a block and its wires without pushing undo."""
        wires_to_remove = []
        for wire in self.wires:
            if wire.source_port and wire.source_port.parent_block is block:
                wires_to_remove.append(wire)
            elif wire.dest_port and wire.dest_port.parent_block is block:
                wires_to_remove.append(wire)

        for wire in wires_to_remove:
            self.remove_wire(wire)

        if block.id in self._block_items:
            item = self._block_items.pop(block.id)
            self._scene.removeItem(item)

        self.blocks = [b for b in self.blocks if b.id != block.id]

        if self.selected_block is block:
            self.selected_block = None
        self._minimap.update()

    def _wire_obstacles(self, exclude_ids):
        """Block bounding rects (canvas coords) for routing, excluding the given
        block ids and reroute nodes."""
        rects = []
        for b in self.blocks:
            if b.id in exclude_ids or getattr(b, "is_reroute", False):
                continue
            bx, by = b.position
            bw, bh = b.size
            rects.append((bx, bx + bw, by - bh, by))
        return rects

    def _annotation_rects(self):
        rects = []
        for a in self.annotations:
            ax, ay = a.position
            aw, ah = a.size
            rects.append((ax, ax + aw, ay - ah, ay))
        return rects

    def auto_route_wire(self, wire):
        """Recompute an 'auto' wire's waypoints to avoid blocks. No-op for
        manual wires or wires missing an endpoint."""
        if getattr(wire, "route_mode", "auto") != "auto":
            return
        if not wire.source_port or not wire.dest_port:
            return
        exclude = set()
        if wire.source_port.parent_block:
            exclude.add(wire.source_port.parent_block.id)
        if wire.dest_port.parent_block:
            exclude.add(wire.dest_port.parent_block.id)
        wire.waypoints = wire_router.route_wire(
            wire.source_port.position,
            wire.dest_port.position,
            self._wire_obstacles(exclude),
            self._annotation_rects(),
        )
        if wire.id in self._wire_items:
            self._wire_items[wire.id].update_path()

    def tidy_all_wires(self):
        """Auto-route every 'auto' wire (manual wires untouched)."""
        self.push_undo()
        for wire in self.wires:
            self.auto_route_wire(wire)

    def menu_auto_route_wire(self, wire):
        """Force a wire back to auto mode and route it (wire-menu action)."""
        self.push_undo()
        wire.route_mode = "auto"
        self.auto_route_wire(wire)

    def clear_wire_waypoints(self, wire):
        """Straighten a wire: drop waypoints, return to auto (wire-menu action)."""
        self.push_undo()
        wire.waypoints = []
        wire.route_mode = "auto"
        if wire.id in self._wire_items:
            self._wire_items[wire.id].update_path()

    def _begin_waypoint_drag(self, wire, wp_idx):
        self._is_dragging_waypoint = True
        self._drag_wire = wire
        self._drag_wp_idx = wp_idx

    def add_wire(self, source_port, dest_port):
        """Add a wire connection between two ports."""
        self.push_undo()
        wire = WireConnection(source_port, dest_port)

        # Set wire color from source block
        if source_port.parent_block:
            wire.color = source_port.parent_block.color

        self.wires.append(wire)

        # (Auto-align disabled — let user position blocks manually)

        item = WireGraphicsItem(wire, self)
        self._scene.addItem(item)
        self._wire_items[wire.id] = item
        # Only auto-route around blocks on create when the setting opts in;
        # the default is straight ("normal") routing.
        if getattr(self, "auto_route_on_create", False):
            self.auto_route_wire(wire)

        # Notify
        src_block = source_port.parent_block if source_port else None
        dst_block = dest_port.parent_block if dest_port else None
        if self.wire_change_callback:
            self.wire_change_callback(src_block, dst_block)

        return wire

    def remove_wire(self, wire):
        """Remove a wire from the canvas."""
        src_block = wire.source_port.parent_block if wire.source_port else None
        dst_block = wire.dest_port.parent_block if wire.dest_port else None

        wire.disconnect()

        if wire.id in self._wire_items:
            item = self._wire_items.pop(wire.id)
            self._scene.removeItem(item)

        self.wires = [w for w in self.wires if w.id != wire.id]

        if self.selected_wire is wire:
            self.selected_wire = None

        # Notify so dynamic ports can update
        if self.wire_change_callback:
            self.wire_change_callback(src_block, dst_block)

    def update_block_graphics(self, block):
        """Refresh a block's visual representation."""
        if block.id in self._block_items:
            item = self._block_items[block.id]
            self._scene.removeItem(item)

        item = BlockGraphicsItem(block, self)
        self._scene.addItem(item)
        self._block_items[block.id] = item

        # Update connected wires
        for wire in self.wires:
            if (wire.source_port and wire.source_port.parent_block is block) or (
                wire.dest_port and wire.dest_port.parent_block is block
            ):
                if wire.id in self._wire_items:
                    self._wire_items[wire.id].update_path()

    def redraw(self):
        """Full redraw of all blocks and wires."""
        # Clear all items except grid
        for item in list(self._block_items.values()):
            self._scene.removeItem(item)
        for item in list(self._wire_items.values()):
            self._scene.removeItem(item)
        self._block_items.clear()
        self._wire_items.clear()

        for block in self.blocks:
            item = BlockGraphicsItem(block, self)
            self._scene.addItem(item)
            self._block_items[block.id] = item

        for wire in self.wires:
            item = WireGraphicsItem(wire, self)
            self._scene.addItem(item)
            self._wire_items[wire.id] = item

        for item in list(self._annotation_items.values()):
            self._scene.removeItem(item)
        self._annotation_items.clear()
        for ann in self.annotations:
            item = AnnotationGraphicsItem(ann, self)
            self._scene.addItem(item)
            self._annotation_items[ann.id] = item

        for item in list(self._text_label_items.values()):
            self._scene.removeItem(item)
        self._text_label_items.clear()
        for lbl in self.text_labels:
            item = TextLabelGraphicsItem(lbl, self)
            self._scene.addItem(item)
            self._text_label_items[lbl.id] = item

    def select_block(self, block):
        """Select a single block."""
        # Deselect previous
        if self.selected_block:
            self.selected_block.set_selected(False)
            self.update_block_graphics(self.selected_block)
        if self.selected_wire:
            self.selected_wire.is_selected = False
            if self.selected_wire.id in self._wire_items:
                self._wire_items[self.selected_wire.id].update_path()
            self.selected_wire = None
        if self.selected_annotation:
            self.selected_annotation.is_selected = False
            self._update_annotation_graphics(self.selected_annotation)
            self.selected_annotation = None

        self.selected_block = block
        if block:
            block.set_selected(True)
            self.update_block_graphics(block)
        self._update_wire_dimming()

    def clear_selection(self):
        """Clear all selection."""
        if self.selected_block:
            # Don't deselect the debug-locked block
            if self.selected_block is self._debug_locked_block:
                pass
            else:
                self.selected_block.set_selected(False)
                self.update_block_graphics(self.selected_block)
                self.selected_block = None
        if self.selected_wire:
            self.selected_wire.is_selected = False
            if self.selected_wire.id in self._wire_items:
                self._wire_items[self.selected_wire.id].update_path()
            self.selected_wire = None
        if self.selected_annotation:
            self.selected_annotation.is_selected = False
            self._update_annotation_graphics(self.selected_annotation)
            self.selected_annotation = None
        if self.selected_text_label:
            self.selected_text_label.is_selected = False
            self._update_text_label_graphics(self.selected_text_label)
            self.selected_text_label = None
        self._deselect_multi()
        self._update_wire_dimming()

    def delete_selected(self):
        """Delete the currently selected block or wire."""
        if self.selected_block:
            self.remove_block(self.selected_block)
        elif self.selected_wire:
            self.push_undo()
            self.remove_wire(self.selected_wire)
        elif self.selected_annotation and not self.selected_annotation.is_locked:
            self.push_undo()
            self.remove_annotation(self.selected_annotation)
        elif self.selected_text_label:
            self.push_undo()
            self.remove_text_label(self.selected_text_label)

    # --- Mouse events ---

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        canvas_pt = (scene_pos.x(), -scene_pos.y())  # Flip Y

        # Middle-click: start box selection
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_box_selecting = True
            self._box_select_start = canvas_pt
            self._middle_box_select = True
            self._middle_box_shift = bool(
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
            self.setCursor(Qt.CursorShape.CrossCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Close annotate popup if open and not in an annotation sub-mode
            if (
                self._annotate_popup_visible
                and not self._annotation_mode
                and not self._text_mode
            ):
                self._toggle_annotate_popup()

            # Wire label click: center view on the off-screen port
            for item in self._wire_label_items:
                target = getattr(item, "_wire_label_target", None)
                if target and item.contains(item.mapFromScene(scene_pos)):
                    self.centerOn(target)
                    self._update_wire_labels()
                    return

            # Cmd+click (macOS) / Ctrl+click: edit block
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                for block in reversed(self.blocks):
                    if block.contains_point(canvas_pt):
                        if self.block_edit_callback:
                            self.block_edit_callback(block)
                        return

            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

            # Shift+click on a block: toggle it in/out of multi-selection
            if shift:
                for block in reversed(self.blocks):
                    if block.contains_point(canvas_pt):
                        if block in self._multi_drag_blocks:
                            self._multi_drag_blocks.remove(block)
                            block.set_selected(False)
                        else:
                            self._multi_drag_blocks.append(block)
                            block.set_selected(True)
                        self.update_block_graphics(block)
                        return

            # Multi-select: click on a selected block/annotation to start dragging group
            if self._multi_drag_blocks or self._multi_drag_annotations:
                hit = False
                for block in self._multi_drag_blocks:
                    if block.contains_point(canvas_pt):
                        self._multi_drag_offset = (
                            canvas_pt[0] - block.position[0],
                            canvas_pt[1] - block.position[1],
                        )
                        self._multi_drag_anchor = block
                        hit = True
                        break
                if not hit:
                    for ann in self._multi_drag_annotations:
                        if ann.contains_point(canvas_pt):
                            self._multi_drag_offset = (
                                canvas_pt[0] - ann.position[0],
                                canvas_pt[1] - ann.position[1],
                            )
                            self._multi_drag_anchor = ann
                            hit = True
                            break
                if hit:
                    self._is_dragging_multi = True
                    self._multi_drag_initial = {
                        id(b): b.position for b in self._multi_drag_blocks
                    }
                    for ann in self._multi_drag_annotations:
                        self._multi_drag_initial[id(ann)] = ann.position
                    self.push_undo()
                    return
                # Clicked outside multi-selection: clear it
                self._deselect_multi()

            # Check for port click (wire drawing)
            for block in self.blocks:
                port = block.find_port_at_point(canvas_pt)
                if port:
                    self._is_drawing_wire = True
                    self._wire_start_port = port
                    return

            # Check for block click (drag)
            for block in reversed(self.blocks):
                if block.contains_point(canvas_pt):
                    # Check resize handle
                    if block.is_on_resize_handle(canvas_pt):
                        self._is_resizing = True
                        self._resize_block = block
                        self.select_block(block)
                        return

                    # Check collapse/expand indicator click (header right side)
                    if not block.is_compact and self._is_collapse_click(
                        block, canvas_pt
                    ):
                        block.toggle_collapsed()
                        self.update_block_graphics(block)
                        # Update connected wires
                        for wire in self.wires:
                            if (
                                wire.source_port
                                and wire.source_port.parent_block is block
                            ) or (
                                wire.dest_port and wire.dest_port.parent_block is block
                            ):
                                if wire.id in self._wire_items:
                                    self._wire_items[wire.id].update_path()
                        self._minimap.update()
                        return

                    self._is_dragging = True
                    self._drag_block = block
                    self._drag_offset = (
                        canvas_pt[0] - block.position[0],
                        canvas_pt[1] - block.position[1],
                    )
                    self.select_block(block)
                    self.push_undo()
                    return

            # Check annotation lock icon click (any annotation under cursor)
            for ann in reversed(self.annotations):
                if ann.lock_icon_at(canvas_pt):
                    ann.is_locked = not ann.is_locked
                    self._update_annotation_graphics(ann)
                    return

            # Check annotation move handle (top-center, moves box only)
            if self.selected_annotation and self.selected_annotation.is_selected:
                if (
                    not self.selected_annotation.is_locked
                    and self.selected_annotation.move_handle_at(canvas_pt)
                ):
                    self._is_dragging_annotation = True
                    self._drag_annotation = self.selected_annotation
                    self._drag_ann_offset = (
                        canvas_pt[0] - self.selected_annotation.position[0],
                        canvas_pt[1] - self.selected_annotation.position[1],
                    )
                    self._drag_ann_blocks = []  # empty = move box only
                    self._drag_ann_children = []
                    self.push_undo()
                    return

            # Check annotation corner handles (only on selected)
            if self.selected_annotation and self.selected_annotation.is_selected:
                if not self.selected_annotation.is_locked:
                    corner = self.selected_annotation.corner_handle_at(canvas_pt)
                    if corner:
                        self._start_annotation_resize(corner, canvas_pt)
                        return

            # Waypoint editing on the already-selected wire (handles first,
            # then a segment click inserts a new waypoint -> manual mode).
            if self.selected_wire is not None:
                wp_idx = self.selected_wire.find_waypoint_at_point(
                    canvas_pt, tolerance=8
                )
                if wp_idx >= 0:
                    self.push_undo()
                    self._begin_waypoint_drag(self.selected_wire, wp_idx)
                    return
                if self.selected_wire.contains_point(canvas_pt, tolerance=6):
                    self.push_undo()
                    idx = self.selected_wire.insert_waypoint(canvas_pt)
                    self.selected_wire.route_mode = "manual"
                    self._begin_waypoint_drag(self.selected_wire, idx)
                    if self.selected_wire.id in self._wire_items:
                        self._wire_items[self.selected_wire.id].update_path()
                    return

            # Check for wire click (before annotations so wires inside boxes work)
            for wire in self.wires:
                if wire.contains_point(canvas_pt):
                    self.clear_selection()
                    self.selected_wire = wire
                    wire.is_selected = True
                    if wire.id in self._wire_items:
                        self._wire_items[wire.id].update_path()
                    return

            # Check text label click (drag or select)
            for lbl in reversed(self.text_labels):
                if lbl.contains_point(canvas_pt):
                    self._select_text_label(lbl)
                    self._is_dragging_text_label = True
                    self._drag_text_label = lbl
                    self._drag_text_offset = (
                        canvas_pt[0] - lbl.position[0],
                        canvas_pt[1] - lbl.position[1],
                    )
                    self.push_undo()
                    return

            # Select mode: start box-selecting
            if self._select_mode:
                self._is_box_selecting = True
                self._box_select_start = canvas_pt
                return

            # Check annotation body click (moves box + contained blocks)
            for ann in reversed(self.annotations):
                if ann.contains_point(canvas_pt):
                    if ann.is_locked:
                        continue  # locked: let click fall through to pan
                    self._select_annotation(ann)
                    self._is_dragging_annotation = True
                    self._drag_annotation = ann
                    self._drag_ann_offset = (
                        canvas_pt[0] - ann.position[0],
                        canvas_pt[1] - ann.position[1],
                    )
                    self._drag_ann_blocks = [
                        b for b in self.blocks if ann.contains_block(b)
                    ]
                    self._drag_ann_children = [
                        a for a in self.annotations if ann.contains_annotation(a)
                    ]
                    self.push_undo()
                    return

            # Annotation mode: start drawing (snap to 1/4 grid)
            if self._annotation_mode:
                self._is_drawing_annotation = True
                gs = self.grid_spacing / 4
                self._annotation_draw_start = (
                    round(canvas_pt[0] / gs) * gs,
                    round(canvas_pt[1] / gs) * gs,
                )
                return

            # Text mode: place a text label
            if self._text_mode:
                self._place_text_label(canvas_pt)
                return

            # Empty canvas click - start panning. Deselection is deferred to
            # mouse release so that a drag-pan keeps the current selection;
            # only a click without drag performs the deselect.
            self._is_panning = True
            self._pan_start = event.pos()
            self._pan_press_pos = event.pos()
            self._pan_moved = False

        elif event.button() == Qt.MouseButton.RightButton:
            try:
                # Block context menu (checked first so blocks inside annotations work)
                for block in reversed(self.blocks):
                    if block.contains_point(canvas_pt):
                        self.select_block(block)
                        self._show_block_context_menu(
                            block, event.globalPosition().toPoint()
                        )
                        return

                # Text label context menu
                for lbl in reversed(self.text_labels):
                    if lbl.contains_point(canvas_pt):
                        self._select_text_label(lbl)
                        self._show_text_label_context_menu(
                            lbl, event.globalPosition().toPoint()
                        )
                        return

                # Annotation context menu
                for ann in reversed(self.annotations):
                    if ann.contains_point(canvas_pt):
                        if not ann.is_locked:
                            self._select_annotation(ann)
                        self._show_annotation_context_menu(
                            ann, event.globalPosition().toPoint()
                        )
                        return

                # Wire context menu
                for wire in self.wires:
                    if wire.contains_point(canvas_pt):
                        self.clear_selection()
                        self.selected_wire = wire
                        wire.is_selected = True
                        if wire.id in self._wire_items:
                            self._wire_items[wire.id].update_path()
                        self._show_wire_context_menu(
                            wire, canvas_pt, event.globalPosition().toPoint()
                        )
                        return

                self._last_context_pos = canvas_pt
                self._show_canvas_context_menu(
                    canvas_pt, event.globalPosition().toPoint()
                )
            except Exception:
                import traceback

                traceback.print_exc()

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        canvas_pt = (scene_pos.x(), -scene_pos.y())

        if self._is_dragging and self._drag_block:
            new_x = canvas_pt[0] - self._drag_offset[0]
            new_y = canvas_pt[1] - self._drag_offset[1]
            # Snap to grid (1/8 grid for compact blocks)
            gs = (
                self.grid_spacing / 8
                if self._drag_block.is_compact
                else self.grid_spacing
            )
            new_x = round(new_x / gs) * gs
            new_y = round(new_y / gs) * gs
            self._drag_block.position = (new_x, new_y)
            self.update_block_graphics(self._drag_block)

            # Update connected wires
            for wire in self.wires:
                if (
                    wire.source_port
                    and wire.source_port.parent_block is self._drag_block
                ) or (
                    wire.dest_port and wire.dest_port.parent_block is self._drag_block
                ):
                    if wire.id in self._wire_items:
                        self._wire_items[wire.id].update_path()

        elif self._is_dragging_waypoint and self._drag_wire is not None:
            wps = self._drag_wire.waypoints
            if 0 <= self._drag_wp_idx < len(wps):
                wps[self._drag_wp_idx] = canvas_pt
                if self._drag_wire.id in self._wire_items:
                    self._wire_items[self._drag_wire.id].update_path()

        elif self._is_drawing_wire and self._wire_start_port:
            # Draw preview wire
            start_pos = self._wire_start_port.position
            if self._preview_wire_item:
                self._scene.removeItem(self._preview_wire_item)

            pts = WireConnection.compute_bezier_points(start_pos, canvas_pt)
            path = QPainterPath()
            path.moveTo(pts[0][0], -pts[0][1])
            for pt in pts[1:]:
                path.lineTo(pt[0], -pt[1])

            self._preview_wire_item = self._scene.addPath(
                path, QPen(QColor(77, 178, 255), 1.5, Qt.PenStyle.DashLine)
            )

        elif self._is_resizing and self._resize_block:
            block = self._resize_block
            bx, by = block.position
            half_gs = self.grid_spacing / 2
            # Floor: never shrink below the block's default size (and respect
            # port-count height for non-compact blocks). When collapsed, use
            # the *visible* (connected) port count and ignore the expanded
            # default height — otherwise resizing snaps a folded block back
            # to its full expanded size.
            def_w, def_h = getattr(block, "default_size", block.size)
            if block.is_compact:
                min_w, min_h = def_w, def_h
            elif block.is_collapsed:
                n_vis = max(
                    len(block.get_visible_ports("input")),
                    len(block.get_visible_ports("output")),
                    1,
                )
                min_w = def_w
                min_h = max(22 + 35, 22 + 10 + n_vis * 20)
            else:
                max_ports = max(len(block.input_ports), len(block.output_ports))
                port_min_h = 22 + 10 + max_ports * 20
                min_w = def_w
                min_h = max(def_h, port_min_h)
            new_w = max(min_w, round((canvas_pt[0] - bx) / half_gs) * half_gs)
            new_h = max(min_h, round((by - canvas_pt[1]) / half_gs) * half_gs)
            block.size = (new_w, new_h)
            self.update_block_graphics(block)

        elif self._is_dragging_annotation and self._drag_annotation:
            ann = self._drag_annotation
            gs = self.grid_spacing / 4
            new_x = round((canvas_pt[0] - self._drag_ann_offset[0]) / gs) * gs
            new_y = round((canvas_pt[1] - self._drag_ann_offset[1]) / gs) * gs
            dx = new_x - ann.position[0]
            dy = new_y - ann.position[1]
            ann.position = (new_x, new_y)
            # Move contained blocks
            for b in self._drag_ann_blocks:
                b.position = (b.position[0] + dx, b.position[1] + dy)
                self.update_block_graphics(b)
                for wire in self.wires:
                    if (wire.source_port and wire.source_port.parent_block is b) or (
                        wire.dest_port and wire.dest_port.parent_block is b
                    ):
                        if wire.id in self._wire_items:
                            self._wire_items[wire.id].update_path()
            # Move contained child annotations
            for child in self._drag_ann_children:
                child.position = (child.position[0] + dx, child.position[1] + dy)
                self._update_annotation_graphics(child)
                self._sync_parented_labels(child)
            self._update_annotation_graphics(ann)
            self._sync_parented_labels(ann)
            return

        elif self._is_resizing_annotation and self._resize_annotation:
            gs = self.grid_spacing / 4
            fx, fy = self._resize_ann_fixed
            cx = round(canvas_pt[0] / gs) * gs
            cy = round(canvas_pt[1] / gs) * gs
            new_x = min(fx, cx)
            new_y = max(fy, cy)  # Y-up: top is max
            new_w = max(40, abs(cx - fx))
            new_h = max(30, abs(cy - fy))
            self._resize_annotation.position = (new_x, new_y)
            self._resize_annotation.size = (new_w, new_h)
            self._update_annotation_graphics(self._resize_annotation)
            self._sync_parented_labels(self._resize_annotation)
            return

        elif self._is_drawing_annotation and self._annotation_draw_start:
            sx, sy = self._annotation_draw_start
            gs = self.grid_spacing / 4
            cx = round(canvas_pt[0] / gs) * gs
            cy = round(canvas_pt[1] / gs) * gs
            scene_x = min(sx, cx)
            scene_y_top = max(sy, cy)  # canvas Y-up
            w = abs(cx - sx)
            h = abs(cy - sy)
            if self._preview_annotation:
                self._scene.removeItem(self._preview_annotation)
            pen = QPen(QColor(70, 130, 200), 1.5, Qt.PenStyle.DashLine)
            brush = QBrush(QColor(70, 130, 200, 30))
            self._preview_annotation = self._scene.addRect(
                scene_x, -scene_y_top, w, h, pen, brush
            )
            self._preview_annotation.setZValue(-50)
            return

        elif self._is_dragging_multi and (
            self._multi_drag_blocks or self._multi_drag_annotations
        ):
            gs = self.grid_spacing
            anchor = self._multi_drag_anchor
            anchor_init = self._multi_drag_initial[id(anchor)]
            new_ax = round((canvas_pt[0] - self._multi_drag_offset[0]) / gs) * gs
            new_ay = round((canvas_pt[1] - self._multi_drag_offset[1]) / gs) * gs
            dx = new_ax - anchor_init[0]
            dy = new_ay - anchor_init[1]
            for b in self._multi_drag_blocks:
                init_pos = self._multi_drag_initial[id(b)]
                b.position = (init_pos[0] + dx, init_pos[1] + dy)
                self.update_block_graphics(b)
                for wire in self.wires:
                    if (wire.source_port and wire.source_port.parent_block is b) or (
                        wire.dest_port and wire.dest_port.parent_block is b
                    ):
                        if wire.id in self._wire_items:
                            self._wire_items[wire.id].update_path()
            for ann in self._multi_drag_annotations:
                init_pos = self._multi_drag_initial.get(id(ann))
                if init_pos:
                    ann.position = (init_pos[0] + dx, init_pos[1] + dy)
                    self._update_annotation_graphics(ann)
                    self._sync_parented_labels(ann)

        elif self._is_dragging_text_label and self._drag_text_label:
            gs = self.grid_spacing / 4
            lbl = self._drag_text_label
            new_x = round((canvas_pt[0] - self._drag_text_offset[0]) / gs) * gs
            new_y = round((canvas_pt[1] - self._drag_text_offset[1]) / gs) * gs
            lbl.position = (new_x, new_y)
            # While actively dragged, label detaches from its parent annotation
            # (re-parenting is resolved on mouse release).
            if lbl.parent_annotation_id is not None:
                lbl.parent_annotation_id = None
                lbl.relative_position = None
            self._update_text_label_graphics(lbl)

        elif self._is_box_selecting and self._box_select_start:
            sx, sy = self._box_select_start
            cx, cy = canvas_pt
            scene_x = min(sx, cx)
            scene_y_top = max(sy, cy)
            w = abs(cx - sx)
            h = abs(cy - sy)
            if self._preview_box_select:
                self._scene.removeItem(self._preview_box_select)
            pen = QPen(QColor(51, 128, 255), 1.5, Qt.PenStyle.DashLine)
            brush = QBrush(QColor(51, 128, 255, 25))
            self._preview_box_select = self._scene.addRect(
                scene_x, -scene_y_top, w, h, pen, brush
            )
            self._preview_box_select.setZValue(100)
            return

        elif self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            press = getattr(self, "_pan_press_pos", None)
            if press is not None:
                total = event.pos() - press
                if abs(total.x()) + abs(total.y()) > 3:
                    self._pan_moved = True

        self._last_mouse_pos = canvas_pt
        self._update_hover(canvas_pt)

    def _update_hover(self, canvas_pt):
        """Update hover highlights for blocks, ports, and wires."""
        # Skip hover updates during active interactions
        if (
            self._is_dragging
            or self._is_drawing_wire
            or self._is_panning
            or self._is_resizing
            or self._is_dragging_annotation
            or self._is_resizing_annotation
            or self._is_box_selecting
            or self._is_dragging_multi
            or self._is_dragging_text_label
        ):
            return

        new_port = None
        new_block = None
        new_wire = None

        # Check ports first (highest priority)
        for block in self.blocks:
            port = block.find_port_at_point(canvas_pt)
            if port:
                new_port = port
                new_block = block
                break

        # Then blocks
        if not new_port:
            for block in reversed(self.blocks):
                if block.contains_point(canvas_pt):
                    new_block = block
                    break

        # Then wires
        if not new_port and not new_block:
            for wire in self.wires:
                if wire.contains_point(canvas_pt):
                    new_wire = wire
                    break

        # Update cursor
        if new_port:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif new_block and new_block.is_on_resize_handle(canvas_pt):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif new_block:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif new_wire:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        # Update hover highlight on blocks
        changed = False
        if new_block is not self._hover_block:
            old = self._hover_block
            self._hover_block = new_block
            if old and old.id in self._block_items:
                self._block_items[old.id].setOpacity(1.0)
            if new_block and new_block.id in self._block_items:
                self._block_items[new_block.id].setOpacity(0.85)
            changed = True

        # Update hover highlight on wires
        if new_wire is not self._hover_wire:
            old_wire = self._hover_wire
            self._hover_wire = new_wire
            if old_wire and old_wire.id in self._wire_items:
                self._wire_items[old_wire.id].setPen(
                    self._wire_items[old_wire.id].pen()
                )  # force redraw
                self._wire_items[old_wire.id].update_path()
            if new_wire and new_wire.id in self._wire_items:
                self._wire_items[new_wire.id].update_path()

        self._hover_port = new_port
        self._update_wire_dimming()

    def _update_wire_dimming(self):
        """Dim wires not connected to the focused (hovered or selected) block."""
        # Determine the focus block(s)
        focus_blocks = set()
        if self._hover_block:
            focus_blocks.add(self._hover_block)
        if self.selected_block:
            focus_blocks.add(self.selected_block)
        for b in self._multi_drag_blocks:
            focus_blocks.add(b)

        if not focus_blocks:
            # No focus — restore all wires to full opacity
            for item in self._wire_items.values():
                item.setOpacity(1.0)
            return

        for wire in self.wires:
            wid = wire.id
            if wid not in self._wire_items:
                continue
            src_block = wire.source_port.parent_block if wire.source_port else None
            dst_block = wire.dest_port.parent_block if wire.dest_port else None
            connected = src_block in focus_blocks or dst_block in focus_blocks
            self._wire_items[wid].setOpacity(1.0 if connected else 0.12)

    def _update_wire_labels(self):
        """Show floating tags pointing to off-screen Start blocks and
        annotation rectangles.  Click a tag to center the view on its target.
        (Method name is preserved so resize/scroll callbacks keep working.)"""
        for item in self._wire_label_items:
            self._scene.removeItem(item)
        self._wire_label_items.clear()
        self._placed_label_rects = []

        vp_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if vp_rect.width() <= 0 or vp_rect.height() <= 0:
            return

        targets = []  # list of (label_text, target_scene_pt)

        # --- Start blocks ---
        for block in getattr(self, "blocks", []) or []:
            if getattr(block, "definition_name", "") != "StartBlock":
                continue
            x, y = block.position
            w, h = block.size
            # Block center in scene coords (canvas Y-up \u2192 scene Y-down)
            center = QPointF(x + w / 2, -(y - h / 2))
            if vp_rect.contains(center):
                continue
            label = block.display_name or "Start"
            targets.append((label, center))

        # --- Annotation rectangles ---
        for ann in getattr(self, "annotations", []) or []:
            x, y = ann.position
            w, h = ann.size
            center = QPointF(x + w / 2, -(y - h / 2))
            ann_rect = QRectF(x, -y, w, h)
            if vp_rect.intersects(ann_rect):
                continue
            text = (ann.text or "").strip().splitlines()[0] if ann.text else ""
            if not text:
                text = "(annotation)"
            elif len(text) > 24:
                text = text[:23] + "\u2026"
            targets.append((text, center))

        if not targets:
            return

        for text, target_pt in targets:
            edge_pt, arrow = self._viewport_edge_point(vp_rect, target_pt)
            self._add_wire_label(
                edge_pt, f"{arrow} {text}", None, target_scene_pt=target_pt
            )

    def _viewport_edge_point(self, vp_rect, target_pt):
        """Compute where a marker should sit on the viewport edge so it
        points toward target_pt, plus a directional arrow glyph."""
        cx = vp_rect.center().x()
        cy = vp_rect.center().y()
        tx, ty = target_pt.x(), target_pt.y()
        dx, dy = tx - cx, ty - cy

        # Directional arrow \u2014 pick the dominant component
        if abs(dx) >= abs(dy):
            arrow = "\u2192" if dx > 0 else "\u2190"
        else:
            arrow = "\u2193" if dy > 0 else "\u2191"

        # Clamp target to viewport rectangle to find the nearest edge point
        margin = 20
        inset = vp_rect.adjusted(margin, margin, -margin, -margin)
        x = max(inset.left(), min(tx, inset.right()))
        y = max(inset.top(), min(ty, inset.bottom()))
        return QPointF(x, y), arrow

    def _add_wire_label(self, scene_pt, text, wire, target_scene_pt=None):
        """Add a small floating label at scene_pt for a wire."""
        # Text item to measure bounds
        label = QGraphicsTextItem(text)
        label.setDefaultTextColor(QColor(theme.palette.text))
        font = QFont(SANS_FONT)
        font.setPixelSize(9)
        label.setFont(font)
        label.setCursor(Qt.CursorShape.PointingHandCursor)
        lw = label.boundingRect().width()
        lh = label.boundingRect().height()
        pad = 2

        # Compute initial position (centered on entry point)
        lx = scene_pt.x() - lw / 2
        ly = scene_pt.y() - lh - 2

        # Clamp to viewport so the label stays fully visible
        vp_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        inset = 4
        min_x = vp_rect.left() + inset
        max_x = vp_rect.right() - lw - 2 * pad - inset
        min_y = vp_rect.top() + inset
        max_y = vp_rect.bottom() - lh - 2 * pad - inset
        lx = max(min_x, min(lx, max_x))
        ly = max(min_y, min(ly, max_y))

        # Avoid overlapping labels already placed this pass: scan candidate
        # positions (vertical first, then horizontal columns) and pick the
        # closest free slot to the desired position.
        placed = getattr(self, "_placed_label_rects", None)
        if placed is None:
            placed = []
            self._placed_label_rects = placed
        gap = 2
        rect_w = lw + 2 * pad
        rect_h = lh + 2 * pad

        def _overlaps(test_lx, test_ly):
            l = test_lx - pad
            t = test_ly - pad
            r = l + rect_w
            b = t + rect_h
            for pr in placed:
                if (
                    l < pr.right() + gap
                    and r + gap > pr.left()
                    and t < pr.bottom() + gap
                    and b + gap > pr.top()
                ):
                    return True
            return False

        if _overlaps(lx, ly):
            orig_ly = ly
            orig_lx = lx
            y_step = rect_h + gap
            x_step = rect_w + gap
            found = False
            # Try shifting along Y in the current column, alternating down/up
            for k in range(1, 96):
                for cand_y in (orig_ly + k * y_step, orig_ly - k * y_step):
                    if min_y <= cand_y <= max_y and not _overlaps(orig_lx, cand_y):
                        ly = cand_y
                        found = True
                        break
                if found:
                    break
            # Fall back to shifting to an adjacent column at the desired Y
            if not found:
                for k in range(1, 32):
                    for cand_x in (orig_lx + k * x_step, orig_lx - k * x_step):
                        if not (min_x <= cand_x <= max_x):
                            continue
                        cand_y = max(min_y, min(orig_ly, max_y))
                        if not _overlaps(cand_x, cand_y):
                            lx = cand_x
                            ly = cand_y
                            found = True
                            break
                    if found:
                        break

        # Background pill — semi-opaque so it sits over wires without hiding
        # what's behind. Colour per theme so it reads on light or dark canvas.
        bg_rect = QRectF(lx - pad, ly - pad, lw + 2 * pad, lh + 2 * pad)
        placed.append(bg_rect)
        bg = QGraphicsPathItem(
            _make_rounded_rect_path(
                bg_rect.x(), bg_rect.y(), bg_rect.width(), bg_rect.height(), 4
            )
        )
        if get_theme_name() == "dark":
            bg_fill = QColor(theme.palette.panel_bg)
            bg_fill.setAlpha(220)
            bg_pen = QColor(theme.palette.border)
        else:
            bg_fill = QColor(255, 255, 255, 210)
            bg_pen = QColor(180, 180, 180)
        bg.setBrush(QBrush(bg_fill))
        bg.setPen(QPen(bg_pen, 0.5))
        bg.setZValue(49)
        self._scene.addItem(bg)
        self._wire_label_items.append(bg)

        label.setPos(lx, ly)
        label.setZValue(50)
        self._scene.addItem(label)
        self._wire_label_items.append(label)

        # Store target for click-to-navigate
        if target_scene_pt:
            bg._wire_label_target = target_scene_pt
            label._wire_label_target = target_scene_pt

    def mouseReleaseEvent(self, event):
        # Middle-click box select release
        if event.button() == Qt.MouseButton.MiddleButton and getattr(
            self, "_middle_box_select", False
        ):
            if self._preview_box_select:
                self._scene.removeItem(self._preview_box_select)
                self._preview_box_select = None
            if self._box_select_start:
                scene_pos = self.mapToScene(event.pos())
                canvas_pt = (scene_pos.x(), -scene_pos.y())
                sx, sy = self._box_select_start
                ex, ey = canvas_pt
                rx_min, rx_max = min(sx, ex), max(sx, ex)
                ry_min, ry_max = min(sy, ey), max(sy, ey)
                shift = getattr(self, "_middle_box_shift", False)
                if not shift:
                    self._deselect_multi()
                for block in self.blocks:
                    bx, by = block.position
                    bw, bh = block.size
                    if (
                        bx >= rx_min
                        and bx + bw <= rx_max
                        and by - bh >= ry_min
                        and by <= ry_max
                    ):
                        if shift and block in self._multi_drag_blocks:
                            self._multi_drag_blocks.remove(block)
                            block.set_selected(False)
                        elif block not in self._multi_drag_blocks:
                            self._multi_drag_blocks.append(block)
                            block.set_selected(True)
                        self.update_block_graphics(block)
                for ann in self.annotations:
                    if ann.is_locked:
                        continue
                    ax, ay = ann.position
                    aw, ah = ann.size
                    if (
                        ax >= rx_min
                        and ax + aw <= rx_max
                        and ay - ah >= ry_min
                        and ay <= ry_max
                    ):
                        if shift and ann in self._multi_drag_annotations:
                            self._multi_drag_annotations.remove(ann)
                            ann.is_selected = False
                        elif ann not in self._multi_drag_annotations:
                            self._multi_drag_annotations.append(ann)
                            ann.is_selected = True
                        self._update_annotation_graphics(ann)
            self._is_box_selecting = False
            self._box_select_start = None
            self._middle_box_select = False
            self._middle_box_shift = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_drawing_wire and self._wire_start_port:
                # Check if released on a compatible port
                scene_pos = self.mapToScene(event.pos())
                canvas_pt = (scene_pos.x(), -scene_pos.y())

                for block in self.blocks:
                    port = block.find_port_at_point(canvas_pt)
                    if port and port is not self._wire_start_port:
                        if self._wire_start_port.is_compatible_with(port):
                            # Determine source/dest
                            if self._wire_start_port.direction == "output":
                                src, dst = self._wire_start_port, port
                            else:
                                src, dst = port, self._wire_start_port

                            # Check if already connected
                            already = False
                            for w in self.wires:
                                if w.source_port is src and w.dest_port is dst:
                                    already = True
                                    break

                            # Input ports can only have one connection
                            if (
                                not already
                                and dst.direction == "input"
                                and dst.is_connected
                            ):
                                # Remove existing connection silently (suppress dynamic port rebuild)
                                self._replacing_wire = True
                                for w in list(self.wires):
                                    if w.dest_port is dst:
                                        self.remove_wire(w)
                                        break
                                self._replacing_wire = False

                            if not already:
                                self.add_wire(src, dst)

                # Remove preview
                if self._preview_wire_item:
                    self._scene.removeItem(self._preview_wire_item)
                    self._preview_wire_item = None

                self._is_drawing_wire = False
                self._wire_start_port = None

            if self._is_drawing_annotation:
                if self._preview_annotation:
                    self._scene.removeItem(self._preview_annotation)
                    self._preview_annotation = None
                if self._annotation_draw_start:
                    scene_pos = self.mapToScene(event.pos())
                    gs = self.grid_spacing / 4
                    ex = round(scene_pos.x() / gs) * gs
                    ey = round((-scene_pos.y()) / gs) * gs
                    sx, sy = self._annotation_draw_start
                    w = abs(ex - sx)
                    h = abs(ey - sy)
                    if w > 20 and h > 20:
                        pos = (min(sx, ex), max(sy, ey))
                        text, ok = QInputDialog.getText(self, "Annotation", "Label:")
                        ann = AnnotationRect(pos, (w, h))
                        if ok and text:
                            ann.text = text
                        self.annotations.append(ann)
                        item = AnnotationGraphicsItem(ann, self)
                        self._scene.addItem(item)
                        self._annotation_items[ann.id] = item
                        self._select_annotation(ann)
                        self.push_undo()
                self._is_drawing_annotation = False
                self._annotation_draw_start = None
                self._annotation_mode = False
                self._annotation_btn.setChecked(False)
                self._annotation_btn.setStyleSheet(self._ann_btn_style_normal)
                self.setCursor(Qt.CursorShape.ArrowCursor)

            if self._is_dragging_text_label:
                dropped_lbl = self._drag_text_label
                self._is_dragging_text_label = False
                self._drag_text_label = None
                if dropped_lbl is not None:
                    self._snap_text_label_after_drag(dropped_lbl)
                self._minimap.update()

            if self._is_dragging_annotation:
                self._is_dragging_annotation = False
                self._drag_annotation = None
                self._drag_ann_blocks = []
                self._drag_ann_children = []
                self._minimap.update()

            if self._is_resizing_annotation:
                self._is_resizing_annotation = False
                self._resize_annotation = None
                self._resize_corner = None
                self._resize_ann_fixed = None
                self._minimap.update()

            if self._is_box_selecting:
                if self._preview_box_select:
                    self._scene.removeItem(self._preview_box_select)
                    self._preview_box_select = None
                if self._box_select_start:
                    scene_pos = self.mapToScene(event.pos())
                    canvas_pt = (scene_pos.x(), -scene_pos.y())
                    sx, sy = self._box_select_start
                    ex, ey = canvas_pt
                    # Selection rect in canvas coords (Y-up)
                    rx_min = min(sx, ex)
                    rx_max = max(sx, ex)
                    ry_min = min(sy, ey)
                    ry_max = max(sy, ey)
                    # Find blocks and unlocked annotations within rect
                    self._deselect_multi()
                    for block in self.blocks:
                        bx, by = block.position
                        bw, bh = block.size
                        if (
                            bx >= rx_min
                            and bx + bw <= rx_max
                            and by - bh >= ry_min
                            and by <= ry_max
                        ):
                            self._multi_drag_blocks.append(block)
                            block.set_selected(True)
                            self.update_block_graphics(block)
                    for ann in self.annotations:
                        if ann.is_locked:
                            continue
                        ax, ay = ann.position
                        aw, ah = ann.size
                        if (
                            ax >= rx_min
                            and ax + aw <= rx_max
                            and ay - ah >= ry_min
                            and ay <= ry_max
                        ):
                            self._multi_drag_annotations.append(ann)
                            ann.is_selected = True
                            self._update_annotation_graphics(ann)
                self._is_box_selecting = False
                self._box_select_start = None
                self._select_mode = False
                self._select_btn.setChecked(False)
                self._select_btn.setStyleSheet(self._ann_btn_style_normal)
                self.setCursor(Qt.CursorShape.ArrowCursor)

            if self._is_dragging_multi:
                self._is_dragging_multi = False
                self._multi_drag_initial = {}
                self._minimap.update()

            if self._is_panning and not getattr(self, "_pan_moved", False):
                # Pure click on empty canvas: deselect annotations/text labels
                # (block/wire selection is handled elsewhere).
                if self.selected_annotation:
                    ann = self.selected_annotation
                    ann.is_selected = False
                    self.selected_annotation = None
                    self._update_annotation_graphics(ann)
                if self.selected_text_label:
                    lbl = self.selected_text_label
                    lbl.is_selected = False
                    self.selected_text_label = None
                    self._update_text_label_graphics(lbl)
                if self.selected_block:
                    self.selected_block.set_selected(False)
                    self.update_block_graphics(self.selected_block)
                    self.selected_block = None
                if self.selected_wire:
                    self.selected_wire.is_selected = False
                    if self.selected_wire.id in self._wire_items:
                        self._wire_items[self.selected_wire.id].update_path()
                    self.selected_wire = None

            if self._is_dragging_waypoint:
                self._is_dragging_waypoint = False
                self._drag_wire = None
                self._drag_wp_idx = -1
            self._is_dragging = False
            self._drag_block = None
            self._is_panning = False
            self._pan_moved = False
            self._pan_press_pos = None
            self._is_resizing = False
            self._resize_block = None
            self._update_scene_rect()
            self._minimap.update()

    def mouseDoubleClickEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        canvas_pt = (scene_pos.x(), -scene_pos.y())

        if self.selected_wire is not None:
            wp_idx = self.selected_wire.find_waypoint_at_point(canvas_pt, tolerance=8)
            if wp_idx >= 0:
                self.push_undo()
                self.selected_wire.remove_waypoint(wp_idx)
                if not self.selected_wire.waypoints:
                    self.selected_wire.route_mode = "auto"
                if self.selected_wire.id in self._wire_items:
                    self._wire_items[self.selected_wire.id].update_path()
                return

        for block in reversed(self.blocks):
            if block.contains_point(canvas_pt):
                if self.block_double_click_callback:
                    self.block_double_click_callback(block)
                return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._minimap.reposition()
        self._update_wire_labels()
        self._reposition_annotation_btn()
        self._reposition_zoom_btns()
        if hasattr(self, "resize_callback") and self.resize_callback:
            self.resize_callback()

    def scrollContentsBy(self, dx, dy):
        try:
            super().scrollContentsBy(dx, dy)
            self._minimap.update()
            self._update_wire_labels()
        except (RuntimeError, AttributeError):
            pass  # Guard against calls during Qt/Python shutdown

    # ---------- shared zoom helper ----------

    def _zoom_at(self, factor, view_pt):
        """Multiply the view scale by `factor` keeping `view_pt` (a viewport
        QPoint) anchored under the same scene point. Clamped to 0.33×–3.0×."""
        if factor <= 0 or factor == 1.0:
            return
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        if new_scale < 1 / 3.0 or new_scale > 3.0:
            return
        old_scene_pos = self.mapToScene(view_pt)
        self.scale(factor, factor)
        new_scene_pos = self.mapToScene(view_pt)
        delta = old_scene_pos - new_scene_pos
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + int(delta.x() * self.transform().m11())
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + int(delta.y() * self.transform().m22())
        )

    def _pan_by(self, dx, dy):
        """Pan the view by dx, dy viewport pixels (positive = scroll right /
        down). Trackpad two-finger swipe routes here."""
        if dx:
            h = self.horizontalScrollBar()
            h.setValue(h.value() - int(dx))
        if dy:
            v = self.verticalScrollBar()
            v.setValue(v.value() - int(dy))

    # ---------- wheel / trackpad routing ----------

    def wheelEvent(self, event):
        """Cross-platform wheel routing:
        • trackpad two-finger swipe (pixelDelta non-zero, no modifier) → pan
        • mouse wheel (angleDelta, no pixelDelta) → zoom under cursor
        • Ctrl/Cmd + scroll → zoom under cursor (any source)
        """
        try:
            pixel = event.pixelDelta()
            angle = event.angleDelta()
            mods = event.modifiers()
            zoom_mod = bool(
                mods
                & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
            )

            # Decide: zoom or pan.
            use_pixel_pan = (not pixel.isNull()) and not zoom_mod

            if use_pixel_pan:
                self._pan_by(pixel.x(), pixel.y())
                event.accept()
                return

            # Zoom path. Prefer angleDelta.y() (mouse wheel ticks), else fall
            # back to pixel.y() for trackpad+modifier so Ctrl-pinch still works.
            ay = angle.y() if not angle.isNull() else pixel.y()
            if ay == 0:
                event.ignore()
                return
            # Smaller step for high-resolution wheels (typical tick = 120):
            #   |ay| ≥ 120 → 15%, otherwise scale proportionally.
            step = min(0.30, max(0.02, abs(ay) / 800.0))
            factor = (1.0 + step) if ay > 0 else 1.0 / (1.0 + step)
            self._zoom_at(factor, event.position().toPoint())
            event.accept()
        except Exception:
            pass  # Prevent crash on edge cases

    # ---------- pinch gesture (touchpad zoom) ----------

    def event(self, ev):
        """Intercept pinch gestures on Windows/Linux (`QGestureEvent`) and
        native pinch on macOS (`QNativeGestureEvent`)."""
        t = ev.type()
        if t == QEvent.Type.NativeGesture:
            try:
                gt = ev.gestureType()
                if gt == Qt.NativeGestureType.ZoomNativeGesture:
                    # ev.value() is the incremental zoom delta this frame
                    # (e.g. 0.02 = +2%); convert to a multiplier.
                    factor = 1.0 + float(ev.value())
                    pos = (
                        ev.position()
                        if hasattr(ev, "position")
                        else QPointF(ev.localPos())
                    )
                    self._zoom_at(factor, pos.toPoint())
                    return True
            except Exception:
                pass
        elif t == QEvent.Type.Gesture:
            pinch = ev.gesture(Qt.GestureType.PinchGesture)
            if pinch is not None:
                self._handle_pinch(pinch)
                ev.accept()
                return True
        return super().event(ev)

    def _handle_pinch(self, pinch):
        """Apply a Qt PinchGesture (Windows / Linux touchpad)."""
        try:
            state = pinch.state()
            if state == Qt.GestureState.GestureStarted:
                self._pinch_start_scale = self.transform().m11()
                return
            if pinch.changeFlags() & QPinchGesture.ChangeFlag.ScaleFactorChanged:
                # scaleFactor() is per-update; multiplying incrementally is fine.
                factor = float(pinch.scaleFactor())
                if factor <= 0:
                    return
                # Pinch centerPoint is in widget coords.
                center = pinch.centerPoint().toPoint()
                # Translate from global → viewport coords if needed.
                self._zoom_at(factor, self.mapFromGlobal(self.mapToGlobal(center)))
        except Exception:
            pass

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.delete_callback:
                self.delete_callback()
            else:
                self.delete_selected()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Z:
                self.undo()
            elif event.key() == Qt.Key.Key_Y:
                self.redo()
            elif event.key() == Qt.Key.Key_A:
                self.auto_layout()
            elif event.key() == Qt.Key.Key_Equal:
                if self.transform().m11() * 1.15 <= 3.0:
                    self.scale(1.15, 1.15)
            elif event.key() == Qt.Key.Key_Minus:
                if self.transform().m11() / 1.15 >= 1 / 3.0:
                    self.scale(1 / 1.15, 1 / 1.15)
            elif event.key() == Qt.Key.Key_0:
                self.fit_to_content()
        else:
            super().keyPressEvent(event)

    # --- Context menus ---

    def _is_collapse_click(self, block, canvas_pt):
        """Check if click is on the collapse/expand triangle in the header."""
        bx, by = block.position
        bw, _ = block.size
        header_h = 22
        # Triangle is at right side of header, ~14px from right edge
        tri_x = bx + bw - 14
        tri_y = by - header_h / 2
        return abs(canvas_pt[0] - tri_x) < 10 and abs(canvas_pt[1] - tri_y) < 10

    def _show_block_context_menu(self, block, global_pos):
        """Show context menu for a block."""
        if self.block_context_menu_callback:
            self.block_context_menu_callback(block, global_pos)

    def _show_wire_context_menu(self, wire, canvas_pt, global_pos):
        """Show context menu for a wire."""
        menu = QMenu()

        def _insert_reroute():
            try:
                self.push_undo()
                src_port = wire.source_port
                dst_port = wire.dest_port
                if not src_port or not dst_port:
                    return
                # Remove original wire
                self.remove_wire(wire)
                # Create reroute node at click position
                gs = self.grid_spacing
                rx = round(canvas_pt[0] / gs) * gs
                ry = round(canvas_pt[1] / gs) * gs
                reroute = self.registry.create_block("RerouteNode", (rx - 6, ry + 6))
                self.blocks.append(reroute)
                item = BlockGraphicsItem(reroute, self)
                self._scene.addItem(item)
                self._block_items[reroute.id] = item
                # Wire: source -> reroute input
                in_port = reroute.get_input_port("in")
                w1 = WireConnection(src_port, in_port)
                w1.color = (
                    src_port.parent_block.color
                    if src_port.parent_block
                    else (0.5, 0.5, 0.5)
                )
                self.wires.append(w1)
                wi1 = WireGraphicsItem(w1, self)
                self._scene.addItem(wi1)
                self._wire_items[w1.id] = wi1
                # Wire: reroute output -> dest
                out_port = reroute.get_output_port("out")
                w2 = WireConnection(out_port, dst_port)
                w2.color = w1.color
                self.wires.append(w2)
                wi2 = WireGraphicsItem(w2, self)
                self._scene.addItem(wi2)
                self._wire_items[w2.id] = wi2
                self._update_scene_rect()
            except Exception:
                import traceback

                traceback.print_exc()

        def _delete_wire():
            try:
                self.push_undo()
                self.remove_wire(wire)
            except Exception:
                import traceback

                traceback.print_exc()

        menu.addAction("Auto-route", lambda: self.menu_auto_route_wire(wire))
        menu.addAction(
            "Clear waypoints / Straighten", lambda: self.clear_wire_waypoints(wire)
        )
        menu.addSeparator()
        menu.addAction("Delete Wire", _delete_wire)
        menu.exec(global_pos)

    def _show_canvas_context_menu(self, canvas_pt, global_pos):
        """Show context menu for empty canvas."""
        if self.canvas_context_menu_callback:
            self.canvas_context_menu_callback(canvas_pt, global_pos)

    # --- Undo/Redo ---

    def push_undo(self):
        """Save current state to undo stack."""
        state = self._serialize_state()
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        if self.state_changed_callback:
            self.state_changed_callback()

    def undo(self):
        """Restore previous state."""
        if not self._undo_stack:
            return
        current = self._serialize_state()
        self._redo_stack.append(current)
        state = self._undo_stack.pop()
        self._restore_state(state)

    def redo(self):
        """Restore next state."""
        if not self._redo_stack:
            return
        current = self._serialize_state()
        self._undo_stack.append(current)
        state = self._redo_stack.pop()
        self._restore_state(state)

    def _serialize_state(self):
        """Serialize current canvas state for undo/redo.

        Deep-copied so the snapshot is fully detached from the live blocks:
        `block.to_struct()` returns the live `parameters` dict by reference, so
        without this a later in-place edit (e.g. `_update_dynamic_ports`
        rebinding `inputPortDefs` after a wire change) would silently mutate
        snapshots already sitting in the undo stack.
        """
        state = {
            "blocks": [block.to_struct() for block in self.blocks],
            "wires": [wire.to_struct() for wire in self.wires],
            "annotations": [a.to_struct() for a in self.annotations],
            "textLabels": [t.to_struct() for t in self.text_labels],
        }
        return copy.deepcopy(state)

    def _restore_state(self, state):
        """Restore canvas from serialized state, preserving camera position."""
        # Save current view
        saved_transform = self.transform()
        saved_hscroll = self.horizontalScrollBar().value()
        saved_vscroll = self.verticalScrollBar().value()

        self.clear_selection()
        # Clear current
        for item in list(self._block_items.values()):
            self._scene.removeItem(item)
        for item in list(self._wire_items.values()):
            self._scene.removeItem(item)
        self._block_items.clear()
        self._wire_items.clear()
        self.blocks.clear()
        self.wires.clear()
        for item in list(self._annotation_items.values()):
            self._scene.removeItem(item)
        self._annotation_items.clear()
        self.annotations.clear()

        # Suppress scroll_to_start inside load_from_struct. exact=True so undo/
        # redo restore the precise prior visual state (saved colors included)
        # rather than re-deriving them from the registry like a file load.
        self._restoring_state = True
        self.load_from_struct(state, exact=True)
        self._restoring_state = False

        # Restore camera position
        self.setTransform(saved_transform)
        self.horizontalScrollBar().setValue(saved_hscroll)
        self.verticalScrollBar().setValue(saved_vscroll)

    # --- View control ---

    def fit_to_content(self):
        """Zoom to fit all blocks."""
        if not self.blocks:
            return
        min_x = min(b.position[0] for b in self.blocks) - 50
        max_x = max(b.position[0] + b.size[0] for b in self.blocks) + 50
        min_y = min(b.position[1] - b.size[1] for b in self.blocks) - 50
        max_y = max(b.position[1] for b in self.blocks) + 50

        rect = QRectF(min_x, -max_y, max_x - min_x, max_y - min_y)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def center_on_content(self):
        """Center the view on all blocks without changing zoom."""
        if not self.blocks:
            self.centerOn(0, 0)
            return
        min_x = min(b.position[0] for b in self.blocks)
        max_x = max(b.position[0] + b.size[0] for b in self.blocks)
        min_y = min(b.position[1] - b.size[1] for b in self.blocks)
        max_y = max(b.position[1] for b in self.blocks)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        self.centerOn(cx, -cy)

    def reset_view(self):
        """Reset to default view centered on the Start block (or all content)."""
        self.resetTransform()
        # Find the Start block
        start_block = None
        for blk in self.blocks:
            if (
                blk.definition_name
                and self.registry
                and self.registry.has(blk.definition_name)
            ):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isStart"):
                    start_block = blk
                    break
        if start_block:
            bx, by = start_block.position
            bw, bh = start_block.size
            cx = bx + bw / 2
            cy = -(by - bh / 2)
            self.centerOn(cx, cy)
        else:
            self.center_on_content()

    def get_view_center(self):
        """Get the center of the current view in canvas coordinates."""
        center = self.mapToScene(self.viewport().rect().center())
        return (center.x(), -center.y())

    def scroll_to_block(self, block):
        """Center the view on a specific block."""
        x, y = block.position
        w, h = block.size
        self.centerOn(x + w / 2, -(y - h / 2))

    def ensure_block_visible_right(self, block):
        """If block is not comfortably inside the viewport, pan so it appears centered."""
        x, y = block.position
        w, h = block.size
        # Block center in scene coords
        bx = x + w / 2
        by = -(y - h / 2)

        # Current visible area in scene coords, shrunk by 15% margin
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        margin_x = visible.width() * 0.15
        margin_y = visible.height() * 0.15
        inner = visible.adjusted(margin_x, margin_y, -margin_x, -margin_y)

        if inner.contains(QPointF(bx, by)):
            return  # Comfortably visible

        self.centerOn(bx, by)

    def scroll_to_start(self):
        """Scroll so the Start block appears on the left side of the viewport.

        Uses a deferred call to ensure the viewport has its final size.
        """
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._do_scroll_to_start)

    def _do_scroll_to_start(self):
        """Deferred scroll: position Start block ~15% from the left edge."""
        target = None
        for block in self.blocks:
            if block.definition_name == "StartBlock":
                target = block
                break
        if target is None and self.blocks:
            target = self.blocks[0]
        if target is None:
            return

        x, y = target.position
        w, h = target.size
        # Block center in scene coords
        bx = x + w / 2
        by = -(y - h / 2)

        vw = self.viewport().width()
        scale = self.transform().m11() if self.transform().m11() != 0 else 1.0
        # Shift viewport center to the right so the block sits ~15% from left
        offset_x = (vw * 0.35) / scale
        self.centerOn(bx + offset_x, by)

    # --- Auto layout ---

    def auto_layout(self):
        """Automatically arrange blocks in a left-to-right flow."""
        if not self.blocks:
            return

        self.push_undo()

        # Build adjacency for topological sort
        block_map = {b.id: b for b in self.blocks}
        in_degree = {b.id: 0 for b in self.blocks}
        adj = {b.id: [] for b in self.blocks}

        for wire in self.wires:
            if wire.source_port and wire.dest_port:
                src_id = wire.source_port.parent_block.id
                dst_id = wire.dest_port.parent_block.id
                if src_id in adj and dst_id in in_degree:
                    adj[src_id].append(dst_id)
                    in_degree[dst_id] += 1

        # BFS topological sort into layers
        queue = [bid for bid, deg in in_degree.items() if deg == 0]
        layers = []
        visited = set()

        while queue:
            layers.append(list(queue))
            next_queue = []
            for bid in queue:
                visited.add(bid)
                for nb in adj[bid]:
                    in_degree[nb] -= 1
                    if in_degree[nb] == 0 and nb not in visited:
                        next_queue.append(nb)
            queue = next_queue

        # Add unvisited blocks
        remaining = [b.id for b in self.blocks if b.id not in visited]
        if remaining:
            layers.append(remaining)

        # Position blocks
        x_start = 100
        y_start = 800
        x_gap = 200
        y_gap = 30

        for layer_idx, layer in enumerate(layers):
            x = x_start + layer_idx * x_gap
            for row_idx, bid in enumerate(layer):
                block = block_map[bid]
                y = y_start - row_idx * (block.size[1] + y_gap)
                block.position = (x, y)

        self.redraw()

    # --- Serialization ---

    def to_struct(self):
        """Serialize canvas to dict for JSON export."""
        blocks_data = [b.to_struct() for b in self.blocks]
        wires_data = [w.to_struct() for w in self.wires]
        return {
            "blocks": blocks_data,
            "wires": wires_data,
            "annotations": [a.to_struct() for a in self.annotations],
            "textLabels": [t.to_struct() for t in self.text_labels],
        }

    def load_from_struct(self, data, exact=False):
        """Load canvas from a dict (deserialized JSON).

        `exact=True` (used by undo/redo restore) honors every saved per-block
        color verbatim. File loads keep `exact=False` so known blocks re-derive
        their color from the registry and track the evolving palette scheme.
        """
        self.clear_selection()

        # Clear
        for item in list(self._block_items.values()):
            self._scene.removeItem(item)
        for item in list(self._wire_items.values()):
            self._scene.removeItem(item)
        self._block_items.clear()
        self._wire_items.clear()
        self.blocks.clear()
        self.wires.clear()

        block_map = {}

        # Load blocks
        for bd in data.get("blocks", []):
            def_name = bd.get("definition", "")
            pos = tuple(bd.get("position", [0, 0]))

            if self.registry.has(def_name):
                block = self.registry.create_block(def_name, pos)
            else:
                block = BlockNode(def_name, pos)
                block.display_name = bd.get("displayName", def_name)

            block.id = bd.get("id", block.id)
            if "blockId" in bd:
                block.block_id = bd["blockId"]
            if "size" in bd and not block.is_compact:
                block.size = tuple(bd["size"])
            # Prefer the registry-assigned (category-based) color so the
            # palette and the canvas stay in sync as the color scheme
            # evolves. Exceptions: unknown defs (use saved) and super
            # blocks (user-customizable per instance; saved wins).
            is_sub = def_name.startswith("SubPipeline_") or (
                self.registry.has(def_name)
                and self.registry.get(def_name).get("isSubPipeline")
            )
            if "color" in bd and (exact or not self.registry.has(def_name) or is_sub):
                block.color = tuple(bd["color"])
            if "gradientEnd" in bd:
                block.gradient_end = tuple(bd["gradientEnd"])
            if "parameters" in bd:
                block.parameters.update(bd["parameters"])
            if bd.get("isCustomName"):
                block.is_custom_name = True
                block.display_name = bd.get("displayName", block.display_name)
            if bd.get("isCollapsed"):
                block.is_collapsed = True

            # Restore dynamic ports from parameters. The code definition is
            # authoritative for label/case/type of its own (static) ports — so
            # defn changes (e.g. displayName, preserveCase, wire type)
            # propagate to already-saved workflows. Truly dynamic ports (names
            # not in the defn) keep their saved label, case and type.
            if "inputPortDefs" in block.parameters:
                defn_in = {p.name: p for p in block.input_ports}
                block.input_ports = []
                for i, pd in enumerate(block.parameters["inputPortDefs"]):
                    if isinstance(pd, dict):
                        name = pd.get("name", "")
                        p = Port(
                            name=name,
                            direction="input",
                            type_=pd.get("type", "any"),
                            required=pd.get("required", False),
                            description=pd.get("description", ""),
                        )
                        dpn = defn_in.get(name)
                        if dpn is not None:
                            p.display_name = dpn.display_name
                            p.preserve_case = dpn.preserve_case
                            p.type = dpn.type
                        else:
                            p.display_name = pd.get("displayName", p.name)
                            p.preserve_case = pd.get(
                                "preserveCase", name not in FRAMEWORK_PORT_NAMES
                            )
                        p.parent_block = block
                        p.index = i + 1
                        block.input_ports.append(p)

            if "outputPortDefs" in block.parameters:
                defn_out = {p.name: p for p in block.output_ports}
                block.output_ports = []
                seen_names = set()
                has_dupes = False
                for pd in block.parameters["outputPortDefs"]:
                    if isinstance(pd, dict):
                        n = pd.get("name", "")
                        if n in seen_names:
                            has_dupes = True
                            break
                        seen_names.add(n)
                # Detect the port name prefix used (out, ref, etc.)
                prefixes = set()
                for pd in block.parameters["outputPortDefs"]:
                    if isinstance(pd, dict):
                        n = pd.get("name", "")
                        prefix = n.rstrip("0123456789")
                        if prefix and prefix not in ("Run",):
                            prefixes.add(prefix)
                port_prefix = prefixes.pop() if len(prefixes) == 1 else "out"
                seq = 0
                for i, pd in enumerate(block.parameters["outputPortDefs"]):
                    if isinstance(pd, dict):
                        old_name = pd.get("name", "")
                        if has_dupes and old_name != "Run":
                            seq += 1
                            new_name = f"{port_prefix}{seq}"
                        else:
                            new_name = old_name
                        p = Port(
                            name=new_name,
                            direction="output",
                            type_=pd.get("type", "any"),
                            required=True,
                            description=pd.get("description", ""),
                        )
                        dpn = defn_out.get(old_name)
                        if dpn is not None:
                            p.display_name = dpn.display_name
                            p.preserve_case = dpn.preserve_case
                            p.type = dpn.type
                        else:
                            p.display_name = pd.get("displayName", p.name)
                            p.preserve_case = pd.get(
                                "preserveCase", new_name not in FRAMEWORK_PORT_NAMES
                            )
                        p.parent_block = block
                        p.index = i + 1
                        block.output_ports.append(p)

            # Clear stale displayText on blocks that already have output ports,
            # but keep it for blocks whose definition includes displayText
            # (e.g. filter blocks, text blocks that show user-configured text).
            if block.output_ports and "displayText" in block.parameters:
                defn = (
                    self.registry.get(def_name) if self.registry.has(def_name) else {}
                )
                defn_defaults = defn.get("defaultParameters", {})
                if "displayText" not in defn_defaults and not defn.get(
                    "parameterDefinitions"
                ):
                    block.parameters.pop("displayText")

            # Style attrs aren't serialized per-block; carry the canvas's
            # current `base_*` values so the look survives load/undo.
            block.font_size = self.base_font_size
            block.port_font_size = self.base_port_font_size
            block.node_shape = self.base_node_shape

            self.blocks.append(block)
            block_map[block.id] = block

            item = BlockGraphicsItem(block, self)
            self._scene.addItem(item)
            self._block_items[block.id] = item

        # Build description-based lookup for blocks whose output ports
        # were renumbered due to duplicate names (backward compat).
        # Maps block_id → {description → port} for unambiguous matching.
        _desc_port_map = {}
        for blk in self.blocks:
            if "outputPortDefs" in blk.parameters:
                seen = set()
                has_dupes = False
                for pd in blk.parameters["outputPortDefs"]:
                    if isinstance(pd, dict):
                        n = pd.get("name", "")
                        if n in seen:
                            has_dupes = True
                            break
                        seen.add(n)
                if has_dupes:
                    _desc_port_map[blk.id] = {
                        p.description: p for p in blk.output_ports if p.description
                    }

        # Assign block IDs to blocks loaded from older layouts (backward compat)
        max_id = 0
        for blk in self.blocks:
            if blk.block_id is not None and blk.block_id > max_id:
                max_id = blk.block_id
        for blk in self.blocks:
            if blk.block_id is None:
                max_id += 1
                blk.block_id = max_id
        self._next_block_id = max_id + 1

        # Load wires
        for wd in data.get("wires", []):
            src_block_id = wd.get("sourceBlock")
            src_port_name = wd.get("sourcePort")
            dst_block_id = wd.get("destBlock")
            dst_port_name = wd.get("destPort")

            if src_block_id in block_map and dst_block_id in block_map:
                src_block = block_map[src_block_id]
                dst_block = block_map[dst_block_id]
                dst_port = dst_block.get_input_port(dst_port_name)

                # For blocks with renumbered output ports (had duplicate names),
                # match by description against the destination port display name
                src_port = None
                if src_block_id in _desc_port_map and dst_port:
                    desc_map = _desc_port_map[src_block_id]
                    src_port = desc_map.get(dst_port.display_name)
                if src_port is None:
                    src_port = src_block.get_output_port(src_port_name)

                if src_port and dst_port:
                    wire = WireConnection(src_port, dst_port)
                    if "id" in wd:
                        wire.id = wd["id"]
                    if "waypoints" in wd and wd["waypoints"]:
                        wire.waypoints = [tuple(wp) for wp in wd["waypoints"]]
                    wire.route_mode = wd.get("routeMode", "auto")
                    wire.color = src_block.color

                    self.wires.append(wire)
                    item = WireGraphicsItem(wire, self)
                    self._scene.addItem(item)
                    self._wire_items[wire.id] = item

        # On collapsed blocks, a port's visible_index depends on which other
        # ports are connected. Each WireGraphicsItem.update_path() above ran
        # against the partial connection set at the moment it was created, so
        # earlier wires can end up pointing to stale port y-coords once more
        # wires connect. Refresh every wire path now that all connections are
        # in place. (Manifests as wires snapping into the right positions
        # only after clicking a block to trigger an incremental redraw.)
        for _item in self._wire_items.values():
            _item.update_path()

        # A collapsed block shows only its *connected* ports, but each block's
        # graphics were built (above) before any wires were restored — so
        # collapsed blocks rendered with no ports/labels. Redraw them now that
        # all connections exist. File loads also run _refresh_dynamic_ports
        # afterward, which masked this; undo/redo restore do not, so without
        # this their port labels vanish.
        for blk in self.blocks:
            if blk.is_collapsed and blk.id in self._block_items:
                self.update_block_graphics(blk)

        # Load annotations
        for item in list(self._annotation_items.values()):
            self._scene.removeItem(item)
        self._annotation_items.clear()
        self.annotations = []
        for ad in data.get("annotations", []):
            ann = AnnotationRect.from_struct(ad)
            self.annotations.append(ann)
            item = AnnotationGraphicsItem(ann, self)
            self._scene.addItem(item)
            self._annotation_items[ann.id] = item

        # Load text labels
        for item in list(self._text_label_items.values()):
            self._scene.removeItem(item)
        self._text_label_items.clear()
        self.text_labels = []
        for td in data.get("textLabels", []):
            lbl = TextLabel.from_struct(td)
            self.text_labels.append(lbl)
            item = TextLabelGraphicsItem(lbl, self)
            self._scene.addItem(item)
            self._text_label_items[lbl.id] = item

        # Fit scene to content and center view
        self._update_scene_rect()
        if not getattr(self, "_restoring_state", False):
            self.scroll_to_start()

    # --- Annotation helpers ---

    def _select_annotation(self, ann):
        """Select an annotation rectangle."""
        self.clear_selection()
        self.selected_annotation = ann
        ann.is_selected = True
        self._update_annotation_graphics(ann)

    def _update_annotation_graphics(self, ann):
        """Refresh an annotation's visual representation."""
        if ann.id in self._annotation_items:
            self._scene.removeItem(self._annotation_items[ann.id])
        item = AnnotationGraphicsItem(ann, self)
        self._scene.addItem(item)
        self._annotation_items[ann.id] = item

    def remove_annotation(self, ann):
        """Remove an annotation from the canvas."""
        if ann.id in self._annotation_items:
            self._scene.removeItem(self._annotation_items.pop(ann.id))
        self.annotations = [a for a in self.annotations if a.id != ann.id]
        if self.selected_annotation is ann:
            self.selected_annotation = None
        # Un-parent any labels that referenced this annotation
        for lbl in self.text_labels:
            if lbl.parent_annotation_id == ann.id:
                lbl.parent_annotation_id = None
                lbl.relative_position = None
                self._update_text_label_graphics(lbl)

    def _start_annotation_resize(self, corner, canvas_pt):
        """Begin resizing an annotation from the given corner."""
        ann = self.selected_annotation
        self._is_resizing_annotation = True
        self._resize_annotation = ann
        self._resize_corner = corner
        x, y = ann.position
        w, h = ann.size
        # Fixed corner is diagonal opposite
        opposites = {
            "tl": (x + w, y - h),
            "tr": (x, y - h),
            "bl": (x + w, y),
            "br": (x, y),
        }
        self._resize_ann_fixed = opposites[corner]
        self.push_undo()

    def _show_annotation_context_menu(self, ann, global_pos):
        """Show context menu for an annotation."""
        menu = QMenu(self)
        lock_label = "Unlock" if ann.is_locked else "Lock"
        menu.addAction(
            lock_label, self._safe_action(lambda: self._toggle_annotation_lock(ann))
        )
        menu.addSeparator()
        menu.addAction(
            "Edit Text...", self._safe_action(lambda: self._edit_annotation_text(ann))
        )
        menu.addAction(
            "Outline Color...",
            self._safe_action(lambda: self._edit_annotation_color(ann, "outline")),
        )
        menu.addAction(
            "Fill Color...",
            self._safe_action(lambda: self._edit_annotation_color(ann, "fill")),
        )
        menu.addSeparator()
        delete_action = menu.addAction(
            "Delete", self._safe_action(lambda: self.remove_annotation(ann))
        )
        if ann.is_locked:
            delete_action.setEnabled(False)
        menu.exec(global_pos)

    def _toggle_annotation_lock(self, ann):
        """Toggle lock state of an annotation."""
        ann.is_locked = not ann.is_locked
        if ann.is_locked:
            ann.is_selected = False
            if self.selected_annotation is ann:
                self.selected_annotation = None
        self._update_annotation_graphics(ann)

    def _edit_annotation_text(self, ann):
        """Edit annotation label text."""
        text, ok = QInputDialog.getText(self, "Annotation", "Label:", text=ann.text)
        if ok:
            ann.text = text
            self._update_annotation_graphics(ann)

    def _edit_annotation_color(self, ann, which):
        """Edit annotation outline or fill color."""
        if which == "outline":
            current = QColor(*[int(c * 255) for c in ann.outline_color])
            color = QColorDialog.getColor(current, self, "Outline Color")
            if color.isValid():
                ann.outline_color = (color.redF(), color.greenF(), color.blueF())
        else:
            current = QColor(
                *[int(c * 255) for c in ann.fill_color[:3]],
                int(ann.fill_color[3] * 255) if len(ann.fill_color) > 3 else 64,
            )
            color = QColorDialog.getColor(
                current,
                self,
                "Fill Color",
                QColorDialog.ColorDialogOption.ShowAlphaChannel,
            )
            if color.isValid():
                ann.fill_color = (
                    color.redF(),
                    color.greenF(),
                    color.blueF(),
                    color.alphaF(),
                )
        self._update_annotation_graphics(ann)

    # --- Text label helpers ---

    def _place_text_label(self, canvas_pt):
        """Place a new text label at the given position."""
        text, ok = QInputDialog.getText(self, "Text Label", "Text:")
        if ok and text:
            gs = self.grid_spacing / 4
            x = round(canvas_pt[0] / gs) * gs
            y = round(canvas_pt[1] / gs) * gs
            lbl = TextLabel((x, y), text)
            lbl.font_size = self.base_annotation_font_size
            self.text_labels.append(lbl)
            item = TextLabelGraphicsItem(lbl, self)
            self._scene.addItem(item)
            self._text_label_items[lbl.id] = item
            self._select_text_label(lbl)
            self.push_undo()
        # Exit text mode
        self._text_mode = False
        self._ann_text_btn.setChecked(False)
        self._ann_text_btn.setStyleSheet(self._ann_btn_style_normal)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _select_text_label(self, lbl):
        """Select a text label."""
        # Deselect previous
        if self.selected_text_label:
            self.selected_text_label.is_selected = False
            self._update_text_label_graphics(self.selected_text_label)
        self.selected_text_label = lbl
        lbl.is_selected = True
        self._update_text_label_graphics(lbl)

    def _update_text_label_graphics(self, lbl):
        if lbl.id in self._text_label_items:
            self._scene.removeItem(self._text_label_items[lbl.id])
        item = TextLabelGraphicsItem(lbl, self)
        self._scene.addItem(item)
        self._text_label_items[lbl.id] = item

    def remove_text_label(self, lbl):
        if lbl.id in self._text_label_items:
            self._scene.removeItem(self._text_label_items.pop(lbl.id))
        self.text_labels = [t for t in self.text_labels if t.id != lbl.id]
        if self.selected_text_label is lbl:
            self.selected_text_label = None

    def _safe_action(self, func):
        """Wrap a menu action callback to prevent crashes from unhandled exceptions."""

        def wrapper():
            try:
                func()
            except Exception:
                import traceback

                traceback.print_exc()

        return wrapper

    def _show_text_label_context_menu(self, lbl, global_pos):
        menu = QMenu(self)
        menu.addAction(
            "Edit Text...", self._safe_action(lambda: self._edit_text_label_text(lbl))
        )
        menu.addAction(
            "Font Size...",
            self._safe_action(lambda: self._edit_text_label_font_size(lbl)),
        )
        menu.addAction(
            "Color...", self._safe_action(lambda: self._edit_text_label_color(lbl))
        )
        menu.addSeparator()
        is_parented = bool(lbl.parent_annotation_id) and any(
            a.id == lbl.parent_annotation_id for a in self.annotations
        )
        wl_action = menu.addAction(
            "Width Limit...",
            self._safe_action(lambda: self._edit_text_label_width_limit(lbl)),
        )
        wl_action.setEnabled(not is_parented)
        if is_parented:
            menu.addAction(
                "Unlink from Annotation",
                self._safe_action(lambda: self._unparent_text_label(lbl)),
            )
        menu.addSeparator()
        menu.addAction("Delete", self._safe_action(lambda: self.remove_text_label(lbl)))
        menu.exec(global_pos)

    def _edit_text_label_width_limit(self, lbl):
        current = int(lbl.width_limit) if lbl.width_limit else 0
        val, ok = QInputDialog.getInt(
            self,
            "Width Limit",
            "Wrap width in pixels (0 = no limit):",
            current,
            0,
            4000,
        )
        if ok:
            lbl.width_limit = float(val) if val > 0 else None
            self._update_text_label_graphics(lbl)
            self.push_undo()

    def _unparent_text_label(self, lbl):
        lbl.parent_annotation_id = None
        lbl.relative_position = None
        self._update_text_label_graphics(lbl)
        self.push_undo()

    def _innermost_annotation_containing_point(self, canvas_pt):
        """Return the smallest-area annotation whose body contains canvas_pt."""
        hits = [a for a in self.annotations if a.contains_point(canvas_pt)]
        if not hits:
            return None
        return min(hits, key=lambda a: a.size[0] * a.size[1])

    def _snap_text_label_after_drag(self, lbl):
        """After dropping a dragged label, (un)parent to containing annotation."""
        # Use the label's top-left position as the snap reference
        ann = self._innermost_annotation_containing_point(lbl.position)
        if ann is not None:
            pad = TextLabel.ANNOTATION_PADDING
            ax, ay = ann.position
            aw, ah = ann.size
            # Clamp inside annotation interior so the wrapped text fits
            min_x = ax + pad
            max_x = ax + aw - pad
            min_y = ay - ah + pad
            max_y = ay - pad
            nx = min(max(lbl.position[0], min_x), max(min_x, max_x))
            ny = min(max(lbl.position[1], min_y), max(min_y, max_y))
            lbl.position = (nx, ny)
            lbl.parent_annotation_id = ann.id
            lbl.relative_position = (nx - ax, ay - ny)
            lbl.width_limit = None
        else:
            lbl.parent_annotation_id = None
            lbl.relative_position = None
        self._update_text_label_graphics(lbl)

    def _sync_parented_labels(self, ann):
        """Reposition and redraw labels parented to the given annotation."""
        ax, ay = ann.position
        for lbl in self.text_labels:
            if lbl.parent_annotation_id != ann.id:
                continue
            if lbl.relative_position is not None:
                rx, ry = lbl.relative_position
                lbl.position = (ax + rx, ay - ry)
            self._update_text_label_graphics(lbl)

    def _edit_text_label_text(self, lbl):
        text, ok = QInputDialog.getText(self, "Text Label", "Text:", text=lbl.text)
        if ok and text:
            lbl.text = text
            self._update_text_label_graphics(lbl)

    def _edit_text_label_font_size(self, lbl):
        size, ok = QInputDialog.getInt(
            self, "Font Size", "Size (px):", lbl.font_size, 6, 72
        )
        if ok:
            lbl.font_size = size
            self._update_text_label_graphics(lbl)

    def _edit_text_label_color(self, lbl):
        current = QColor(
            int(lbl.color[0] * 255), int(lbl.color[1] * 255), int(lbl.color[2] * 255)
        )
        color = QColorDialog.getColor(current, self, "Text Color")
        if color.isValid():
            lbl.color = (color.redF(), color.greenF(), color.blueF())
            self._update_text_label_graphics(lbl)
