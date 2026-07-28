"""BlockPalettePanel - LabView-style block palette sidebar with category tree and visual grid."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QListWidget, QListWidgetItem, QStyledItemDelegate, QAbstractItemView,
    QListView, QStyle, QPushButton, QLabel, QLineEdit, QToolButton, QScrollArea,
    QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize, QPoint, QMimeData, QRect
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QDrag, QPen, QBrush, QPixmap,
    QLinearGradient, QIcon, QPolygon,
)
from collections import Counter
import colorsys
import sys as _sys
from theme import theme

_SANS_FONT = "Helvetica Neue" if _sys.platform == "darwin" else "Segoe UI"
# Segoe UI renders larger than Helvetica Neue at the same point size.
_TILE_FONT_SIZE = 9 if _sys.platform == "darwin" else 7
_BADGE_FONT_SIZE = 7 if _sys.platform == "darwin" else 6
_TILE_W = 185  # fits the widest label (~154px) with margin; keeps panel narrow
_TILE_H = 32
_GRID_W = _TILE_W + 6
_GRID_H = _TILE_H + 6

_SEPARATOR_KEY = "__separator__"


class _TreeSeparatorDelegate(QStyledItemDelegate):
    """Draws separator items as a thin horizontal line in the category tree."""

    def paint(self, painter, option, index):
        if index.data(Qt.ItemDataRole.UserRole) == _SEPARATOR_KEY:
            painter.save()
            y = option.rect.center().y()
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawLine(option.rect.left() + 4, y,
                             option.rect.right() - 4, y)
            painter.restore()
            return
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        if index.data(Qt.ItemDataRole.UserRole) == _SEPARATOR_KEY:
            return QSize(0, 7)
        return super().sizeHint(option, index)


def _make_separator_item():
    """Create a non-selectable tree separator item."""
    item = QTreeWidgetItem([""])
    item.setData(0, Qt.ItemDataRole.UserRole, _SEPARATOR_KEY)
    item.setFlags(Qt.ItemFlag.NoItemFlags)
    return item


# Shared constants (duplicated from workflow_app to avoid circular imports)
CATEGORY_ORDER = [
    "DataIO", "Variables", "FlowControl", "Preprocessing", "Filters",
    "Template", "Detection", "FeatureExtraction", "Analysis", "Utility",
    "SubPipeline",
]
CATEGORY_LABELS = {
    "DataIO": "IO", "Variables": "Variables",
    "FlowControl": "Flow Control", "Preprocessing": "Preprocessing",
    "Filters": "Filters",
    "Template": "Template", "Detection": "Detection",
    "FeatureExtraction": "Feature Extraction",
    "Analysis": "Analysis", "Utility": "Utility",
    "SubPipeline": "Super Blocks",
}
_IO_INPUT_KEYWORDS = {"load", "import", "read", "constant", "toggle",
                      "unpack", "filepath", "text"}

BLOCK_MIME_TYPE = "application/x-npsview-block"


# Per-section hue/saturation/value overrides. Tuple format:
#   (hue_0..1, sat_top, sat_bottom, val_top, val_bottom)
# When a section's category is in this map, its blocks use this colour band
# instead of the position-derived rainbow hue. The within-section gradient
# (top → bottom of the band) still indicates block position.
CATEGORY_HUE_OVERRIDES = {
    "Variables":     (0.13,  0.95, 0.80, 0.95, 0.85),  # vibrant yellow (~FFCC00)
    "Preprocessing": (0.611, 0.75, 0.55, 0.80, 0.95),  # blue (~3366CC)
}

# Per-defName fixed colours — these blocks ignore section colouring and
# always render with the listed RGB. Used to preserve historic block
# identities (e.g. Start is universally green).
BLOCK_COLOR_OVERRIDES = {
    "StartBlock": (0.38, 0.77, 0.39),
    "Priority":   (0.20, 0.60, 1.00),  # #3399FF
}


def _make_block_brush(color, rect, def_name):
    """Return a QBrush for a block tile — gradient for super blocks, solid otherwise."""
    r, g, b = color
    qcolor = QColor(int(r * 255), int(g * 255), int(b * 255))
    if def_name and def_name.startswith("SubPipeline_"):
        grad = QLinearGradient(rect.left(), 0, rect.right(), 0)
        # Vibrant complementary shift: rotate hue significantly
        r2 = min(0.95, max(0.1, 1.0 - r * 0.5))
        g2 = min(0.95, max(0.1, g * 0.4 + 0.3))
        b2 = min(0.95, max(0.1, b * 0.5 + 0.4))
        grad.setColorAt(0.0, qcolor)
        grad.setColorAt(1.0, QColor(int(r2 * 255), int(g2 * 255), int(b2 * 255)))
        return QBrush(grad), qcolor
    return QBrush(qcolor), qcolor


def _is_io_input(defn):
    """Check if a DataIO block belongs to the Input sub-category."""
    return any(kw in defn["name"].lower() for kw in _IO_INPUT_KEYWORDS)


class BlockItemDelegate(QStyledItemDelegate):
    """Paints block items as colored rounded rectangles with white text."""

    def paint(self, painter, option, index):
        try:
            color = index.data(Qt.ItemDataRole.UserRole + 1)
            display_name = index.data(Qt.ItemDataRole.DisplayRole)
            if not color or not display_name:
                super().paint(painter, option, index)
                return

            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = option.rect.adjusted(3, 3, -3, -3)
            def_name = index.data(Qt.ItemDataRole.UserRole)

            # Block color (gradient for super blocks)
            brush, qcolor = _make_block_brush(color, rect, def_name)
            painter.setBrush(brush)
            painter.setPen(QPen(qcolor.darker(140), 1))
            painter.drawRoundedRect(rect, 5, 5)

            # Text
            painter.setPen(QColor(255, 255, 255))
            font = QFont(_SANS_FONT, _TILE_FONT_SIZE)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, display_name)

            # Selection highlight
            if option.state & QStyle.StateFlag.State_Selected:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(60, 140, 255), 2))
                painter.drawRoundedRect(rect, 5, 5)

            painter.restore()
        except Exception:
            # Prevent Python exceptions from propagating into Qt C++ paint
            # callbacks, which causes SIGABRT crashes on macOS.
            try:
                painter.restore()
            except Exception:
                pass

    def sizeHint(self, option, index):
        return QSize(_TILE_W, _TILE_H)


class SummaryItemDelegate(QStyledItemDelegate):
    """Paints block tiles with instance-count badges for the Summary tab."""

    def paint(self, painter, option, index):
        try:
            color = index.data(Qt.ItemDataRole.UserRole + 1)
            display_name = index.data(Qt.ItemDataRole.DisplayRole)
            count = index.data(Qt.ItemDataRole.UserRole + 2)
            if not color or not display_name:
                super().paint(painter, option, index)
                return

            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = option.rect.adjusted(3, 3, -3, -3)
            def_name = index.data(Qt.ItemDataRole.UserRole)

            # Block color (gradient for super blocks)
            brush, qcolor = _make_block_brush(color, rect, def_name)
            painter.setBrush(brush)
            painter.setPen(QPen(qcolor.darker(140), 1))
            painter.drawRoundedRect(rect, 5, 5)

            # Block name text
            painter.setPen(QColor(255, 255, 255))
            font = QFont(_SANS_FONT, _TILE_FONT_SIZE)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, display_name)

            # Count badge (top-right corner)
            if count and count > 0:
                badge_text = str(count)
                badge_font = QFont(_SANS_FONT, _BADGE_FONT_SIZE)
                badge_font.setBold(True)
                fm = QFontMetrics(badge_font)
                tw = fm.horizontalAdvance(badge_text)
                bw = max(tw + 6, 16)
                bh = 14
                bx = rect.right() - bw + 2
                by = rect.top() - 2

                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(qcolor.darker(140), 1))
                painter.drawRoundedRect(QRect(bx, by, bw, bh), 7, 7)

                painter.setPen(qcolor.darker(160))
                painter.setFont(badge_font)
                painter.drawText(QRect(bx, by, bw, bh),
                                 Qt.AlignmentFlag.AlignCenter, badge_text)

            # Selection highlight
            if option.state & QStyle.StateFlag.State_Selected:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(60, 140, 255), 2))
                painter.drawRoundedRect(rect, 5, 5)

            painter.restore()
        except Exception:
            try:
                painter.restore()
            except Exception:
                pass

    def sizeHint(self, option, index):
        return QSize(_TILE_W, _TILE_H)


RECENT_REMOVE_MIME = "application/x-npsview-recent-remove"
SUPER_BLOCK_DELETE_MIME = "application/x-npsview-superblock-delete"


class BlockGrid(QListWidget):
    """Visual grid of draggable block tiles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setIconSize(QSize(0, 0))
        self.setGridSize(QSize(_GRID_W, _GRID_H))
        self.setSpacing(2)
        self.setMovement(QListView.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setItemDelegate(BlockItemDelegate())
        self.setWordWrap(True)
        self.apply_theme()

    def apply_theme(self):
        p = theme.palette
        # Section grids paint transparent so the section body's (darker) panel
        # color shows through full-width behind the single-column tiles.
        bg = "transparent" if getattr(self, "_transparent_bg", False) else p.panel_bg
        self.setStyleSheet(
            f"QListWidget {{ background: {bg}; border: none; padding: 6px 0px; }}"
            "QListWidget::item { background: transparent; border: none; }"
            "QListWidget::item:selected { background: transparent; }"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_items()

    def _center_items(self):
        """Center tiles when only 1 column fits; keep tight left packing once
        2+ columns fit so the panel doesn't have to be unnecessarily wide to
        reveal the second column.

        Uses self.width() (stable across setViewportMargins). Only applies
        when the pad actually changes, to avoid recursing through resizeEvent.
        """
        sb = self.verticalScrollBar()
        sb_w = sb.sizeHint().width() if sb is not None and sb.isVisible() else 0
        avail = max(0, self.width() - sb_w)
        spacing = self.spacing()
        effective = _GRID_W + spacing
        cols = max(1, (avail + spacing) // effective)
        if cols >= 2:
            pad = 0
        else:
            pad = max(0, (avail - _GRID_W) // 2)
        if getattr(self, '_last_center_pad', None) == pad:
            return
        self._last_center_pad = pad
        self.setViewportMargins(pad, 0, pad, 0)

    def set_blocks(self, definitions):
        """Populate the grid with block definitions."""
        self.clear()
        for defn in definitions:
            if defn.get("isReroute") or not defn.get("displayName"):
                continue
            item = QListWidgetItem(defn["displayName"])
            item.setData(Qt.ItemDataRole.UserRole, defn["name"])
            item.setData(Qt.ItemDataRole.UserRole + 1, tuple(defn["color"]))
            item.setSizeHint(QSize(_TILE_W, _TILE_H))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            self.addItem(item)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
        def_name = item.data(Qt.ItemDataRole.UserRole)
        color = item.data(Qt.ItemDataRole.UserRole + 1)
        display_name = item.data(Qt.ItemDataRole.DisplayRole)

        mime = QMimeData()
        mime.setData(BLOCK_MIME_TYPE, def_name.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)

        # Render drag pixmap
        pixmap = self._render_block_pixmap(display_name, color, def_name)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        drag.exec(Qt.DropAction.CopyAction)

    def _render_block_pixmap(self, display_name, color, def_name=None):
        """Create a small pixmap that looks like the block tile."""
        w, h = 120, 32
        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRect(0, 0, w - 1, h - 1)
        brush, qcolor = _make_block_brush(color, rect, def_name)
        painter.setBrush(brush)
        painter.setPen(QPen(qcolor.darker(140), 1))
        painter.drawRoundedRect(rect, 5, 5)

        painter.setPen(QColor(255, 255, 255))
        font = QFont(_SANS_FONT, _TILE_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, display_name)

        painter.end()
        return pixmap


class _RemoveDropZone(QLabel):
    """A drop target that appears when dragging from the Recent grid.

    Dragging a block tile onto this zone removes it from the recent list.
    """

    def __init__(self, palette, parent=None):
        super().__init__("Drop here to remove", parent)
        self._palette = palette
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setFixedHeight(36)
        self._set_idle_style()
        self.hide()

    def _set_idle_style(self):
        self.setStyleSheet(
            "QLabel { background: #e8e8e8; color: #888; border: 2px dashed #bbb;"
            " border-radius: 6px; font-size: 11px; }")

    def _set_hover_style(self):
        self.setStyleSheet(
            "QLabel { background: #ffdddd; color: #c00; border: 2px dashed #c00;"
            " border-radius: 6px; font-size: 11px; font-weight: bold; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(RECENT_REMOVE_MIME):
            event.acceptProposedAction()
            self._set_hover_style()

    def dragLeaveEvent(self, event):
        self._set_idle_style()

    def dropEvent(self, event):
        data = event.mimeData().data(RECENT_REMOVE_MIME)
        def_name = bytes(data).decode("utf-8")
        self._palette.remove_recent(def_name)
        self._set_idle_style()
        event.acceptProposedAction()


class RecentBlockGrid(BlockGrid):
    """Block grid that encodes drags with a special MIME type for removal."""

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
        def_name = item.data(Qt.ItemDataRole.UserRole)
        color = item.data(Qt.ItemDataRole.UserRole + 1)
        display_name = item.data(Qt.ItemDataRole.DisplayRole)

        mime = QMimeData()
        # Include both the block MIME (for dropping onto canvas) and the
        # recent-remove MIME (for dropping onto the remove zone).
        mime.setData(BLOCK_MIME_TYPE, def_name.encode("utf-8"))
        mime.setData(RECENT_REMOVE_MIME, def_name.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)
        pixmap = self._render_block_pixmap(display_name, color)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

        # Show remove zone while dragging
        palette = self._get_palette()
        if palette:
            palette._drop_zone.show()

        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

        if palette:
            palette._drop_zone.hide()

    def _get_palette(self):
        w = self.parent()
        while w is not None:
            if isinstance(w, BlockPalettePanel):
                return w
            w = w.parent()
        return None


class SuperBlockGrid(BlockGrid):
    """Block grid for Super Block category — supports delete via drag."""

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
        def_name = item.data(Qt.ItemDataRole.UserRole)
        color = item.data(Qt.ItemDataRole.UserRole + 1)
        display_name = item.data(Qt.ItemDataRole.DisplayRole)

        mime = QMimeData()
        mime.setData(BLOCK_MIME_TYPE, def_name.encode("utf-8"))
        mime.setData(SUPER_BLOCK_DELETE_MIME, def_name.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)
        pixmap = self._render_block_pixmap(display_name, color)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

        palette = self._get_palette()
        if palette:
            palette._sb_delete_zone.show()

        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

        if palette:
            palette._sb_delete_zone.hide()

    def _get_palette(self):
        w = self.parent()
        while w is not None:
            if isinstance(w, BlockPalettePanel):
                return w
            w = w.parent()
        return None


class _SuperBlockDeleteZone(QLabel):
    """Drop target that deletes a Super Block template."""

    delete_requested = None  # set by palette to a callback

    def __init__(self, parent=None):
        super().__init__("Drop here to delete", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setFixedHeight(36)
        self._set_idle_style()
        self.hide()

    def _set_idle_style(self):
        self.setStyleSheet(
            "QLabel { background: #e8e8e8; color: #888; border: 2px dashed #bbb;"
            " border-radius: 6px; font-size: 11px; }")

    def _set_hover_style(self):
        self.setStyleSheet(
            "QLabel { background: #ffdddd; color: #c00; border: 2px dashed #c00;"
            " border-radius: 6px; font-size: 11px; font-weight: bold; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(SUPER_BLOCK_DELETE_MIME):
            event.acceptProposedAction()
            self._set_hover_style()

    def dragLeaveEvent(self, event):
        self._set_idle_style()

    def dropEvent(self, event):
        data = event.mimeData().data(SUPER_BLOCK_DELETE_MIME)
        def_name = bytes(data).decode("utf-8")
        self._set_idle_style()
        event.acceptProposedAction()
        if self.delete_requested:
            self.delete_requested(def_name)


class SummaryBlockGrid(BlockGrid):
    """Block grid for the Summary tab — shows blocks used in the current
    layout with instance-count badges.  Draggable onto the canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(SummaryItemDelegate())

    def set_summary(self, definitions, counts):
        """Populate the grid with block definitions and their counts.

        Args:
            definitions: list of block definition dicts.
            counts: dict mapping defName -> instance count.
        """
        self.clear()
        for defn in definitions:
            if defn.get("isReroute") or not defn.get("displayName"):
                continue
            name = defn["name"]
            item = QListWidgetItem(defn["displayName"])
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setData(Qt.ItemDataRole.UserRole + 1, tuple(defn["color"]))
            item.setData(Qt.ItemDataRole.UserRole + 2, counts.get(name, 0))
            item.setSizeHint(QSize(_TILE_W, _TILE_H))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            self.addItem(item)


def _make_section_grid(grid_cls=None):
    """Create a SummaryBlockGrid (or subclass) configured for inline use inside
    an accordion section: single-column, no internal scrollbar, height
    auto-sized to fit all blocks (outer scroll area handles overflow)."""
    cls = grid_cls or SummaryBlockGrid
    g = cls()
    # Force single column by fixing width to one grid cell plus spacing.
    g.setFixedWidth(_GRID_W + g.spacing() * 2)
    g.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    g.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    g.setFrameShape(QFrame.Shape.NoFrame)
    g._transparent_bg = True
    g.apply_theme()
    return g


def _fit_grid_height(grid):
    """Resize a section grid to fit exactly its current item count
    (single-column assumption). Grid cells are _GRID_H tall and the grid
    stylesheet adds ~6px top/bottom padding; budget a little slack so the
    last tile isn't clipped by a missing scrollbar."""
    n = grid.count()
    if n == 0:
        grid.setFixedHeight(0)
        return
    spacing = grid.spacing()
    h = n * (_GRID_H + spacing) + spacing * 2 + 14
    grid.setFixedHeight(h)


_CHEVRON_PX = 14  # display size of the section-header expand triangle


def _make_chevron_icon(expanded, color, px=_CHEVRON_PX * 2):
    """Render a filled down/right triangle as a QIcon for section headers.

    Drawn at 2x and downscaled by the iconSize so it stays crisp on retina.
    `color` is a QColor; the triangle has no outline."""
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))
    m = px * 0.24
    if expanded:  # ▼
        pts = [QPoint(round(m), round(m * 1.15)),
               QPoint(round(px - m), round(m * 1.15)),
               QPoint(round(px / 2), round(px - m * 1.15))]
    else:         # ▶
        pts = [QPoint(round(m * 1.15), round(m)),
               QPoint(round(m * 1.15), round(px - m)),
               QPoint(round(px - m * 1.15), round(px / 2))]
    p.drawPolygon(QPolygon(pts))
    p.end()
    return QIcon(pm)


class _PaletteSection(QWidget):
    """Collapsible section: clickable header + inline single-column block grid."""

    def __init__(self, title, grid, parent=None):
        super().__init__(parent)
        self._title = title
        self._grid = grid
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # QPushButton honors `text-align: left` in QSS reliably; QToolButton
        # in TextOnly mode tends to center regardless. The expand triangle is
        # a rendered icon (not a text glyph) so its size is independent of the
        # title font.
        self.header = QPushButton(f" {self._title}")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setFlat(True)
        self.header.setFixedHeight(24)
        self.header.setIconSize(QSize(_CHEVRON_PX, _CHEVRON_PX))
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.toggled.connect(self._on_toggled)
        self._refresh_chevron(True)
        v.addWidget(self.header)
        # Full-width body behind the (single-column) grid so the open-section
        # panel color reads edge-to-edge under the header, not just behind the
        # tile column. Collapsing hides the whole body.
        self._body = QWidget()
        self._body.setObjectName("sectionBody")
        self._body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(self._body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        # Center the fixed-width tile column horizontally within the panel.
        bl.addWidget(self._grid, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(self._body)

    @property
    def grid(self):
        return self._grid

    @property
    def title(self):
        return self._title

    def _refresh_chevron(self, expanded):
        """Re-render the triangle icon — white on the accent-highlighted open
        header, primary text color otherwise."""
        p = theme.palette
        color = QColor(p.section_open_text if expanded else p.text)
        self.header.setIcon(_make_chevron_icon(expanded, color))

    def _on_toggled(self, checked):
        self._body.setVisible(checked)
        self._refresh_chevron(checked)

    def set_expanded(self, expanded):
        if self.header.isChecked() != expanded:
            self.header.setChecked(expanded)

    def is_expanded(self):
        return self.header.isChecked()

    def apply_theme(self):
        p = theme.palette
        self.header.setStyleSheet(
            f"QPushButton {{ background: {p.alt_base_bg}; color: {p.text};"
            f" border: none; padding: 0 6px; text-align: left;"
            f" font-size: 11px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover};"
            f" border-radius: 5px; }}"
            f"QPushButton:checked {{ background: {p.section_open_bg};"
            f" color: {p.section_open_text};"
            f" border-top-left-radius: 5px; border-top-right-radius: 5px; }}"
            f"QPushButton:checked:hover {{ background: {p.section_open_bg};"
            f" border-top-left-radius: 5px; border-top-right-radius: 5px;"
            f" border-bottom-left-radius: 0; border-bottom-right-radius: 0; }}"
        )
        self._refresh_chevron(self.header.isChecked())
        # Bottom corners rounded so header (rounded top) + body read as one
        # small-radius card.
        self._body.setStyleSheet(
            f"#sectionBody {{ background: {p.section_open_panel};"
            f" border-bottom-left-radius: 5px;"
            f" border-bottom-right-radius: 5px; }}")
        self._grid.apply_theme()


class BlockPalettePanel(QWidget):
    """Left sidebar — single accordion of collapsible category sections."""

    MAX_RECENT = 10

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self._recent_names = []  # ordered list of def_names, most recent first
        self._menu_config = None  # custom menu config (set by rebuild_from_config)
        self._display_overrides = {}  # defName -> custom displayName
        self._canvas_ref = None   # set externally to the active WorkflowCanvas
        # Min fits a single tile column plus the always-on scrollbar and a
        # small margin; panel width remains user-adjustable via the splitter.
        self.setMinimumWidth(_GRID_W + 28)
        self.setMaximumWidth(500)

        # Sections: ordered list of (key, _PaletteSection). The key is e.g.
        # "__recent__", a (cat, sub) tuple, or a CATEGORY_ORDER string —
        # preserved so existing helpers (`refresh_current_view`,
        # `_refresh_recent_grid`, etc.) keep working.
        self._sections = []
        self._sections_by_key = {}

        # Hidden in-memory tree — drives section generation and lets legacy
        # helpers (rebuild_from_config, …) keep working unchanged. Never
        # shown in the layout.
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.hide()
        self.tree.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Search bar — sits in a padded wrapper so it drops below the toolbar
        # with breathing room on all sides.
        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("Search blocks...")
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.setFixedHeight(26)
        self._search_bar.textChanged.connect(self._on_search_changed)
        search_wrap = QWidget()
        search_wrap_layout = QVBoxLayout(search_wrap)
        search_wrap_layout.setContentsMargins(8, 12, 8, 8)
        search_wrap_layout.setSpacing(0)
        search_wrap_layout.addWidget(self._search_bar)
        layout.addWidget(search_wrap)

        # "Clear All" — visible only when the Recent section is expanded.
        self._clear_btn = QPushButton("Clear Recent")
        self._clear_btn.setFixedHeight(22)
        self._clear_btn.clicked.connect(self._clear_recent)
        self._clear_btn.hide()
        layout.addWidget(self._clear_btn)

        # Single scroll area hosting the accordion of category sections.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._sections_host = QWidget()
        self._sections_layout = QVBoxLayout(self._sections_host)
        self._sections_layout.setContentsMargins(2, 2, 2, 2)
        self._sections_layout.setSpacing(2)
        self._sections_layout.addStretch(1)  # tail spacer
        self._scroll.setWidget(self._sections_host)
        layout.addWidget(self._scroll, 1)

        # Drop zones for the two existing behaviors (kept; positioned at panel
        # bottom so drag-to-remove still works from any section).
        self._drop_zone = _RemoveDropZone(self)
        layout.addWidget(self._drop_zone)
        self._sb_delete_zone = _SuperBlockDeleteZone()
        layout.addWidget(self._sb_delete_zone)

        # Build per-category sections. Aliases below preserve the old
        # `self._recent_grid` / `self.grid` / etc. references so app-level
        # code that pokes those attributes continues to work.
        self._build_sections()
        recent_sec = self._sections_by_key.get("__recent__")
        summary_sec = self._sections_by_key.get("__summary__")
        self._recent_grid = recent_sec.grid if recent_sec else RecentBlockGrid()
        self._summary_grid = summary_sec.grid if summary_sec else SummaryBlockGrid()
        # Generic alias for legacy callers that expect a "current category" grid;
        # accordion shows everything so this is just a sentinel.
        self.grid = self._summary_grid
        # Super-block section's grid (if any) for legacy reference.
        sb_sec = next((s for k, s in self._sections if isinstance(k, tuple)
                       and k[0] == "SubPipeline"), None)
        self._sb_grid = sb_sec.grid if sb_sec else SuperBlockGrid()
        # Toggle Clear-Recent visibility with the Recent section's expansion.
        if recent_sec is not None:
            recent_sec.header.toggled.connect(self._clear_btn.setVisible)
            self._clear_btn.setVisible(recent_sec.is_expanded())

        # Initial fill of all sections.
        self.refresh_current_view()

        self.apply_theme()

    # ----- Accordion -------------------------------------------------------

    def _clear_sections(self):
        """Tear down all existing section widgets (including the QFrame
        separators) while keeping the trailing stretch in the layout."""
        # Remove every widget child except the stretch (which has no widget).
        # `setParent(None)` re-orphans the widget and would briefly show it as
        # a top-level window if the event loop spins before `deleteLater` runs
        # (e.g. when reload_modules is called from run_all, which then issues
        # `QApplication.processEvents`). Hiding first avoids the flash.
        while self._sections_layout.count() > 1:
            item = self._sections_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._sections.clear()
        self._sections_by_key.clear()

    def _section_title_for(self, key):
        """Human-readable header text for a section key."""
        if key == "__recent__":  return "Recent"
        if key == "__summary__": return "Summary"
        if key == "__all__":     return "All"
        if isinstance(key, tuple):
            cat, sub = key
            label = CATEGORY_LABELS.get(cat, cat.replace("_", " "))
            return f"{label} — {sub.capitalize()}" if sub else label
        return str(key).replace("_", " ")

    def _grid_cls_for(self, key):
        """Pick the right grid subclass so existing drag/drop/badge behavior
        is preserved (Recent has remove-on-drag; SubPipeline allows delete)."""
        if key == "__recent__":
            return RecentBlockGrid
        if isinstance(key, tuple) and key[0] == "SubPipeline":
            return SuperBlockGrid
        return SummaryBlockGrid

    def _build_sections(self):
        """Rebuild the accordion: populate the hidden tree, then mirror each
        top-level entry into a _PaletteSection."""
        self.tree.clear()
        self._build_tree()
        self._rebuild_sections_from_tree()

    def _rebuild_sections_from_tree(self):
        """Mirror the (already-populated) hidden tree into accordion sections."""
        self._clear_sections()
        # Recent opens by default (it persists with the workflow file); the
        # rest start collapsed and single-open keeps only one expanded.
        default_open = {"__recent__"}
        # Iterate top-level tree items; flatten DataIO sub-children as their
        # own sections so users see Input/Output side by side.
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            key = it.data(0, Qt.ItemDataRole.UserRole)
            if key == _SEPARATOR_KEY:
                # Keep a thin spacer line between groups for visual rhythm.
                line = QFrame(self._sections_host)
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFixedHeight(1)
                line.setStyleSheet(f"background: {theme.palette.border};")
                self._sections_layout.insertWidget(
                    self._sections_layout.count() - 1, line)
                continue
            if it.childCount() > 0:
                for ci in range(it.childCount()):
                    ch = it.child(ci)
                    ck = ch.data(0, Qt.ItemDataRole.UserRole)
                    self._add_section(ck, default_open=ck in default_open)
            else:
                self._add_section(key, default_open=key in default_open)

    def _add_section(self, key, default_open=False):
        title = self._section_title_for(key)
        # Parent the section + grid to the host up front so they never exist
        # as orphan top-level widgets (which can briefly flash as empty
        # windows during a reload triggered by run_all).
        grid = _make_section_grid(self._grid_cls_for(key))
        grid.setParent(self._sections_host)
        sec = _PaletteSection(title, grid, parent=self._sections_host)
        sec.set_expanded(default_open)
        sec.header.toggled.connect(
            lambda checked, s=sec: self._on_section_toggled(s, checked))
        self._sections.append((key, sec))
        self._sections_by_key[key] = sec
        self._sections_layout.insertWidget(
            self._sections_layout.count() - 1, sec)
        sec.apply_theme()

    def _on_section_toggled(self, sec, checked):
        """Enforce single-open accordion: opening one section collapses the
        rest. Skipped while a search is active (search expands every matching
        section at once)."""
        if not checked:
            return
        if self._search_bar.text().strip():
            return
        for _, other in self._sections:
            if other is not sec and other.is_expanded():
                other.set_expanded(False)

    def _resolve_section_blocks(self, key):
        """Return the list of defns that populate this section, in display
        order. Returns [] if nothing applies. Pure — does not mutate any
        grid or definition."""
        if key == "__recent__":
            return [self.registry.get(n) for n in self._recent_names
                    if self.registry.has(n)]
        if key == "__summary__":
            counts = self._canvas_block_counts()
            return [self.registry.get(n) for n in sorted(counts, key=str.lower)
                    if self.registry.has(n)]
        if key == "__all__":
            if self._menu_config is not None:
                blocks = []
                for entry in self._menu_config:
                    if entry["category"] == "__unused__":
                        continue
                    blocks.extend(entry.get("blocks", []))
                    for sc in entry.get("subcategories", []):
                        blocks.extend(sc["blocks"])
                defs = self._resolve_blocks(blocks)
            else:
                defs = [d for defs_ in self.registry.list_by_category().values()
                        for d in defs_ if not d.get("isReroute")]
            return sorted(defs, key=lambda d: d["displayName"].lower())
        if isinstance(key, tuple) and key[0] == "SubPipeline":
            defs = self.registry.list_by_category().get("SubPipeline", [])
            return sorted(defs, key=lambda d: d["displayName"].lower())
        cat, sub = (key if isinstance(key, tuple) else (key, None))
        config_defs = self._get_config_blocks(cat, sub)
        if config_defs is not None:
            return list(config_defs)
        groups = self.registry.list_by_category()
        defs = groups.get(cat, [])
        if cat == "DataIO":
            if sub == "input":
                defs = [d for d in defs if _is_io_input(d)]
            elif sub == "output":
                defs = [d for d in defs if not _is_io_input(d)]
        return sorted([d for d in defs if not d.get("isReroute")],
                      key=lambda d: d["displayName"].lower())

    def _is_meta_section(self, key):
        """True for sections that should not participate in color assignment
        (they aggregate or duplicate other sections)."""
        if key in ("__all__", "__recent__", "__summary__"):
            return True
        if isinstance(key, tuple) and key[0] == "SubPipeline":
            return True
        return False

    def _apply_section_colors(self):
        """Color block defns by their visible accordion position: each real
        section gets a distinct hue (starting at red for the first section);
        within a section, blocks shade slightly by their order in the list.

        Pure-position based — adding a new block to a section slots it into
        the next shade; the block's name is irrelevant. Mutates the
        registry defns directly so subsequent `_resolve_blocks` calls (which
        copy from the registry) see the new color.

        If the same defName appears in more than one section, the FIRST
        section wins. That gives each block one stable color across the
        palette and keeps the canvas color consistent."""
        real = [(k, sec) for k, sec in self._sections
                if not self._is_meta_section(k)]
        n_cats = max(1, len(real))
        seen = set()
        for ci, (key, sec) in enumerate(real):
            defs = self._resolve_section_blocks(key) or []
            # Strip dupes against already-colored defNames so a stray
            # cross-section entry doesn't recolor an earlier section's
            # block. Also skip super blocks — they keep their per-template
            # color (user-customizable via the dedicated picker).
            fresh = []
            for d in defs:
                name = d.get("name")
                if not name or name in seen:
                    continue
                if (name.startswith("SubPipeline_")
                        or d.get("isSubPipeline")
                        or d.get("category") == "SubPipeline"):
                    continue
                seen.add(name)
                fresh.append(name)
            n = max(1, len(fresh))
            cat = key[0] if isinstance(key, tuple) else key
            override = CATEGORY_HUE_OVERRIDES.get(cat)
            if override:
                hue, sat0, sat1, val0, val1 = override
            else:
                hue = (ci / n_cats) % 1.0
                sat0, sat1, val0, val1 = 0.70, 0.55, 0.55, 0.75
            for bi, name in enumerate(fresh):
                if not self.registry.has(name):
                    continue
                if name in BLOCK_COLOR_OVERRIDES:
                    self.registry.get(name)["color"] = list(
                        BLOCK_COLOR_OVERRIDES[name])
                    continue
                t = bi / max(1, n - 1) if n > 1 else 0.0
                sat = sat0 + (sat1 - sat0) * t
                val = val0 + (val1 - val0) * t
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                self.registry.get(name)["color"] = [r, g, b]

    def _refresh_section(self, key, sec):
        """Populate one section's grid for its key. Reads block defns via
        `_resolve_section_blocks` so color application stays consistent."""
        grid = sec.grid
        defs = self._resolve_section_blocks(key) or []
        if key == "__recent__":
            grid.set_blocks(defs)
        elif isinstance(key, tuple) and key[0] == "SubPipeline":
            grid.set_blocks(defs)
        else:
            grid.set_summary(defs, self._canvas_block_counts())
        _fit_grid_height(grid)

    # ----- Theme -----------------------------------------------------------

    def apply_theme(self):
        """Re-apply theme-driven styles to this panel and all sections."""
        p = theme.palette

        self.setStyleSheet(
            f"BlockPalettePanel {{ background: {p.panel_bg}; }}"
        )

        self._search_bar.setStyleSheet(
            f"QLineEdit {{ border: 1px solid {p.border}; border-radius: 4px;"
            f" padding: 2px 6px; font-size: 11px;"
            f" background: {p.base_bg}; color: {p.text}; }}"
            f"QLineEdit:focus {{ border-color: {p.accent}; }}"
        )

        self._clear_btn.setStyleSheet(
            f"QPushButton {{ background: {p.button_bg}; color: {p.button_text};"
            f" border: none; font-size: 10px; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover}; }}"
        )

        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: {p.panel_bg}; border: none; }}"
            f"QScrollArea > QWidget > QWidget {{ background: {p.panel_bg}; }}"
            + theme.qss.scrollbar
        )

        for _, sec in self._sections:
            sec.apply_theme()

    def _build_tree(self):
        """Build the category tree from the registry."""
        groups = self.registry.list_by_category()

        # "All" node
        all_item = QTreeWidgetItem(["All"])
        all_item.setData(0, Qt.ItemDataRole.UserRole, "__all__")
        self.tree.addTopLevelItem(all_item)

        # "Recent" node
        recent_item = QTreeWidgetItem(["Recent"])
        recent_item.setData(0, Qt.ItemDataRole.UserRole, "__recent__")
        self.tree.addTopLevelItem(recent_item)

        # "Summary" node
        summary_item = QTreeWidgetItem(["Summary"])
        summary_item.setData(0, Qt.ItemDataRole.UserRole, "__summary__")
        self.tree.addTopLevelItem(summary_item)

        # Divider
        self.tree.addTopLevelItem(_make_separator_item())

        used = set()
        for cat in CATEGORY_ORDER:
            if cat not in groups:
                continue
            used.add(cat)
            label = CATEGORY_LABELS.get(cat, cat)

            if cat == "DataIO":
                # IO with Input / Output children
                io_item = QTreeWidgetItem([label])
                io_item.setData(0, Qt.ItemDataRole.UserRole, ("DataIO", None))
                self.tree.addTopLevelItem(io_item)

                input_item = QTreeWidgetItem(["Input"])
                input_item.setData(0, Qt.ItemDataRole.UserRole, ("DataIO", "input"))
                io_item.addChild(input_item)

                output_item = QTreeWidgetItem(["Output"])
                output_item.setData(0, Qt.ItemDataRole.UserRole, ("DataIO", "output"))
                io_item.addChild(output_item)

                io_item.setExpanded(True)
            else:
                cat_item = QTreeWidgetItem([label])
                cat_item.setData(0, Qt.ItemDataRole.UserRole, (cat, None))
                self.tree.addTopLevelItem(cat_item)

        # Extra categories
        for cat in sorted(groups.keys()):
            if cat in used:
                continue
            cat_item = QTreeWidgetItem([cat.replace("_", " ")])
            cat_item.setData(0, Qt.ItemDataRole.UserRole, (cat, None))
            self.tree.addTopLevelItem(cat_item)

    # ------------------------------------------------------------------
    # Rebuild from custom menu config
    # ------------------------------------------------------------------

    def rebuild_from_config(self, config, registry):
        """Rebuild the palette tree and grid from a menu config.

        Args:
            config: list of dicts with keys category, label, blocks.
                    blocks is a list of {"defName": str, "displayName": str}.
                    DataIO entries may have "subcategories" with Input/Output.
            registry: BlockRegistry (for definition lookup).
        """
        self._menu_config = config
        self.registry = registry

        # Build display name overrides
        self._display_overrides.clear()
        for entry in config:
            for blk in entry.get("blocks", []):
                if registry.has(blk["defName"]):
                    orig = registry.get(blk["defName"])["displayName"]
                    if blk["displayName"] != orig:
                        self._display_overrides[blk["defName"]] = blk["displayName"]
            for sc in entry.get("subcategories", []):
                for blk in sc["blocks"]:
                    if registry.has(blk["defName"]):
                        orig = registry.get(blk["defName"])["displayName"]
                        if blk["displayName"] != orig:
                            self._display_overrides[blk["defName"]] = blk["displayName"]

        self.tree.clear()
        self._build_tree_from_config(config)

        # Mirror the new tree into accordion sections, then re-fill all
        # sections with their per-category content. Section grid aliases
        # (`self._recent_grid`, `self._summary_grid`, `self._sb_grid`,
        # `self.grid`) may now point to the OLD grids that were torn down;
        # re-establish them against the freshly-built sections.
        self._rebuild_sections_from_tree()
        recent_sec = self._sections_by_key.get("__recent__")
        summary_sec = self._sections_by_key.get("__summary__")
        if recent_sec is not None:
            self._recent_grid = recent_sec.grid
        if summary_sec is not None:
            self._summary_grid = summary_sec.grid
            self.grid = self._summary_grid
        sb_sec = next((s for k, s in self._sections if isinstance(k, tuple)
                       and k[0] == "SubPipeline"), None)
        if sb_sec is not None:
            self._sb_grid = sb_sec.grid
        self.refresh_current_view()
        self.apply_theme()

    def _build_tree_from_config(self, config):
        """Build tree nodes from a menu config list."""
        # "All" and "Recent" nodes
        all_item = QTreeWidgetItem(["All"])
        all_item.setData(0, Qt.ItemDataRole.UserRole, "__all__")
        self.tree.addTopLevelItem(all_item)

        recent_item = QTreeWidgetItem(["Recent"])
        recent_item.setData(0, Qt.ItemDataRole.UserRole, "__recent__")
        self.tree.addTopLevelItem(recent_item)

        summary_item = QTreeWidgetItem(["Summary"])
        summary_item.setData(0, Qt.ItemDataRole.UserRole, "__summary__")
        self.tree.addTopLevelItem(summary_item)

        # Divider
        self.tree.addTopLevelItem(_make_separator_item())

        for entry in config:
            if entry["category"] == "__unused__":
                continue  # hide Unused from palette tree
            label = entry["label"]
            cat_key = entry["category"]

            cat_item = QTreeWidgetItem([label])
            cat_item.setData(0, Qt.ItemDataRole.UserRole, (cat_key, None))
            self.tree.addTopLevelItem(cat_item)

            subcats = entry.get("subcategories", [])
            if subcats:
                for sc in subcats:
                    sub_item = QTreeWidgetItem([sc["label"]])
                    sub_item.setData(0, Qt.ItemDataRole.UserRole,
                                     (cat_key, sc["label"].lower()))
                    cat_item.addChild(sub_item)
                cat_item.setExpanded(True)

    def _get_display_name(self, defn):
        """Return the display name for a definition, respecting overrides."""
        return self._display_overrides.get(defn["name"], defn["displayName"])

    def _get_config_blocks(self, cat_key, sub=None):
        """Get block definitions for a category from the current config.

        Handles subcategories: when *sub* matches a subcategory label
        (case-insensitive), only that subcategory's blocks are returned.
        When *sub* is None, all blocks (direct + all subcategory) are
        returned.
        """
        if self._menu_config is None:
            return None  # fall back to registry-based lookup

        for entry in self._menu_config:
            if entry["category"] != cat_key:
                continue

            subcats = entry.get("subcategories", [])

            if sub is not None and subcats:
                # Find matching subcategory by label
                for sc in subcats:
                    if sc["label"].lower() == sub.lower():
                        return self._resolve_blocks(sc["blocks"])
                return []

            # No sub filter -> return everything (direct + all subcats)
            all_blocks = list(entry.get("blocks", []))
            for sc in subcats:
                all_blocks.extend(sc["blocks"])
            return self._resolve_blocks(all_blocks)

        return []

    def _resolve_blocks(self, block_list):
        """Convert a list of {"defName", "displayName"} dicts to full defs."""
        defs = []
        for blk in block_list:
            if self.registry.has(blk["defName"]):
                defn = dict(self.registry.get(blk["defName"]))
                defn["displayName"] = blk["displayName"]
                defs.append(defn)
        return defs

    # --- Public helpers ---

    def refresh_current_view(self):
        """Re-render every accordion section (updates instance counts and
        recent / summary lists)."""
        # Step 1: assign per-section colors so every grid below reads the
        # freshly-computed defn colors. Section index drives the hue;
        # position drives the shade.
        self._apply_section_colors()
        # Step 2: push the new defn colors to any canvas block instances
        # so existing workflows update too (otherwise palette and canvas
        # drift apart until the workflow is reloaded).
        if self._canvas_ref is not None and hasattr(self._canvas_ref, "blocks"):
            for blk in self._canvas_ref.blocks:
                if self.registry.has(blk.definition_name):
                    blk.color = tuple(
                        self.registry.get(blk.definition_name)["color"])
                    if hasattr(self._canvas_ref, "update_block_graphics"):
                        self._canvas_ref.update_block_graphics(blk)
        # Step 3: populate grids.
        for key, sec in self._sections:
            self._refresh_section(key, sec)
        # Re-apply any active search filter after content changes.
        q = self._search_bar.text() if hasattr(self, "_search_bar") else ""
        if q:
            self._on_search_changed(q)

    # --- Recent blocks ---

    def record_use(self, def_name):
        """Record a block as recently used (most recent first)."""
        if def_name in self._recent_names:
            self._recent_names.remove(def_name)
        self._recent_names.insert(0, def_name)
        if len(self._recent_names) > self.MAX_RECENT:
            self._recent_names = self._recent_names[:self.MAX_RECENT]
        self._refresh_recent_grid()

    def remove_recent(self, def_name):
        """Remove a single block from the recent list."""
        if def_name in self._recent_names:
            self._recent_names.remove(def_name)
        self._refresh_recent_grid()

    def _clear_recent(self):
        """Clear all recent blocks."""
        self._recent_names.clear()
        self._refresh_recent_grid()

    def _refresh_sb_grid(self):
        """Update the super block section."""
        for key, sec in self._sections:
            if isinstance(key, tuple) and key[0] == "SubPipeline":
                self._refresh_section(key, sec)
                return

    def _refresh_recent_grid(self):
        """Update the Recent section."""
        sec = self._sections_by_key.get("__recent__")
        if sec is not None:
            self._refresh_section("__recent__", sec)

    def _canvas_block_counts(self):
        """Return a Counter of defName -> instance count on the canvas."""
        canvas = self._canvas_ref
        if canvas is None or not hasattr(canvas, "blocks"):
            return Counter()
        return Counter(b.definition_name for b in canvas.blocks
                       if not getattr(b, "is_reroute", False))

    def _on_search_changed(self, text):
        """Filter sections by search text: hide sections with no match and
        force-expand the ones that do match."""
        query = text.strip().lower()
        if not query:
            # Restore: re-show all sections, re-populate, and restore the
            # expansion state captured when the search began — otherwise every
            # section the search force-expanded stays open.
            snapshot = getattr(self, "_pre_search_expansion", None)
            for key, sec in self._sections:
                sec.setVisible(True)
                # Re-populate without filter
                self._refresh_section(key, sec)
                if snapshot is not None and key in snapshot:
                    sec.set_expanded(snapshot[key])
            self._pre_search_expansion = None
            return
        # Snapshot the pre-search accordion state on the first keystroke so
        # clearing the box restores it (search force-expands matching sections).
        if getattr(self, "_pre_search_expansion", None) is None:
            self._pre_search_expansion = {
                k: s.is_expanded() for k, s in self._sections}
        counts = self._canvas_block_counts()
        for key, sec in self._sections:
            # Skip the Recent / Summary / Super Block special sections from
            # the search results (they're orthogonal to category browsing).
            if key in ("__recent__", "__summary__") or (
                    isinstance(key, tuple) and key[0] == "SubPipeline"):
                sec.setVisible(False)
                continue
            # Pull the section's normal block list, then filter.
            self._refresh_section(key, sec)
            matched = []
            for i in range(sec.grid.count()):
                it = sec.grid.item(i)
                name = it.text() or ""
                def_name = it.data(Qt.ItemDataRole.UserRole) or ""
                if query in name.lower() or query in def_name.lower():
                    matched.append(self.registry.get(def_name)
                                   if self.registry.has(def_name) else None)
            matched = [d for d in matched if d is not None]
            sec.grid.set_summary(matched, counts) if hasattr(sec.grid, "set_summary") \
                else sec.grid.set_blocks(matched)
            _fit_grid_height(sec.grid)
            sec.setVisible(bool(matched))
            if matched:
                sec.set_expanded(True)
