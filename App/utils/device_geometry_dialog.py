"""Dialog for defining and visualizing NPS device geometry — drag-and-drop canvas."""

import json
from utils.paths import project_root
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QDoubleSpinBox, QPushButton, QFrame,
    QRadioButton, QSizePolicy, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QMimeData, QRectF, QPointF
from PySide6.QtGui import (
    QDrag, QColor, QPixmap, QPainter, QPen, QBrush,
)
from utils.dialog_style import apply_dialog_style
from theme import theme

# ── Theme ──────────────────────────────────────────────────────────
BG_CANVAS  = "#eef2f7"
TEXT_DIM   = "#888888"
BORDER     = "#cccccc"
NODE_CLR   = "#6AAED6"
PORE_SIZE  = "#3B7AAF"
PORE_CONTR = "#C06040"
PORE_RECOV = "#50A060"
SELECT_CLR = "#E8A030"
ELEC_CLR   = "#D4A017"  # Gold for electrodes
ELEC_SEL   = "#FF8C00"  # Dark orange for selected electrode

PORE_COLORS = {"sizing": PORE_SIZE, "contraction": PORE_CONTR, "recovery": PORE_RECOV}
MIME_TYPE = "application/x-nps-component"


def _comp_color(comp):
    if comp["type"] == "Node":
        return NODE_CLR
    return PORE_COLORS.get(comp.get("property", "sizing"), PORE_SIZE)


# ── Palette chip ───────────────────────────────────────────────────
class _PaletteChip(QWidget):
    def __init__(self, comp_type, color, parent=None):
        super().__init__(parent)
        self.comp_type = comp_type
        self._color = QColor(color)
        self.setFixedSize(80, 30)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(f"Drag to add {comp_type}")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(QPen(self._color.darker(115), 1))
        p.drawRoundedRect(2, 2, self.width() - 4, self.height() - 4, 4, 4)
        p.setPen(QColor("white"))
        f = p.font()
        f.setPixelSize(11)
        f.setBold(True)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.comp_type)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(MIME_TYPE, self.comp_type.encode())
            drag.setMimeData(mime)
            pix = QPixmap(self.size())
            pix.fill(Qt.GlobalColor.transparent)
            self.render(pix)
            drag.setPixmap(pix)
            drag.setHotSpot(event.pos())
            drag.exec(Qt.DropAction.CopyAction)


# ── Floating property editor ───────────────────────────────────────
class _FloatingEditor(QFrame):
    """Small floating panel that appears below the selected component."""

    def __init__(self, parent_dialog):
        super().__init__(parent_dialog.canvas)
        self.dlg = parent_dialog
        p = theme.palette
        self.setStyleSheet(
            f"QFrame {{ background: {p.base_bg}; border: 1px solid {p.border};"
            f" border-radius: 5px; }}"
            "QLabel { border: none; }"
            f"QDoubleSpinBox {{ border: 1px solid {p.border}; border-radius: 3px;"
            f"  padding: 2px 4px; background: {p.base_bg}; font-size: 11px; }}"
            "QRadioButton { border: none; font-size: 10px; }"
        )
        self.setFixedSize(300, 105)
        self.hide()
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(4)
        self.title = QLabel("--")
        self.title.setStyleSheet("color: #4A90C4; font-weight: bold; font-size: 11px;")
        title_row.addWidget(self.title)
        title_row.addStretch()
        btn_del = QPushButton("Remove")
        btn_del.setFixedSize(70, 26)
        btn_del.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_del.setStyleSheet("font-size: 9px; padding: 2px 4px;")
        btn_del.clicked.connect(self._on_delete)
        title_row.addWidget(btn_del)
        layout.addLayout(title_row)

        row = QHBoxLayout()
        row.setSpacing(6)

        def _make_spin(label_text):
            col = QVBoxLayout()
            col.setSpacing(1)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #888; font-size: 9px;")
            col.addWidget(lbl)
            sb = QDoubleSpinBox()
            sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            sb.setRange(0.1, 100000)
            sb.setDecimals(2)
            sb.setFixedWidth(65)
            col.addWidget(sb)
            return col, sb

        cw, self.sp_width = _make_spin("Width (\u00b5m)")
        row.addLayout(cw)
        cl, self.sp_length = _make_spin("Length (\u00b5m)")
        row.addLayout(cl)

        self._prop_group = QFrame()
        _pg = theme.palette
        self._prop_group.setStyleSheet(
            f"QFrame {{ border: 1px solid {_pg.border}; border-radius: 4px;"
            f" background: {_pg.panel_bg}; }}"
            "QRadioButton { border: none; font-size: 10px; }"
        )
        prop_layout = QVBoxLayout(self._prop_group)
        prop_layout.setContentsMargins(6, 4, 6, 4)
        prop_layout.setSpacing(4)
        self._radio_buttons = {}
        for prop in ["sizing", "contraction", "recovery"]:
            rb = QRadioButton(prop)
            rb.toggled.connect(self._on_prop_changed)
            prop_layout.addWidget(rb)
            self._radio_buttons[prop] = rb
        self._radio_buttons["sizing"].setChecked(True)
        row.addWidget(self._prop_group)

        layout.addLayout(row)

        self.sp_width.valueChanged.connect(self._on_changed)
        self.sp_length.valueChanged.connect(self._on_changed)

    def show_for(self, idx, rect):
        """Position below the component rect and populate."""
        c = self.dlg._components[idx]
        self._updating = True
        self.title.setText(f"{c['type']}  #{idx + 1}")
        self.sp_width.setValue(c["width"])
        self.sp_length.setValue(c["length"])
        is_pore = c["type"] == "Pore"
        self._prop_group.setVisible(is_pore)
        if is_pore:
            prop = c.get("property", "sizing")
            self._radio_buttons[prop].setChecked(True)
        # Adjust size based on whether property radios are visible
        self.setFixedSize(300 if is_pore else 210, 105 if is_pore else 80)
        self._updating = False

        # Position: centered below rect, fixed Y at 65% of canvas height
        cx = rect.center().x() - self.width() / 2
        canvas = self.dlg.canvas
        cy = int(canvas.height() * 0.65)
        canvas = self.dlg.canvas
        cx = max(4, min(cx, canvas.width() - self.width() - 4))
        cy = min(cy, canvas.height() - self.height() - 4)
        self.move(int(cx), int(cy))
        self.show()

    def _on_changed(self):
        if self._updating:
            return
        idx = self.dlg._selected_idx
        if 0 <= idx < len(self.dlg._components):
            self.dlg._components[idx]["width"] = self.sp_width.value()
            self.dlg._components[idx]["length"] = self.sp_length.value()
            self.dlg.canvas.update()
            self.dlg._update_output_panel()
            # Reposition after geometry change
            rects = self.dlg.canvas._layout_rects()
            if idx < len(rects):
                self.show_for(idx, rects[idx])

    def _on_prop_changed(self, checked):
        if self._updating or not checked:
            return
        idx = self.dlg._selected_idx
        if 0 <= idx < len(self.dlg._components):
            if self.dlg._components[idx]["type"] == "Pore":
                for prop, rb in self._radio_buttons.items():
                    if rb.isChecked():
                        self.dlg._components[idx]["property"] = prop
                        break
                self.dlg.canvas.update()
                self.dlg._update_output_panel()

    def _on_delete(self):
        self.dlg._delete_selected()


# ── Floating electrode editor ─────────────────────────────────────
class _FloatingElectrodeEditor(QFrame):
    """Small floating panel for selected electrode pair."""

    def __init__(self, parent_dialog):
        super().__init__(parent_dialog.canvas)
        self.dlg = parent_dialog
        p = theme.palette
        self.setStyleSheet(
            f"QFrame {{ background: {p.base_bg}; border: 1px solid {p.border};"
            f" border-radius: 5px; }}"
            "QLabel { border: none; }"
        )
        self.setFixedSize(220, 62)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(4)
        self.title = QLabel("--")
        self.title.setStyleSheet("color: #B8860B; font-weight: bold; font-size: 11px;")
        title_row.addWidget(self.title)
        title_row.addStretch()
        btn_del = QPushButton("Remove")
        btn_del.setFixedSize(70, 26)
        btn_del.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_del.setStyleSheet("font-size: 9px; padding: 2px 4px;")
        btn_del.clicked.connect(self._on_delete)
        title_row.addWidget(btn_del)
        layout.addLayout(title_row)

        self.lz_label = QLabel("")
        self.lz_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.lz_label)

    def show_for(self, pair_idx):
        """Position near the electrode pair and populate."""
        pair = self.dlg._electrode_pairs[pair_idx]
        self.title.setText(f"Electrode Pair #{pair_idx + 1}")

        # Compute Lz for this pair
        lz = self.dlg._compute_lz(pair)
        self.lz_label.setText(f"Lz = {lz:g} \u00b5m")

        # Position near the electrode
        junctions = self.dlg.canvas._junction_xs()
        rects = self.dlg.canvas._layout_rects()
        if not junctions or not rects:
            self.hide()
            return

        j_left = pair["left"]
        j_right = pair["right"]
        if j_left < len(junctions) and j_right < len(junctions):
            cx = (junctions[j_left] + junctions[j_right]) / 2 - self.width() / 2
        else:
            cx = self.dlg.canvas.width() / 2 - self.width() / 2

        canvas = self.dlg.canvas
        cy = int(canvas.height() * 0.65)
        cx = max(4, min(cx, canvas.width() - self.width() - 4))
        cy = min(cy, canvas.height() - self.height() - 4)
        self.move(int(cx), int(cy))
        self.show()

    def _on_delete(self):
        self.dlg._delete_selected_electrode()


# ── Device canvas ──────────────────────────────────────────────────
class _DeviceCanvas(QWidget):
    def __init__(self, parent_dialog):
        super().__init__()
        self.dlg = parent_dialog
        self.setAcceptDrops(True)
        self.setMinimumSize(500, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._hover_idx = -1
        self._drop_idx = -1
        self._dragging_elec = None  # (pair_idx, "left"|"right") while dragging
        # Component reorder drag state
        self._dragging_comp = -1     # index of component being dragged
        self._drag_press_pos = None  # press position to detect drag start
        self._drag_active = False    # True once drag threshold exceeded
        # Zoom / pan state
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF()

    def _to_widget(self, scene_pt):
        """Convert scene coordinates to widget coordinates (apply zoom+pan)."""
        return QPointF(scene_pt.x() * self._zoom + self._pan_offset.x(),
                       scene_pt.y() * self._zoom + self._pan_offset.y())

    def _to_scene(self, widget_pt):
        """Convert widget coordinates to scene coordinates (invert zoom+pan)."""
        return QPointF((widget_pt.x() - self._pan_offset.x()) / self._zoom,
                       (widget_pt.y() - self._pan_offset.y()) / self._zoom)

    def _base_layout_rects(self):
        """Compute component rects in scene (unzoomed) coordinates."""
        comps = self.dlg._components
        if not comps:
            return []
        lengths = [c["length"] for c in comps]
        widths = [c["width"] for c in comps]
        total_l = sum(lengths)
        max_w = max(widths) if widths else 1

        margin_x, margin_y = 40, 50
        canvas_w = self.width() - 2 * margin_x
        canvas_h = self.height() - 2 * margin_y
        center_y = self.height() / 2 - 10

        scale_x = canvas_w / total_l if total_l > 0 else 1
        scale_y = canvas_h / (max_w * 1.5) if max_w > 0 else 1
        scale = min(scale_x, scale_y)

        rects = []
        x = margin_x + (canvas_w - total_l * scale) / 2
        for c in comps:
            w_px = c["length"] * scale
            h_px = c["width"] * scale
            y = center_y - h_px / 2
            rects.append(QRectF(x, y, w_px, h_px))
            x += w_px
        return rects

    def _layout_rects(self):
        """Compute component rects in widget (zoomed+panned) coordinates."""
        base = self._base_layout_rects()
        if self._zoom == 1.0 and self._pan_offset == QPointF(0, 0):
            return base
        result = []
        for r in base:
            tl = self._to_widget(r.topLeft())
            br = self._to_widget(r.bottomRight())
            result.append(QRectF(tl, br))
        return result

    def fit_view(self):
        """Reset zoom and pan to fit all components."""
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self.update()

    def _junction_xs(self):
        """Return x-coordinates for each junction point (boundary between components)."""
        rects = self._layout_rects()
        if not rects:
            return []
        xs = [rects[0].left()]
        for r in rects:
            xs.append(r.right())
        return xs

    def _electrode_at(self, pos):
        """Return (pair_idx, side) if pos is near an electrode marker, else None."""
        junctions = self._junction_xs()
        rects = self._layout_rects()
        if not junctions or not rects:
            return None

        top = min(r.top() for r in rects)
        bottom = max(r.bottom() for r in rects)
        hit_radius = 10

        for pair_idx, pair in enumerate(self.dlg._electrode_pairs):
            for side in ["left", "right"]:
                j = pair[side]
                if 0 <= j < len(junctions):
                    x = junctions[j]
                    if abs(pos.x() - x) < hit_radius and top - 20 < pos.y() < bottom + 20:
                        return (pair_idx, side)
        return None

    def _drop_index_at(self, pos):
        rects = self._layout_rects()
        if not rects:
            return 0
        for i, r in enumerate(rects):
            if pos.x() < r.center().x():
                return i
        return len(rects)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME_TYPE):
            self._drop_idx = self._drop_index_at(event.position())
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_idx = -1
        self.update()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(MIME_TYPE):
            comp_type = bytes(event.mimeData().data(MIME_TYPE)).decode()
            idx = self._drop_index_at(event.position())
            self.dlg._insert_component(comp_type, idx)
            self._drop_idx = -1
            event.acceptProposedAction()

    def wheelEvent(self, event):
        """Zoom in/out centered on mouse position."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = self._zoom * factor
        new_zoom = max(0.3, min(new_zoom, 5.0))

        # Zoom toward mouse position
        mouse = event.position()
        self._pan_offset = mouse - (mouse - self._pan_offset) * (new_zoom / self._zoom)
        self._zoom = new_zoom
        self.update()
        # Reposition floating editors
        self.dlg._editor.hide()
        self.dlg._elec_editor.hide()

    def mousePressEvent(self, event):
        # Right-button or middle-button: start pan
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_start = event.position() - self._pan_offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Check electrodes first (they're on top visually)
            elec = self._electrode_at(event.position())
            if elec is not None:
                self.dlg._select_electrode(elec[0])
                self._dragging_elec = elec
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

            # Check components — record press for potential drag-to-reorder
            rects = self._layout_rects()
            for i, r in enumerate(rects):
                hit = r.adjusted(-4, -4, 4, 4)
                if hit.contains(event.position()):
                    self.dlg._select(i)
                    self._dragging_comp = i
                    self._drag_press_pos = event.position()
                    self._drag_active = False
                    return
            self.dlg._select(-1)

    def mouseMoveEvent(self, event):
        # Pan
        if self._panning:
            self._pan_offset = event.position() - self._pan_start
            self.update()
            return

        # Component reorder drag
        if self._dragging_comp >= 0:
            if not self._drag_active:
                # Start drag once mouse moves beyond threshold
                delta = event.position() - self._drag_press_pos
                if abs(delta.x()) > 8 or abs(delta.y()) > 8:
                    self._drag_active = True
                    self.dlg._editor.hide()
                    self.dlg._elec_editor.hide()
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._drag_active:
                self._drop_idx = self._drop_index_at(event.position())
                self.update()
            return

        if self._dragging_elec is not None:
            # Snap to nearest junction
            junctions = self._junction_xs()
            if junctions:
                x = event.position().x()
                nearest_j = min(range(len(junctions)), key=lambda i: abs(junctions[i] - x))
                pair_idx, side = self._dragging_elec
                if pair_idx < len(self.dlg._electrode_pairs):
                    self.dlg._electrode_pairs[pair_idx][side] = nearest_j
                    self.update()
                    self.dlg._update_output_panel()
                    # Update floating editor if visible
                    if self.dlg._elec_editor.isVisible():
                        self.dlg._elec_editor.show_for(pair_idx)
            return

        # Normal hover logic (with expanded hit area matching click)
        rects = self._layout_rects()
        old = self._hover_idx
        self._hover_idx = -1
        for i, r in enumerate(rects):
            hit = r.adjusted(-4, -4, 4, 4)
            if hit.contains(event.position()):
                self._hover_idx = i
                break

        # Check if hovering over electrode
        elec = self._electrode_at(event.position())
        if elec is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._hover_idx >= 0:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._hover_idx != old:
            self.update()

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        # Component reorder drop
        if self._dragging_comp >= 0:
            src = self._dragging_comp
            dst = self._drop_idx
            self._dragging_comp = -1
            self._drag_press_pos = None
            self._drop_idx = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._drag_active and dst >= 0 and dst != src and dst != src + 1:
                self.dlg._move_component(src, dst)
            self._drag_active = False
            self.update()
            return
        if self._dragging_elec is not None:
            # Ensure left <= right in the pair
            pair_idx = self._dragging_elec[0]
            if pair_idx < len(self.dlg._electrode_pairs):
                pair = self.dlg._electrode_pairs[pair_idx]
                if pair["left"] > pair["right"]:
                    pair["left"], pair["right"] = pair["right"], pair["left"]
            self._dragging_elec = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            self.dlg._update_output_panel()
            if self.dlg._selected_elec_pair >= 0:
                self.dlg._elec_editor.show_for(self.dlg._selected_elec_pair)

    def leaveEvent(self, event):
        if self._hover_idx != -1:
            self._hover_idx = -1
            self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.dlg._selected_elec_pair >= 0:
                self.dlg._delete_selected_electrode()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Background
        p.fillRect(self.rect(), QColor(BG_CANVAS))

        comps = self.dlg._components
        if not comps:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QColor(TEXT_DIM))
            f = p.font()
            f.setPixelSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Drag Node or Pore here to build device")
            p.end()
            return

        rects = self._layout_rects()
        sel = self.dlg._selected_idx

        for i, (comp, rect) in enumerate(zip(comps, rects)):
            qc = QColor(_comp_color(comp))

            # Fade the component being dragged
            if self._drag_active and i == self._dragging_comp:
                qc.setAlpha(90)

            # Flat fill with subtle top highlight
            if i == self._hover_idx and i != sel:
                fill = qc.lighter(115)
            else:
                fill = qc

            p.setBrush(QBrush(fill))

            if i == sel and not self._drag_active:
                p.setPen(QPen(QColor(SELECT_CLR), 2.5))
            else:
                pen_c = qc.darker(120)
                pen_c.setAlpha(qc.alpha())
                p.setPen(QPen(pen_c, 0.8))

            # Sharp rectangles — no rounded corners
            p.drawRect(rect)

        # ── Draw electrodes ──
        self._draw_electrodes(p, rects)

        # Drop indicator
        if self._drop_idx >= 0:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if rects:
                if self._drop_idx < len(rects):
                    dx = rects[self._drop_idx].left()
                else:
                    dx = rects[-1].right()
            else:
                dx = self.width() / 2
            p.setPen(QPen(QColor(SELECT_CLR), 2, Qt.PenStyle.DashLine))
            p.drawLine(int(dx), 30, int(dx), self.height() - 30)

        # Total length annotation
        if rects:
            total_l = sum(c["length"] for c in comps)
            p.setPen(QColor(TEXT_DIM))
            f = p.font()
            f.setPixelSize(10)
            p.setFont(f)
            bot = max(r.bottom() for r in rects) + 14
            p.drawText(QRectF(0, bot, self.width(), 20),
                       Qt.AlignmentFlag.AlignCenter,
                       f"Total length: {total_l:g} \u00b5m")

        # Legend (top-right corner of canvas)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        legend = [("Node", NODE_CLR), ("Sizing", PORE_SIZE),
                  ("Contraction", PORE_CONTR), ("Recovery", PORE_RECOV),
                  ("Electrode", ELEC_CLR)]
        f = p.font()
        f.setPixelSize(10)
        p.setFont(f)
        lx = self.width() - 108
        ly = 10
        for label, clr in legend:
            p.setBrush(QBrush(QColor(clr)))
            p.setPen(QPen(QColor(clr).darker(115), 0.8))
            p.drawRect(QRectF(lx, ly + 1, 10, 10))
            p.setPen(QColor(TEXT_DIM))
            p.drawText(QRectF(lx + 14, ly, 90, 14),
                       Qt.AlignmentFlag.AlignVCenter, label)
            ly += 16

        p.end()

    def _draw_electrodes(self, p, rects):
        """Render electrode pairs as rectangular markers at junction points."""
        pairs = self.dlg._electrode_pairs
        if not pairs or not rects:
            return

        junctions = self._junction_xs()
        if not junctions:
            return

        top = min(r.top() for r in rects)
        bottom = max(r.bottom() for r in rects)
        marker_ext = 14  # extension beyond channel
        bar_half_w = 1.5 # half-width of the rectangular bar

        sel_pair = self.dlg._selected_elec_pair

        for pair_idx, pair in enumerate(pairs):
            is_selected = (pair_idx == sel_pair)
            base_color = QColor(ELEC_SEL if is_selected else ELEC_CLR)

            j_left = pair["left"]
            j_right = pair["right"]

            # Draw bracket above channel connecting the pair with Lz label
            if 0 <= j_left < len(junctions) and 0 <= j_right < len(junctions):
                x_left = junctions[j_left]
                x_right = junctions[j_right]
                bracket_y = top - marker_ext - 10 - pair_idx * 18
                bracket_color = QColor(base_color)
                bracket_color.setAlpha(160)
                p.setPen(QPen(bracket_color, 1.5))
                # Horizontal line
                p.drawLine(QPointF(x_left, bracket_y), QPointF(x_right, bracket_y))
                # Small vertical ticks
                tick_h = 4
                p.drawLine(QPointF(x_left, bracket_y - tick_h), QPointF(x_left, bracket_y + tick_h))
                p.drawLine(QPointF(x_right, bracket_y - tick_h), QPointF(x_right, bracket_y + tick_h))

                # Lz annotation (spacing value only)
                lz = self.dlg._compute_lz(pair)
                p.setPen(QColor(TEXT_DIM))
                f = p.font()
                f.setPixelSize(9)
                p.setFont(f)
                mid_x = (x_left + x_right) / 2
                p.drawText(QRectF(mid_x - 50, bracket_y - 16, 100, 14),
                           Qt.AlignmentFlag.AlignCenter,
                           f"{lz:g} \u00b5m")

            # Draw electrode markers as rectangular bars
            for side in ["left", "right"]:
                j = pair[side]
                if j < 0 or j >= len(junctions):
                    continue
                x = junctions[j]
                color = QColor(ELEC_SEL if is_selected else ELEC_CLR)

                y_top = top - marker_ext
                y_bot = bottom + marker_ext
                bar_rect = QRectF(x - bar_half_w, y_top, bar_half_w * 2, y_bot - y_top)

                p.setBrush(QBrush(color))
                p.setPen(QPen(color.darker(120), 0.8))
                p.drawRect(bar_rect)


# ── Main dialog ────────────────────────────────────────────────────
class DeviceGeometryDialog(QDialog):

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Device Geometry")
        self.setMinimumSize(1000, 440)
        self.resize(1100, 520)
        apply_dialog_style(self)

        self._components = []
        self._selected_idx = -1
        self._electrode_pairs = []       # [{"left": junction_idx, "right": junction_idx}, ...]
        self._selected_elec_pair = -1    # index into _electrode_pairs, or -1

        self._setup_ui()
        self._load_params(params)
        self.canvas.update()
        self._update_output_panel()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Middle: canvas + output properties ──
        mid = QHBoxLayout()
        mid.setSpacing(10)

        self.canvas = _DeviceCanvas(self)
        self.canvas.setStyleSheet("border: 1px solid #ccc; border-radius: 6px;")
        mid.addWidget(self.canvas, stretch=1)

        self._editor = _FloatingEditor(self)
        self._elec_editor = _FloatingElectrodeEditor(self)

        # Right column: properties + device parameters
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        # Output properties panel
        p = theme.palette
        out_frame = QFrame()
        out_frame.setStyleSheet(
            f"QFrame {{ background: {p.base_bg}; border: 1px solid {p.border};"
            f" border-radius: 6px; }}")
        out_frame.setFixedWidth(200)
        out_layout = QVBoxLayout(out_frame)
        out_layout.setContentsMargins(10, 10, 10, 10)
        out_layout.setSpacing(4)

        out_title = QLabel("Device Properties")
        out_title.setStyleSheet("font-weight: bold; font-size: 12px; border: none;")
        out_layout.addWidget(out_title)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(
            f"background: {p.border}; border: none; max-height: 1px;")
        out_layout.addWidget(sep)

        self._out_labels = {}
        for key, display in [
            ("H", "H (channel height)"),
            ("De", "De (eff. diameter)"),
            ("De_c", "De_c (contraction eff. diameter)"),
            ("L", "L (total length)"),
            ("Lz", "Lz (effective length)"),
            ("Ls", "Ls (sizing length)"),
            ("Ws", "Ws (sizing width)"),
            ("Lc", "Lc (contraction length)"),
            ("Wc", "Wc (contraction width)"),
            ("Lr", "Lr (recovery length)"),
            ("Wr", "Wr (recovery width)"),
        ]:
            name_lbl = QLabel(display)
            name_lbl.setStyleSheet(
                f"color: {p.text_muted}; font-size: 10px; border: none;")
            out_layout.addWidget(name_lbl)
            val_lbl = QLabel("--")
            val_lbl.setStyleSheet("font-size: 11px; font-weight: bold; border: none;")
            val_lbl.setWordWrap(True)
            out_layout.addWidget(val_lbl)
            self._out_labels[key] = val_lbl

        right_col.addWidget(out_frame)

        # Device parameters panel
        dev_frame = QFrame()
        dev_frame.setStyleSheet(
            f"QFrame {{ background: {p.base_bg}; border: 1px solid {p.border};"
            f" border-radius: 6px; }}"
            "QLabel { border: none; }")
        dev_frame.setFixedWidth(200)
        dev_layout = QVBoxLayout(dev_frame)
        dev_layout.setContentsMargins(10, 8, 10, 8)
        dev_layout.setSpacing(4)

        dev_layout.addWidget(self._dim_label("Channel Height (H):"))
        self.ch_height = QDoubleSpinBox()
        self.ch_height.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.ch_height.setRange(0.1, 10000)
        self.ch_height.setDecimals(2)
        self.ch_height.setSuffix(" \u00b5m")
        self.ch_height.setValue(30)
        self.ch_height.valueChanged.connect(self._update_output_panel)
        dev_layout.addWidget(self.ch_height)

        dev_layout.addWidget(self._dim_label("Effective Diameter (De):"))
        self.De_np = QDoubleSpinBox()
        self.De_np.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.De_np.setRange(0.1, 10000)
        self.De_np.setDecimals(2)
        self.De_np.setSuffix(" \u00b5m")
        self.De_np.setValue(20)
        self.De_np.valueChanged.connect(self._update_output_panel)
        dev_layout.addWidget(self.De_np)

        right_col.addWidget(dev_frame)

        mid.addLayout(right_col)

        root.addLayout(mid, stretch=1)

        # ── Bottom bar: New + Fit (left) | palette (center) | Save + OK (right) ──
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        # Left: New + Fit
        btn_new = QPushButton("New")
        btn_new.setFixedWidth(70)
        btn_new.setToolTip("Clear all components and start a new geometry")
        btn_new.clicked.connect(self._new_geometry)
        bottom.addWidget(btn_new)

        btn_fit = QPushButton("Fit")
        btn_fit.setFixedWidth(50)
        btn_fit.setToolTip("Reset zoom and pan to fit all components")
        btn_fit.clicked.connect(self.canvas.fit_view)
        bottom.addWidget(btn_fit)

        # Center: palette chips
        bottom.addStretch()
        lbl = QLabel("Add Component:")
        lbl.setStyleSheet("color: #888; font-size: 11px;")
        bottom.addWidget(lbl)
        bottom.addWidget(_PaletteChip("Node", NODE_CLR))
        bottom.addWidget(_PaletteChip("Pore", PORE_SIZE))

        # Electrode button (click action, not drag)
        btn_elec = QPushButton("Electrode")
        btn_elec.setFixedSize(80, 30)
        btn_elec.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_elec.setToolTip("Add an electrode pair")
        btn_elec.setStyleSheet(
            f"QPushButton {{ background: {ELEC_CLR}; color: white; font-size: 11px;"
            f"  font-weight: bold; border: 1px solid {ELEC_CLR}; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: {ELEC_SEL}; }}"
        )
        btn_elec.clicked.connect(self._add_electrode_pair)
        bottom.addWidget(btn_elec)
        bottom.addStretch()

        # Right: Save + OK
        btn_save = QPushButton("Save")
        btn_save.setFixedWidth(70)
        btn_save.clicked.connect(self._save_geometry)
        bottom.addWidget(btn_save)

        btn_ok = QPushButton("OK")
        btn_ok.setObjectName("confirmBtn")
        btn_ok.setFixedWidth(70)
        btn_ok.clicked.connect(self.accept)
        bottom.addWidget(btn_ok)

        root.addLayout(bottom)

    @staticmethod
    def _dim_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #888; font-size: 10px; border: none;")
        return lbl

    # ── Data ──
    def _load_params(self, params):
        components = params.get("components")
        if components:
            self._components = [dict(c) for c in components]
        self.ch_height.setValue(params.get("ch_height", 30))
        self.De_np.setValue(params.get("De_np", 20))
        # Load electrode pairs
        electrodes = params.get("electrode_pairs")
        if electrodes:
            self._electrode_pairs = [dict(e) for e in electrodes]

    def _move_component(self, src, dst):
        """Move a component from index *src* to insert position *dst*.

        *dst* uses the same convention as _drop_index_at: the gap index
        where the component will land.  After the move the component
        occupies index  dst  (if dst <= src)  or  dst-1  (if dst > src).
        Electrode junction indices are remapped accordingly.
        """
        if src < 0 or src >= len(self._components):
            return
        n = len(self._components)
        # Build old→new index mapping for components
        order = list(range(n))
        comp = order.pop(src)
        new_dst = dst if dst <= src else dst - 1
        order.insert(new_dst, comp)
        # old_to_new[old_idx] = new_idx
        old_to_new = [0] * n
        for new_i, old_i in enumerate(order):
            old_to_new[old_i] = new_i

        # Reorder the components list
        self._components = [self._components[i] for i in order]

        # Remap electrode junctions.
        # Junction j sits between component j-1 and j (0 = left edge,
        # n = right edge).  Build a mapping from old junction to new.
        # A junction j in the old layout corresponds to "left edge of
        # old component j" (or right edge of the last one for j == n).
        # After reorder, old component j is now at old_to_new[j].
        junction_map = {}
        # Left edges: junction j → left edge of wherever old component j went
        for old_j in range(n):
            junction_map[old_j] = old_to_new[old_j]
        # Right edge: junction n → n (always the far right)
        junction_map[n] = n

        for pair in self._electrode_pairs:
            for side in ["left", "right"]:
                pair[side] = junction_map.get(pair[side], pair[side])
            if pair["left"] > pair["right"]:
                pair["left"], pair["right"] = pair["right"], pair["left"]

        # Remove degenerate electrode pairs
        self._electrode_pairs = [p for p in self._electrode_pairs
                                 if p["left"] != p["right"]]

        self._select(new_dst)
        self.canvas.update()
        self._update_output_panel()

    def _insert_component(self, comp_type, index):
        defaults = {
            "Node": {"type": "Node", "width": 85, "length": 50},
            "Pore": {"type": "Pore", "width": 25, "length": 300, "property": "sizing"},
        }
        comp = defaults.get(comp_type)
        if not comp:
            return
        self._components.insert(index, dict(comp))
        # Adjust electrode junction indices: junctions >= index+1 shift by +1
        for pair in self._electrode_pairs:
            for side in ["left", "right"]:
                if pair[side] > index:
                    pair[side] += 1
        self._select(index)
        self.canvas.update()
        self._update_output_panel()

    def _select(self, idx):
        self._selected_idx = idx
        # Deselect electrode when selecting a component
        self._selected_elec_pair = -1
        self._elec_editor.hide()
        if 0 <= idx < len(self._components):
            rects = self.canvas._layout_rects()
            if idx < len(rects):
                self._editor.show_for(idx, rects[idx])
        else:
            self._editor.hide()
        self.canvas.update()

    def _select_electrode(self, pair_idx):
        """Select an electrode pair by index."""
        self._selected_elec_pair = pair_idx
        # Deselect component
        self._selected_idx = -1
        self._editor.hide()
        if 0 <= pair_idx < len(self._electrode_pairs):
            self._elec_editor.show_for(pair_idx)
        else:
            self._elec_editor.hide()
        self.canvas.update()

    def _delete_selected(self):
        idx = self._selected_idx
        if 0 <= idx < len(self._components):
            self._components.pop(idx)
            # Adjust electrode junction indices
            n_junctions = len(self._components) + 1
            pairs_to_remove = []
            for pi, pair in enumerate(self._electrode_pairs):
                for side in ["left", "right"]:
                    j = pair[side]
                    if j == idx + 1:
                        pair[side] = idx  # merge boundary
                    elif j > idx + 1:
                        pair[side] -= 1
                # Clamp to valid range
                pair["left"] = min(pair["left"], n_junctions - 1)
                pair["right"] = min(pair["right"], n_junctions - 1)
                # Remove degenerate pairs
                if pair["left"] == pair["right"]:
                    pairs_to_remove.append(pi)
            for pi in reversed(pairs_to_remove):
                self._electrode_pairs.pop(pi)
            self._select(-1)
            self.canvas.update()
            self._update_output_panel()

    def _add_electrode_pair(self):
        """Add a new electrode pair spanning the full device."""
        n = len(self._components)
        if n == 0:
            QMessageBox.information(self, "No Components",
                                    "Add at least one component before placing electrodes.")
            return
        # Default: span the full device (junction 0 to junction N)
        pair = {"left": 0, "right": n}
        self._electrode_pairs.append(pair)
        self._select_electrode(len(self._electrode_pairs) - 1)
        self.canvas.update()
        self._update_output_panel()

    def _delete_selected_electrode(self):
        idx = self._selected_elec_pair
        if 0 <= idx < len(self._electrode_pairs):
            self._electrode_pairs.pop(idx)
            self._selected_elec_pair = -1
            self._elec_editor.hide()
            self.canvas.update()
            self._update_output_panel()

    def _compute_lz(self, pair):
        """Compute effective length (sum of component lengths) between electrode pair junctions."""
        j_left = pair["left"]
        j_right = pair["right"]
        if j_left > j_right:
            j_left, j_right = j_right, j_left
        # Sum lengths of components between junction j_left and j_right
        # Junction j_left is at the left edge of component j_left
        # Junction j_right is at the right edge of component j_right - 1
        total = 0
        for i in range(j_left, min(j_right, len(self._components))):
            total += self._components[i]["length"]
        return total

    def _update_output_panel(self):
        comps = self._components
        if not comps:
            for lbl in self._out_labels.values():
                lbl.setText("--")
            return

        def _v(val):
            return f"{val:g} \u00b5m"

        def _arr(vals):
            if not vals:
                return "--"
            strs = [f"{v:g}" for v in vals]
            return f"[{', '.join(strs)}] \u00b5m"

        de_np = self.De_np.value()
        self._out_labels["De"].setText(_v(de_np))
        self._out_labels["H"].setText(_v(self.ch_height.value()))
        self._out_labels["L"].setText(_v(sum(c["length"] for c in comps)))

        # De_c = De_np * (Wc / Ws)^0.5
        sizing = [c for c in comps if c["type"] == "Pore" and c.get("property") == "sizing"]
        contr = [c for c in comps if c["type"] == "Pore" and c.get("property") == "contraction"]
        ws = sizing[0]["width"] if sizing else 0
        wc = contr[0]["width"] if contr else ws
        de_c = de_np * (wc / ws) ** 0.5 if ws > 0 else de_np
        self._out_labels["De_c"].setText(_v(round(de_c, 2)))

        # Lz: effective lengths from electrode pairs
        if self._electrode_pairs:
            lz_vals = [self._compute_lz(p) for p in self._electrode_pairs]
            self._out_labels["Lz"].setText(_arr(lz_vals))
        else:
            self._out_labels["Lz"].setText("--")

        sizing = [c for c in comps if c["type"] == "Pore" and c.get("property") == "sizing"]
        self._out_labels["Ws"].setText(_arr([c["width"] for c in sizing]))
        self._out_labels["Ls"].setText(_arr([c["length"] for c in sizing]))

        contr = [c for c in comps if c["type"] == "Pore" and c.get("property") == "contraction"]
        self._out_labels["Lc"].setText(_arr([c["length"] for c in contr]))
        self._out_labels["Wc"].setText(_arr([c["width"] for c in contr]))

        recov = [c for c in comps if c["type"] == "Pore" and c.get("property") == "recovery"]
        self._out_labels["Wr"].setText(_arr([c["width"] for c in recov]))
        self._out_labels["Lr"].setText(_arr([c["length"] for c in recov]))

    def _new_geometry(self):
        """Clear all components and start a fresh geometry."""
        reply = QMessageBox.question(
            self, "New Geometry",
            "Clear all components and start a new geometry?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._components.clear()
        self._electrode_pairs.clear()
        self._selected_idx = -1
        self._selected_elec_pair = -1
        self._editor.hide()
        self._elec_editor.hide()
        self.ch_height.setValue(30)
        self.De_np.setValue(20)
        self.canvas.fit_view()
        self._update_output_panel()

    def _save_geometry(self):
        import os
        root = project_root()
        default_folder = os.path.join(root, "SavedTemplates", "Geometry")
        os.makedirs(default_folder, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Device Geometry", default_folder, "JSON Files (*.json)")
        if not path:
            return
        data = self.get_result()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Saved", f"Geometry saved to:\n{path}")

    def get_result(self):
        comps = [dict(c) for c in self._components]
        ch_h = self.ch_height.value()
        de_np = self.De_np.value()

        sizing = [c for c in comps if c["type"] == "Pore" and c.get("property") == "sizing"]
        contr = [c for c in comps if c["type"] == "Pore" and c.get("property") == "contraction"]
        recov = [c for c in comps if c["type"] == "Pore" and c.get("property") == "recovery"]

        ws = sizing[0]["width"] if sizing else 25
        wc = contr[0]["width"] if contr else ws
        de_c = de_np * (wc / ws) ** 0.5 if ws > 0 else de_np

        # Electrode data
        electrode_pairs = [dict(p) for p in self._electrode_pairs]
        lz_vals = [self._compute_lz(p) for p in self._electrode_pairs]
        total_l = sum(c["length"] for c in comps)

        return {
            "components": comps,
            "De": de_np,
            "De_c": de_c,
            "H": ch_h,
            "L": total_l,
            "Lz": lz_vals if lz_vals else [total_l],
            "Ws": [c["width"] for c in sizing],
            "Ls": [c["length"] for c in sizing],
            "Lc": [c["length"] for c in contr],
            "Wc": [c["width"] for c in contr],
            "Wr": [c["width"] for c in recov],
            "Lr": [c["length"] for c in recov],
            "ch_height": ch_h,
            "De_np": de_np,
            "electrode_pairs": electrode_pairs,
        }
