"""Interactive UI for assembling a per-sample training label array.

Plots the input signal with color-underlaid labeled regions and a matplotlib
navigation toolbar (pan / zoom / home). Class cards use drag-and-drop: drag
source chips between cards to merge/split classes, rename classes inline,
add or delete class slots.  Class ID 0 is reserved for background samples.

Drag-and-drop is implemented with plain QLabel chips and custom mouse event
handling — NOT QListWidget — because Qt 6 / PySide6 crashes inside
QListView::mouseMoveEvent when initiating a drag from a QListWidget subclass
(observed reproducibly on macOS).
"""

import os
from utils.paths import project_root
import json
import datetime
import warnings
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QWidget, QSplitter, QMessageBox, QLineEdit,
    QFrame, QScrollArea, QApplication,
)
from PySide6.QtCore import Qt, QMimeData, Signal, QTimer, QPoint
from PySide6.QtGui import QDrag
from theme import theme
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from matplotlib.patches import Patch

from utils.dialog_style import apply_dialog_style
from utils.decimate import minmax_decimate
from utils.zones import is_multizone, zone_columns

_SOURCE_NAMES = ("single", "coincident", "noise", "uncertain")
_PALETTE = [
    "#2ca02c", "#d62728", "#ff7f0e", "#9467bd",
    "#17becf", "#e377c2", "#8c564b", "#bcbd22",
    "#1f77b4", "#ffbb78",
]
_MIME = "application/x-nps-source"

MAX_PLOT_POINTS = 5000


def _build_label_array(data_len, source_regions, assignments):
    labels = np.zeros(data_len, dtype=np.int32)
    for src, regions in source_regions.items():
        cid = assignments.get(src, 0)
        if cid == 0 or regions is None:
            continue
        r = np.atleast_2d(regions)
        if r.size == 0:
            continue
        for row in r:
            if len(row) < 2:
                continue
            s = max(0, int(row[0]))
            e = min(data_len, int(row[1]))
            if e > s:
                labels[s:e] = cid
    return labels


def _build_csv_text(zones, labels):
    """CSV text for training-data export.

    *zones* is a list of 1-D value arrays (one per zone); *labels* is the
    shared 1-D int label array. Single zone -> ``sample_index,value,label_id``.
    Multizone -> ``sample_index,value_zone1,...,value_zoneN,label_id``.
    """
    n = len(labels)
    out = []
    if len(zones) == 1:
        out.append("sample_index,value,label_id")
        z = zones[0]
        for i in range(n):
            out.append(f"{i},{float(z[i]):.6f},{int(labels[i])}")
    else:
        header = ("sample_index,"
                  + ",".join(f"value_zone{zi + 1}" for zi in range(len(zones)))
                  + ",label_id")
        out.append(header)
        for i in range(n):
            vals = ",".join(f"{float(z[i]):.6f}" for z in zones)
            out.append(f"{i},{vals},{int(labels[i])}")
    return "\n".join(out) + "\n"


class _SourceChip(QLabel):
    """Draggable pill-shaped label representing one source."""

    def __init__(self, key, label, zone, dialog):
        super().__init__(label)
        self._key = key
        self._label_text = label
        self._zone = zone
        self._dialog = dialog
        self._press_pos = None
        p = theme.palette
        self.setStyleSheet(
            f"QLabel {{ background: {p.base_bg}; border: 1px solid {p.border};"
            f" border-radius: 10px; padding: 4px 10px; color: {p.text}; }}"
            f"QLabel:hover {{ background: {p.check_bg};"
            f" border-color: {p.check_border}; }}"
        )
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def key(self):
        return self._key

    def label_text(self):
        return self._label_text

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        delta = event.position().toPoint() - self._press_pos
        if delta.manhattanLength() < QApplication.startDragDistance():
            return
        self._press_pos = None
        mime = QMimeData()
        mime.setData(_MIME, str(self._key).encode())
        mime.setText(self._label_text)
        drag = QDrag(self)
        drag.setMimeData(mime)
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        dialog = self._dialog
        drag.exec(Qt.DropAction.CopyAction)
        # IMPORTANT: do not access self.* after drag.exec() — this chip may be
        # about to be replaced by the pending move. Delegate cleanup to the
        # dialog, which schedules the actual move on the outer event loop.
        dialog._finalize_drop()

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)


class _DropZone(QWidget):
    """Container inside a class card that holds source chips and accepts drops."""

    def __init__(self, card, dialog):
        super().__init__()
        self._card = card
        self._dialog = dialog
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        self.setStyleSheet(
            f"_DropZone {{ background: {theme.palette.panel_bg};"
            f" border: 1px dashed {theme.palette.border}; border-radius: 3px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout = lay
        self._chips = {}

    def add_chip(self, key, label):
        chip = _SourceChip(key, label, self, self._dialog)
        self._chips[key] = chip
        self._layout.addWidget(chip)

    def take_chip(self, key):
        chip = self._chips.pop(key, None)
        if chip is None:
            return False
        self._layout.removeWidget(chip)
        chip.hide()
        chip.deleteLater()
        return True

    def source_keys(self):
        return list(self._chips.keys())

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_MIME):
            event.acceptProposedAction()
            self.setStyleSheet(
                f"_DropZone {{ background: {theme.palette.check_bg};"
                f" border: 1px dashed {theme.palette.check_border};"
                f" border-radius: 3px; }}"
            )
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            f"_DropZone {{ background: {theme.palette.panel_bg};"
            f" border: 1px dashed {theme.palette.border}; border-radius: 3px; }}"
        )

    def dropEvent(self, event):
        self.setStyleSheet(
            f"_DropZone {{ background: {theme.palette.panel_bg};"
            f" border: 1px dashed {theme.palette.border}; border-radius: 3px; }}"
        )
        mime = event.mimeData()
        if not mime.hasFormat(_MIME):
            event.ignore()
            return
        src_chip = event.source()
        if not isinstance(src_chip, _SourceChip):
            event.ignore()
            return
        src_zone = src_chip._zone
        if src_zone is self:
            event.ignore()
            return
        key = bytes(mime.data(_MIME)).decode()
        label = mime.text()
        # Just record the intent. The actual move happens after drag.exec()
        # has fully unwound (see _SourceChip.mouseMoveEvent → _finalize_drop).
        self._dialog._pending_move = (src_zone, self, key, label)
        event.acceptProposedAction()


class _ClassCard(QFrame):
    changed = Signal()
    delete_requested = Signal(object)

    def __init__(self, dialog, name, color, items,
                 is_background=False, is_placeholder=False):
        super().__init__()
        self._dialog = dialog
        self._is_background = is_background
        self._is_placeholder = is_placeholder
        self._color = color
        self._had_items = bool(items)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(160)
        self._apply_frame_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self._layout = lay

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self._hdr_layout = hdr

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Click to cycle color")
        self._color_btn.clicked.connect(self._cycle_color)
        self._apply_color_style()
        if is_background or is_placeholder:
            self._color_btn.setVisible(False)
        hdr.addWidget(self._color_btn)

        if is_background:
            self._header_label = QLabel("Ignored / Background")
            self._header_label.setStyleSheet(
                f"font-weight: bold; color: {theme.palette.text_muted};")
            hdr.addWidget(self._header_label)
            self._name_edit = None
        elif is_placeholder:
            self._header_label = QLabel("+ New Class")
            self._header_label.setStyleSheet(
                "font-weight: bold; color: #0066cc;")
            hdr.addWidget(self._header_label)
            self._name_edit = None
        else:
            self._header_label = None
            self._name_edit = QLineEdit(name)
            self._name_edit.setPlaceholderText("class name")
            self._name_edit.setStyleSheet("font-weight: bold;")
            self._name_edit.editingFinished.connect(self.changed.emit)
            hdr.addWidget(self._name_edit)

        hdr.addStretch()

        self._del_btn = None
        if not is_background and not is_placeholder:
            self._del_btn = self._make_delete_btn()
            hdr.addWidget(self._del_btn)

        lay.addLayout(hdr)

        self._zone = _DropZone(self, dialog)
        for key, label in items:
            self._zone.add_chip(key, label)
        lay.addWidget(self._zone, 1)

    def _apply_color_style(self):
        self._color_btn.setStyleSheet(
            f"QPushButton {{ background: {self._color};"
            f" border: 1px solid #666; border-radius: 3px; }}"
        )

    def _apply_frame_style(self):
        p = theme.palette
        if self._is_placeholder:
            style = (
                f"_ClassCard {{ background: {p.check_bg};"
                f" border: 2px dashed {p.check_border}; border-radius: 5px; }}"
            )
        elif self._is_background:
            style = (
                f"_ClassCard {{ background: {p.panel_bg};"
                f" border: 1px solid {p.border}; border-radius: 5px; }}"
            )
        else:
            style = (
                f"_ClassCard {{ background: {p.base_bg};"
                f" border: 1px solid {p.border}; border-radius: 5px; }}"
            )
        self.setStyleSheet(style)

    def _cycle_color(self):
        try:
            idx = _PALETTE.index(self._color)
        except ValueError:
            idx = -1
        self._color = _PALETTE[(idx + 1) % len(_PALETTE)]
        self._apply_color_style()
        self.changed.emit()

    def _make_delete_btn(self):
        btn = QPushButton("×")
        btn.setFixedSize(22, 22)
        btn.setToolTip("Delete this class (sources move to Ignored)")
        btn.setStyleSheet(
            "QPushButton { color: #a00; font-size: 16px; font-weight: bold;"
            " border: none; background: transparent; }"
            "QPushButton:hover { background: #fee; border-radius: 3px; }"
        )
        btn.clicked.connect(lambda: self.delete_requested.emit(self))
        return btn

    def promote_from_placeholder(self, color):
        """Transform a placeholder card into a real class card in place."""
        if not self._is_placeholder:
            return
        self._is_placeholder = False
        self._color = color
        self._apply_frame_style()

        if self._header_label is not None:
            self._hdr_layout.removeWidget(self._header_label)
            self._header_label.hide()
            self._header_label.deleteLater()
            self._header_label = None

        self._color_btn.setVisible(True)
        self._apply_color_style()

        self._name_edit = QLineEdit("")
        self._name_edit.setPlaceholderText("class name")
        self._name_edit.setStyleSheet("font-weight: bold;")
        self._name_edit.editingFinished.connect(self.changed.emit)
        self._hdr_layout.insertWidget(1, self._name_edit)

        self._del_btn = self._make_delete_btn()
        self._hdr_layout.addWidget(self._del_btn)

    def take_source(self, key):
        ok = self._zone.take_chip(key)
        return ok

    def add_source(self, key, label):
        self._zone.add_chip(key, label)
        self._had_items = True

    def source_keys(self):
        return self._zone.source_keys()

    def name(self):
        if self._is_background or self._name_edit is None:
            return ""
        return self._name_edit.text().strip()

    def color(self):
        return self._color

    def is_background(self):
        return self._is_background

    def is_placeholder(self):
        return self._is_placeholder

    def is_empty(self):
        return len(self._zone.source_keys()) == 0

    def had_items(self):
        return self._had_items


class _TrainingDataDialog(QDialog):
    def __init__(self, data, source_regions, sample_rate, init_settings=None):
        super().__init__()
        self.setWindowTitle("Training Data Labeler")
        self.setMinimumSize(1200, 780)
        apply_dialog_style(self)

        self.confirmed = False
        self._data = np.asarray(data, dtype=float)
        self._is_mz = is_multizone(self._data)
        self._zones = zone_columns(self._data)
        self._n = self._zones[0].shape[0]
        self._source_regions = source_regions
        self._sample_rate = sample_rate if sample_rate and sample_rate > 1 else 0
        self._exported_csv = ""
        self._labels = np.zeros(self._n, dtype=np.int32)
        self._classes = []

        self._class_cards = []
        self._background_card = None
        self._next_color_idx = 0
        self._plot_initialized = False
        self._pending_move = None

        self._init_settings = init_settings or {}

        self._build_ui()
        self._init_cards()
        self._refresh_plot()

    def _source_label(self, src):
        r = self._source_regions.get(src)
        n = 0 if r is None else len(np.atleast_2d(r))
        return f"{src.capitalize()} ({n})"

    def _connected_sources(self):
        return [s for s in _SOURCE_NAMES
                if self._source_regions.get(s) is not None]

    def _build_ui(self):
        main = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        plot_widget = QWidget()
        plot_lay = QVBoxLayout(plot_widget)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.setSpacing(0)
        self._fig = Figure(figsize=(10, 4), dpi=100)
        style_mpl_figure(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        plot_lay.addWidget(self._toolbar)
        plot_lay.addWidget(self._canvas)
        splitter.addWidget(plot_widget)

        bottom = QWidget()
        bot_lay = QVBoxLayout(bottom)

        cls_grp = QGroupBox(
            "Label Classes  —  drag source chips between cards to combine/split; "
            "rename inline; click × to delete a class (its sources move to Ignored)."
        )
        cls_lay = QVBoxLayout(cls_grp)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(220)

        self._cards_host = QWidget()
        self._cards_lay = QHBoxLayout(self._cards_host)
        self._cards_lay.setSpacing(8)
        self._cards_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._placeholder_card = None

        scroll.setWidget(self._cards_host)
        cls_lay.addWidget(scroll)

        self._summary_label = QLabel()
        self._summary_label.setStyleSheet("padding: 4px; font-size: 12px;")
        self._summary_label.setTextFormat(Qt.TextFormat.RichText)
        cls_lay.addWidget(self._summary_label)

        bot_lay.addWidget(cls_grp)

        btn_lay = QHBoxLayout()
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self._export_csv)
        btn_lay.addWidget(btn_export)

        btn_lay.addStretch()

        btn_ok = QPushButton("OK")
        btn_ok.setObjectName("confirmBtn")
        btn_ok.setMinimumWidth(90)
        btn_ok.clicked.connect(self._confirm)
        btn_lay.addWidget(btn_ok)

        btn_close = QPushButton("Close")
        btn_close.setMinimumWidth(90)
        btn_close.clicked.connect(self.reject)
        btn_lay.addWidget(btn_close)
        bot_lay.addLayout(btn_lay)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter)

    def _next_color(self):
        color = _PALETTE[self._next_color_idx % len(_PALETTE)]
        self._next_color_idx += 1
        return color

    def _init_cards(self):
        self._background_card = _ClassCard(
            self, name="", color="#dddddd", items=[], is_background=True)
        self._background_card.changed.connect(self._refresh_plot)
        self._cards_lay.addWidget(self._background_card)

        connected = self._connected_sources()

        classes_spec = None
        ignored = []
        if isinstance(self._init_settings.get("classes"), list):
            classes_spec = []
            for c in self._init_settings["classes"]:
                if not isinstance(c, dict):
                    continue
                name = c.get("name", "")
                color = c.get("color")
                srcs = [s for s in c.get("sources", []) if s in connected]
                classes_spec.append((name, color, srcs))
            ignored = [s for s in self._init_settings.get("ignored", [])
                       if s in connected]
        elif isinstance(self._init_settings.get("class_map"), dict):
            cmap = self._init_settings["class_map"]
            groups = {}
            order = []
            for src in connected:
                nm = (cmap.get(src) or "").strip()
                if not nm:
                    ignored.append(src)
                else:
                    if nm not in groups:
                        groups[nm] = []
                        order.append(nm)
                    groups[nm].append(src)
            classes_spec = [(nm, None, groups[nm]) for nm in order]

        placed = set()
        if classes_spec is not None:
            for name, color, srcs in classes_spec:
                srcs = [s for s in srcs if s not in placed]
                if not srcs:
                    continue
                col = color if color in _PALETTE else self._next_color()
                items = [(s, self._source_label(s)) for s in srcs]
                self._add_class_card(name, col, items)
                placed.update(srcs)
            for s in ignored:
                if s in placed:
                    continue
                self._background_card.add_source(s, self._source_label(s))
                placed.add(s)
            for s in connected:
                if s in placed:
                    continue
                items = [(s, self._source_label(s))]
                self._add_class_card(s, self._next_color(), items)
        else:
            for s in connected:
                items = [(s, self._source_label(s))]
                self._add_class_card(s, self._next_color(), items)

        self._install_placeholder_card()
        self._cards_lay.addStretch()

    def _add_class_card(self, name, color, items):
        card = _ClassCard(self, name=name, color=color, items=items)
        card.changed.connect(self._refresh_plot)
        card.delete_requested.connect(self._on_delete_card)
        self._class_cards.append(card)
        if self._placeholder_card is not None and \
                self._cards_lay.indexOf(self._placeholder_card) >= 0:
            idx = self._cards_lay.indexOf(self._placeholder_card)
            self._cards_lay.insertWidget(idx, card)
        else:
            self._cards_lay.addWidget(card)
        return card

    def _install_placeholder_card(self, index=None):
        card = _ClassCard(self, name="", color=self._next_color(),
                          items=[], is_placeholder=True)
        card.changed.connect(self._refresh_plot)
        card.delete_requested.connect(self._on_delete_card)
        self._placeholder_card = card
        if index is None:
            self._cards_lay.addWidget(card)
        else:
            self._cards_lay.insertWidget(index, card)

    def _promote_placeholder(self, card):
        """A chip was dropped on the placeholder — turn it into a real class
        and spawn a fresh placeholder after it."""
        if card is not self._placeholder_card:
            return
        card.promote_from_placeholder(self._next_color())
        self._class_cards.append(card)
        # Insert a new placeholder after the newly-promoted card
        self._install_placeholder_card(self._cards_lay.indexOf(card) + 1)
        if card._name_edit is not None:
            card._name_edit.setFocus()

    def _on_delete_card(self, card):
        if card.is_background():
            return
        for key in list(card.source_keys()):
            card.take_source(key)
            self._background_card.add_source(key, self._source_label(key))
        if card in self._class_cards:
            self._class_cards.remove(card)
        card.hide()
        card.deleteLater()
        self._refresh_plot()

    def _finalize_drop(self):
        """Called from the drag source's mouseMoveEvent after drag.exec()
        returns. Schedules the actual move so it runs on the outer event
        loop, well after mouseMoveEvent unwinds — the only way to guarantee
        the drag source widget isn't touched after deletion."""
        pending = self._pending_move
        self._pending_move = None
        if pending is None:
            return
        QTimer.singleShot(0, lambda p=pending: self._do_move(*p))

    def _do_move(self, src_zone, dst_zone, key, label):
        try:
            src_card = src_zone._card
            dst_card = dst_zone._card
        except RuntimeError:
            return
        if src_card is dst_card:
            return
        if not src_card.take_source(key):
            return
        # If the drop landed on the placeholder, promote it to a real class
        # and spawn a new placeholder in its place (dynamic-port pattern).
        if dst_card.is_placeholder():
            self._promote_placeholder(dst_card)
        dst_card.add_source(key, label)
        self._prune_empty_cards()
        self._refresh_plot()

    def _prune_empty_cards(self):
        to_remove = [c for c in self._class_cards
                     if c.had_items() and c.is_empty()]
        for c in to_remove:
            self._class_cards.remove(c)
            c.hide()
            c.deleteLater()

    def _compute_assignments(self):
        assignments = {s: 0 for s in _SOURCE_NAMES}
        classes = []
        cid = 0
        for card in self._class_cards:
            keys = card.source_keys()
            if not keys:
                continue
            cid += 1
            name = card.name() or f"class{cid}"
            color = card.color()
            classes.append((cid, name, color, list(keys)))
            for k in keys:
                assignments[k] = cid
        return assignments, classes

    def _refresh_plot(self):
        assignments, classes = self._compute_assignments()

        xlim = ylim = None
        if self._plot_initialized:
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()

        self._ax.clear()
        n = self._n
        if n == 0:
            self._canvas.draw()
            return

        if self._sample_rate > 1:
            t = np.arange(n) / self._sample_rate
            xlabel = "Time (s)"
        else:
            t = np.arange(n)
            xlabel = "Sample index"

        id_to_color = {cid: color for cid, _, color, _ in classes}
        for src in _SOURCE_NAMES:
            cid = assignments.get(src, 0)
            if cid == 0:
                continue
            regs = self._source_regions.get(src)
            if regs is None:
                continue
            r = np.atleast_2d(regs)
            if r.size == 0:
                continue
            color = id_to_color.get(cid)
            for row in r:
                if len(row) < 2:
                    continue
                s = max(0, int(row[0]))
                e = min(n, int(row[1]))
                if e > s:
                    x0 = t[s]
                    x1 = t[e - 1] if e - 1 < n else t[-1]
                    self._ax.axvspan(x0, x1, color=color, alpha=0.28,
                                     zorder=0, linewidth=0)

        if self._is_mz:
            d = self._data
            n_zones = d.shape[1]
            zmd = (d - d.mean(axis=0)) / 1e6
            min_spacing = 0
            for zi in range(n_zones - 1):
                req = 2 * (np.max(zmd[:, zi + 1]) - np.min(zmd[:, zi]))
                min_spacing = max(min_spacing, req)
            if min_spacing <= 0:
                ranges = np.max(zmd, axis=0) - np.min(zmd, axis=0)
                min_spacing = 0.1 * np.mean(ranges)
            for zi in range(n_zones):
                offset = (n_zones - zi) * min_spacing / 2
                yz = zmd[:, zi] + offset
                tx, ty = minmax_decimate(t, yz, MAX_PLOT_POINTS)
                self._ax.plot(tx, ty, linewidth=0.6, zorder=3)
                self._ax.annotate(
                    f"Zone {zi + 1}", xy=(t[0], yz[0]),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7, va="center", zorder=4)
            self._ax.set_xlabel(xlabel, fontsize=9)
            self._ax.set_ylabel("Resistance [MΩ] (offset)", fontsize=9)
        else:
            self._ax.plot(t, self._zones[0], color="#222",
                          linewidth=0.6, zorder=3)
            self._ax.set_xlabel(xlabel, fontsize=9)
            self._ax.set_ylabel("Amplitude", fontsize=9)
        self._ax.tick_params(labelsize=8)

        if classes:
            handles = [
                Patch(facecolor=color, alpha=0.45, edgecolor="none",
                      label=f"{name} (id={cid})")
                for cid, name, color, _ in classes
            ]
            self._ax.legend(handles=handles, loc="upper right",
                            fontsize=8, framealpha=0.85)

        self._ax.set_title("Input signal with labeled regions", fontsize=10)

        if xlim is not None:
            self._ax.set_xlim(xlim)
            self._ax.set_ylim(ylim)
        else:
            self._ax.set_xlim(t[0], t[-1])
            self._ax.margins(y=0.05)

        try:
            self._fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

        if not self._plot_initialized:
            try:
                self._toolbar.update()
                self._toolbar.push_current()
            except Exception:
                pass
            self._plot_initialized = True

        labels = _build_label_array(n, self._source_regions, assignments)
        parts = [
            f'<span style="color:{theme.palette.text}">Background (0): '
            f'{int(np.sum(labels == 0))} samples</span>'
        ]
        for cid, name, color, _ in classes:
            cnt = int(np.sum(labels == cid))
            parts.append(
                f'<span style="color:{color}"><b>{name} ({cid})</b>: '
                f'{cnt} samples</span>'
            )
        self._summary_label.setText("  |  ".join(parts))

        self._labels = labels
        self._classes = [(cid, name, color) for cid, name, color, _ in classes]

    def _confirm(self):
        if not self._classes:
            reply = QMessageBox.question(
                self, "No Classes Defined",
                "No classes have sources assigned — all samples will be "
                "background (0).\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._autosave()
        self.confirmed = True
        self.accept()

    def _export_csv(self):
        root = project_root()
        default_dir = os.path.join(root, "Output")
        os.makedirs(default_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"training_labels_{timestamp}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Training Data CSV",
            os.path.join(default_dir, default_name),
            "CSV Files (*.csv)"
        )
        if not path:
            return

        labels = self._labels
        classes = self._classes

        with open(path, "w") as f:
            f.write(_build_csv_text(self._zones, labels))

        meta_path = path.rsplit(".", 1)[0] + "_meta.json"
        meta = {
            "source_file": os.path.basename(path),
            "sample_rate": self._sample_rate,
            "num_samples": int(self._n),
            "label_map": {"0": "background",
                          **{str(cid): name for cid, name, _ in classes}},
            "class_counts": {
                "background": int(np.sum(labels == 0)),
                **{name: int(np.sum(labels == cid))
                   for cid, name, _ in classes},
            },
            "classes": self._serialize_classes(),
            "date_exported": datetime.datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._exported_csv = path
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {self._n} samples to:\n{path}\n\n"
            f"Metadata saved to:\n{meta_path}"
        )

    def _serialize_classes(self):
        out = []
        for card in self._class_cards:
            keys = card.source_keys()
            if not keys:
                continue
            out.append({
                "name": card.name(),
                "color": card.color(),
                "sources": keys,
            })
        return out

    def _ignored_sources(self):
        return list(self._background_card.source_keys())

    def _autosave(self):
        root = project_root()
        folder = os.path.join(root, "SavedTemplates", "Training")
        os.makedirs(folder, exist_ok=True)
        settings = {
            "classes": self._serialize_classes(),
            "ignored": self._ignored_sources(),
        }
        try:
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass


def training_data_ui(data, single, coincident, noise, uncertain,
                     sample_rate=1, init_settings=None):
    """Entry point called by the runner.

    Returns dict with keys:
      'labels'  : int32 array, same length as data (0 = background)
      'csvFile' : path to exported CSV (empty string if not exported)
    """
    data = np.asarray(data, dtype=float) if data is not None else np.array([])
    if data.size == 0:
        raise ValueError("No data provided to TrainingData block.")

    source_regions = {}
    any_regions = False
    for name, val in [("single", single), ("coincident", coincident),
                      ("noise", noise), ("uncertain", uncertain)]:
        if val is None:
            source_regions[name] = None
            continue
        arr = np.atleast_2d(val)
        if arr.size == 0:
            source_regions[name] = None
        else:
            source_regions[name] = arr
            any_regions = True

    if not any_regions:
        raise ValueError("No labeled regions provided to TrainingData block.")

    dlg = _TrainingDataDialog(data, source_regions, sample_rate,
                              init_settings=init_settings)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Training Data dialog without confirming.")

    return {
        "labels": dlg._labels,
        "csvFile": dlg._exported_csv,
        "classes": dlg._serialize_classes(),
        "ignored": dlg._ignored_sources(),
    }
