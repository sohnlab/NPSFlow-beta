"""Interactive 2D Plot dialog with collapsible-section settings panel.

Provides per-series plot type, line/marker/scatter/bar/histogram/box options,
labels, axes, curve fit, annotations, reference lines, subplots, colorbar,
figure export, and CSV export — all in a scrollable accordion layout.
"""

import sys
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QPushButton, QColorDialog,
    QWidget, QFileDialog, QLabel, QGroupBox, QScrollArea, QFrame,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

# ---------------------------------------------------------------- constants
_SANS = "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"

_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_LINESTYLES = [
    ("-", "Solid"), ("--", "Dashed"), ("-.", "Dash-dot"),
    (":", "Dotted"), ("None", "None"),
]
_MARKERS = [
    ("None", "None"), (".", "Point"), ("o", "Circle"), ("s", "Square"),
    ("^", "Triangle Up"), ("v", "Triangle Down"), ("D", "Diamond"),
    ("*", "Star"), ("+", "Plus"), ("x", "Cross"),
]
_PLOT_TYPES = [
    ("line", "Line"), ("scatter", "Scatter"), ("bar", "Bar"),
    ("histogram", "Histogram"), ("box", "Box Plot"), ("stem", "Stem"),
    ("step", "Step"), ("area", "Area"),
]
_FIT_TYPES = [
    ("none", "None"), ("poly", "Polynomial"), ("exp", "Exponential"),
    ("power", "Power"), ("log", "Logarithmic"),
]
_COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Spectral", "coolwarm", "RdYlBu", "RdBu", "PiYG",
    "jet", "rainbow", "turbo", "hot", "cool",
]
_FONT_FAMILIES = ["sans-serif", "serif", "monospace", "cursive", "fantasy"]

# Accordion section header style
_SECTION_STYLE = """
    CollapsibleSection { border: none; }
    CollapsibleSection > QPushButton#sectionHeader {
        text-align: left;
        padding: 4px 8px;
        border: none;
        border-bottom: 1px solid #ccc;
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                     stop:0 #f0f0f0, stop:1 #e4e4e4);
        font-weight: bold;
        font-size: 11px;
    }
    CollapsibleSection > QPushButton#sectionHeader:hover {
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                     stop:0 #e8e8e8, stop:1 #d8d8d8);
    }
"""

# Compact panel stylesheet — constrains widget sizes for narrow panel
_PANEL_STYLE = """
    QDoubleSpinBox, QSpinBox { max-width: 70px; min-width: 50px; }
    QLineEdit { max-width: 140px; }
    QLabel { font-size: 11px; }
    QCheckBox { font-size: 11px; spacing: 3px; margin: 1px 0; }
    QGroupBox { font-size: 11px; padding-top: 10px; margin-top: 4px; }
    QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }
    QPushButton { padding: 3px 8px; }
"""

# --------------------------------------------------------- collapsible section

class CollapsibleSection(QWidget):
    """A section with a clickable header that shows/hides its content."""

    def __init__(self, title, parent=None, expanded=False):
        super().__init__(parent)
        self.setStyleSheet(_SECTION_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._btn = QPushButton(f"  \u25B6  {title}")
        self._btn.setObjectName("sectionHeader")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(self.toggle)
        layout.addWidget(self._btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(6)
        layout.addWidget(self._content)

        self._title = title
        self._expanded = False
        if expanded:
            self.expand()
        else:
            self.collapse()

    def content_layout(self):
        return self._content_layout

    def toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        self._expanded = True
        self._content.show()
        self._btn.setText(f"  \u25BC  {self._title}")

    def collapse(self):
        self._expanded = False
        self._content.hide()
        self._btn.setText(f"  \u25B6  {self._title}")


# ----------------------------------------------------------- helper widgets

class NoScrollComboBox(QComboBox):
    """QComboBox that ignores scroll-wheel events — select by click only."""
    def wheelEvent(self, event):
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores scroll-wheel events."""
    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores scroll-wheel events."""
    def wheelEvent(self, event):
        event.ignore()


class _ScaleFactorEdit(QLineEdit):
    """QLineEdit that accepts numeric values including scientific notation."""
    valueChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 1.0
        self.setText("1")
        self.setMaximumWidth(100)
        self.editingFinished.connect(self._on_edited)

    def _on_edited(self):
        try:
            v = float(self.text())
        except ValueError:
            self.setText(self._format(self._value))
            return
        if v != self._value:
            self._value = v
            self.valueChanged.emit()

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = float(v)
        self.setText(self._format(self._value))

    @staticmethod
    def _format(v):
        if v == 0:
            return "0"
        if abs(v) >= 0.001 and abs(v) < 1e6:
            s = f"{v:g}"
        else:
            s = f"{v:.6e}"
        return s


def _make_combo(*items, data=None):
    c = NoScrollComboBox()
    if data:
        for d, label in data:
            c.addItem(label, d)
    else:
        for item in items:
            c.addItem(item)
    return c


def _make_color_btn(hex_color="#000000"):
    btn = QPushButton()
    btn.setFixedSize(36, 18)
    btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #888; padding: 0;")
    return btn

def _pick_color(parent, current_hex, callback):
    color = QColorDialog.getColor(QColor(current_hex), parent, "Pick Color")
    if color.isValid():
        callback(color.name())

def _make_fs_spin(value=12, lo=6, hi=48):
    s = NoScrollSpinBox(); s.setRange(lo, hi); s.setValue(value)
    s.setMaximumWidth(60)
    return s

def _make_font_combo():
    c = NoScrollComboBox()
    c.setMaximumWidth(110)
    for fam in _FONT_FAMILIES:
        c.addItem(fam)
    return c

def _set_btn_color(btn, hex_c):
    btn.setStyleSheet(f"background-color: {hex_c}; border: 1px solid #888; padding: 0;")


# ============================================================= main dialog

class PlotDataDialog(QDialog):
    """Full-featured interactive 2D plot dialog with accordion settings."""

    def __init__(self, series_data, parent=None, block=None):
        super().__init__(parent)
        self.setWindowTitle("2D Plot")
        self.resize(1250, 720)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._block = block
        self._series_data = series_data
        self._blocking = False
        self._annotations = []
        self._ref_lines = []
        self._ax2 = None
        self._colorbar = None
        self._colorbar_mappable = None
        self._fit_texts = []
        self._subplot_axes = []

        self._settings = self._init_series_settings()

        # Subplot configuration
        n = len(series_data)
        self._subplot_assignments = [list(range(n))]
        self._n_subplots = 1

        self.fig = Figure(figsize=(8, 5), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._hover_idx = None
        self._hover_highlight = None
        self.toolbar = NavigationToolbar(self.canvas, self)

        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setWidgetResizable(True)
        self._canvas_scroll.setWidget(self.canvas)
        self._canvas_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._canvas_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(self.toolbar)
        left_l.addWidget(self._canvas_scroll, stretch=1)

        right = self._build_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 350])

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(splitter)

        self._restore_plot_settings()
        self._replot()
        self._sync_controls_to_series(0)

    # --------------------------------------------------- init
    def _init_series_settings(self):
        settings = []
        for i, sd in enumerate(self._series_data):
            settings.append({
                "color": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                "plot_type": "line", "line_width": 1.5, "line_style": "-",
                "marker": "None", "marker_size": 6,
                "marker_fill": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                "marker_edge": "#000000", "marker_edge_width": 0.5,
                "opacity": 1.0,
                "visible": True, "x_offset": 0.0, "y_offset": 0.0,
                "normalize": False, "secondary_y": False, "exclude_outliers": False,
                "scatter_color_by": "uniform", "scatter_cmap": "viridis",
                "scatter_size": 20, "bar_width": 0.8,
                "hist_bins": 30, "hist_alpha": 0.7, "box_showfliers": True,
                "fit_type": "none", "fit_degree": 1,
                "fit_color": "#ff0000", "fit_show_eq": True, "fit_show_r2": True,
            })
        return settings

    def _on_continue(self):
        self._save_plot_settings()
        self.accept()

    # --------------------------------------------------- settings persistence
    def _collect_plot_settings(self):
        cfg = {}
        cfg["series"] = list(self._settings)
        cfg["figure"] = {
            "n_subplots": self._n_subplots,
            "arrange": self._arrange_combo.currentText(),
            "share_x": self._share_x_cb.isChecked(),
            "fig_width": self._fig_width.value(),
            "fig_height": self._fig_height.value(),
            "dialog_width": self.width(),
            "dialog_height": self.height(),
            "fig_size_pinned": self.canvas.maximumWidth() < 16777215,
            "fig_bg": self._fig_bg_hex,
            "axes_bg": self._axes_bg_hex,
            "grid": self._grid_cb.isChecked(),
            "grid_color": self._grid_color_hex,
            "grid_style": self._grid_style.currentData(),
            "subplot_assignments": self._subplot_assignments,
        }
        cfg["labels"] = {
            "title": self._title_edit.text(),
            "title_fs": self._title_fs.value(),
            "title_color": self._title_color_hex,
            "title_bold": self._title_bold.isChecked(),
            "title_italic": self._title_italic.isChecked(),
            "title_tex": self._title_tex.isChecked(),
            "xlabel": self._xlabel_edit.text(),
            "xlabel_fs": self._xlabel_fs.value(),
            "xlabel_color": self._xlabel_color_hex,
            "xlabel_bold": self._xlabel_bold.isChecked(),
            "xlabel_italic": self._xlabel_italic.isChecked(),
            "xlabel_tex": self._xlabel_tex.isChecked(),
            "ylabel": self._ylabel_edit.text(),
            "ylabel_fs": self._ylabel_fs.value(),
            "ylabel_color": self._ylabel_color_hex,
            "ylabel_bold": self._ylabel_bold.isChecked(),
            "ylabel_italic": self._ylabel_italic.isChecked(),
            "ylabel_tex": self._ylabel_tex.isChecked(),
            "label_font": self._label_font.currentText(),
        }
        cfg["ticks"] = {
            "font": self._tick_font.currentText(),
            "fs": self._tick_fs.value(),
            "color": self._tick_color_hex,
            "rotation_x": self._tick_rotation_x.value(),
            "rotation_y": self._tick_rotation_y.value(),
            "minor_show": self._minor_tick_show.isChecked(),
        }
        cfg["axes"] = {
            "x_scale_factor": self._x_scale_factor.value(),
            "x_auto": self._x_auto.isChecked(),
            "x_min": self._x_min.value(), "x_max": self._x_max.value(),
            "x_scale": self._x_scale.currentText(),
            "x_invert": self._x_invert.isChecked(),
            "y_scale_factor": self._y_scale_factor.value(),
            "y_auto": self._y_auto.isChecked(),
            "y_min": self._y_min.value(), "y_max": self._y_max.value(),
            "y_scale": self._y_scale.currentText(),
            "y_invert": self._y_invert.isChecked(),
            "aspect": self._aspect.currentText(),
        }
        cfg["legend"] = {
            "show": self._legend_cb.isChecked(),
            "pos": self._legend_pos.currentText(),
            "fs": self._legend_fs.value(),
        }
        cfg["series_names"] = [sd["label"] for sd in self._series_data]
        cfg["annotations"] = list(self._annotations)
        cfg["ref_lines"] = list(self._ref_lines)
        return cfg

    def _save_plot_settings(self):
        if not self._block:
            return
        self._block.parameters["plotSettings"] = self._collect_plot_settings()

    def _restore_plot_settings(self):
        if not self._block:
            return
        cfg = self._block.parameters.get("plotSettings")
        if not cfg:
            return
        self._blocking = True
        try:
            # Per-series settings (handle legacy line_alpha/marker_alpha)
            saved_series = cfg.get("series", [])
            for i, s in enumerate(saved_series):
                if i < len(self._settings):
                    # Migrate legacy separate alpha to unified opacity
                    if "line_alpha" in s and "opacity" not in s:
                        s["opacity"] = s.pop("line_alpha")
                        s.pop("marker_alpha", None)
                    self._settings[i].update(s)

            # Figure
            fig = cfg.get("figure", {})
            self._n_subplots = fig.get("n_subplots", 1)
            self._n_subplots_spin.setValue(self._n_subplots)
            idx = self._arrange_combo.findText(fig.get("arrange", "Vertical"))
            if idx >= 0: self._arrange_combo.setCurrentIndex(idx)
            self._share_x_cb.setChecked(fig.get("share_x", True))
            self._fig_width.setValue(fig.get("fig_width", 8.0))
            self._fig_height.setValue(fig.get("fig_height", 5.0))
            self._fig_bg_hex = fig.get("fig_bg", "#ffffff")
            _set_btn_color(self._fig_bg_btn, self._fig_bg_hex)
            self._axes_bg_hex = fig.get("axes_bg", "#ffffff")
            _set_btn_color(self._axes_bg_btn, self._axes_bg_hex)
            self._grid_cb.setChecked(fig.get("grid", True))
            self._grid_color_hex = fig.get("grid_color", "#cccccc")
            _set_btn_color(self._grid_color_btn, self._grid_color_hex)
            gs = fig.get("grid_style", ":")
            for i in range(self._grid_style.count()):
                if self._grid_style.itemData(i) == gs:
                    self._grid_style.setCurrentIndex(i); break
            self._subplot_assignments = fig.get("subplot_assignments", [list(range(len(self._series_data)))])

            dw = fig.get("dialog_width")
            dh = fig.get("dialog_height")
            if dw and dh:
                self.resize(int(dw), int(dh))
            if fig.get("fig_size_pinned", False):
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, self._apply_fig_size)

            # Labels
            lbl = cfg.get("labels", {})
            self._title_edit.setText(lbl.get("title", ""))
            self._title_fs.setValue(lbl.get("title_fs", 14))
            self._title_color_hex = lbl.get("title_color", "#000000")
            _set_btn_color(self._title_color_btn, self._title_color_hex)
            self._title_bold.setChecked(lbl.get("title_bold", False))
            self._title_italic.setChecked(lbl.get("title_italic", False))
            self._title_tex.setChecked(lbl.get("title_tex", False))
            self._xlabel_edit.setText(lbl.get("xlabel", "X"))
            self._xlabel_fs.setValue(lbl.get("xlabel_fs", 12))
            self._xlabel_color_hex = lbl.get("xlabel_color", "#000000")
            _set_btn_color(self._xlabel_color_btn, self._xlabel_color_hex)
            self._xlabel_bold.setChecked(lbl.get("xlabel_bold", False))
            self._xlabel_italic.setChecked(lbl.get("xlabel_italic", False))
            self._xlabel_tex.setChecked(lbl.get("xlabel_tex", False))
            self._ylabel_edit.setText(lbl.get("ylabel", "Y"))
            self._ylabel_fs.setValue(lbl.get("ylabel_fs", 12))
            self._ylabel_color_hex = lbl.get("ylabel_color", "#000000")
            _set_btn_color(self._ylabel_color_btn, self._ylabel_color_hex)
            self._ylabel_bold.setChecked(lbl.get("ylabel_bold", False))
            self._ylabel_italic.setChecked(lbl.get("ylabel_italic", False))
            self._ylabel_tex.setChecked(lbl.get("ylabel_tex", False))
            idx = self._label_font.findText(lbl.get("label_font", "sans-serif"))
            if idx >= 0: self._label_font.setCurrentIndex(idx)

            # Ticks
            tk = cfg.get("ticks", {})
            idx = self._tick_font.findText(tk.get("font", "sans-serif"))
            if idx >= 0: self._tick_font.setCurrentIndex(idx)
            self._tick_fs.setValue(tk.get("fs", 10))
            self._tick_color_hex = tk.get("color", "#000000")
            _set_btn_color(self._tick_color_btn, self._tick_color_hex)
            self._tick_rotation_x.setValue(tk.get("rotation_x", 0))
            self._tick_rotation_y.setValue(tk.get("rotation_y", 0))
            self._minor_tick_show.setChecked(tk.get("minor_show", False))

            # Axes
            ax = cfg.get("axes", {})
            self._x_scale_factor.setValue(ax.get("x_scale_factor", 1.0))
            self._x_auto.setChecked(ax.get("x_auto", True))
            self._x_min.setValue(ax.get("x_min", 0))
            self._x_max.setValue(ax.get("x_max", 0))
            idx = self._x_scale.findText(ax.get("x_scale", "Linear"))
            if idx >= 0: self._x_scale.setCurrentIndex(idx)
            self._x_invert.setChecked(ax.get("x_invert", False))
            self._y_scale_factor.setValue(ax.get("y_scale_factor", 1.0))
            self._y_auto.setChecked(ax.get("y_auto", True))
            self._y_min.setValue(ax.get("y_min", 0))
            self._y_max.setValue(ax.get("y_max", 0))
            idx = self._y_scale.findText(ax.get("y_scale", "Linear"))
            if idx >= 0: self._y_scale.setCurrentIndex(idx)
            self._y_invert.setChecked(ax.get("y_invert", False))
            idx = self._aspect.findText(ax.get("aspect", "Auto"))
            if idx >= 0: self._aspect.setCurrentIndex(idx)

            # Legend
            leg = cfg.get("legend", {})
            self._legend_cb.setChecked(leg.get("show", True))
            idx = self._legend_pos.findText(leg.get("pos", "best"))
            if idx >= 0: self._legend_pos.setCurrentIndex(idx)
            self._legend_fs.setValue(leg.get("fs", 10))

            # Series names
            saved_names = cfg.get("series_names", [])
            for i, name in enumerate(saved_names):
                if i < len(self._series_data):
                    self._series_data[i]["label"] = name
                    self._series_combo.setItemText(i, name)

            self._annotations = cfg.get("annotations", [])
            self._ref_lines = cfg.get("ref_lines", [])
        finally:
            self._blocking = False

    # ========================================================= PANEL BUILDER
    def _build_panel(self):
        panel = QWidget()
        panel.setStyleSheet(_PANEL_STYLE)
        panel.setMinimumWidth(280)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Top bar: Plot type + Series selector ---
        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(6, 4, 6, 2)
        top_l.setSpacing(2)
        # Plot type row
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._plot_type_combo = NoScrollComboBox()
        for key, label in _PLOT_TYPES:
            self._plot_type_combo.addItem(label, key)
        self._plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)
        type_row.addWidget(self._plot_type_combo, stretch=1)
        top_l.addLayout(type_row)
        # Series row
        ser_row = QHBoxLayout()
        ser_row.addWidget(QLabel("Series:"))
        self._series_combo = NoScrollComboBox()
        for sd in self._series_data:
            self._series_combo.addItem(sd["label"])
        self._series_combo.currentIndexChanged.connect(self._on_series_changed)
        ser_row.addWidget(self._series_combo, stretch=1)
        top_l.addLayout(ser_row)
        outer.addWidget(top)

        # --- Scrollable accordion ---
        self._panel_scroll = scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        self._sections_layout = QVBoxLayout(container)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(0)

        self._sections = {}
        self._build_section_series_type()   # 1
        self._build_section_labels()        # 2
        self._build_section_axes()          # 3
        self._build_section_ticks()         # 4
        self._build_section_figure()        # 5
        self._build_section_fit()           # 6
        self._build_section_legend()        # 7
        self._build_section_annotate()      # 8
        self._build_section_export()        # 9

        self._sections_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll, stretch=1)

        # --- Bottom bar ---
        bot = QWidget()
        bot_l = QHBoxLayout(bot)
        bot_l.setContentsMargins(8, 4, 8, 4)
        btn_cont = QPushButton("Continue")
        btn_cont.clicked.connect(self._on_continue)
        bot_l.addStretch()
        bot_l.addWidget(btn_cont)
        outer.addWidget(bot)

        return panel

    # ==================== 1. Series & Type ===================================
    def _build_section_series_type(self):
        sec = CollapsibleSection("Series && Type", self, expanded=False)
        self._sections["series"] = sec
        L = sec.content_layout()

        # --- Core series controls ---
        f = QFormLayout()
        self._series_name_edit = QLineEdit()
        self._series_name_edit.editingFinished.connect(self._on_series_name_changed)
        f.addRow("Name:", self._series_name_edit)
        self._line_color_btn = QPushButton()
        self._line_color_btn.setFixedSize(48, 24)
        self._line_color_btn.clicked.connect(self._pick_line_color)
        f.addRow("Color:", self._line_color_btn)
        self._visible_cb = QCheckBox("Visible"); self._visible_cb.setChecked(True)
        self._visible_cb.stateChanged.connect(self._apply_data_settings)
        f.addRow(self._visible_cb)
        L.addLayout(f)

        # --- Line & Marker (shown for line/step/area/stem) ---
        self._line_marker_w = QWidget()
        lmf = QFormLayout(self._line_marker_w)
        lmf.setContentsMargins(0, 4, 0, 0)
        self._line_width = NoScrollDoubleSpinBox()
        self._line_width.setRange(0.1, 10); self._line_width.setSingleStep(0.25)
        self._line_width.setValue(1.5); self._line_width.valueChanged.connect(self._apply_series_style)
        self._line_style_combo = NoScrollComboBox()
        for key, label in _LINESTYLES:
            self._line_style_combo.addItem(label, key)
        self._line_style_combo.currentIndexChanged.connect(self._apply_series_style)
        self._marker_combo = NoScrollComboBox()
        for key, label in _MARKERS:
            self._marker_combo.addItem(label, key)
        self._marker_combo.currentIndexChanged.connect(self._apply_series_style)
        self._marker_size = NoScrollDoubleSpinBox()
        self._marker_size.setRange(1, 30); self._marker_size.setValue(6)
        self._marker_size.valueChanged.connect(self._apply_series_style)
        self._marker_fill_btn = _make_color_btn(_DEFAULT_COLORS[0])
        self._marker_fill_btn.clicked.connect(self._pick_marker_fill)
        self._marker_edge_btn = _make_color_btn("#000000")
        self._marker_edge_btn.clicked.connect(self._pick_marker_edge)
        self._marker_edge_width = NoScrollDoubleSpinBox()
        self._marker_edge_width.setRange(0, 5); self._marker_edge_width.setSingleStep(0.25)
        self._marker_edge_width.setValue(0.5)
        self._marker_edge_width.valueChanged.connect(self._apply_series_style)
        lmf.addRow("Width:", self._line_width)
        lmf.addRow("Style:", self._line_style_combo)
        lmf.addRow("Marker:", self._marker_combo)
        lmf.addRow("Size:", self._marker_size)
        fill_edge = QHBoxLayout()
        fill_edge.addWidget(QLabel("Fill:")); fill_edge.addWidget(self._marker_fill_btn)
        fill_edge.addSpacing(6)
        fill_edge.addWidget(QLabel("Edge:")); fill_edge.addWidget(self._marker_edge_btn)
        fill_edge.addSpacing(6)
        fill_edge.addWidget(QLabel("W:")); fill_edge.addWidget(self._marker_edge_width)
        lmf.addRow("Marker:", fill_edge)
        L.addWidget(self._line_marker_w)

        # --- Opacity (single control) ---
        of = QFormLayout()
        self._opacity = NoScrollDoubleSpinBox()
        self._opacity.setRange(0, 1); self._opacity.setSingleStep(0.1)
        self._opacity.setValue(1.0); self._opacity.setDecimals(2)
        self._opacity.valueChanged.connect(self._apply_series_style)
        of.addRow("Opacity:", self._opacity)
        L.addLayout(of)

        # --- Scatter options ---
        self._scatter_w = QWidget()
        sf = QFormLayout(self._scatter_w); sf.setContentsMargins(0, 4, 0, 0)
        self._scatter_size = NoScrollDoubleSpinBox()
        self._scatter_size.setRange(1, 200); self._scatter_size.setValue(20)
        self._scatter_size.valueChanged.connect(self._apply_series_style)
        self._scatter_fill_btn = _make_color_btn(_DEFAULT_COLORS[0])
        self._scatter_fill_btn.clicked.connect(self._pick_marker_fill)
        self._scatter_edge_btn = _make_color_btn("#000000")
        self._scatter_edge_btn.clicked.connect(self._pick_marker_edge)
        self._scatter_edge_width = NoScrollDoubleSpinBox()
        self._scatter_edge_width.setRange(0, 5); self._scatter_edge_width.setSingleStep(0.25)
        self._scatter_edge_width.setValue(0.5)
        self._scatter_edge_width.valueChanged.connect(self._apply_series_style)
        self._scatter_color_by = NoScrollComboBox()
        self._scatter_color_by.addItems(["Uniform", "By X", "By Y"])
        self._scatter_color_by.currentIndexChanged.connect(self._apply_series_style)
        self._scatter_cmap = NoScrollComboBox(); self._scatter_cmap.addItems(_COLORMAPS)
        self._scatter_cmap.currentIndexChanged.connect(self._apply_series_style)
        sf.addRow("Point Size:", self._scatter_size)
        sc_fill_edge = QHBoxLayout()
        sc_fill_edge.addWidget(QLabel("Fill:")); sc_fill_edge.addWidget(self._scatter_fill_btn)
        sc_fill_edge.addSpacing(6)
        sc_fill_edge.addWidget(QLabel("Edge:")); sc_fill_edge.addWidget(self._scatter_edge_btn)
        sc_fill_edge.addSpacing(6)
        sc_fill_edge.addWidget(QLabel("W:")); sc_fill_edge.addWidget(self._scatter_edge_width)
        sf.addRow("Marker:", sc_fill_edge)
        sf.addRow("Color By:", self._scatter_color_by)
        sf.addRow("Colormap:", self._scatter_cmap)
        L.addWidget(self._scatter_w)

        # --- Bar options ---
        self._bar_w = QWidget()
        bf = QFormLayout(self._bar_w); bf.setContentsMargins(0, 4, 0, 0)
        self._bar_width = NoScrollDoubleSpinBox()
        self._bar_width.setRange(0.01, 5); self._bar_width.setSingleStep(0.1)
        self._bar_width.setValue(0.8); self._bar_width.valueChanged.connect(self._apply_series_style)
        bf.addRow("Bar Width:", self._bar_width)
        L.addWidget(self._bar_w)

        # --- Histogram options ---
        self._hist_w = QWidget()
        hf = QFormLayout(self._hist_w); hf.setContentsMargins(0, 4, 0, 0)
        self._hist_bins = NoScrollSpinBox(); self._hist_bins.setRange(2, 500); self._hist_bins.setValue(30)
        self._hist_bins.valueChanged.connect(self._apply_series_style)
        self._hist_alpha = NoScrollDoubleSpinBox()
        self._hist_alpha.setRange(0, 1); self._hist_alpha.setSingleStep(0.1); self._hist_alpha.setValue(0.7)
        self._hist_alpha.valueChanged.connect(self._apply_series_style)
        hf.addRow("Bins:", self._hist_bins)
        hf.addRow("Alpha:", self._hist_alpha)
        L.addWidget(self._hist_w)

        # --- Box plot options ---
        self._box_w = QWidget()
        bxf = QFormLayout(self._box_w); bxf.setContentsMargins(0, 4, 0, 0)
        self._box_showfliers = QCheckBox("Show Outliers")
        self._box_showfliers.setChecked(True)
        self._box_showfliers.toggled.connect(self._apply_series_style)
        bxf.addRow(self._box_showfliers)
        L.addWidget(self._box_w)

        # --- Data Transform ---
        L.addSpacing(4)
        lbl = QLabel("Data Transform"); lbl.setStyleSheet("font-weight: bold;"); L.addWidget(lbl)
        tf = QFormLayout()
        self._x_offset = NoScrollDoubleSpinBox(); self._x_offset.setDecimals(6); self._x_offset.setRange(-1e15, 1e15)
        self._x_offset.valueChanged.connect(self._apply_data_settings)
        tf.addRow("X Offset:", self._x_offset)
        self._y_offset = NoScrollDoubleSpinBox(); self._y_offset.setDecimals(6); self._y_offset.setRange(-1e15, 1e15)
        self._y_offset.valueChanged.connect(self._apply_data_settings)
        tf.addRow("Y Offset:", self._y_offset)
        self._exclude_outliers_cb = QCheckBox("Exclude Outliers (IQR)")
        self._exclude_outliers_cb.stateChanged.connect(self._apply_data_settings)
        tf.addRow(self._exclude_outliers_cb)
        self._normalize_cb = QCheckBox("Normalize (0\u20131)")
        self._normalize_cb.stateChanged.connect(self._apply_data_settings)
        tf.addRow(self._normalize_cb)
        self._secondary_y_cb = QCheckBox("Secondary Y-axis (right)")
        self._secondary_y_cb.stateChanged.connect(self._apply_data_settings)
        tf.addRow(self._secondary_y_cb)
        L.addLayout(tf)

        self._update_style_visibility("line")
        self._sections_layout.addWidget(sec)

    # ==================== 2. Labels ==========================================
    def _build_section_labels(self):
        sec = CollapsibleSection("Labels", self, expanded=False)
        self._sections["labels"] = sec
        L = sec.content_layout()

        # Global font family
        ff = QFormLayout()
        self._label_font = _make_font_combo()
        self._label_font.currentIndexChanged.connect(self._apply_labels)
        ff.addRow("Font:", self._label_font)
        L.addLayout(ff)

        def add_label_group(heading, default="", default_fs=12):
            lbl = QLabel(heading); lbl.setStyleSheet("font-weight: bold;")
            L.addWidget(lbl)
            f = QFormLayout()
            edit = QLineEdit(default)
            fs = _make_fs_spin(default_fs)
            color_hex = "#000000"
            color_btn = _make_color_btn(color_hex)
            bold = QCheckBox("B"); italic = QCheckBox("I"); tex = QCheckBox("TeX")
            if heading == "Title":
                bold.setChecked(True)
            f.addRow("Text:", edit)
            row = QHBoxLayout()
            row.addWidget(QLabel("Size:")); row.addWidget(fs, stretch=1)
            bold.setFixedWidth(30); row.addWidget(bold)
            italic.setFixedWidth(26); row.addWidget(italic)
            tex.setFixedWidth(40); row.addWidget(tex)
            f.addRow("Style:", row)
            cr = QHBoxLayout()
            cr.addWidget(QLabel("Color:")); cr.addWidget(color_btn); cr.addStretch()
            f.addRow(cr)
            L.addLayout(f)
            return edit, fs, color_btn, color_hex, bold, italic, tex

        self._title_edit, self._title_fs, self._title_color_btn, \
            self._title_color_hex, self._title_bold, self._title_italic, self._title_tex = \
            add_label_group("Title", default_fs=14)
        self._title_color_btn.clicked.connect(lambda: _pick_color(
            self, self._title_color_hex, self._set_title_color))
        L.addSpacing(4)
        self._xlabel_edit, self._xlabel_fs, self._xlabel_color_btn, \
            self._xlabel_color_hex, self._xlabel_bold, self._xlabel_italic, self._xlabel_tex = \
            add_label_group("X Label", "X")
        self._xlabel_color_btn.clicked.connect(lambda: _pick_color(
            self, self._xlabel_color_hex, self._set_xlabel_color))
        L.addSpacing(4)
        self._ylabel_edit, self._ylabel_fs, self._ylabel_color_btn, \
            self._ylabel_color_hex, self._ylabel_bold, self._ylabel_italic, self._ylabel_tex = \
            add_label_group("Y Label", "Y")
        self._ylabel_color_btn.clicked.connect(lambda: _pick_color(
            self, self._ylabel_color_hex, self._set_ylabel_color))

        for w in (self._title_edit, self._xlabel_edit, self._ylabel_edit):
            w.editingFinished.connect(self._apply_labels)
        for w in (self._title_fs, self._xlabel_fs, self._ylabel_fs):
            w.valueChanged.connect(self._apply_labels)
        for w in (self._title_bold, self._title_italic, self._title_tex,
                  self._xlabel_bold, self._xlabel_italic, self._xlabel_tex,
                  self._ylabel_bold, self._ylabel_italic, self._ylabel_tex):
            w.stateChanged.connect(self._apply_labels)

        self._sections_layout.addWidget(sec)

    # ==================== 3. Axes ============================================
    def _build_section_axes(self):
        sec = CollapsibleSection("Axes", self, expanded=False)
        self._sections["axes"] = sec
        L = sec.content_layout()
        for axis_name, prefix in [("X Axis", "x"), ("Y Axis", "y")]:
            lbl = QLabel(axis_name); lbl.setStyleSheet("font-weight: bold;"); L.addWidget(lbl)
            f = QFormLayout()
            sf = _ScaleFactorEdit()
            sf.valueChanged.connect(self._replot)
            f.addRow("Scale Factor:", sf)
            auto = QCheckBox("Auto Limits"); auto.setChecked(True)
            auto.stateChanged.connect(self._on_axis_auto_changed)
            f.addRow(auto)
            mn = NoScrollDoubleSpinBox(); mn.setDecimals(6); mn.setRange(-1e15, 1e15); mn.setEnabled(False)
            mx = NoScrollDoubleSpinBox(); mx.setDecimals(6); mx.setRange(-1e15, 1e15); mx.setEnabled(False)
            f.addRow("Min:", mn); f.addRow("Max:", mx)
            scale = NoScrollComboBox(); scale.addItems(["Linear", "Log"])
            scale.currentIndexChanged.connect(self._apply_axes)
            f.addRow("Scale:", scale)
            inv = QCheckBox("Invert"); inv.stateChanged.connect(self._apply_axes)
            f.addRow(inv)
            setattr(self, f"_{prefix}_scale_factor", sf)
            setattr(self, f"_{prefix}_auto", auto)
            setattr(self, f"_{prefix}_min", mn)
            setattr(self, f"_{prefix}_max", mx)
            setattr(self, f"_{prefix}_scale", scale)
            setattr(self, f"_{prefix}_invert", inv)
            L.addLayout(f)
            L.addSpacing(4)

        af = QFormLayout()
        self._aspect = NoScrollComboBox(); self._aspect.addItems(["Auto", "Equal"])
        self._aspect.currentIndexChanged.connect(self._apply_axes)
        af.addRow("Aspect:", self._aspect)
        L.addLayout(af)
        btn = QPushButton("Apply Limits"); btn.clicked.connect(self._apply_axes)
        L.addWidget(btn)

        # --- Colorbar ---
        L.addSpacing(6)
        lbl2 = QLabel("Colorbar"); lbl2.setStyleSheet("font-weight: bold;"); L.addWidget(lbl2)
        cf = QFormLayout()
        self._cbar_enable = QCheckBox("Show Colorbar")
        self._cbar_enable.stateChanged.connect(self._replot)
        cf.addRow(self._cbar_enable)
        self._cbar_label = QLineEdit(); self._cbar_label.setPlaceholderText("Colorbar label")
        self._cbar_label.editingFinished.connect(self._replot)
        cf.addRow("Label:", self._cbar_label)
        self._cbar_orient = NoScrollComboBox(); self._cbar_orient.addItems(["vertical", "horizontal"])
        self._cbar_orient.currentIndexChanged.connect(self._replot)
        cf.addRow("Orient:", self._cbar_orient)
        L.addLayout(cf)

        self._sections_layout.addWidget(sec)

    # ==================== 4. Ticks ===========================================
    def _build_section_ticks(self):
        sec = CollapsibleSection("Ticks", self, expanded=False)
        self._sections["ticks"] = sec
        L = sec.content_layout()
        f = QFormLayout()
        self._tick_font = _make_font_combo()
        self._tick_font.currentIndexChanged.connect(self._apply_ticks)
        f.addRow("Font:", self._tick_font)
        self._tick_fs = _make_fs_spin(10, 6, 36)
        self._tick_fs.valueChanged.connect(self._apply_ticks)
        self._tick_color_hex = "#000000"
        self._tick_color_btn = _make_color_btn(self._tick_color_hex)
        self._tick_color_btn.clicked.connect(lambda: _pick_color(
            self, self._tick_color_hex, self._set_tick_color))
        sct = QHBoxLayout()
        sct.addWidget(QLabel("Size:")); sct.addWidget(self._tick_fs, stretch=1)
        sct.addWidget(QLabel("Color:")); sct.addWidget(self._tick_color_btn)
        f.addRow(sct)
        self._tick_rotation_x = NoScrollSpinBox(); self._tick_rotation_x.setRange(-90, 90)
        self._tick_rotation_x.valueChanged.connect(self._apply_ticks)
        f.addRow("X Rotation:", self._tick_rotation_x)
        self._tick_rotation_y = NoScrollSpinBox(); self._tick_rotation_y.setRange(-90, 90)
        self._tick_rotation_y.valueChanged.connect(self._apply_ticks)
        f.addRow("Y Rotation:", self._tick_rotation_y)
        self._minor_tick_show = QCheckBox("Show Minor Ticks")
        self._minor_tick_show.stateChanged.connect(self._apply_ticks)
        f.addRow(self._minor_tick_show)
        L.addLayout(f)
        self._sections_layout.addWidget(sec)

    # ==================== 5. Figure ==========================================
    def _build_section_figure(self):
        sec = CollapsibleSection("Figure", self, expanded=False)
        self._sections["figure"] = sec
        L = sec.content_layout()

        # --- Background ---
        lbl = QLabel("Background"); lbl.setStyleSheet("font-weight: bold;"); L.addWidget(lbl)
        fb = QFormLayout()
        self._fig_bg_hex = "#ffffff"
        self._fig_bg_btn = _make_color_btn(self._fig_bg_hex)
        self._fig_bg_btn.clicked.connect(lambda: _pick_color(
            self, self._fig_bg_hex, self._set_fig_bg))
        fb.addRow("Figure:", self._fig_bg_btn)
        self._axes_bg_hex = "#ffffff"
        self._axes_bg_btn = _make_color_btn(self._axes_bg_hex)
        self._axes_bg_btn.clicked.connect(lambda: _pick_color(
            self, self._axes_bg_hex, self._set_axes_bg))
        fb.addRow("Axes:", self._axes_bg_btn)
        L.addLayout(fb)
        L.addSpacing(4)

        # --- Grid ---
        lbl2 = QLabel("Grid"); lbl2.setStyleSheet("font-weight: bold;"); L.addWidget(lbl2)
        fg = QFormLayout()
        self._grid_cb = QCheckBox("Show Grid"); self._grid_cb.setChecked(True)
        self._grid_cb.stateChanged.connect(self._apply_figure_settings)
        fg.addRow(self._grid_cb)
        self._grid_color_hex = "#cccccc"
        self._grid_color_btn = _make_color_btn(self._grid_color_hex)
        self._grid_color_btn.clicked.connect(lambda: _pick_color(
            self, self._grid_color_hex, self._set_grid_color))
        self._grid_style = NoScrollComboBox()
        for key, label in _LINESTYLES[:4]:
            self._grid_style.addItem(label, key)
        self._grid_style.setCurrentIndex(3)
        self._grid_style.currentIndexChanged.connect(self._apply_figure_settings)
        gr = QHBoxLayout()
        gr.addWidget(QLabel("Color:")); gr.addWidget(self._grid_color_btn)
        gr.addWidget(QLabel("Style:")); gr.addWidget(self._grid_style, stretch=1)
        fg.addRow(gr)
        L.addLayout(fg)
        L.addSpacing(4)

        # --- Subplots ---
        lbl3 = QLabel("Subplots"); lbl3.setStyleSheet("font-weight: bold;"); L.addWidget(lbl3)
        fl = QFormLayout()
        self._n_subplots_spin = NoScrollSpinBox()
        self._n_subplots_spin.setRange(1, max(len(self._series_data), 6))
        self._n_subplots_spin.setValue(1)
        self._n_subplots_spin.valueChanged.connect(self._on_subplot_count_changed)
        fl.addRow("Count:", self._n_subplots_spin)
        self._arrange_combo = NoScrollComboBox()
        self._arrange_combo.addItems(["Vertical", "Horizontal", "Grid"])
        self._arrange_combo.currentIndexChanged.connect(self._replot)
        fl.addRow("Arrange:", self._arrange_combo)
        self._share_x_cb = QCheckBox("Share X axis"); self._share_x_cb.setChecked(True)
        self._share_x_cb.stateChanged.connect(self._replot)
        fl.addRow(self._share_x_cb)
        L.addLayout(fl)

        self._subplot_assign_container = QVBoxLayout()
        L.addLayout(self._subplot_assign_container)
        self._subplot_assign_combos = []
        self._rebuild_subplot_assignments()
        L.addSpacing(4)

        # --- Figure size (inches only) ---
        lbl4 = QLabel("Size"); lbl4.setStyleSheet("font-weight: bold;"); L.addWidget(lbl4)
        fs = QFormLayout()
        self._fig_width = NoScrollDoubleSpinBox(); self._fig_width.setDecimals(1)
        self._fig_width.setSingleStep(0.5); self._fig_width.setRange(2, 30)
        self._fig_width.setValue(self.fig.get_figwidth())
        self._fig_height = NoScrollDoubleSpinBox(); self._fig_height.setDecimals(1)
        self._fig_height.setSingleStep(0.5); self._fig_height.setRange(2, 30)
        self._fig_height.setValue(self.fig.get_figheight())
        fs.addRow("Width (in):", self._fig_width)
        fs.addRow("Height (in):", self._fig_height)
        sz_row = QHBoxLayout()
        btn_sz = QPushButton("Apply"); btn_sz.clicked.connect(self._apply_fig_size)
        btn_rst = QPushButton("Reset"); btn_rst.clicked.connect(self._reset_fig_size)
        sz_row.addWidget(btn_sz); sz_row.addWidget(btn_rst)
        fs.addRow(sz_row)
        L.addLayout(fs)

        self._sections_layout.addWidget(sec)

    # ==================== 6. Curve Fit =======================================
    def _build_section_fit(self):
        sec = CollapsibleSection("Curve Fit", self, expanded=False)
        L = sec.content_layout()
        f = QFormLayout()
        self._fit_type = NoScrollComboBox()
        for key, label in _FIT_TYPES:
            self._fit_type.addItem(label, key)
        self._fit_type.currentIndexChanged.connect(self._apply_fit_settings)
        f.addRow("Fit:", self._fit_type)
        self._fit_degree = NoScrollSpinBox(); self._fit_degree.setRange(1, 10); self._fit_degree.setValue(1)
        self._fit_degree.valueChanged.connect(self._apply_fit_settings)
        f.addRow("Degree:", self._fit_degree)
        self._fit_color_hex = "#ff0000"
        self._fit_color_btn = _make_color_btn(self._fit_color_hex)
        self._fit_color_btn.clicked.connect(lambda: _pick_color(
            self, self._fit_color_hex, self._set_fit_color))
        f.addRow("Color:", self._fit_color_btn)
        row = QHBoxLayout()
        self._fit_show_eq = QCheckBox("Equation"); self._fit_show_eq.setChecked(True)
        self._fit_show_eq.stateChanged.connect(self._apply_fit_settings)
        self._fit_show_r2 = QCheckBox("R\u00b2"); self._fit_show_r2.setChecked(True)
        self._fit_show_r2.stateChanged.connect(self._apply_fit_settings)
        row.addWidget(self._fit_show_eq); row.addWidget(self._fit_show_r2)
        f.addRow("Show:", row)
        L.addLayout(f)
        self._sections_layout.addWidget(sec)

    # ==================== 7. Legend ==========================================
    def _build_section_legend(self):
        sec = CollapsibleSection("Legend", self, expanded=False)
        self._sections["legend"] = sec
        L = sec.content_layout()
        self._legend_cb = QCheckBox("Show Legend"); self._legend_cb.setChecked(True)
        self._legend_cb.stateChanged.connect(self._apply_plot_options)
        L.addWidget(self._legend_cb)
        f = QFormLayout()
        self._legend_pos = NoScrollComboBox()
        self._legend_pos.addItems([
            "best", "upper right", "upper left", "lower left",
            "lower right", "center left", "center right",
            "lower center", "upper center", "center"])
        self._legend_pos.currentTextChanged.connect(self._apply_plot_options)
        f.addRow("Position:", self._legend_pos)
        self._legend_fs = _make_fs_spin(10, 6, 24)
        self._legend_fs.valueChanged.connect(self._apply_plot_options)
        f.addRow("Font Size:", self._legend_fs)
        L.addLayout(f)
        self._sections_layout.addWidget(sec)

    # ==================== 8. Annotate ========================================
    def _build_section_annotate(self):
        sec = CollapsibleSection("Annotate", self, expanded=False)
        L = sec.content_layout()

        lbl = QLabel("Text Annotation"); lbl.setStyleSheet("font-weight: bold;"); L.addWidget(lbl)
        f = QFormLayout()
        self._annot_text = QLineEdit(); self._annot_text.setPlaceholderText("Annotation text...")
        f.addRow("Text:", self._annot_text)
        self._annot_x = NoScrollDoubleSpinBox(); self._annot_x.setDecimals(4); self._annot_x.setRange(-1e12, 1e12)
        self._annot_y = NoScrollDoubleSpinBox(); self._annot_y.setDecimals(4); self._annot_y.setRange(-1e12, 1e12)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("X:")); pr.addWidget(self._annot_x, stretch=1)
        pr.addWidget(QLabel("Y:")); pr.addWidget(self._annot_y, stretch=1)
        f.addRow(pr)
        self._annot_fs = _make_fs_spin(12, 6, 36)
        self._annot_color_hex = "#000000"
        self._annot_color_btn = _make_color_btn(self._annot_color_hex)
        self._annot_color_btn.clicked.connect(lambda: _pick_color(
            self, self._annot_color_hex, self._set_annot_color))
        sc_a = QHBoxLayout()
        sc_a.addWidget(QLabel("Size:")); sc_a.addWidget(self._annot_fs)
        sc_a.addWidget(QLabel("Color:")); sc_a.addWidget(self._annot_color_btn)
        f.addRow(sc_a)
        self._annot_bg_hex = "#faffa0"
        self._annot_bg_btn = _make_color_btn(self._annot_bg_hex)
        self._annot_bg_btn.clicked.connect(lambda: _pick_color(
            self, self._annot_bg_hex, self._set_annot_bg))
        self._annot_border_hex = "#808080"
        self._annot_border_btn = _make_color_btn(self._annot_border_hex)
        self._annot_border_btn.clicked.connect(lambda: _pick_color(
            self, self._annot_border_hex, self._set_annot_border))
        br = QHBoxLayout()
        br.addWidget(self._annot_bg_btn); br.addWidget(QLabel("bg")); br.addStretch()
        br.addWidget(self._annot_border_btn); br.addWidget(QLabel("border"))
        f.addRow("Box:", br)
        L.addLayout(f)
        r1 = QHBoxLayout()
        b1 = QPushButton("Add"); b1.clicked.connect(self._add_annotation); r1.addWidget(b1)
        b2 = QPushButton("Clear All"); b2.clicked.connect(self._clear_annotations); r1.addWidget(b2)
        L.addLayout(r1)

        L.addSpacing(6)
        lbl2 = QLabel("Reference Lines"); lbl2.setStyleSheet("font-weight: bold;"); L.addWidget(lbl2)
        f2 = QFormLayout()
        self._ref_orient = NoScrollComboBox(); self._ref_orient.addItems(["Horizontal", "Vertical"])
        f2.addRow("Direction:", self._ref_orient)
        self._ref_value = NoScrollDoubleSpinBox(); self._ref_value.setDecimals(6); self._ref_value.setRange(-1e15, 1e15)
        f2.addRow("Value:", self._ref_value)
        self._ref_color_hex = "#888888"
        self._ref_color_btn = _make_color_btn(self._ref_color_hex)
        self._ref_color_btn.clicked.connect(lambda: _pick_color(
            self, self._ref_color_hex, self._set_ref_color))
        self._ref_style = NoScrollComboBox()
        for key, label in _LINESTYLES[:4]:
            self._ref_style.addItem(label, key)
        csr = QHBoxLayout()
        csr.addWidget(QLabel("Color:")); csr.addWidget(self._ref_color_btn)
        csr.addWidget(QLabel("Style:")); csr.addWidget(self._ref_style, stretch=1)
        f2.addRow(csr)
        self._ref_label = QLineEdit(); self._ref_label.setPlaceholderText("Optional label")
        f2.addRow("Label:", self._ref_label)
        L.addLayout(f2)
        r2 = QHBoxLayout()
        b3 = QPushButton("Add"); b3.clicked.connect(self._add_ref_line); r2.addWidget(b3)
        b4 = QPushButton("Clear All"); b4.clicked.connect(self._clear_ref_lines); r2.addWidget(b4)
        L.addLayout(r2)
        self._sections_layout.addWidget(sec)

    # ==================== 9. Export ==========================================
    def _build_section_export(self):
        sec = CollapsibleSection("Export", self, expanded=False)
        L = sec.content_layout()
        f = QFormLayout()
        self._dpi_spin = NoScrollSpinBox(); self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setValue(300); self._dpi_spin.setFixedWidth(70)
        f.addRow("DPI:", self._dpi_spin)
        L.addLayout(f)
        btn_save = QPushButton("Save Figure..."); btn_save.clicked.connect(self._save_figure)
        L.addWidget(btn_save)
        btn_csv = QPushButton("Export Data as CSV..."); btn_csv.clicked.connect(self._export_csv)
        L.addWidget(btn_csv)
        self._sections_layout.addWidget(sec)

    # ======================================================= helpers
    def _tex_wrap(self, text, use_tex):
        if not use_tex or not text: return text
        if text.startswith("$") and text.endswith("$"): return text
        return f"${text}$"

    def _cur_idx(self):
        return max(0, self._series_combo.currentIndex())

    # ================================================ series switching
    def _on_series_name_changed(self):
        if self._blocking:
            return
        idx = self._cur_idx()
        name = self._series_name_edit.text().strip()
        if not name or idx < 0 or idx >= len(self._series_data):
            return
        self._series_data[idx]["label"] = name
        self._series_combo.setItemText(idx, name)
        for cbs in self._subplot_assign_combos:
            if idx < len(cbs):
                cbs[idx].setText(name)
        self._replot()

    def _on_series_changed(self, idx):
        if 0 <= idx < len(self._settings):
            self._sync_controls_to_series(idx)

    def _sync_controls_to_series(self, idx):
        if idx < 0 or idx >= len(self._settings): return
        s = self._settings[idx]
        self._blocking = True
        self._series_name_edit.setText(self._series_data[idx]["label"])
        for i in range(self._plot_type_combo.count()):
            if self._plot_type_combo.itemData(i) == s["plot_type"]:
                self._plot_type_combo.setCurrentIndex(i); break
        _set_btn_color(self._line_color_btn, s["color"])
        self._visible_cb.setChecked(s["visible"])
        self._line_width.setValue(s["line_width"])
        for i in range(self._line_style_combo.count()):
            if self._line_style_combo.itemData(i) == s["line_style"]:
                self._line_style_combo.setCurrentIndex(i); break
        for i in range(self._marker_combo.count()):
            if self._marker_combo.itemData(i) == s["marker"]:
                self._marker_combo.setCurrentIndex(i); break
        self._marker_size.setValue(s["marker_size"])
        _set_btn_color(self._marker_fill_btn, s["marker_fill"])
        _set_btn_color(self._marker_edge_btn, s["marker_edge"])
        self._marker_edge_width.setValue(s["marker_edge_width"])
        self._opacity.setValue(s.get("opacity", 1.0))
        self._scatter_size.setValue(s["scatter_size"])
        _set_btn_color(self._scatter_fill_btn, s["marker_fill"])
        _set_btn_color(self._scatter_edge_btn, s["marker_edge"])
        self._scatter_edge_width.setValue(s["marker_edge_width"])
        cbm = {"uniform": 0, "x": 1, "y": 2}
        self._scatter_color_by.setCurrentIndex(cbm.get(s["scatter_color_by"], 0))
        ci = _COLORMAPS.index(s["scatter_cmap"]) if s["scatter_cmap"] in _COLORMAPS else 0
        self._scatter_cmap.setCurrentIndex(ci)
        self._bar_width.setValue(s["bar_width"])
        self._hist_bins.setValue(s["hist_bins"])
        self._hist_alpha.setValue(s["hist_alpha"])
        self._box_showfliers.setChecked(s.get("box_showfliers", True))
        self._x_offset.setValue(s["x_offset"]); self._y_offset.setValue(s["y_offset"])
        self._exclude_outliers_cb.setChecked(s.get("exclude_outliers", False))
        self._normalize_cb.setChecked(s["normalize"])
        self._secondary_y_cb.setChecked(s["secondary_y"])
        for i in range(self._fit_type.count()):
            if self._fit_type.itemData(i) == s["fit_type"]:
                self._fit_type.setCurrentIndex(i); break
        self._fit_degree.setValue(s["fit_degree"])
        self._fit_degree.setEnabled(s["fit_type"] == "poly")
        self._fit_color_hex = s["fit_color"]
        _set_btn_color(self._fit_color_btn, s["fit_color"])
        self._fit_show_eq.setChecked(s["fit_show_eq"])
        self._fit_show_r2.setChecked(s["fit_show_r2"])
        self._blocking = False
        self._update_style_visibility(s["plot_type"])

    def _update_style_visibility(self, pt):
        self._line_marker_w.setVisible(pt in ("line", "step", "area", "stem"))
        self._scatter_w.setVisible(pt == "scatter")
        self._bar_w.setVisible(pt == "bar")
        self._hist_w.setVisible(pt == "histogram")
        self._box_w.setVisible(pt == "box")

    # ================================================ display data
    def _get_display_data(self, idx):
        sd = self._series_data[idx]; s = self._settings[idx]
        x, y = sd["x"].copy(), sd["y"].copy()
        if s.get("exclude_outliers", False):
            mask = np.ones(len(x), dtype=bool)
            for arr in (x, y):
                finite = arr[np.isfinite(arr)]
                if len(finite) > 3:
                    q1, q3 = np.percentile(finite, [25, 75])
                    iqr = q3 - q1
                    mask &= (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
            x, y = x[mask], y[mask]
        if s["normalize"]:
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            if ymax - ymin > 0: y = (y - ymin) / (ymax - ymin)
        x = x + s["x_offset"]
        y = y + s["y_offset"]
        xsf = self._x_scale_factor.value()
        ysf = self._y_scale_factor.value()
        if xsf != 1.0: x = x * xsf
        if ysf != 1.0: y = y * ysf
        return x, y

    # ------------------------------------------------ subplot helpers
    def _compute_subplot_grid(self, n):
        arr = self._arrange_combo.currentText() if self._arrange_combo else "Vertical"
        if arr == "Vertical": return n, 1
        if arr == "Horizontal": return 1, n
        nc = math.ceil(math.sqrt(n)); return math.ceil(n / nc), nc

    def _plot_ref_lines(self, ax):
        for rl in self._ref_lines:
            (ax.axhline if rl["orient"] == "h" else ax.axvline)(
                **{("y" if rl["orient"] == "h" else "x"): rl["value"]},
                color=rl["color"], linestyle=rl["style"], label=rl.get("label", ""))

    def _plot_series_on_ax(self, ax, series_indices):
        for i in series_indices:
            if i >= len(self._series_data): continue
            sd, s = self._series_data[i], self._settings[i]
            if not s["visible"]: continue
            x, y = self._get_display_data(i)
            self._plot_one(ax, x, y, i, s, sd)
            if s["fit_type"] != "none" and len(x) > 1:
                self._plot_fit(ax, x, y, i, s)

    def _rebuild_subplot_assignments(self):
        while self._subplot_assign_container.count():
            item = self._subplot_assign_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        self._subplot_assign_combos = []
        n = self._n_subplots
        ns = len(self._series_data)

        while len(self._subplot_assignments) < n:
            self._subplot_assignments.append([])
        self._subplot_assignments = self._subplot_assignments[:n]

        if n <= 1:
            if not self._subplot_assignments or not self._subplot_assignments[0]:
                self._subplot_assignments = [list(range(ns))]

        for sp_idx in range(n):
            title = "Series" if n <= 1 else f"Subplot {sp_idx + 1}"
            grp = QGroupBox(title)
            gl = QVBoxLayout(grp)
            gl.setContentsMargins(4, 2, 4, 2)
            gl.setSpacing(1)
            cbs = []
            for si, sd in enumerate(self._series_data):
                cb = QCheckBox(sd["label"])
                cb.setChecked(si in self._subplot_assignments[sp_idx])
                cb.stateChanged.connect(self._on_subplot_assign_changed)
                gl.addWidget(cb)
                cbs.append(cb)
            self._subplot_assign_combos.append(cbs)
            self._subplot_assign_container.addWidget(grp)

    def _on_subplot_count_changed(self, n):
        self._n_subplots = n
        ns = len(self._series_data)
        self._subplot_assignments = []
        for i in range(n):
            if n == 1:
                self._subplot_assignments.append(list(range(ns)))
            else:
                self._subplot_assignments.append([i] if i < ns else [])
        self._rebuild_subplot_assignments()
        self._replot()

    def _on_subplot_assign_changed(self):
        for sp_idx, cbs in enumerate(self._subplot_assign_combos):
            assigned = []
            for si, cb in enumerate(cbs):
                if cb.isChecked():
                    assigned.append(si)
            self._subplot_assignments[sp_idx] = assigned
        self._replot()

    # ================================================ REPLOT
    def _replot(self):
        self.fig.clear()
        self._ax2 = None; self._subplot_axes = []
        if self._colorbar: self._colorbar = None
        self._colorbar_mappable = None; self._fit_texts = []
        self._hover_highlight = None; self._hover_idx = None

        n_sp = self._n_subplots

        if n_sp > 1:
            nr, nc = self._compute_subplot_grid(n_sp)
            share_x = self._share_x_cb.isChecked() if self._share_x_cb else False
            axes = self.fig.subplots(nr, nc, squeeze=False, sharex=share_x)
            af = axes.ravel().tolist()
            for j in range(n_sp, len(af)): af[j].set_visible(False)
            self._subplot_axes = af[:n_sp]
            self.ax = self._subplot_axes[0]

            for sp_idx in range(n_sp):
                ax = self._subplot_axes[sp_idx]
                assigned = self._subplot_assignments[sp_idx] if sp_idx < len(self._subplot_assignments) else []
                self._plot_series_on_ax(ax, assigned)
                names = [self._series_data[i]["label"] for i in assigned
                         if i < len(self._series_data)]
                ax.set_title(", ".join(names) if names else f"Subplot {sp_idx+1}",
                             fontsize=11)
                self._plot_ref_lines(ax)
                if self._legend_cb.isChecked():
                    h, l = ax.get_legend_handles_labels()
                    filt = [(hh, ll) for hh, ll in zip(h, l) if not ll.startswith("_")]
                    if filt:
                        leg = ax.legend(*zip(*filt), loc=self._legend_pos.currentText(),
                                        fontsize=self._legend_fs.value())
                        leg.set_draggable(True)
        else:
            self.ax = self.fig.add_subplot(111); self._subplot_axes = []
            all_indices = self._subplot_assignments[0] if self._subplot_assignments else list(range(len(self._series_data)))
            visible = [i for i in all_indices if i < len(self._settings) and self._settings[i]["visible"]]
            need_y2 = any(self._settings[i]["secondary_y"] for i in visible)
            if need_y2: self._ax2 = self.ax.twinx()
            box_vis = [i for i in visible if self._settings[i]["plot_type"] == "box"]
            self._box_positions = {si: pos for pos, si in enumerate(box_vis, 1)}
            for i in visible:
                sd, s = self._series_data[i], self._settings[i]
                tgt = self._ax2 if s["secondary_y"] and self._ax2 else self.ax
                x, y = self._get_display_data(i)
                self._plot_one(tgt, x, y, i, s, sd)
                if s["fit_type"] != "none" and len(x) > 1: self._plot_fit(tgt, x, y, i, s)
            self._plot_ref_lines(self.ax)
            if box_vis:
                labels = [self._series_data[i]["label"] for i in box_vis]
                positions = list(range(1, len(box_vis) + 1))
                self.ax.set_xticks(positions)
                self.ax.set_xticklabels(labels)
                self.ax.set_xlim(0.5, len(box_vis) + 0.5)
            self._apply_legend()

        for a in self._annotations:
            self.ax.annotate(a["text"], xy=(a["x"], a["y"]), fontsize=a["fontsize"],
                             color=a["color"], ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.3", fc=a["bg"],
                                       ec=a["border"], alpha=0.9))
        if self._cbar_enable.isChecked() and self._colorbar_mappable is not None:
            self._colorbar = self.fig.colorbar(
                self._colorbar_mappable, ax=self.ax,
                orientation=self._cbar_orient.currentText())
            cl = self._cbar_label.text()
            if cl: self._colorbar.set_label(cl)

        self._apply_labels_to_axes(); self._apply_axes_to_plot()
        # Apply box plot tick labels after all axis formatting
        if not self._subplot_axes and hasattr(self, '_box_positions') and self._box_positions:
            labels = [self._series_data[i]["label"]
                      for i in sorted(self._box_positions, key=self._box_positions.get)]
            positions = sorted(self._box_positions.values())
            self.ax.set_xticks(positions)
            self.ax.set_xticklabels(labels)
            self.ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        self.fig.tight_layout()
        self.toolbar.update()
        self.canvas.draw()

    def _hit_test_series(self, event):
        if event.inaxes is None:
            return None
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return None
        ax = event.inaxes
        tr = ax.transData
        mx_d, my_d = tr.transform((mx, my))
        best_idx, best_dist = None, 20
        assignments = list(range(len(self._series_data)))
        if self._subplot_axes:
            for sp_idx, sp_ax in enumerate(self._subplot_axes):
                if sp_ax is ax and sp_idx < len(self._subplot_assignments):
                    assignments = self._subplot_assignments[sp_idx]
                    break
        elif self._subplot_assignments:
            assignments = self._subplot_assignments[0]
        for i in assignments:
            if i >= len(self._series_data) or i >= len(self._settings):
                continue
            if not self._settings[i]["visible"]:
                continue
            s = self._settings[i]
            x, y = self._get_display_data(i)
            if s["plot_type"] in ("histogram", "box"):
                sd = self._series_data[i]
                hd = y if sd.get("y_connected", True) else x
                if len(hd) > 0 and np.nanmin(hd) <= mx <= np.nanmax(hd):
                    best_idx = i
                continue
            pts = tr.transform(np.column_stack([x, y]))
            dists = np.hypot(pts[:, 0] - mx_d, pts[:, 1] - my_d)
            d = np.min(dists)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _on_hover(self, event):
        idx = self._hit_test_series(event)
        if idx == self._hover_idx:
            return
        if self._hover_highlight is not None:
            try:
                self._hover_highlight.remove()
            except Exception:
                pass
            self._hover_highlight = None
        self._hover_idx = idx
        if idx is not None and idx < len(self._series_data):
            s = self._settings[idx]
            if s["plot_type"] not in ("line", "scatter", "stem", "step"):
                self.canvas.draw_idle()
                return
            x, y = self._get_display_data(idx)
            ax = event.inaxes or self.ax
            self._hover_highlight = ax.scatter(
                x, y, s=80, facecolors="none",
                edgecolors=s["color"], linewidths=1.5, alpha=0.4, zorder=99)
            self.canvas.draw_idle()
        elif self._hover_idx is None:
            self.canvas.draw_idle()

    def _focus_section(self, key):
        for sec in self._sections.values():
            sec.collapse()
        sec = self._sections.get(key)
        if sec:
            sec.expand()
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._panel_scroll.ensureWidgetVisible(sec))

    def _on_click(self, event):
        if event.button != 1:
            return
        if self.toolbar.mode:
            return
        mx, my = event.x, event.y
        for ax in (self._subplot_axes or [self.ax]):
            if not ax.get_visible():
                continue
            title = ax.title
            if title.get_text() and title.get_window_extent(
                    self.canvas.get_renderer()).contains(mx, my):
                self._focus_section("labels"); return
            xlabel = ax.xaxis.label
            if xlabel.get_text() and xlabel.get_window_extent(
                    self.canvas.get_renderer()).contains(mx, my):
                self._focus_section("labels"); return
            ylabel = ax.yaxis.label
            if ylabel.get_text() and ylabel.get_window_extent(
                    self.canvas.get_renderer()).contains(mx, my):
                self._focus_section("labels"); return
            for lbl in ax.get_xticklabels():
                try:
                    if lbl.get_window_extent(
                            self.canvas.get_renderer()).contains(mx, my):
                        self._focus_section("ticks"); return
                except Exception:
                    pass
            for lbl in ax.get_yticklabels():
                try:
                    if lbl.get_window_extent(
                            self.canvas.get_renderer()).contains(mx, my):
                        self._focus_section("ticks"); return
                except Exception:
                    pass
            legend = ax.get_legend()
            if legend:
                try:
                    if legend.get_window_extent(
                            self.canvas.get_renderer()).contains(mx, my):
                        self._focus_section("legend"); return
                except Exception:
                    pass
        idx = self._hit_test_series(event)
        if idx is not None and 0 <= idx < len(self._series_data):
            self._series_combo.setCurrentIndex(idx)
            self._focus_section("series")

    def _plot_one(self, ax, x, y, idx, s, sd):
        pt, c, lb = s["plot_type"], s["color"], sd["label"]
        mfc = s.get("marker_fill", c)
        mec = s.get("marker_edge", "#000000")
        mew = s.get("marker_edge_width", 0.5)
        la = s.get("opacity", 1.0)
        ma = la
        if pt == "line":
            ax.plot(x, y, color=c, linewidth=s["line_width"], linestyle=s["line_style"],
                    alpha=la, marker=s["marker"], markersize=s["marker_size"],
                    markerfacecolor=(*mcolors.to_rgb(mfc), ma),
                    markeredgecolor=(*mcolors.to_rgb(mec), ma),
                    markeredgewidth=mew, label=lb)
        elif pt == "scatter":
            cb = s["scatter_color_by"]
            if cb == "uniform":
                ax.scatter(x, y, s=s["scatter_size"],
                           facecolors=(*mcolors.to_rgb(mfc), ma),
                           edgecolors=(*mcolors.to_rgb(mec), ma),
                           linewidths=mew, label=lb)
            else:
                cv = {"x": x, "y": y}.get(cb, y)
                sc = ax.scatter(x, y, s=s["scatter_size"], c=cv, cmap=s["scatter_cmap"],
                                edgecolors=(*mcolors.to_rgb(mec), ma),
                                linewidths=mew, alpha=ma, label=lb)
                self._colorbar_mappable = sc
        elif pt == "bar":
            ax.bar(x, y, width=s["bar_width"], color=c, alpha=la, label=lb)
        elif pt == "histogram":
            hd = y if sd.get("y_connected", True) else x
            ax.hist(hd, bins=s["hist_bins"], alpha=s["hist_alpha"] * la,
                    color=c, edgecolor="black", label=lb)
        elif pt == "box":
            hd = y if sd.get("y_connected", True) else x
            hd = hd[np.isfinite(hd)]
            pos = self._box_positions.get(idx, idx + 1)
            bp = ax.boxplot(hd, positions=[pos], widths=0.6, patch_artist=True,
                            showfliers=s.get("box_showfliers", True))
            for patch in bp["boxes"]:
                patch.set_facecolor(c); patch.set_alpha(la)
            for element in ("whiskers", "caps", "medians"):
                plt.setp(bp[element], color="black")
            if s.get("box_showfliers", True):
                plt.setp(bp["fliers"], markeredgecolor=c, alpha=ma)
            ax.plot([], [], color=c, label=lb, linewidth=6, alpha=la)
        elif pt == "stem":
            ml, sl, bl = ax.stem(x, y, label=lb)
            plt.setp(sl, color=c, linewidth=s["line_width"], alpha=la)
            plt.setp(ml, color=mfc, markersize=s["marker_size"],
                     markeredgecolor=mec, markeredgewidth=mew, alpha=ma)
            plt.setp(bl, color="grey", linewidth=0.5)
        elif pt == "step":
            ax.step(x, y, where="mid", color=c, linewidth=s["line_width"],
                    linestyle=s["line_style"], alpha=la, label=lb)
        elif pt == "area":
            ax.fill_between(x, y, alpha=0.3 * la, color=c, label=lb)
            ax.plot(x, y, color=c, linewidth=s["line_width"],
                    linestyle=s["line_style"], alpha=la)

    def _plot_fit(self, ax, x, y, idx, s):
        ft, c = s["fit_type"], s["fit_color"]
        m = np.isfinite(x) & np.isfinite(y); xf, yf = x[m], y[m]
        if len(xf) < 2: return
        xfit = np.linspace(xf.min(), xf.max(), 300); eq = ""; r2 = None
        try:
            if ft == "poly":
                d = s["fit_degree"]; co = np.polyfit(xf, yf, d)
                yfit = np.polyval(co, xfit); yp = np.polyval(co, xf)
                ts = []
                for i, cv in enumerate(co):
                    pw = d - i
                    if abs(cv) < 1e-15: continue
                    cs = f"{cv:.4g}"
                    ts.append(cs if pw == 0 else f"{cs}x" if pw == 1 else f"{cs}x^{pw}")
                eq = "y = " + " + ".join(ts) if ts else "y = 0"
            elif ft == "exp":
                ef = lambda x, a, b: a * np.exp(b * x)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    po, _ = curve_fit(ef, xf, yf, p0=[1, 0.01], maxfev=10000)
                yfit = ef(xfit, *po); yp = ef(xf, *po)
                eq = f"y = {po[0]:.4g} exp({po[1]:.4g} x)"
            elif ft == "power":
                p = (xf > 0) & (yf > 0)
                if p.sum() < 2: return
                lc = np.polyfit(np.log(xf[p]), np.log(yf[p]), 1)
                a, b = np.exp(lc[1]), lc[0]
                yfit = a * xfit ** b; yp = a * xf ** b
                eq = f"y = {a:.4g} x^{b:.4g}"
            elif ft == "log":
                p = xf > 0
                if p.sum() < 2: return
                co = np.polyfit(np.log(xf[p]), yf[p], 1)
                xfit = xfit[xfit > 0]; yfit = co[0] * np.log(xfit) + co[1]
                yp = co[0] * np.log(xf[p]) + co[1]; yf = yf[p]
                eq = f"y = {co[0]:.4g} ln(x) + {co[1]:.4g}"
            else: return
            ssr = np.sum((yf - yp) ** 2); sst = np.sum((yf - np.mean(yf)) ** 2)
            r2 = 1 - ssr / sst if sst > 0 else 0
            ax.plot(xfit, yfit, color=c, linestyle="--", linewidth=1.5)
            parts = []
            if s["fit_show_eq"] and eq: parts.append(eq)
            if s["fit_show_r2"] and r2 is not None: parts.append(f"R\u00b2 = {r2:.6f}")
            if parts:
                ax.text(0.02, 0.98 - idx * 0.10, "\n".join(parts), transform=ax.transAxes,
                        fontsize=9, verticalalignment="top", color=c,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, alpha=0.8))
        except Exception: pass

    # ================================================ APPLY CALLBACKS
    def _apply_labels(self):
        if self._blocking: return
        self._apply_labels_to_axes(); self.fig.tight_layout(); self.canvas.draw()

    def _apply_labels_to_axes(self):
        matplotlib.rcParams['text.usetex'] = False
        tt = self._tex_wrap(self._title_edit.text(), self._title_tex.isChecked())
        xl = self._tex_wrap(self._xlabel_edit.text(), self._xlabel_tex.isChecked())
        yl = self._tex_wrap(self._ylabel_edit.text(), self._ylabel_tex.isChecked())
        font = self._label_font.currentText()
        tkw = lambda fs, b, i, ch: dict(
            fontsize=fs.value(),
            fontweight="bold" if b.isChecked() else "normal",
            fontstyle="italic" if i.isChecked() else "normal",
            color=ch, fontfamily=font)
        t_kw = tkw(self._title_fs, self._title_bold, self._title_italic, self._title_color_hex)
        x_kw = tkw(self._xlabel_fs, self._xlabel_bold, self._xlabel_italic, self._xlabel_color_hex)
        y_kw = tkw(self._ylabel_fs, self._ylabel_bold, self._ylabel_italic, self._ylabel_color_hex)
        if self._subplot_axes:
            self.fig.suptitle(tt, **t_kw)
            for ax in self._subplot_axes:
                if ax.get_visible():
                    ax.set_xlabel(xl, **x_kw); ax.set_ylabel(yl, **y_kw)
        else:
            self.ax.set_title(tt, **t_kw); self.ax.set_xlabel(xl, **x_kw)
            self.ax.set_ylabel(yl, **y_kw)

    def _on_plot_type_changed(self, ci):
        if self._blocking: return
        pt = self._plot_type_combo.itemData(ci)
        self._update_style_visibility(pt)
        for s in self._settings:
            s["plot_type"] = pt
        self._replot()

    def _pick_line_color(self):
        idx = self._cur_idx(); cur = self._settings[idx]["color"]
        c = QColorDialog.getColor(QColor(cur), self, "Series Color")
        if c.isValid():
            h = c.name(); self._settings[idx]["color"] = h
            _set_btn_color(self._line_color_btn, h); self._replot()

    def _pick_marker_fill(self):
        idx = self._cur_idx(); cur = self._settings[idx]["marker_fill"]
        c = QColorDialog.getColor(QColor(cur), self, "Marker Fill Color")
        if c.isValid():
            h = c.name(); self._settings[idx]["marker_fill"] = h
            _set_btn_color(self._marker_fill_btn, h)
            _set_btn_color(self._scatter_fill_btn, h)
            self._replot()

    def _pick_marker_edge(self):
        idx = self._cur_idx(); cur = self._settings[idx]["marker_edge"]
        c = QColorDialog.getColor(QColor(cur), self, "Marker Edge Color")
        if c.isValid():
            h = c.name(); self._settings[idx]["marker_edge"] = h
            _set_btn_color(self._marker_edge_btn, h)
            _set_btn_color(self._scatter_edge_btn, h)
            self._replot()

    def _apply_series_style(self):
        if self._blocking: return
        s = self._settings[self._cur_idx()]
        s["line_width"] = self._line_width.value()
        s["line_style"] = self._line_style_combo.currentData() or "-"
        s["marker"] = self._marker_combo.currentData() or "None"
        s["marker_size"] = self._marker_size.value()
        s["opacity"] = self._opacity.value()
        s["scatter_size"] = self._scatter_size.value()
        # Sync marker edge width from whichever widget is visible
        if s["plot_type"] == "scatter":
            s["marker_edge_width"] = self._scatter_edge_width.value()
            self._marker_edge_width.setValue(s["marker_edge_width"])
        else:
            s["marker_edge_width"] = self._marker_edge_width.value()
            self._scatter_edge_width.setValue(s["marker_edge_width"])
        cbk = ["uniform", "x", "y"]
        s["scatter_color_by"] = cbk[self._scatter_color_by.currentIndex()]
        s["scatter_cmap"] = self._scatter_cmap.currentText()
        s["bar_width"] = self._bar_width.value()
        s["hist_bins"] = self._hist_bins.value()
        s["hist_alpha"] = self._hist_alpha.value()
        s["box_showfliers"] = self._box_showfliers.isChecked()
        self._replot()

    def _apply_plot_options(self):
        if self._blocking: return
        if not self._subplot_axes: self._apply_legend()
        self.canvas.draw()

    def _apply_legend(self):
        axes = [self.ax] + ([self._ax2] if self._ax2 else [])
        if self._legend_cb.isChecked():
            h, l = [], []
            for a in axes:
                hh, ll = a.get_legend_handles_labels(); h.extend(hh); l.extend(ll)
            f = [(hh, ll) for hh, ll in zip(h, l) if not ll.startswith("_")]
            fs, loc = self._legend_fs.value(), self._legend_pos.currentText()
            if f:
                leg = self.ax.legend(*zip(*f), loc=loc, fontsize=fs)
                leg.set_draggable(True)
            elif self.ax.get_legend():
                self.ax.get_legend().remove()
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()

    def _on_axis_auto_changed(self):
        self._x_min.setEnabled(not self._x_auto.isChecked())
        self._x_max.setEnabled(not self._x_auto.isChecked())
        self._y_min.setEnabled(not self._y_auto.isChecked())
        self._y_max.setEnabled(not self._y_auto.isChecked())

    def _apply_axes(self):
        if self._blocking: return
        self._apply_axes_to_plot(); self.fig.tight_layout(); self.canvas.draw()

    def _apply_axes_to_plot(self):
        xs = "log" if self._x_scale.currentIndex() == 1 else "linear"
        ys = "log" if self._y_scale.currentIndex() == 1 else "linear"
        xm, ym = not self._x_auto.isChecked(), not self._y_auto.isChecked()
        xi, yi = self._x_invert.isChecked(), self._y_invert.isChecked()
        asp = "equal" if self._aspect.currentIndex() == 1 else "auto"
        for ax in (self._subplot_axes or [self.ax]):
            if not ax.get_visible(): continue
            ax.set_xscale(xs); ax.set_yscale(ys)
            if xm: ax.set_xlim(self._x_min.value(), self._x_max.value())
            if ym: ax.set_ylim(self._y_min.value(), self._y_max.value())
            if xi and not ax.xaxis_inverted(): ax.invert_xaxis()
            elif not xi and ax.xaxis_inverted(): ax.invert_xaxis()
            if yi and not ax.yaxis_inverted(): ax.invert_yaxis()
            elif not yi and ax.yaxis_inverted(): ax.invert_yaxis()
            ax.set_aspect(asp)
        self._apply_figure_to_plot()
        self._apply_ticks_to_plot()

    def _apply_ticks(self):
        if self._blocking: return
        self._apply_ticks_to_plot(); self.fig.tight_layout(); self.canvas.draw()

    def _apply_ticks_to_plot(self):
        tfs = self._tick_fs.value()
        tc = self._tick_color_hex
        tfam = self._tick_font.currentText()
        rx = self._tick_rotation_x.value()
        ry = self._tick_rotation_y.value()
        mn_show = self._minor_tick_show.isChecked()
        for ax in (self._subplot_axes or [self.ax]):
            if not ax.get_visible(): continue
            ax.tick_params(axis='both', which='major', labelsize=tfs,
                           labelcolor=tc, length=6, width=1, direction='out')
            for lbl in ax.get_xticklabels():
                lbl.set_fontfamily(tfam); lbl.set_rotation(rx)
            for lbl in ax.get_yticklabels():
                lbl.set_fontfamily(tfam); lbl.set_rotation(ry)
            if mn_show:
                ax.minorticks_on()
                ax.tick_params(axis='both', which='minor', length=3,
                               width=0.5, direction='out')
            else:
                ax.minorticks_off()

    def _apply_figure_settings(self):
        if self._blocking: return
        self._apply_figure_to_plot(); self.fig.tight_layout(); self.canvas.draw()

    def _apply_figure_to_plot(self):
        self.fig.set_facecolor(self._fig_bg_hex)
        for ax in (self._subplot_axes or [self.ax]):
            if not ax.get_visible(): continue
            ax.set_facecolor(self._axes_bg_hex)
        g = self._grid_cb.isChecked()
        gc = self._grid_color_hex
        gs = self._grid_style.currentData() or ":"
        for ax in (self._subplot_axes or [self.ax]):
            if not ax.get_visible(): continue
            ax.grid(g, which='major', color=gc, alpha=0.5, linestyle=gs, linewidth=0.5)

    def _apply_data_settings(self):
        if self._blocking: return
        s = self._settings[self._cur_idx()]
        s["visible"] = self._visible_cb.isChecked()
        s["x_offset"] = self._x_offset.value(); s["y_offset"] = self._y_offset.value()
        s["exclude_outliers"] = self._exclude_outliers_cb.isChecked()
        s["normalize"] = self._normalize_cb.isChecked()
        s["secondary_y"] = self._secondary_y_cb.isChecked()
        self._replot()

    def _apply_fit_settings(self):
        if self._blocking: return
        s = self._settings[self._cur_idx()]
        s["fit_type"] = self._fit_type.currentData() or "none"
        s["fit_degree"] = self._fit_degree.value()
        s["fit_show_eq"] = self._fit_show_eq.isChecked()
        s["fit_show_r2"] = self._fit_show_r2.isChecked()
        self._fit_degree.setEnabled(s["fit_type"] == "poly")
        self._replot()

    # --- color setters ---
    def _set_fit_color(self, h):
        self._fit_color_hex = h; _set_btn_color(self._fit_color_btn, h)
        self._settings[self._cur_idx()]["fit_color"] = h; self._replot()
    def _set_title_color(self, h):
        self._title_color_hex = h; _set_btn_color(self._title_color_btn, h); self._apply_labels()
    def _set_xlabel_color(self, h):
        self._xlabel_color_hex = h; _set_btn_color(self._xlabel_color_btn, h); self._apply_labels()
    def _set_ylabel_color(self, h):
        self._ylabel_color_hex = h; _set_btn_color(self._ylabel_color_btn, h); self._apply_labels()
    def _set_tick_color(self, h):
        self._tick_color_hex = h; _set_btn_color(self._tick_color_btn, h); self._apply_ticks()
    def _set_fig_bg(self, h):
        self._fig_bg_hex = h; _set_btn_color(self._fig_bg_btn, h); self._apply_figure_settings()
    def _set_axes_bg(self, h):
        self._axes_bg_hex = h; _set_btn_color(self._axes_bg_btn, h); self._apply_figure_settings()
    def _set_grid_color(self, h):
        self._grid_color_hex = h; _set_btn_color(self._grid_color_btn, h); self._apply_figure_settings()
    def _set_annot_color(self, h):
        self._annot_color_hex = h; _set_btn_color(self._annot_color_btn, h)
    def _set_annot_bg(self, h):
        self._annot_bg_hex = h; _set_btn_color(self._annot_bg_btn, h)
    def _set_annot_border(self, h):
        self._annot_border_hex = h; _set_btn_color(self._annot_border_btn, h)
    def _set_ref_color(self, h):
        self._ref_color_hex = h; _set_btn_color(self._ref_color_btn, h)

    # --- annotations ---
    def _add_annotation(self):
        t = self._annot_text.text().strip()
        if not t: return
        self._annotations.append({"text": t, "x": self._annot_x.value(), "y": self._annot_y.value(),
            "fontsize": self._annot_fs.value(), "color": self._annot_color_hex,
            "bg": self._annot_bg_hex, "border": self._annot_border_hex})
        self._replot()
    def _clear_annotations(self): self._annotations.clear(); self._replot()
    def _add_ref_line(self):
        self._ref_lines.append({"orient": "h" if self._ref_orient.currentIndex() == 0 else "v",
            "value": self._ref_value.value(), "color": self._ref_color_hex,
            "style": self._ref_style.currentData() or "-", "label": self._ref_label.text().strip()})
        self._replot()
    def _clear_ref_lines(self): self._ref_lines.clear(); self._replot()

    # --- figure size ---
    def _apply_fig_size(self):
        w, h = self._fig_width.value(), self._fig_height.value()
        screen_dpi = self.screen().logicalDotsPerInch() if self.screen() else 96
        self.fig.set_size_inches(w, h)
        px_w = int(w * screen_dpi)
        px_h = int(h * screen_dpi)
        self.canvas.setMinimumSize(px_w, px_h)
        self.canvas.setMaximumSize(px_w, px_h)
        try:
            self.fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.canvas.draw()

    def _reset_fig_size(self):
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setMaximumSize(16777215, 16777215)

    def _save_figure(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;JPEG (*.jpg);;All Files (*)")
        if p: self.fig.savefig(p, dpi=self._dpi_spin.value(), bbox_inches="tight", facecolor="white")

    def _export_csv(self):
        import pandas as pd
        cols = {}
        for i, (sd, s) in enumerate(zip(self._series_data, self._settings)):
            x, y = self._get_display_data(i); pf = sd["label"] or f"Series {i+1}"
            cols[f"{pf}_x"] = x; cols[f"{pf}_y"] = y
        ml = max(len(v) for v in cols.values()) if cols else 0
        for k in cols:
            d = ml - len(cols[k])
            if d > 0: cols[k] = np.concatenate([cols[k], np.full(d, np.nan)])
        df = pd.DataFrame(cols)
        p, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)")
        if p:
            if not p.lower().endswith(".csv"): p += ".csv"
            df.to_csv(p, index=False)


# ============================================================ entry point

def plot_data_dialog(series_data, parent=None, block=None):
    """Open the interactive 2D Plot dialog."""
    dlg = PlotDataDialog(series_data, parent, block=block)
    dlg.exec()
    plt.close(dlg.fig)
