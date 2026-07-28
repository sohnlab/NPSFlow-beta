"""Interactive pulse template rectangularization.

Translates MATLAB PulseTemplateProcessing.m: derivative-threshold-based
rectangularization with draggable thresholds and peak exclusion. Filtering
is applied only from a wired filter config (Filter UI block); there is no
built-in filter stage.
Uses a QDialog with embedded FigureCanvas and side panel.
"""

import numpy as np
from scipy.signal import find_peaks

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QWidget, QGroupBox, QCheckBox, QSpinBox, QRadioButton, QButtonGroup,
    QSlider,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from utils.tpsettings_io import get_zone
from utils.edge_refine import fwhm_refine_boundaries as _fwhm_refine_boundaries

from theme import theme


def _intersect_zone_into_list(zones, new_zone):
    """Add *new_zone* to *zones*, intersecting any overlapping zones.

    When the new zone overlaps with existing zones, the overlapping ones
    are replaced by their intersection with the new zone (narrowing).
    Non-overlapping zones are kept as-is.
    """
    a, b = min(new_zone), max(new_zone)
    kept = []
    for z in zones:
        lo, hi = z
        if lo < b and a < hi:
            a = max(a, lo)
            b = min(b, hi)
        else:
            kept.append(z)
    if a < b:
        kept.append((a, b))
    return kept


def _rect_segment(filtered, s, e, method):
    """Compute rectangularized values for one segment."""
    seg = filtered[s:e]
    if len(seg) == 0:
        return np.zeros(0)
    if method == 'linear':
        seg_len = e - s
        if seg_len < 2:
            return np.full(seg_len, seg[0])
        t = np.arange(seg_len, dtype=float)
        t_mean = t.mean()
        y_mean = seg.mean()
        denom = np.sum((t - t_mean) ** 2)
        if denom > 0:
            a = np.sum((t - t_mean) * (seg - y_mean)) / denom
            b_val = y_mean - a * t_mean
            return a * t + b_val
        return np.full(seg_len, y_mean)
    elif method == 'mean':
        return np.full(e - s, np.mean(seg))
    else:  # median
        return np.full(e - s, np.median(seg))


class _CircleOverlay(QWidget):
    """Lightweight Qt overlay that draws a dashed circle following the cursor."""

    RADIUS = 25

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._cx = -1
        self._cy = -1
        self._has_peak = False
        self._draw_radius = self.RADIUS
        self.hide()

    def set_pos(self, cx, cy, has_peak=False, draw_radius=None):
        self._cx = cx
        self._cy = cy
        self._has_peak = has_peak
        if draw_radius is not None:
            self._draw_radius = draw_radius
        r = self._draw_radius + 4
        self.setGeometry(cx - r, cy - r, 2 * r, 2 * r)
        self.update()

    def paintEvent(self, event):
        if self._cx < 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._draw_radius
        center_x = self.width() // 2
        center_y = self.height() // 2
        if self._has_peak:
            color = QColor(255, 60, 60, 180)
        else:
            color = QColor(100, 0, 170, 140)
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center_x - r, center_y - r, 2 * r, 2 * r)
        p.end()


class _ThresholdDialog(QDialog):
    """Stage 2 dialog: interactive thresholding and rectangularization."""

    # Click-mode constants
    MODE_NONE = ''
    MODE_MARK_PEAK = 'mark_peak'
    MODE_SELECT_ZONE = 'select_zone'
    MODE_DRAW_ZONE = 'draw_zone'
    MODE_SET_SEGMENT = 'set_segment'
    MODE_PEAK_ADJUST = 'peak_adjust'

    @property
    def _HINT_IDLE(self):
        p = theme.palette
        return (
            f"color: {p.text_muted}; font-size: 11px; padding: 4px 8px; "
            f"border: 1px solid {p.border}; border-radius: 3px; "
            f"background: {p.alt_base_bg}; min-height: 18px;")

    @property
    def _HINT_ACTIVE(self):
        p = theme.palette
        return (
            f"color: {p.accent_text}; font-size: 11px; padding: 4px 8px; "
            f"border: 1px solid {p.accent}; border-radius: 3px; "
            f"background: {p.accent}; min-height: 18px; font-weight: bold;")

    def __init__(self, original_template, filtered_template, sample_rate,
                 exclude_end_zones=False, end_zone_percent=5,
                 filter_config=None, padding_info=None,
                 loaded_settings=None, geometry=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pulse Template Processing - Thresholding")
        self.resize(1400, 800)
        self._geometry = geometry  # optional device geometry struct
        # Step gating: max step the user has reached (or pre-unlocked via loaded settings).
        self._max_unlocked_step = 3 if loaded_settings else 0

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._original_template = original_template.copy()
        self._filtered = filtered_template.copy()
        self._filter_config = filter_config or {}
        self._padding_info = padding_info or {'pad_length': 0, 'method': 'replicate'}
        self._n = len(filtered_template)
        self._x = np.arange(self._n)
        self._template_diff = np.diff(filtered_template)
        self._x_diff = np.arange(len(self._template_diff))
        self._exclude_end_zones = exclude_end_zones
        self._end_zone_percent = end_zone_percent

        self._pos_thresh = 0.0
        self._neg_thresh = 0.0
        self._key_peaks = {}              # {template_idx: "regular"|"min"|"max"}
        self._excl_zones = []               # list of (start, end) tuples
        self._global_rect_method = 'mean'  # applied global method
        self._segment_methods = {}          # segment_start -> method override
        self._edge_method = 'fwhm'          # 'peaks' | 'fwhm' slicing indices
        self._apex_locations = np.array([], dtype=int)
        self._rect_template = filtered_template.copy()
        self._peak_locations = np.array([], dtype=int)
        self._peak_sequence = np.array([], dtype=int)
        self._last_all_idx = np.array([0, self._n - 1])
        self.confirmed = False
        self.restart_requested = False

        # Interaction state
        self._dragging_line = None
        self._click_mode = self.MODE_NONE
        self._zone_first_x = None       # for draw_zone two-click
        self._hover_target = None
        self._art_select_zones = []     # selectable zone spans
        self._peak_adjust_mode = False
        self._added_peaks = set()       # manually-added diff-indices
        self._removed_peaks = set()     # manually-removed diff-indices

        self._loaded_settings = loaded_settings

        self._cursor_circle = None
        self._build_ui()
        self._cursor_circle = _CircleOverlay(self.canvas)
        self._setup_interaction()
        if loaded_settings:
            self._apply_loaded_settings(loaded_settings)
        else:
            self._reset_thresholds()
        self._update_all()

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left: canvas
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(13, 8), tight_layout=False)
        style_mpl_figure(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.95,
                                 bottom=0.08, hspace=0.30)
        self.canvas = FigureCanvas(self.fig)
        # Equal-height subplots that share the x-axis: pan/zoom on either
        # plot updates the other automatically.
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.30)
        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_bot = self.fig.add_subplot(gs[1], sharex=self.ax_top)
        self._nav_toolbar = NavigationToolbar(self.canvas, self)
        left_layout.addWidget(self._nav_toolbar)
        left_layout.addWidget(self.canvas)

        # Status/hint bar below plots (always visible, reserved space)
        self._lbl_hint = QLabel("")
        self._lbl_hint.setStyleSheet(self._HINT_IDLE)
        self._lbl_hint.setWordWrap(True)
        self._lbl_hint.setFixedHeight(28)
        left_layout.addWidget(self._lbl_hint)

        main_layout.addWidget(left_widget, stretch=4)

        # Right: wizard-style panel
        panel = QWidget()
        panel.setFixedWidth(210)
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(5, 5, 5, 5)
        vl.setSpacing(4)

        # Collect all toggle buttons for mutual exclusion
        self._toggle_buttons = {}

        # Step indicator
        self._wizard_step = 0  # 0-based: 0=Threshold, 1=KeyPeaks, 2=Exclusion, 3=Rect

        # Clickable stepper pills: 1 — 2 — 3 — 4.
        # Locked steps (beyond _max_unlocked_step) are disabled.
        self._step_btns = []
        step_row = QHBoxLayout()
        step_row.setSpacing(2)
        for i in range(4):
            b = QPushButton(str(i + 1))
            b.setCheckable(True)
            b.setFixedSize(36, 28)
            b.setObjectName("wizardStepBtn")
            b.clicked.connect(lambda _checked, idx=i: self._jump_to_step(idx))
            self._step_btns.append(b)
            step_row.addWidget(b)
            if i < 3:
                sep = QLabel("─")
                sep.setStyleSheet(f"color: {theme.palette.text_muted};")
                step_row.addWidget(sep)
        step_row.addStretch()
        vl.addLayout(step_row)

        self._lbl_step = QLabel()
        self._lbl_step.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._lbl_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self._lbl_step)

        self._lbl_step_desc = QLabel()
        self._lbl_step_desc.setWordWrap(True)
        self._lbl_step_desc.setStyleSheet(
            f"color: {theme.palette.text}; font-size: 11px;")
        vl.addWidget(self._lbl_step_desc)

        vl.addSpacing(6)

        # ---- Step 0: Thresholds ----
        self._step0 = QWidget()
        s0l = QVBoxLayout(self._step0)
        s0l.setContentsMargins(0, 0, 0, 0)
        s0l.setSpacing(4)
        self._lbl_thresh_pos = QLabel()
        self._lbl_thresh_neg = QLabel()
        s0l.addWidget(self._lbl_thresh_pos)
        s0l.addWidget(self._lbl_thresh_neg)

        # Live counter: detected positive / negative peaks
        self._lbl_peak_count = QLabel("Peaks: + 0  / − 0")
        self._lbl_peak_count.setStyleSheet(f"color: {theme.palette.text};")
        s0l.addWidget(self._lbl_peak_count)

        # Geometry-predicted sequence readout (for verification).
        self._lbl_predicted_seq = QLabel("")
        self._lbl_predicted_seq.setWordWrap(True)
        self._lbl_predicted_seq.setStyleSheet(
            f"color: {theme.palette.text_muted}; font-size: 10px;")
        s0l.addWidget(self._lbl_predicted_seq)
        self._update_predicted_seq_label()

        s0l.addSpacing(8)

        # Auto-threshold buttons
        g_auto = QGroupBox("Auto-threshold")
        l_auto = QVBoxLayout(g_auto)
        l_auto.setContentsMargins(6, 4, 6, 4)
        btn_3sigma = QPushButton("±3σ")
        btn_3sigma.setToolTip("Set thresholds to ±3 × std(derivative)")
        btn_3sigma.clicked.connect(lambda: self._auto_threshold_sigma(3.0))
        l_auto.addWidget(btn_3sigma)
        self._btn_auto_geom = QPushButton("Match Geometry")
        counts = self._expected_peak_counts
        if counts:
            n_pos, n_neg = counts
            self._btn_auto_geom.setToolTip(
                f"Align thresholds to the geometry-predicted pattern "
                f"({n_pos} ↑ / {n_neg} ↓ width steps).")
        else:
            self._btn_auto_geom.setToolTip(
                "Connect a Device Geometry input to enable this")
            self._btn_auto_geom.setEnabled(False)
        self._btn_auto_geom.clicked.connect(self._auto_threshold_geometry)
        l_auto.addWidget(self._btn_auto_geom)
        s0l.addWidget(g_auto)

        s0l.addStretch()
        vl.addWidget(self._step0)

        # ---- Step 1: Key Peaks ----
        self._step1 = QWidget()
        s1l = QVBoxLayout(self._step1)
        s1l.setContentsMargins(0, 0, 0, 0)
        s1l.setSpacing(4)

        g2 = QGroupBox("Key Peaks")
        l2 = QVBoxLayout(g2)
        l2.setContentsMargins(6, 4, 6, 4)
        self._lbl_key_peaks = QLabel("Marked: 0")
        l2.addWidget(self._lbl_key_peaks)

        # Property selector for new key peaks
        prop_row = QHBoxLayout()
        self._radio_kp_regular = QRadioButton("Regular")
        self._radio_kp_max = QRadioButton("Max")
        self._radio_kp_min = QRadioButton("Min")
        self._radio_kp_regular.setChecked(True)
        prop_row.addWidget(self._radio_kp_regular)
        prop_row.addWidget(self._radio_kp_max)
        prop_row.addWidget(self._radio_kp_min)
        l2.addLayout(prop_row)

        btn2a = self._make_toggle("Mark", self.MODE_MARK_PEAK,
                                  "Click a detected peak to mark as key peak.")
        l2.addWidget(btn2a)
        btn_auto = QPushButton("Auto (max/min)")
        btn_auto.setToolTip(
            "Auto-mark the largest positive and largest negative "
            "derivative peaks as key peaks.")
        btn_auto.clicked.connect(self._on_auto_key_peaks)
        l2.addWidget(btn_auto)
        btn_clr2 = QPushButton("Clear Peaks")
        btn_clr2.clicked.connect(self._on_clear_key_peaks)
        l2.addWidget(btn_clr2)
        s1l.addWidget(g2)
        s1l.addStretch()
        vl.addWidget(self._step1)

        # ---- Step 2: Exclusion ----
        self._step2 = QWidget()
        s2l = QVBoxLayout(self._step2)
        s2l.setContentsMargins(0, 0, 0, 0)
        s2l.setSpacing(4)

        # Edit Peak — add / remove individual peaks on the derivative plot.
        g_pa = QGroupBox("Edit Peak")
        l_pa = QVBoxLayout(g_pa)
        l_pa.setContentsMargins(6, 4, 6, 4)
        btn_peaks = QPushButton("✦ Peaks")
        btn_peaks.setToolTip("Add/remove peaks on derivative plot")
        btn_peaks.setStyleSheet(
            "QPushButton { color: #6400aa; font-weight: bold; }")
        btn_peaks.clicked.connect(self._toggle_peak_adjust)
        l_pa.addWidget(btn_peaks)
        rad_row = QHBoxLayout()
        rad_row.addWidget(QLabel("Radius:"))
        self._slider_cursor_radius = QSlider(Qt.Orientation.Horizontal)
        self._slider_cursor_radius.setRange(5, 30)
        self._slider_cursor_radius.setValue(15)
        self._lbl_cursor_radius = QLabel("15")
        self._slider_cursor_radius.valueChanged.connect(
            lambda v: self._lbl_cursor_radius.setText(str(v)))
        rad_row.addWidget(self._slider_cursor_radius, 1)
        rad_row.addWidget(self._lbl_cursor_radius)
        l_pa.addLayout(rad_row)
        s2l.addWidget(g_pa)

        g3 = QGroupBox("Zone Exclusion")
        l3 = QVBoxLayout(g3)
        l3.setContentsMargins(6, 4, 6, 4)
        self._lbl_zones = QLabel("Zones: 0")
        l3.addWidget(self._lbl_zones)
        row3 = QHBoxLayout()
        btn3a = self._make_toggle("Select", self.MODE_SELECT_ZONE,
                                  "Click a region between key peaks to toggle exclusion.")
        row3.addWidget(btn3a)
        btn3b = self._make_toggle("Draw", self.MODE_DRAW_ZONE,
                                  "Click two positions to define a zone.")
        row3.addWidget(btn3b)
        l3.addLayout(row3)
        btn_clr3 = QPushButton("Clear Zones")
        btn_clr3.clicked.connect(self._on_clear_zones)
        l3.addWidget(btn_clr3)
        self._chk_end_zones = QCheckBox("End Zones")
        self._chk_end_zones.setChecked(self._exclude_end_zones)
        self._chk_end_zones.stateChanged.connect(self._on_end_zone_changed)
        l3.addWidget(self._chk_end_zones)
        ez_row = QHBoxLayout()
        ez_row.addWidget(QLabel("Size (%):"))
        self._spin_end_zone = QSpinBox()
        self._spin_end_zone.setRange(1, 49)
        self._spin_end_zone.setValue(self._end_zone_percent)
        self._spin_end_zone.setSuffix("%")
        self._spin_end_zone.setEnabled(self._exclude_end_zones)
        self._spin_end_zone.valueChanged.connect(self._on_end_zone_changed)
        ez_row.addWidget(self._spin_end_zone)
        l3.addLayout(ez_row)
        s2l.addWidget(g3)

        s2l.addStretch()
        vl.addWidget(self._step2)

        # ---- Step 3: Rect. Method ----
        self._step3 = QWidget()
        s3l = QVBoxLayout(self._step3)
        s3l.setContentsMargins(0, 0, 0, 0)
        s3l.setSpacing(4)
        g4 = QGroupBox("Rect. Method")
        l4 = QVBoxLayout(g4)
        l4.setContentsMargins(6, 4, 6, 4)
        self._rect_method_group = QButtonGroup(self)
        for i, label in enumerate(["Mean", "Median", "Linear"]):
            rb = QRadioButton(label)
            self._rect_method_group.addButton(rb, i)
            l4.addWidget(rb)
        self._rect_method_group.button(0).setChecked(True)
        row4 = QHBoxLayout()
        btn4b = self._make_toggle("Select", self.MODE_SET_SEGMENT,
                                  "Click a segment to set its method.")
        row4.addWidget(btn4b)
        btn4a = QPushButton("Apply All")
        btn4a.clicked.connect(self._on_apply_all_method)
        row4.addWidget(btn4a)
        l4.addLayout(row4)
        lbl_edge = QLabel("Edge Indices:")
        l4.addWidget(lbl_edge)
        edge_row = QHBoxLayout()
        self._radio_edge_peaks = QRadioButton("Peaks")
        self._radio_edge_fwhm = QRadioButton("FWHM")
        self._edge_method_group = QButtonGroup(self)
        self._edge_method_group.addButton(self._radio_edge_peaks)
        self._edge_method_group.addButton(self._radio_edge_fwhm)
        (self._radio_edge_fwhm if self._edge_method == 'fwhm'
         else self._radio_edge_peaks).setChecked(True)
        self._edge_method_group.buttonClicked.connect(
            self._on_edge_method_changed)
        edge_row.addWidget(self._radio_edge_peaks)
        edge_row.addWidget(self._radio_edge_fwhm)
        l4.addLayout(edge_row)
        s3l.addWidget(g4)

        # Final output sequence readout (actual detected peaks after exclusions)
        g_out = QGroupBox("Output Sequence")
        l_out = QVBoxLayout(g_out)
        l_out.setContentsMargins(6, 4, 6, 4)
        self._lbl_output_seq = QLabel("")
        self._lbl_output_seq.setWordWrap(True)
        self._lbl_output_seq.setTextFormat(Qt.TextFormat.RichText)
        l_out.addWidget(self._lbl_output_seq)
        s3l.addWidget(g_out)

        s3l.addStretch()
        vl.addWidget(self._step3)

        # ---- Bottom buttons: all rows aligned to the same 2-column width ----
        BTN_W = 90
        ROW_SPACING = 6

        def _make_row():
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(ROW_SPACING)
            return row

        # Save Settings (last step) and Load Settings (Step 1) share the same
        # slot above the 2x2 grid — only one is visible at a time.
        vl.addSpacing(6)
        self._btn_save_tp = QPushButton("Save Settings")
        self._btn_save_tp.setFixedWidth(BTN_W * 2 + ROW_SPACING)
        self._btn_save_tp.clicked.connect(self._on_save_tpsettings)
        save_row = _make_row()
        save_row.addWidget(self._btn_save_tp)
        save_row.addStretch()
        vl.addLayout(save_row)

        self._btn_load_tp = QPushButton("Load Settings")
        self._btn_load_tp.setFixedWidth(BTN_W * 2 + ROW_SPACING)
        self._btn_load_tp.clicked.connect(self._on_load_tpsettings)
        load_row = _make_row()
        load_row.addWidget(self._btn_load_tp)
        load_row.addStretch()
        vl.addLayout(load_row)

        vl.addSpacing(6)

        # Row 1: [Restart] [Export]
        from utils.export_figure_ui import make_export_button
        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        self._btn_export.setFixedWidth(BTN_W)
        self._btn_restart = QPushButton("Restart")
        self._btn_restart.setFixedWidth(BTN_W)
        self._btn_restart.clicked.connect(self._on_restart_step1)
        r1 = _make_row()
        r1.addWidget(self._btn_restart)
        r1.addWidget(self._btn_export)
        r1.addStretch()
        vl.addLayout(r1)

        # Row 2: [Back] [Next]
        self._btn_back = QPushButton("Back")
        self._btn_back.setFixedWidth(BTN_W)
        self._btn_back.clicked.connect(self._on_wizard_back)
        self._btn_next = QPushButton("Next")
        self._btn_next.setFixedWidth(BTN_W)
        self._btn_next.setObjectName("confirmBtn")
        self._btn_next.clicked.connect(self._on_wizard_next)
        r2 = _make_row()
        r2.addWidget(self._btn_back)
        r2.addWidget(self._btn_next)
        r2.addStretch()
        vl.addLayout(r2)

        main_layout.addWidget(panel)

        # All step containers
        self._step_widgets = [self._step0, self._step1, self._step2, self._step3]
        self._step_titles = [
            "Step 1/4: Thresholds",
            "Step 2/4: Key Peaks",
            "Step 3/4: Exclusion",
            "Step 4/4: Rect. Method",
        ]
        self._step_descs = [
            "Drag the green (upper) and red (lower) threshold lines "
            "on the plot to set peak detection levels.",
            "Click detected peaks to mark them as key peaks (P1, P2...). "
            "These define segment boundaries for rectangularization. "
            "Skip if not needed.",
            "Exclude individual peaks or define exclusion zones "
            "to remove unwanted detections.",
            "Choose the rectangularization method for each segment. "
            "Review the result in the bottom plot.",
        ]
        self._show_wizard_step(0)

    def _make_toggle(self, label, mode, hint):
        """Create a checkable button that sets the click mode."""
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.clicked.connect(lambda checked, m=mode, h=hint, b=btn:
                            self._on_toggle(b, m, h, checked))
        self._toggle_buttons[mode] = btn
        return btn

    def _on_toggle(self, btn, mode, hint, checked):
        """Mutual-exclusion toggle: uncheck others, set mode."""
        if checked:
            for m, b in self._toggle_buttons.items():
                if b is not btn:
                    b.setChecked(False)
            self._click_mode = mode
            self._set_hint(hint)
        else:
            self._click_mode = self.MODE_NONE
            self._set_hint("")
        # Reset partial states
        self._zone_first_x = None
        self._plot()

    # --------------------------------------------------------- wizard nav
    def _show_wizard_step(self, step):
        """Show only the controls for the given wizard step."""
        self._wizard_step = step
        for i, w in enumerate(self._step_widgets):
            w.setVisible(i == step)
        self._lbl_step.setText(self._step_titles[step])
        self._lbl_step_desc.setText(self._step_descs[step])

        # Save button only on last step; Load Settings only on Step 1.
        is_last = step == len(self._step_widgets) - 1
        self._btn_save_tp.setVisible(is_last)
        self._btn_load_tp.setVisible(step == 0)

        # Navigation buttons
        self._btn_back.setEnabled(step > 0)
        if step < len(self._step_widgets) - 1:
            self._btn_next.setText("Next")
        else:
            self._btn_next.setText("Confirm")

        # Stepper pills: current checked, locked steps disabled.
        for i, b in enumerate(getattr(self, "_step_btns", [])):
            b.blockSignals(True)
            b.setChecked(i == step)
            b.setEnabled(i <= self._max_unlocked_step)
            b.blockSignals(False)

        # Deactivate any active tool and auto-activate per step
        self._deactivate_tool()
        if step == 0:
            self._update_thresh_labels()
        elif step == 1:
            # Auto-activate Mark mode
            btn = self._toggle_buttons.get(self.MODE_MARK_PEAK)
            if btn:
                btn.setChecked(True)
                self._click_mode = self.MODE_MARK_PEAK
                self._set_hint("Click a detected peak to mark as key peak.")

    def _revert_mode_after_action(self):
        """After a one-shot action completes, drop back to inspect mode."""
        active = self._click_mode
        if active and active in self._toggle_buttons:
            btn = self._toggle_buttons[active]
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._click_mode = self.MODE_NONE
        self._zone_first_x = None
        self._set_hint("")

    def _jump_to_step(self, idx):
        """Jump to a previously-confirmed step via the stepper pill."""
        if 0 <= idx <= self._max_unlocked_step:
            self._show_wizard_step(idx)
        else:
            # Re-sync the visual state of the locked pill.
            self._show_wizard_step(self._wizard_step)

    def _on_wizard_back(self):
        if self._wizard_step > 0:
            self._show_wizard_step(self._wizard_step - 1)

    def _on_wizard_next(self):
        if self._wizard_step < len(self._step_widgets) - 1:
            # Confirming the current step unlocks the next.
            self._max_unlocked_step = max(
                self._max_unlocked_step, self._wizard_step + 1)
            self._show_wizard_step(self._wizard_step + 1)
        else:
            # Last step — Confirm
            self._on_finish()

    def _update_thresh_labels(self):
        """Update threshold value labels + live peak count on step 0."""
        if hasattr(self, '_lbl_thresh_pos'):
            self._lbl_thresh_pos.setText(
                f"Upper: {self._pos_thresh:.3e}")
            self._lbl_thresh_neg.setText(
                f"Lower: {self._neg_thresh:.3e}")
        if hasattr(self, '_lbl_peak_count'):
            pos_idx, neg_idx = self._get_effective_peaks()
            text = f"Peaks: + {len(pos_idx)}  / − {len(neg_idx)}"
            counts = self._expected_peak_counts
            if counts:
                n_pos_exp, n_neg_exp = counts
                # Geometry-vs-actual polarity is unknown without alignment,
                # so accept either sign convention by checking both.
                total_exp = n_pos_exp + n_neg_exp
                total_got = len(pos_idx) + len(neg_idx)
                ok = (
                    (len(pos_idx) == n_pos_exp and len(neg_idx) == n_neg_exp)
                    or (len(pos_idx) == n_neg_exp and len(neg_idx) == n_pos_exp)
                ) and total_got == total_exp
                marker = (" ✓" if ok
                          else f"  (expected {n_pos_exp} / {n_neg_exp})")
                text += marker
                color = ("#2a7a2a" if ok else theme.palette.text)
                self._lbl_peak_count.setStyleSheet(f"color: {color};")
            self._lbl_peak_count.setText(text)

    def _set_hint(self, text):
        """Set hint text with style matching the current tool state."""
        if self._click_mode != self.MODE_NONE and text:
            self._lbl_hint.setText(f"{text}  [click button or Esc to cancel]")
            self._lbl_hint.setStyleSheet(self._HINT_ACTIVE)
        else:
            self._lbl_hint.setText(text)
            self._lbl_hint.setStyleSheet(self._HINT_IDLE)

    def _deactivate_tool(self):
        """Deactivate the current tool mode."""
        for b in self._toggle_buttons.values():
            b.setChecked(False)
        self._click_mode = self.MODE_NONE
        self._peak_adjust_mode = False
        if self._cursor_circle is not None:
            self._cursor_circle.hide()
        self._set_hint("")
        self._zone_first_x = None
        self._plot()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self._peak_adjust_mode:
                self._peak_adjust_mode = False
                if self._cursor_circle is not None:
                    self._cursor_circle.hide()
                self._set_hint("")
                self._update_all()
                return
            if self._click_mode != self.MODE_NONE:
                self._deactivate_tool()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------ peak adjust helpers
    def _get_effective_peaks(self):
        """Return (pos_idx, neg_idx) arrays accounting for manual adds/removes."""
        td = self._template_diff
        pks_pos_idx, _ = find_peaks(td, height=self._pos_thresh)
        pks_neg_idx, _ = find_peaks(-td, height=abs(self._neg_thresh))

        pos_set = set(pks_pos_idx.tolist())
        neg_set = set(pks_neg_idx.tolist())

        for di in self._added_peaks:
            if 0 <= di < len(td):
                if td[di] >= 0:
                    pos_set.add(di)
                else:
                    neg_set.add(di)

        pos_set -= self._removed_peaks
        neg_set -= self._removed_peaks

        return (np.array(sorted(pos_set), dtype=int),
                np.array(sorted(neg_set), dtype=int))

    def _raw_peak_locations(self):
        """Threshold-detected peaks (template-index space), independent of
        exclusion zones. Key-peak sequence indices are computed against this
        list so save and load reference the same, reproducible ordering —
        exclusions remove peaks from _peak_locations and would otherwise shift
        the indices between save (post-exclusion) and load (pre-exclusion)."""
        pks_pos_idx, pks_neg_idx = self._get_effective_peaks()
        idx = np.unique(np.concatenate([pks_pos_idx + 1, pks_neg_idx + 1]))
        return idx[(idx > 0) & (idx < self._n - 1)]

    def _toggle_peak_adjust(self):
        """Toggle peak adjustment mode on/off."""
        self._peak_adjust_mode = not self._peak_adjust_mode
        if self._peak_adjust_mode:
            self._set_hint("Click to add/remove peaks on derivative plot.")
            self._lbl_hint.setStyleSheet(self._HINT_ACTIVE)
        else:
            if self._cursor_circle is not None:
                self._cursor_circle.hide()
            self._set_hint("")
            self._update_all()

    def _find_nearest_peak_display(self, event):
        """Find nearest effective peak to cursor in display-pixel distance.

        Returns (diff_index, dist_px) or (None, inf).
        """
        pks_pos, pks_neg = self._get_effective_peaks()
        all_peaks = np.concatenate([pks_pos, pks_neg])
        if len(all_peaks) == 0:
            return None, float('inf')

        td = self._template_diff
        ax = self.ax_top
        cursor_disp = ax.transData.transform((event.xdata, event.ydata))

        best_idx = None
        best_dist = float('inf')
        for di in all_peaks:
            pk_disp = ax.transData.transform((di, td[di]))
            dist = np.sqrt((cursor_disp[0] - pk_disp[0]) ** 2 +
                           (cursor_disp[1] - pk_disp[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = int(di)
        return best_idx, best_dist

    def _handle_peak_adjust_click(self, event):
        """Add or remove a peak at the click location."""
        td = self._template_diff
        ax = self.ax_top
        radius_px = self._slider_cursor_radius.value()

        # Check if click is near an existing effective peak -> remove
        nearest_di, dist = self._find_nearest_peak_display(event)
        if nearest_di is not None and dist <= radius_px:
            # Remove this peak
            self._removed_peaks.add(nearest_di)
            self._added_peaks.discard(nearest_di)
            # Also remove from key peaks (key peaks use 1-based template idx)
            kp_key = nearest_di + 1
            self._key_peaks.pop(kp_key, None)
            self._update_all()
            return

        # No existing peak nearby — find strongest derivative sample in circle
        cursor_disp = ax.transData.transform((event.xdata, event.ydata))
        left_data = ax.transData.inverted().transform(
            (cursor_disp[0] - radius_px, cursor_disp[1]))
        right_data = ax.transData.inverted().transform(
            (cursor_disp[0] + radius_px, cursor_disp[1]))
        lo = max(0, int(round(left_data[0])))
        hi = min(len(td), int(round(right_data[0])) + 1)
        if hi <= lo:
            return

        best_di = None
        best_abs = -1
        for ci in range(lo, hi):
            pk_disp = ax.transData.transform((ci, td[ci]))
            d = np.sqrt((cursor_disp[0] - pk_disp[0]) ** 2 +
                        (cursor_disp[1] - pk_disp[1]) ** 2)
            if d <= radius_px and abs(td[ci]) > best_abs:
                best_abs = abs(td[ci])
                best_di = ci

        if best_di is None:
            return

        self._added_peaks.add(best_di)
        self._removed_peaks.discard(best_di)
        self._update_all()

    def _update_peak_hover(self, event):
        """Move cursor circle on derivative plot during peak adjust mode."""
        if event.inaxes != self.ax_top or event.xdata is None:
            self._cursor_circle.hide()
            return

        ax = self.ax_top
        cursor_disp = ax.transData.transform((event.xdata, event.ydata))
        dpr = self.canvas.devicePixelRatioF()
        canvas_h = self.canvas.height()
        cx = int(cursor_disp[0] / dpr)
        cy = int(canvas_h - cursor_disp[1] / dpr)

        radius_px = self._slider_cursor_radius.value()
        nearest_di, dist = self._find_nearest_peak_display(event)
        has_peak = nearest_di is not None and dist <= radius_px

        self._cursor_circle.set_pos(cx, cy, has_peak,
                                    draw_radius=int(radius_px / dpr))
        self._cursor_circle.show()

    # --------------------------------------------------------------- setup
    def _setup_interaction(self):
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)

    # ---------------------------------------------------------------- plot
    def _plot(self):
        td = self._template_diff
        x_diff = self._x_diff
        n = self._n

        self.ax_top.clear()
        self.ax_top.plot(x_diff, td, 'b', linewidth=0.8)
        self._hl_pos = self.ax_top.axhline(
            self._pos_thresh, color=(0, 0.6, 0),
            linewidth=2, linestyle='-', label='Upper threshold')
        self._hl_neg = self.ax_top.axhline(
            self._neg_thresh, color=(0.8, 0, 0),
            linewidth=2, linestyle='-', label='Lower threshold')

        pks_pos_idx, pks_neg_idx = self._get_effective_peaks()

        # Separate key peaks from regular peaks for different markers
        kp_indices_set = set(self._key_peaks.keys())  # template-level indices (1-based)
        kp_diff_set = {p - 1 for p in kp_indices_set}  # diff-array indices (0-based)

        # Regular (non-key) positive peaks — circles
        reg_pos_mask = np.array([p not in kp_diff_set for p in pks_pos_idx]) if len(pks_pos_idx) else np.array([], dtype=bool)
        pos_x = x_diff[pks_pos_idx[reg_pos_mask]] if np.any(reg_pos_mask) else []
        pos_y = td[pks_pos_idx[reg_pos_mask]] if np.any(reg_pos_mask) else []
        self._art_pks_pos = self.ax_top.plot(
            pos_x, pos_y, 'o', color=(0, 0.6, 0), markersize=6,
            label='Pos peaks')[0]

        # Regular (non-key) negative peaks — squares
        reg_neg_mask = np.array([p not in kp_diff_set for p in pks_neg_idx]) if len(pks_neg_idx) else np.array([], dtype=bool)
        neg_x = x_diff[pks_neg_idx[reg_neg_mask]] if np.any(reg_neg_mask) else []
        neg_y = td[pks_neg_idx[reg_neg_mask]] if np.any(reg_neg_mask) else []
        self._art_pks_neg = self.ax_top.plot(
            neg_x, neg_y, 's', color=(0.8, 0, 0), markersize=6,
            label='Neg peaks')[0]

        # Excluded markers
        detected_idx = set(pks_pos_idx + 1) | set(pks_neg_idx + 1)
        all_excluded = self._effective_excluded(detected_idx)
        excl_list = [p for p in all_excluded
                     if p in detected_idx and 1 <= p - 1 < len(td)]
        excl_x = [x_diff[p - 1] for p in excl_list] if excl_list else []
        excl_y = [td[p - 1] for p in excl_list] if excl_list else []
        self._art_excl = self.ax_top.plot(
            excl_x, excl_y, 'kx', markersize=8,
            linewidth=2, label='Excluded')[0]


        # Key peak markers: ▲ max, ▼ min, ★ regular (replace circle/square)
        max_abs = np.max(np.abs(td)) if len(td) > 0 else 1.0
        if max_abs == 0:
            max_abs = 1.0
        y_top = 1.1 * max_abs
        self._art_large_lines = []
        for i, lp in enumerate(sorted(self._key_peaks)):
            dx = lp - 1
            if 0 <= dx < len(td):
                prop = self._key_peaks[lp]
                pk_y = td[dx]
                color = (0, 0.6, 0) if pk_y >= 0 else (0.8, 0, 0)
                if prop == "max":
                    marker, ms = "^", 8
                elif prop == "min":
                    marker, ms = "v", 8
                else:
                    marker, ms = "*", 10
                prop_suffix = f" ({prop})" if prop != "regular" else ""
                line = self.ax_top.axvline(
                    dx, color='black', linewidth=0.8,
                    linestyle='--', alpha=0.4)
                self.ax_top.plot(dx, pk_y, marker, color=color,
                                markersize=ms, markeredgecolor="none",
                                zorder=6)
                # Label sits to the RIGHT of the marker, vertically centered
                # on the marker tip — avoids being clipped by the y-extents.
                txt = self.ax_top.text(
                    dx, pk_y,
                    f"  P{i + 1}{prop_suffix} ",
                    color='black', fontsize=9,
                    fontweight='bold', va='center', ha='left',
                    clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor='black',
                              alpha=0.85, linewidth=0.8))
                self._art_large_lines.append((line, txt, lp))

        # Exclusion zones
        self._art_zones = []
        for (zs, ze) in self._excl_zones:
            ds = max(zs - 1, 0)
            de = min(ze - 1, len(td) - 1)
            span = self.ax_top.axvspan(
                ds, de, alpha=0.15, color='red', zorder=0)
            self._art_zones.append((span, ds, de))

        # End zones (drawn differently from manual zones)
        for (zs, ze) in self._get_end_zones():
            ds = max(zs - 1, 0)
            de = min(ze - 1, len(td) - 1)
            self.ax_top.axvspan(ds, de, alpha=0.10, color='gray', zorder=0,
                                hatch='///', linewidth=0)

        # Partial markers for two-click modes
        if self._zone_first_x is not None:
            self.ax_top.axvline(self._zone_first_x - 1, color='orange',
                                linewidth=2, linestyle='-', alpha=0.8)

        # Selectable zone regions when Select mode is active
        self._art_select_zones = []
        if self._click_mode == self.MODE_SELECT_ZONE:
            boundaries = self._get_select_zone_boundaries()
            for i in range(len(boundaries) - 1):
                zs, ze = boundaries[i], boundaries[i + 1]
                is_excluded = self._is_region_excluded(zs, ze)
                if not is_excluded:
                    ds = max(zs - 1, 0)
                    de = min(ze - 1, len(td) - 1)
                    span = self.ax_top.axvspan(
                        ds, de, alpha=0.06, color='blue', zorder=0,
                        linestyle='--', linewidth=0.5, edgecolor='blue')
                    self._art_select_zones.append((span, zs, ze))

        # Hidden hover-ring overlay for peak highlighting
        self._hover_ring, = self.ax_top.plot(
            [], [], 'o', markersize=12, markerfacecolor='none',
            markeredgecolor='#FFD700', markeredgewidth=2.5,
            visible=False, zorder=15)
        self._hover_target = None

        self.ax_top.set_ylabel('Diff')
        self.ax_top.set_title('Template Derivative with Thresholds')
        # No legend on top plot
        self.ax_top.set_ylim(-y_top, y_top)
        self.ax_top.set_xlim(0, len(td))

        # Bottom plot
        self.ax_bot.clear()

        # Segment spans (alternating background for each segment)
        all_idx = self._last_all_idx
        self._segment_spans = []
        seg_colors = ['#a0c0e8', '#e8d0a0']
        n = self._n
        for k in range(len(all_idx) - 1):
            s = int(all_idx[k])
            e = int(all_idx[k + 1]) if k < len(all_idx) - 2 else n
            span = self.ax_bot.axvspan(
                s, e, alpha=0.10, color=seg_colors[k % 2], zorder=0)
            self._segment_spans.append((span, s, e))
            if s > 0:
                self.ax_bot.axvline(
                    s, color='#999', linewidth=0.8,
                    linestyle=':', alpha=0.6, zorder=1)

        self.ax_bot.plot(self._x, self._filtered, color='grey',
                         linewidth=0.8, label='Filtered')
        self.ax_bot.plot(self._x, self._rect_template, color='teal',
                         linewidth=1.5, label='Rectangularized')
        self.ax_bot.set_xlabel('Sample Index')
        self.ax_bot.set_ylabel('Amplitude')
        self.ax_bot.set_title('Rectangularized Pulse Template')
        # No legend on bottom plot
        self.ax_bot.set_xlim(0, self._n)

        self.canvas.draw_idle()

    # --------------------------------------------------------- exclusion
    def _get_end_zones(self):
        """Return end zone ranges based on current settings."""
        if not self._chk_end_zones.isChecked():
            return []
        pct = self._spin_end_zone.value() / 100.0
        zone_size = int(self._n * pct)
        zones = []
        if zone_size > 0:
            zones.append((0, zone_size - 1))
            zones.append((self._n - zone_size, self._n - 1))
        return zones

    def _effective_excluded(self, detected_idx):
        excluded = set()
        key_set = set(self._key_peaks)
        all_zones = list(self._excl_zones) + self._get_end_zones()
        for pk in detected_idx:
            if pk in key_set:
                continue  # key peaks are never excluded by zones
            for (zs, ze) in all_zones:
                if zs <= pk <= ze:
                    excluded.add(pk)
                    break
        return excluded

    # ------------------------------------------------------- update / rect
    def _update_all(self):
        # Refresh threshold + peak-count labels (Step 0) on every redraw.
        self._update_thresh_labels()

        td = self._template_diff
        n = self._n

        pks_pos_idx, pks_neg_idx = self._get_effective_peaks()

        idx_pos = pks_pos_idx + 1
        idx_neg = pks_neg_idx + 1

        all_idx = np.unique(np.concatenate([[0], idx_pos, idx_neg, [n - 1]]))
        detected_idx = set(pks_pos_idx + 1) | set(pks_neg_idx + 1)
        excluded = self._effective_excluded(detected_idx)
        all_idx = np.array([i for i in all_idx
                            if i not in excluded or i in (0, n - 1)])
        all_idx = np.sort(all_idx)
        apex_idx = all_idx
        if self._edge_method == 'fwhm':
            all_idx = _fwhm_refine_boundaries(self._filtered, apex_idx)
        self._last_all_idx = all_idx

        self._peak_locations = all_idx[(all_idx > 0) & (all_idx < n - 1)]
        self._apex_locations = apex_idx[(apex_idx > 0) & (apex_idx < n - 1)]

        # Rectangularize with per-segment method overrides
        global_method = self._global_rect_method
        rect = np.zeros(n)
        if len(all_idx) < 2:
            rect[:] = np.median(self._filtered)
        else:
            for k in range(len(all_idx) - 1):
                s = int(all_idx[k])
                e = int(all_idx[k + 1]) if k < len(all_idx) - 2 else n
                method = self._segment_methods.get(s, global_method)
                rect[s:e] = _rect_segment(self._filtered, s, e, method)
        self._rect_template = rect

        # Peak sequence — signs come from the derivative at the APEX, not the
        # refined slicing index; refinement preserves count and order.
        seq = []
        for pk in self._apex_locations:
            if pk - 1 < len(td):
                seq.append(1 if td[pk - 1] >= 0 else 0)
            else:
                seq.append(0)
        self._peak_sequence = np.array(seq, dtype=int)

        # Labels
        self._lbl_key_peaks.setText(f"Marked: {len(self._key_peaks)}")
        end_zones = self._get_end_zones()
        zone_text = f"Zones: {len(self._excl_zones)}"
        if end_zones:
            zone_text += f" (+{len(end_zones)} end)"
        self._lbl_zones.setText(zone_text)
        self._update_output_seq_label()

        self._plot()

    # -------------------------------------------------------- interaction
    def _on_press(self, event):
        if event.ydata is None:
            return

        if event.inaxes == self.ax_top:
            # Peak adjust click — step 1 (Key Peaks)
            if self._peak_adjust_mode and self._wizard_step == 2:
                self._handle_peak_adjust_click(event)
                return

            # Threshold drag — only on step 0
            if self._wizard_step == 0:
                y = event.ydata
                ylims = self.ax_top.get_ylim()
                tol_y = 0.03 * (ylims[1] - ylims[0])
                if abs(y - self._pos_thresh) < tol_y:
                    self._dragging_line = 'pos'
                    return
                if abs(y - self._neg_thresh) < tol_y:
                    self._dragging_line = 'neg'
                    return

            m = self._click_mode
            # Key peaks — only on step 1
            if m == self.MODE_MARK_PEAK and self._wizard_step == 1:
                self._do_mark_key_peak(event)
                self._revert_mode_after_action()
            # Exclusion tools — only on step 2 (zones only; peak-level exclude removed)
            elif m == self.MODE_SELECT_ZONE and self._wizard_step == 2:
                self._do_select_zone(event)
                self._revert_mode_after_action()
            elif m == self.MODE_DRAW_ZONE and self._wizard_step == 2:
                self._do_draw_zone(event)
                # Draw zone needs two clicks; revert only when the pair completes.
                if self._zone_first_x is None:
                    self._revert_mode_after_action()

        elif event.inaxes == self.ax_bot:
            # Set segment — only on step 3
            if self._click_mode == self.MODE_SET_SEGMENT and self._wizard_step == 3:
                self._do_set_segment(event)

    def _on_motion(self, event):
        if self._peak_adjust_mode and self._wizard_step == 2:
            self._update_peak_hover(event)
            return
        if self._dragging_line is not None:
            if event.inaxes != self.ax_top or event.ydata is None:
                return
            if self._dragging_line == 'pos':
                self._pos_thresh = max(0, event.ydata)
                self._hl_pos.set_ydata([self._pos_thresh, self._pos_thresh])
            else:
                self._neg_thresh = min(0, event.ydata)
                self._hl_neg.set_ydata([self._neg_thresh, self._neg_thresh])
            self._update_thresh_labels()
            self._update_peaks_fast()
            self.canvas.draw_idle()
            return
        self._update_hover(event)

    def _on_release(self, event):
        if self._dragging_line is not None:
            self._dragging_line = None
            self._update_all()
            return
        self._dragging_line = None

    def _update_peaks_fast(self):
        """Lightweight peak marker update during threshold drag.

        Only updates peak/excluded marker data without clearing axes.
        """
        td = self._template_diff
        x_diff = self._x_diff

        pks_pos_idx, pks_neg_idx = self._get_effective_peaks()

        if self._art_pks_pos is not None:
            if len(pks_pos_idx) > 0:
                self._art_pks_pos.set_data(x_diff[pks_pos_idx], td[pks_pos_idx])
                self._art_pks_pos.set_visible(True)
            else:
                self._art_pks_pos.set_data([], [])
        if self._art_pks_neg is not None:
            if len(pks_neg_idx) > 0:
                self._art_pks_neg.set_data(x_diff[pks_neg_idx], td[pks_neg_idx])
                self._art_pks_neg.set_visible(True)
            else:
                self._art_pks_neg.set_data([], [])

        # Update excluded markers
        detected_idx = set(pks_pos_idx + 1) | set(pks_neg_idx + 1)
        all_excluded = self._effective_excluded(detected_idx)
        excl_list = [p for p in all_excluded
                     if p in detected_idx and 1 <= p - 1 < len(td)]
        if self._art_excl is not None:
            if excl_list:
                edi = [p - 1 for p in excl_list]
                self._art_excl.set_data(x_diff[edi], td[edi])
                self._art_excl.set_visible(True)
            else:
                self._art_excl.set_data([], [])

    def _on_axes_leave(self, _event):
        if self._peak_adjust_mode and self._cursor_circle is not None:
            self._cursor_circle.hide()
        if self._hover_target is not None:
            self._clear_hover()
            self._hover_target = None
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self.canvas.draw_idle()

    # ---- Hover helpers ----

    def _update_hover(self, event):
        new_target = self._hit_test(event)
        if new_target == self._hover_target:
            return
        self._clear_hover()
        self._hover_target = new_target
        self._show_hover(new_target)
        if new_target in ('pos_line', 'neg_line'):
            self.canvas.setCursor(Qt.CursorShape.SizeVerCursor)
        elif isinstance(new_target, tuple) and new_target[0] == 'segment':
            if self._click_mode == self.MODE_SET_SEGMENT:
                self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        elif isinstance(new_target, tuple) and new_target[0] == 'select_zone':
            self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        elif new_target is not None:
            self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        self.canvas.draw_idle()

    def _hit_test(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        if event.inaxes == self.ax_top:
            ylim = self.ax_top.get_ylim()
            tol_y = 0.03 * (ylim[1] - ylim[0])
            # Threshold lines (highest priority - always draggable)
            if abs(event.ydata - self._pos_thresh) < tol_y:
                return 'pos_line'
            if abs(event.ydata - self._neg_thresh) < tol_y:
                return 'neg_line'
            # Peak markers
            xlim = self.ax_top.get_xlim()
            tol_x = 0.015 * (xlim[1] - xlim[0])
            for art in (self._art_pks_pos, self._art_pks_neg):
                if art is not None:
                    xd, yd = art.get_data()
                    if len(xd) > 0:
                        dists = np.abs(np.asarray(xd) - event.xdata)
                        idx = np.argmin(dists)
                        if (dists[idx] < tol_x
                                and abs(yd[idx] - event.ydata) < tol_y * 2):
                            return ('peak', float(xd[idx]), float(yd[idx]))
            # Excluded markers
            if self._art_excl is not None:
                xd, yd = self._art_excl.get_data()
                if len(xd) > 0:
                    dists = np.abs(np.asarray(xd) - event.xdata)
                    idx = np.argmin(dists)
                    if (dists[idx] < tol_x
                            and abs(yd[idx] - event.ydata) < tol_y * 2):
                        return ('excl', float(xd[idx]), float(yd[idx]))
            # Large peak lines (narrower hit area than markers)
            tol_line = 0.005 * (xlim[1] - xlim[0])
            for _line, _txt, lp in getattr(self, '_art_large_lines', []):
                dx = lp - 1
                if abs(event.xdata - dx) < tol_line:
                    return ('large', lp)
            # Selectable zone regions (Select mode)
            if self._click_mode == self.MODE_SELECT_ZONE:
                for i, (_span, zs, ze) in enumerate(
                        getattr(self, '_art_select_zones', [])):
                    ds = max(zs - 1, 0)
                    de = min(ze - 1, len(self._template_diff) - 1)
                    if ds <= event.xdata <= de:
                        return ('select_zone', i)
            # Exclusion zones
            for i, (_span, ds, de) in enumerate(
                    getattr(self, '_art_zones', [])):
                if ds <= event.xdata <= de:
                    return ('zone', i)
        elif event.inaxes == self.ax_bot:
            # Segment hover in bottom plot
            for i, (_span, s, e) in enumerate(
                    getattr(self, '_segment_spans', [])):
                if s <= event.xdata <= e:
                    return ('segment', i)
        return None

    def _clear_hover(self):
        if hasattr(self, '_hl_pos') and self._hl_pos:
            self._hl_pos.set_linewidth(2)
        if hasattr(self, '_hl_neg') and self._hl_neg:
            self._hl_neg.set_linewidth(2)
        if hasattr(self, '_art_pks_pos') and self._art_pks_pos:
            self._art_pks_pos.set_markersize(6)
        if hasattr(self, '_art_pks_neg') and self._art_pks_neg:
            self._art_pks_neg.set_markersize(6)
        if hasattr(self, '_art_excl') and self._art_excl:
            self._art_excl.set_markersize(8)
        for line, _txt, _ in getattr(self, '_art_large_lines', []):
            line.set_alpha(0.4)
            line.set_linewidth(0.8)
        for span, _, _ in getattr(self, '_art_zones', []):
            span.set_alpha(0.15)
        for span, _, _ in getattr(self, '_art_select_zones', []):
            span.set_alpha(0.06)
        for span, _, _ in getattr(self, '_segment_spans', []):
            span.set_alpha(0.10)
        if hasattr(self, '_hover_ring') and self._hover_ring:
            self._hover_ring.set_visible(False)

    def _show_hover(self, target):
        if target is None:
            return
        if target == 'pos_line':
            self._hl_pos.set_linewidth(3.5)
        elif target == 'neg_line':
            self._hl_neg.set_linewidth(3.5)
        elif isinstance(target, tuple) and target[0] == 'peak':
            _, px, py = target
            self._hover_ring.set_data([px], [py])
            self._hover_ring.set_visible(True)
        elif isinstance(target, tuple) and target[0] == 'excl':
            _, px, py = target
            self._hover_ring.set_data([px], [py])
            self._hover_ring.set_visible(True)
        elif isinstance(target, tuple) and target[0] == 'large':
            lp = target[1]
            for line, _txt, line_lp in self._art_large_lines:
                if line_lp == lp:
                    line.set_alpha(1.0)
                    line.set_linewidth(1.5)
                    break
        elif isinstance(target, tuple) and target[0] == 'zone':
            idx = target[1]
            if idx < len(self._art_zones):
                self._art_zones[idx][0].set_alpha(0.3)
        elif isinstance(target, tuple) and target[0] == 'select_zone':
            idx = target[1]
            if idx < len(self._art_select_zones):
                self._art_select_zones[idx][0].set_alpha(0.20)
        elif isinstance(target, tuple) and target[0] == 'segment':
            idx = target[1]
            if idx < len(self._segment_spans):
                self._segment_spans[idx][0].set_alpha(0.3)

    # ---- helpers ----
    def _find_nearest_detected_peak(self, x_click):
        if x_click is None:
            return None
        td = self._template_diff
        x_diff = self._x_diff
        pks_pos, pks_neg = self._get_effective_peaks()
        locs = np.sort(np.unique(np.concatenate([pks_pos, pks_neg])))
        if len(locs) == 0:
            return None
        nearest = locs[np.argmin(np.abs(x_diff[locs] - x_click))]
        tidx = nearest + 1
        if tidx in (0, self._n - 1):
            return None
        tol = 0.03 * (x_diff[-1] - x_diff[0])
        if abs(x_diff[nearest] - x_click) > tol:
            return None
        return tidx

    def _find_nearest_key_peak(self, x_click):
        """Return template_idx of the nearest key peak, or None."""
        if x_click is None or not self._key_peaks:
            return None
        diff_x = x_click  # in diff-space
        tol = 0.03 * self._n
        best = None
        best_dist = tol
        for lp in self._key_peaks:
            d = abs((lp - 1) - diff_x)
            if d < best_dist:
                best_dist = d
                best = lp
        return best

    def _snap_template_x(self, x_click):
        """Snap diff-space x to template index."""
        tidx = int(round(x_click)) + 1
        return max(1, min(tidx, self._n - 2))

    def _hovered_peak_tidx(self):
        """Return the template index of the currently hovered peak, or None."""
        ht = self._hover_target
        if isinstance(ht, tuple) and ht[0] in ('peak', 'excl'):
            # ht = ('peak'|'excl', diff_x, diff_y)
            return int(round(ht[1])) + 1
        return None

    # ---- mode handlers ----
    def _get_kp_property(self):
        """Return the currently selected key peak property."""
        if self._radio_kp_max.isChecked():
            return "max"
        elif self._radio_kp_min.isChecked():
            return "min"
        return "regular"

    def _do_mark_key_peak(self, event):
        """Toggle a detected peak as a key peak."""
        tidx = self._hovered_peak_tidx()
        if tidx is None:
            tidx = self._find_nearest_detected_peak(event.xdata)
        if tidx is None:
            return
        if tidx in self._key_peaks:
            del self._key_peaks[tidx]
        else:
            self._key_peaks[tidx] = self._get_kp_property()
        self._update_all()

    def _get_select_zone_boundaries(self):
        """Get the boundary points for selectable zones.

        Returns sorted list of template indices that form zone boundaries:
        [start, key_peak_1, key_peak_2, ..., end].
        When end zones are excluded, start/end are shifted to end zone edges.
        """
        end_zones = self._get_end_zones()
        if end_zones:
            # Start from end zone edges instead of 0 / n-1
            left_edge = end_zones[0][1] + 1   # right edge of left end zone
            right_edge = end_zones[1][0] - 1   # left edge of right end zone
        else:
            left_edge = 0
            right_edge = self._n - 1

        boundaries = [left_edge]
        for lp in sorted(self._key_peaks):
            if left_edge < lp < right_edge:
                boundaries.append(lp)
        boundaries.append(right_edge)
        return boundaries

    def _is_region_excluded(self, zs, ze):
        """Check if a region [zs, ze] is already covered by exclusion zones."""
        for (es, ee) in self._excl_zones:
            if es <= zs and ee >= ze:
                return True
        return False

    def _do_select_zone(self, event):
        """Click a region between key peaks to toggle it as exclusion zone."""
        if event.xdata is None:
            return
        click_x = self._snap_template_x(event.xdata)
        boundaries = self._get_select_zone_boundaries()

        if len(boundaries) < 2:
            self._set_hint("Mark key peaks first to create selectable regions.")
            return

        # Find which region the click falls in
        for i in range(len(boundaries) - 1):
            zs, ze = boundaries[i], boundaries[i + 1]
            if zs <= click_x <= ze:
                if self._is_region_excluded(zs, ze):
                    # Remove the zone that covers this region
                    self._excl_zones = [
                        (es, ee) for (es, ee) in self._excl_zones
                        if not (es <= zs and ee >= ze)]
                else:
                    # Add this region as an exclusion zone
                    self._excl_zones = _intersect_zone_into_list(
                        self._excl_zones, (zs, ze))
                self._update_all()
                return

    def _do_draw_zone(self, event):
        """Two-click free-form zone."""
        if event.xdata is None:
            return
        tx = self._snap_template_x(event.xdata)
        if self._zone_first_x is None:
            self._zone_first_x = tx
            self._set_hint(f"First: {tx}. Click second position.")
            self._plot()
        else:
            zs = min(self._zone_first_x, tx)
            ze = max(self._zone_first_x, tx)
            if zs < ze:
                self._excl_zones = _intersect_zone_into_list(
                    self._excl_zones, (zs, ze))
            self._zone_first_x = None
            self._set_hint("Click two positions to define a zone.")
            self._update_all()

    def _do_set_segment(self, event):
        """Click a segment in the bottom plot to set its rect method."""
        if event.xdata is None:
            return
        tx = int(round(event.xdata))
        tx = max(0, min(tx, self._n - 1))
        all_idx = self._last_all_idx
        if len(all_idx) < 2:
            return
        # Find which segment contains tx
        for k in range(len(all_idx) - 1):
            s = int(all_idx[k])
            e = int(all_idx[k + 1]) if k < len(all_idx) - 2 else self._n
            if s <= tx < e:
                method = self._rect_method_group.checkedButton().text().lower()
                self._segment_methods[s] = method
                self._set_hint(
                    f"Segment [{s}:{e}] set to {method}.")
                self._update_all()
                return

    # --------------------------------------------------------- callbacks
    def _on_save_tpsettings(self):
        """Save the current TPsettings result to a JSON file."""
        from PySide6.QtWidgets import QFileDialog
        from utils.tpsettings_io import save_tpsettings
        from utils.paths import saved_templates_dir

        default_folder = saved_templates_dir("PulseShape")

        result = self._build_result(self._filter_config, self._padding_info)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save TPsettings", default_folder, "JSON Files (*.json)")
        if path:
            save_tpsettings(_wrap_zones([result]), path)
            self._set_hint(f"Saved to {path}")

    def _on_load_tpsettings(self):
        """Load a TPsettings JSON and apply it to the current dialog state."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from utils.tpsettings_io import load_tpsettings
        from utils.paths import saved_templates_dir

        import os
        default_folder = saved_templates_dir("PulseShape")

        path, _ = QFileDialog.getOpenFileName(
            self, "Load TPsettings", default_folder, "JSON Files (*.json)")
        if not path:
            return
        try:
            settings = load_tpsettings(path)
        except Exception as e:
            QMessageBox.warning(self, "Load TPsettings",
                                f"Failed to load:\n{e}")
            return
        # Reduce the canonical zone-grouped struct to a flat single-zone
        # view the dialog can apply (zone 0; multi-zone restore-per-channel
        # is not supported in the interactive dialog).
        self._apply_loaded_settings(get_zone(settings, 0))
        # Loaded settings imply all steps were previously confirmed.
        self._max_unlocked_step = 3
        self._show_wizard_step(self._wizard_step)
        self._set_hint(f"Loaded {os.path.basename(path)}")


    def _on_auto_key_peaks(self):
        """Auto-mark the largest positive and largest negative derivative peaks."""
        td = self._template_diff
        if len(td) == 0:
            return
        # Find detected peaks above/below current thresholds
        pks_pos_idx, pks_neg_idx = self._get_effective_peaks()
        detected_idx = set(pks_pos_idx + 1) | set(pks_neg_idx + 1)
        excluded = self._effective_excluded(detected_idx)

        self._key_peaks.clear()
        # Largest positive peak (not excluded) → "max"
        valid_pos = [i for i in pks_pos_idx if (i + 1) not in excluded]
        if valid_pos:
            best_pos = max(valid_pos, key=lambda i: td[i])
            self._key_peaks[int(best_pos + 1)] = "max"
        # Largest negative peak (not excluded) → "min"
        valid_neg = [i for i in pks_neg_idx if (i + 1) not in excluded]
        if valid_neg:
            best_neg = max(valid_neg, key=lambda i: abs(td[i]))
            self._key_peaks[int(best_neg + 1)] = "min"
        self._update_all()

    def _on_clear_key_peaks(self):
        self._key_peaks.clear()
        self._update_all()

    def _on_clear_zones(self):
        self._excl_zones.clear()
        self._zone_first_x = None
        self._update_all()

    def _on_end_zone_changed(self, _value=None):
        self._spin_end_zone.setEnabled(self._chk_end_zones.isChecked())
        self._update_all()

    def _on_apply_all_method(self):
        """Apply the dropdown method to all segments (clear overrides)."""
        self._global_rect_method = self._rect_method_group.checkedButton().text().lower()
        self._segment_methods.clear()
        self._update_all()

    def _on_edge_method_changed(self, btn):
        self._edge_method = btn.text().lower()
        self._update_all()

    def _apply_loaded_settings(self, settings):
        """Restore dialog state from a previously saved TPsettings dict.

        Key peaks and exclusion zones are mapped by SEQUENCE INDEX, not by
        absolute sample position. This allows settings to be applied to a
        different template with the same peak pattern but different timing.
        """
        # Thresholds — apply first so _update_all detects peaks correctly.
        # Sanity-check against the current template's derivative range:
        # saved thresholds are absolute values, so a template with a different
        # amplitude scale would push the lines off the visible axes.
        thresh = settings.get('threshold', {})
        self._pos_thresh = thresh.get('upper', 0.0)
        self._neg_thresh = thresh.get('lower', 0.0)
        td = self._template_diff
        if len(td) > 0:
            td_extent = float(np.max(np.abs(td)))
            if td_extent > 0:
                # Reset when the loaded thresholds don't fit the current
                # derivative's scale — too big (off-axis) or too small
                # (every peak passes), both come from cross-template loads.
                _floor = td_extent * 0.01
                if (abs(self._pos_thresh) > td_extent
                        or abs(self._neg_thresh) > td_extent
                        or abs(self._pos_thresh) < _floor
                        or abs(self._neg_thresh) < _floor):
                    self._reset_thresholds()
        self._update_thresh_labels()

        # Re-detect peaks with loaded thresholds before mapping. Map against
        # the raw (pre-exclusion) peak list so the sequence indices line up
        # with how they were computed at save time — exclusion zones aren't
        # restored yet, and they remove peaks from _peak_locations.
        self._update_all()
        current_peaks = list(self._raw_peak_locations())

        # Map key peaks. key_peaks/properties/seq_indices share one sorted
        # order. Prefer matching each saved absolute position to the nearest
        # current peak — exact for the same template and robust to exclusion
        # changes (and to settings saved before the seq-index fix). Fall back
        # to the saved sequence index for cross-template loads where timing
        # differs and no nearby peak exists.
        kp_props = settings.get('key_peak_properties', [])
        kp_seq_indices = settings.get('key_peak_seq_indices') or []
        saved_kp = settings.get('key_peaks')
        saved_kp = ([int(p) for p in saved_kp]
                    if saved_kp is not None and len(saved_kp) > 0 else [])
        self._key_peaks = {}

        snap_tol = max(3, int(self._n * 0.01))
        n_kp = max(len(saved_kp), len(kp_seq_indices))
        for j in range(n_kp):
            prop = str(kp_props[j]) if j < len(kp_props) else "regular"
            new_pos = None
            if j < len(saved_kp) and current_peaks:
                nearest = min(current_peaks, key=lambda p: abs(p - saved_kp[j]))
                if abs(int(nearest) - saved_kp[j]) <= snap_tol:
                    new_pos = int(nearest)
            if new_pos is None and j < len(kp_seq_indices):
                seq_idx = kp_seq_indices[j]
                if 0 <= seq_idx < len(current_peaks):
                    new_pos = int(current_peaks[seq_idx])
            if new_pos is not None:
                self._key_peaks[new_pos] = prop

        # Map exclusion zones using saved sequence bounds (key-peak-based)
        excl_seq_bounds = settings.get('exclusion_zone_seq_bounds')
        self._excl_zones = []
        n_diff = len(self._template_diff)
        current_kp = sorted(self._key_peaks.keys())

        if excl_seq_bounds and len(excl_seq_bounds) > 0:
            # Sequence bounds are relative to key peaks
            for s_seq, e_seq in excl_seq_bounds:
                if s_seq < 0:
                    new_s = 0
                elif s_seq < len(current_kp):
                    new_s = int(current_kp[s_seq])
                else:
                    new_s = n_diff
                if e_seq < 0:
                    new_e = 0
                elif e_seq >= len(current_kp):
                    new_e = n_diff
                elif e_seq < len(current_kp):
                    new_e = int(current_kp[e_seq])
                else:
                    new_e = n_diff
                if new_s < new_e:
                    self._excl_zones.append((new_s, new_e))
        else:
            # Fallback: map by matching saved key_peaks
            excl = settings.get('exclusion_zones', [])
            saved_kp = settings.get('key_peaks', [])
            saved_kp = sorted(int(p) for p in saved_kp) if saved_kp is not None and len(saved_kp) > 0 else []
            if excl and saved_kp and current_kp:
                for zs, ze in excl:
                    zs, ze = int(zs), int(ze)
                    if zs <= saved_kp[0]:
                        new_s = 0
                    else:
                        dists = [abs(sp - zs) for sp in saved_kp]
                        idx = dists.index(min(dists))
                        new_s = int(current_kp[idx]) if idx < len(current_kp) else 0
                    if ze >= saved_kp[-1]:
                        new_e = n_diff
                    else:
                        dists = [abs(sp - ze) for sp in saved_kp]
                        idx = dists.index(min(dists))
                        new_e = int(current_kp[idx]) if idx < len(current_kp) else n_diff
                    if new_s < new_e:
                        self._excl_zones.append((new_s, new_e))
            elif excl:
                self._excl_zones = [(int(s), int(e)) for s, e in excl]

        # End zones
        ez = settings.get('end_zones', {})
        if ez.get('enabled', False):
            self._chk_end_zones.setChecked(True)
            self._spin_end_zone.setValue(ez.get('percent', 5))
            self._spin_end_zone.setEnabled(True)
        else:
            self._chk_end_zones.setChecked(False)

        # Rect method
        methods = settings.get('methods', ['median'])
        if methods:
            global_method = methods[0]
            self._global_rect_method = global_method
            # Select the matching radio button
            for btn in self._rect_method_group.buttons():
                if btn.text().lower() == global_method:
                    btn.setChecked(True)
                    break

        # Edge method — settings saved before this field existed sliced at
        # the apexes, so a missing key restores as 'peaks'.
        em = settings.get('edge_method', 'peaks')
        self._edge_method = em if em in ('peaks', 'fwhm') else 'peaks'
        (self._radio_edge_fwhm if self._edge_method == 'fwhm'
         else self._radio_edge_peaks).setChecked(True)

        # Re-slice with the restored edge method: the _update_all above ran
        # under the pre-load method, so boundaries / rect would otherwise be
        # stale until the next interaction.
        self._update_all()

    def _predict_expected_peaks(self):
        """Return [(sign, sample_pos)] from geometry, or None.

        Bracketed with virtual baseline (∞-width) sentinels at both ends:
        first transition is the channel entry (always −) and the last is the
        channel exit (always +).
        """
        seq = self._geometry_width_sequence()
        if not seq:
            return None
        total_len = max(t["position_um"] for t in seq) if seq else 0.0
        if total_len <= 0:
            return None
        scale = self._n / total_len  # samples per length unit
        return [(t["sign"], t["position_um"] * scale) for t in seq]

    @property
    def _expected_peak_counts(self):
        """Return (n_pos_width_steps, n_neg_width_steps) or None."""
        seq = self._predict_expected_peaks()
        if not seq:
            return None
        return (sum(1 for s, _ in seq if s > 0),
                sum(1 for s, _ in seq if s < 0))

    def _geometry_width_sequence(self):
        """Walk geometry components with ∞-width baseline sentinels at both ends.

        Returns list of dicts {sign, from_w, to_w, position_um, length_um}.
        The first entry is the channel-entry transition (baseline → comp[0]),
        always −; the last is the channel-exit transition (comp[-1] → baseline),
        always +. Internal entries cover width changes between adjacent
        components (Node or Pore).
        """
        g = self._geometry
        if not g or not isinstance(g, dict):
            return []
        comps = [c for c in (g.get("components") or [])
                 if isinstance(c, dict) and "width" in c and "length" in c]
        if not comps:
            return []

        out = []
        # Channel entry: baseline (∞) → first component → always narrowing.
        out.append({
            "sign": -1,
            "from_w": float("inf"),
            "to_w": float(comps[0]["width"]),
            "position_um": 0.0,
            "length_um": float(comps[0]["length"]),
        })

        cum = float(comps[0]["length"])
        for i in range(1, len(comps)):
            prev_w = float(comps[i - 1]["width"])
            curr_w = float(comps[i]["width"])
            dw = curr_w - prev_w
            if dw != 0:
                out.append({
                    "sign": 1 if dw > 0 else -1,
                    "from_w": prev_w,
                    "to_w": curr_w,
                    "position_um": cum,
                    "length_um": float(comps[i]["length"]),
                })
            cum += float(comps[i]["length"])

        # Channel exit: last component → baseline (∞) → always widening.
        out.append({
            "sign": 1,
            "from_w": float(comps[-1]["width"]),
            "to_w": float("inf"),
            "position_um": cum,
            "length_um": 0.0,
        })
        return out

    def _update_predicted_seq_label(self):
        """Render the geometry-predicted sequence as a compact text block."""
        if not hasattr(self, "_lbl_predicted_seq"):
            return
        seq = self._geometry_width_sequence()
        if not seq:
            self._lbl_predicted_seq.setText("")
            self._lbl_predicted_seq.setVisible(False)
            return
        signs = "".join("+" if t["sign"] > 0 else "−" for t in seq)
        n_pos = sum(1 for t in seq if t["sign"] > 0)
        n_neg = sum(1 for t in seq if t["sign"] < 0)
        self._lbl_predicted_seq.setVisible(True)
        self._lbl_predicted_seq.setText(
            f"Predicted ({n_pos} ↑ / {n_neg} ↓):  {signs}")

    def _update_output_seq_label(self):
        """Render the final detected peak sequence (post-exclusion) on step 4.

        Uses the same marker style as the review block's Template readout:
        green = positive, red = negative; ▲ max / ▼ min / ★ regular key peak,
        ● non-key."""
        if not hasattr(self, "_lbl_output_seq"):
            return
        seq = self._peak_sequence
        locs = self._apex_locations
        n_pos = int(np.sum(seq == 1))
        n_neg = int(np.sum(seq == 0))
        parts = []
        for i, s in enumerate(seq):
            color = "green" if s == 1 else "red"
            pos = int(locs[i]) if i < len(locs) else None
            pol = self._key_peaks.get(pos)
            if pol == "max":
                sym = "▲"   # ▲
            elif pol == "min":
                sym = "▼"   # ▼
            elif pol is not None:
                sym = "★"   # ★ regular key peak
            else:
                sym = "●"   # ● non-key
            parts.append(
                f'<span style="color:{color}; font-size:13px;">{sym}</span>')
        signs = " ".join(parts) if parts else "—"
        self._lbl_output_seq.setText(
            f"Output ({n_pos} ↑ / {n_neg} ↓):<br>{signs}")

    def _auto_threshold_sigma(self, k=3.0):
        """Set thresholds to ±k·σ of the template derivative."""
        td = self._template_diff
        if len(td) == 0:
            return
        sigma = float(np.std(td))
        if sigma <= 0:
            return
        self._pos_thresh = k * sigma
        self._neg_thresh = -k * sigma
        self._added_peaks.clear()
        self._removed_peaks.clear()
        self._update_thresh_labels()
        self._update_all()

    def _auto_threshold_geometry(self):
        """Pick the highest per-side thresholds such that the predicted
        sign-sequence is still a subsequence of the surviving peaks.

        Idea: detected peaks may include extras the geometry doesn't
        predict — that's fine. What matters is that the geometry's signed
        transitions appear, in order, somewhere in the detected stream.
        We raise t_pos / t_neg as high as possible without breaking that
        containment. Tries both polarity conventions and keeps whichever
        succeeds (preferring higher combined threshold).
        """
        expected = self._predict_expected_peaks()
        if not expected:
            return
        td = self._template_diff
        if len(td) == 0:
            return

        pos_pks, _ = find_peaks(td, height=0.0)
        neg_pks, _ = find_peaks(-td, height=0.0)
        cand = [(int(i), float(td[i]), +1) for i in pos_pks] + \
               [(int(i), float(td[i]), -1) for i in neg_pks]
        cand.sort(key=lambda c: c[0])
        if not cand:
            return

        def _signs_kept(t_pos, t_neg):
            return [c[2] for c in cand
                    if (c[2] > 0 and c[1] >= t_pos) or
                       (c[2] < 0 and -c[1] >= t_neg)]

        def _contains(seq, target):
            i = 0
            for s in seq:
                if i < len(target) and s == target[i]:
                    i += 1
            return i == len(target)

        def _solve(targets):
            if not _contains(_signs_kept(0.0, 0.0), targets):
                return None  # can't satisfy even with all candidates
            t_pos = 0.0
            for mag in sorted({c[1] for c in cand if c[2] > 0}):
                # Bump threshold *just past* this peak's magnitude.
                trial = mag * (1 + 1e-9) if mag > 0 else 1e-15
                if _contains(_signs_kept(trial, 0.0), targets):
                    t_pos = trial
                else:
                    break
            t_neg = 0.0
            for mag in sorted({-c[1] for c in cand if c[2] < 0}):
                trial = mag * (1 + 1e-9) if mag > 0 else 1e-15
                if _contains(_signs_kept(t_pos, trial), targets):
                    t_neg = trial
                else:
                    break
            return (t_pos, t_neg)

        target_signs = [s for s, _ in expected]
        a = _solve(target_signs)
        b = _solve([-s for s in target_signs])
        if a is None and b is None:
            from utils.app_logger import logger
            logger.info(
                "Auto-threshold (geometry): predicted sequence is not a "
                "subsequence of the detected peaks — thresholds unchanged.")
            return
        if a is None:
            t_pos, t_neg = b
        elif b is None:
            t_pos, t_neg = a
        else:
            t_pos, t_neg = a if (a[0] + a[1]) >= (b[0] + b[1]) else b

        self._pos_thresh = t_pos
        self._neg_thresh = -t_neg
        self._added_peaks.clear()
        self._removed_peaks.clear()
        self._update_thresh_labels()
        self._update_all()

    def _reset_thresholds(self):
        td = self._template_diff
        if len(td) > 0:
            max_val = np.max(td)
            min_val = np.min(td)
        else:
            max_val, min_val = 1.0, -1.0
        if max_val <= 0:
            max_val = np.max(np.abs(td)) if len(td) > 0 else 1.0
        if min_val >= 0:
            min_val = -np.max(np.abs(td)) if len(td) > 0 else -1.0
        self._pos_thresh = max_val / 3.0
        self._neg_thresh = min_val / 3.0
        self._update_thresh_labels()

    def _on_restart_step1(self):
        self.restart_requested = True
        self.reject()

    def _on_finish(self):
        self.confirmed = True
        self.accept()

    def _get_key_peak_seq_indices(self, peaks):
        """Map each key peak to its index in the peak_locations array."""
        peaks_list = list(peaks) if peaks is not None else []
        result = []
        for kp in sorted(self._key_peaks.keys()):
            best_idx = -1
            best_dist = float('inf')
            for pi, p in enumerate(peaks_list):
                d = abs(int(p) - int(kp))
                if d < best_dist:
                    best_dist = d
                    best_idx = pi
            result.append(best_idx)
        return result

    def _get_excl_zone_seq_bounds(self, peaks):
        """Map each exclusion zone boundary to a KEY PEAK sequence index.

        Returns list of (start_seq, end_seq) where:
        -1 = before first key peak (template start)
        len(key_peaks) = after last key peak (template end)

        Zones are defined between consecutive key peaks, so key peaks
        are used as the anchors for mapping.
        """
        kp_list = sorted(self._key_peaks.keys())
        result = []
        for zs, ze in self._excl_zones:
            zs, ze = int(zs), int(ze)
            if not kp_list:
                result.append((-1, 0))
                continue
            # Map start
            if zs < kp_list[0]:
                s_seq = -1
            else:
                dists = [abs(p - zs) for p in kp_list]
                s_seq = dists.index(min(dists))
            # Map end
            if ze > kp_list[-1]:
                e_seq = len(kp_list)
            else:
                dists = [abs(p - ze) for p in kp_list]
                e_seq = dists.index(min(dists))
            result.append((s_seq, e_seq))
        return result

    def _build_result(self, filter_config, padding_info):
        """Build the TPsettings result dict from current dialog state."""
        n = self._n
        peaks = self._peak_locations
        td = self._template_diff

        if len(peaks) > 0:
            first_prop = peaks[0] / n
            last_prop = (n - peaks[-1]) / n
        else:
            first_prop = 0.0
            last_prop = 0.0

        pks_pos_idx, pks_neg_idx = self._get_effective_peaks()
        detected_idx = set(pks_pos_idx + 1) | set(pks_neg_idx + 1)
        all_excluded = self._effective_excluded(detected_idx)
        pos_excl = sum(1 for p in all_excluded if p - 1 in set(pks_pos_idx))
        neg_excl = sum(1 for p in all_excluded if p - 1 in set(pks_neg_idx))

        # Build per-segment method list from segment_methods overrides
        all_idx = self._last_all_idx
        global_method = self._global_rect_method
        methods = []
        if len(all_idx) >= 2:
            for k in range(len(all_idx) - 1):
                s = int(all_idx[k])
                methods.append(self._segment_methods.get(s, global_method))
        else:
            methods = [global_method]

        return {
            'template_original': self._original_template,
            'pulse_template_rec': self._rect_template,
            'threshold': {
                'upper': self._pos_thresh,
                'lower': self._neg_thresh,
            },
            'edge_margin': {
                'first_peak_prop': first_prop,
                'last_peak_prop': last_prop,
            },
            'peak_counts': {
                'positive': max(0, len(pks_pos_idx) - pos_excl),
                'negative': max(0, len(pks_neg_idx) - neg_excl),
            },
            'peak_locations': peaks,
            'peak_locations_apex': np.asarray(self._apex_locations, dtype=int),
            'peak_sequence': self._peak_sequence,
            'key_peaks': np.array(sorted(self._key_peaks.keys()), dtype=int),
            'key_peak_properties': [self._key_peaks[k]
                                    for k in sorted(self._key_peaks.keys())],
            'key_peak_seq_indices': self._get_key_peak_seq_indices(
                self._raw_peak_locations()),
            'exclusion_zones': [(s, e) for s, e in self._excl_zones],
            'exclusion_zone_seq_bounds': self._get_excl_zone_seq_bounds(peaks),
            'end_zones': {
                'enabled': self._chk_end_zones.isChecked(),
                'percent': self._spin_end_zone.value(),
                'zones': self._get_end_zones(),
            },
            'methods': methods,
            'edge_method': self._edge_method,
            'filter_padding': padding_info,
            'filter_config': filter_config,
        }

    def get_result(self, filter_config, padding_info):
        if not self.confirmed:
            return _empty_result(self._filtered)
        return self._build_result(filter_config, padding_info)


def _wrap_zones(zone_results):
    """Assemble per-zone result dicts into the canonical TPsettings struct.

    Each entry of *zone_results* is a full per-zone dict from
    ``_build_result`` / ``_empty_result``. ``filter_config`` and
    ``filter_padding`` are identical across zones (one filter chain), so
    they are lifted to the top level and removed from each zone dict.
    Returns ``{num_zones, filter_config, filter_padding, zones: [...]}``.
    """
    zones = [dict(z) for z in zone_results]
    first = zones[0] if zones else {}
    shared = {
        'filter_config': first.get('filter_config', {}),
        'filter_padding': first.get('filter_padding', {}),
    }
    for z in zones:
        z.pop('filter_config', None)
        z.pop('filter_padding', None)
    return {'num_zones': len(zones), **shared, 'zones': zones}


def pulse_template_processing(template_data, sample_rate=None,
                              exclude_end_zones=False, end_zone_percent=5,
                              filter_config_in=None, loaded_settings=None,
                              geometry=None):
    """Interactive pulse template rectangularization.

    Parameters
    ----------
    template_data : np.ndarray
        1-D pulse template signal (or 2-D for multi-channel, processed sequentially).
    sample_rate : float, optional
        Sampling rate in Hz. Required.
    exclude_end_zones : bool
        Whether to exclude end zones by default (False).
    end_zone_percent : int
        Percentage of pulse width for each end zone (default 5).
    filter_config_in : list[dict], optional
        Pre-defined filter chain config (from FilterConfig/Filter UI block).
        When provided, filters are applied automatically; otherwise the
        template is processed unfiltered.

    Returns
    -------
    result : dict
        Canonical zone-grouped struct
        ``{num_zones, filter_config, filter_padding, zones: [...]}``.
        Each ``zones`` entry is a self-contained per-zone dict with keys:
        template_original, pulse_template_rec, threshold, edge_margin,
        peak_counts, peak_locations, peak_sequence, key_peaks,
        exclusion_zones, end_zones, methods. One zone per template column;
        a 1-D template yields a single zone. Use
        ``utils.tpsettings_io.get_zone`` to read an individual zone in the
        legacy flat shape.
    """
    if sample_rate is None:
        raise ValueError('sample_rate is required')

    if isinstance(template_data, dict):
        template_data = template_data.get('pulse_template_rec',
                                          template_data.get('template'))
        if template_data is None:
            raise ValueError('template struct has no template data')

    template_data = np.atleast_1d(template_data).copy()

    if template_data.ndim == 2 and template_data.shape[1] > 1:
        n_ch = template_data.shape[1]
        results = []
        # Restore EVERY zone: when loaded_settings is a zone-grouped struct,
        # hand each channel its own zone via get_zone(.., ch). A flat/None
        # value means no per-zone restore (each channel opens fresh).
        full_loaded = loaded_settings
        zone_grouped = (isinstance(full_loaded, dict)
                        and isinstance(full_loaded.get('zones'), list))
        for ch in range(n_ch):
            from utils.app_logger import logger
            logger.info(f'PulseTemplateProcessing channel {ch + 1} of {n_ch}...')
            ch_settings = (get_zone(full_loaded, ch) if zone_grouped
                           else (full_loaded if ch == 0 else None))
            r = pulse_template_processing(
                template_data[:, ch], sample_rate,
                exclude_end_zones=exclude_end_zones,
                end_zone_percent=end_zone_percent,
                filter_config_in=filter_config_in,
                loaded_settings=ch_settings,
                geometry=geometry)
            # Each recursive call returns a wrapped single-zone struct;
            # collect that one zone dict per channel.
            results.append(get_zone(r, 0))
        return _wrap_zones(results)

    template = template_data.ravel().astype(float)
    # A full zone-grouped struct reaching the single-template path (e.g. a
    # direct single-zone call) is flattened to zone 0 for the dialog.
    if isinstance(loaded_settings, dict) and isinstance(
            loaded_settings.get('zones'), list):
        loaded_settings = get_zone(loaded_settings, 0)
    n = len(template)
    pad_length = min(50, int(np.ceil(n * 0.1)))

    while True:
        if filter_config_in is not None and isinstance(filter_config_in, list) and len(filter_config_in) > 0:
            # Apply pre-defined filter chain non-interactively
            from .filter_ui import _apply_single_filter
            filtered_template = template.copy()
            for fi in filter_config_in:
                filtered_template = _apply_single_filter(
                    filtered_template, sample_rate,
                    fi.get('type', 'lowpass'),
                    fi.get('cutoff1', 0),
                    fi.get('cutoff2', 0),
                    fi.get('notch_mode', 'freqBW'),
                    fi.get('notch_bw', 1.0),
                )
            filter_info = filter_config_in
        else:
            # No filter config wired — use the template as-is. Filtering
            # lives in the Filter UI block (filterConfigIn port).
            filtered_template = template.copy()
            filter_info = []

        filter_config = {
            'num_filters': len(filter_info),
            'filter_types': [fi.get('type', 'unknown') for fi in filter_info],
            'cutoff_frequencies': [
                [fi.get('cutoff1', 0), fi.get('cutoff2', 0)]
                for fi in filter_info],
            'notch_modes': [fi.get('notch_mode', 'freqBW') for fi in filter_info],
            'notch_bws': [fi.get('notch_bw', 1.0) for fi in filter_info],
            'filter_info': filter_info,  # full per-filter params for batch reuse
            'filter_description': ' + '.join(
                fi.get('description', '') for fi in filter_info),
        }
        padding_info = {'pad_length': pad_length, 'method': 'replicate'}

        dlg = _ThresholdDialog(template, filtered_template, sample_rate,
                               exclude_end_zones=exclude_end_zones,
                               end_zone_percent=end_zone_percent,
                               filter_config=filter_config,
                               padding_info=padding_info,
                               loaded_settings=loaded_settings,
                               geometry=geometry)
        loaded_settings = None  # only apply on first iteration
        dlg.exec()

        if dlg.restart_requested:
            continue
        if not dlg.confirmed:
            raise ValueError("User closed Template Settings without confirming.")
        return _wrap_zones([dlg.get_result(filter_config, padding_info)])


def _empty_result(template):
    return {
        'template_original': template.copy(),
        'pulse_template_rec': template.copy(),
        'threshold': {'upper': 0, 'lower': 0},
        'edge_margin': {'first_peak_prop': 0, 'last_peak_prop': 0},
        'peak_counts': {'positive': 0, 'negative': 0},
        'peak_locations': np.array([], dtype=int),
        'peak_locations_apex': np.array([], dtype=int),
        'peak_sequence': np.array([], dtype=int),
        'key_peaks': np.array([], dtype=int),
        'key_peak_properties': [],
        'exclusion_zones': [],
        'end_zones': {'enabled': False, 'percent': 5, 'zones': []},
        'methods': ['mean'],
        'edge_method': 'peaks',
        'filter_padding': {'pad_length': 0, 'method': 'replicate'},
        'filter_config': {},
    }
