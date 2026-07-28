"""Interactive review and classification of detected pulse regions.

Paginated 3x3 grid display of pulse regions with per-pulse classification
(single, coincident, noise, uncertain), hover-based classification buttons,
pulse splitting tool, expand/trim tools, and LP-filtered overlay.
"""

import warnings
import numpy as np
from scipy import signal as sig

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QGroupBox, QWidget, QButtonGroup, QRadioButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure

from theme import theme


class _LineOverlay(QWidget):
    """Lightweight Qt overlay that draws a vertical dashed line."""

    def __init__(self, parent=None, color=QColor(220, 0, 0)):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._x = -1
        self._y0 = 0
        self._y1 = 0
        self._color = color
        self.hide()

    def set_line(self, x, y0, y1):
        self._x = x
        self._y0 = y0
        self._y1 = y1
        self.update()

    def paintEvent(self, event):
        if self._x < 0:
            return
        p = QPainter(self)
        pen = QPen(self._color, 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(self._x, self._y0, self._x, self._y1)
        p.end()


_CLASS_COLORS = {
    'single': '#009900',
    'coincident': '#dd7700',
    'noise': '#cc0000',
    # Distinct purple so uncertain events stand out in the grid (and read
    # differently from still-unclassified gray).
    'uncertain': '#8e44ad',
    'unclassified': '#888888',
}

_CLASS_OPTIONS = ['single', 'coincident', 'noise', 'uncertain']

_TOOL_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #444; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #aaa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #555; color: white; border-color: #555; }"
)

_EXPAND_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #0066aa; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #0066aa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #0066aa; color: white; }"
)

_TRIM_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #cc4400; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #cc4400; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #cc4400; color: white; }"
)

_CANCEL_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #888; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #aaa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #888; color: white; border-color: #888; }"
)


def _lp_filter(data, fs, cutoff=1000):
    """Apply a lowpass Butterworth filter (display only)."""
    nyq = fs / 2
    if cutoff >= nyq:
        return data.copy()
    wn = min(cutoff / nyq, 0.99)
    b, a = sig.butter(4, wn, btype='low')
    try:
        return sig.filtfilt(b, a, data)
    except Exception:
        return data.copy()


# Auto-classify spike-detection tunables. A peak counts as a spike when its
# prominence and its height above the local baseline both exceed _SPIKE_MAD_K
# robust-noise (MAD) units, measured on a baseline-subtracted residual so
# unequal-amplitude spikes and sloped/stepped baselines are handled. Peaks
# closer than _SPIKE_MIN_SEP_S seconds are treated as one spike (merges split
# shoulders). Validated against the Sorter_v5 mzNPS training data: K=6 + a 10 ms
# separation drops spurious ~5-MAD edge/shoulder peaks that otherwise flipped
# genuine coincident events to "uncertain".
_SPIKE_MAD_K = 6.0
_SPIKE_MIN_SEP_S = 0.01
# Auto Split won't separate two events unless the gap between the closing
# End-zone peak and the next opening Start-zone peak is at least this many peak
# widths (FWHM). Keeps pulses that are too close together merged.
_SPLIT_MIN_GAP_WIDTHS = 3.0


def _detect_spikes(seg, fs=None):
    """Detect spikes in a 1-D segment; return ``(indices, widths)``.

    The segment's slow baseline (drift/steps) is removed with a wide median
    filter, leaving narrow spikes in the residual, which is oriented so spikes
    read positive (zone/dataset sign conventions vary). Robust noise =
    1.4826*MAD of the residual; peaks whose prominence and height both clear
    ``_SPIKE_MAD_K`` noise units, and which are at least ``_SPIKE_MIN_SEP_S``
    apart, are returned. ``widths`` are the FWHM (samples) of each peak. Because
    the threshold is relative to noise (not to the largest spike), a small
    second spike beside a big one still registers — so coincident pairs of
    unequal amplitude yield 2 peaks.
    """
    seg = np.asarray(seg, dtype=float)
    n = seg.size
    if n < 5:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    from scipy.ndimage import median_filter
    # Baseline window: wide enough to ride over narrow spikes, narrow enough to
    # follow real drift/steps in the zone trace.
    k = max(11, n // 6)
    base = median_filter(seg, size=k, mode='nearest')
    resid = seg - base
    # Orient so spikes point up (some zones invert, e.g. negative resistance).
    if abs(float(resid.min())) > abs(float(resid.max())):
        resid = -resid
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    if mad <= 0:
        mad = float(np.std(resid)) or 1.0
    thr = _SPIKE_MAD_K * mad
    kw = dict(prominence=thr, height=thr)
    if fs and fs > 0:
        kw['distance'] = max(1, int(round(_SPIKE_MIN_SEP_S * fs)))
    peaks, _ = sig.find_peaks(resid, **kw)
    if len(peaks):
        widths = sig.peak_widths(resid, peaks, rel_height=0.5)[0]
    else:
        widths = np.empty(0, dtype=float)
    return peaks.astype(int), widths


def _count_spikes(seg, fs=None):
    """Count prominent spikes in a 1-D segment (see :func:`_detect_spikes`)."""
    return int(len(_detect_spikes(seg, fs)[0]))


class _PulseReviewDialog(QDialog):
    """Paginated pulse region review and classification dialog."""

    def __init__(self, data, pulse_regions, sample_rate=1,
                 init_labels=None, init_regions=None, init_cutoff=None,
                 zone_roles=False, init_zone_roles=None, init_manual=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Review")
        self.resize(1500, 900)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._fs = sample_rate

        # Per-zone signals (multi-zone mzNPS). 1-D / single-col → one zone, so
        # the single-zone path is identical to before. All zones share length N,
        # so merged-region indexing is unaffected.
        from utils.zones import zone_columns
        self._zones = zone_columns(data)
        self._n_zones = len(self._zones)
        self._active_zone = 0

        # Optional Start / Measurement / End zone-role selectors. Only
        # meaningful for multi-zone data; shown when *zone_roles* is set (the
        # mz Detection Review block). Defaults: Start=Zone 1, Measurement=Zone 2,
        # End=Zone 3, each clamped to the available zone count.
        self._show_zone_roles = bool(zone_roles) and self._n_zones > 1

        def _clamp_zone(z, fallback):
            return z if isinstance(z, int) and 0 <= z < self._n_zones else fallback

        _roles = init_zone_roles or {}
        self._role_start = _clamp_zone(_roles.get('start'), 0)
        self._role_measurement = _clamp_zone(
            _roles.get('measurement'), min(1, self._n_zones - 1))
        self._role_end = _clamp_zone(
            _roles.get('end'), min(2, self._n_zones - 1))
        self._overlay_mode = 'active'  # 'active' | 'all' (multi-zone defaults to 'all' in _build_ui)
        # Display-only LPF cutoff (Hz); user-adjustable, persisted in settings.
        self._lpf_cutoff = float(init_cutoff) if init_cutoff else 1000.0
        self._zones_lp = [_lp_filter(z, sample_rate, cutoff=self._lpf_cutoff)
                          for z in self._zones]

        # Use saved regions+labels if provided (preserves splits/trims/expands),
        # otherwise use the current detected pulse regions.
        if (init_labels and init_regions
                and len(init_labels) == len(init_regions)):
            self._pulse_regions = list(map(list, init_regions))
            self._labels = list(init_labels)
        elif init_labels and len(init_labels) == len(pulse_regions):
            self._pulse_regions = list(map(list, pulse_regions))
            self._labels = list(init_labels)
        else:
            self._pulse_regions = list(map(list, pulse_regions))
            self._labels = None

        self._orig_pulse_regions = [list(r) for r in pulse_regions]

        self._n_total = len(self._pulse_regions)
        self._page_size = 9  # 3x3 grid
        self._total_pages = max(1, int(np.ceil(self._n_total / self._page_size)))
        if self._labels is None:
            self._labels = ['unclassified'] * self._n_total
        # Per-region manual flag: True where the user set the label via the
        # in-plot floating buttons. Auto Classify leaves these untouched.
        if init_manual is not None and len(init_manual) == self._n_total:
            self._manual = [bool(v) for v in init_manual]
        else:
            self._manual = [False] * self._n_total
        self._page = 0
        self.confirmed = False

        # Tinted cell background for uncertain events — base_bg mixed toward the
        # uncertain color so those cells are easy to spot on either theme.
        from matplotlib.colors import to_rgb
        _base = np.array(to_rgb(theme.palette.base_bg))
        _unc = np.array(to_rgb(_CLASS_COLORS['uncertain']))
        self._uncertain_bg = tuple(0.85 * _base + 0.15 * _unc)

        # Active-zone signals drive all existing tools / indexing / rendering.
        self._data = self._zones[self._active_zone]
        self._data_lp = self._zones_lp[self._active_zone]

        # Hover / classification overlay state
        self._hover_idx = None  # grid index being hovered
        self._class_overlay = None  # QWidget overlay

        # Line overlays (created after canvas exists)
        self._split_line = None   # red dashed line for split
        self._trim_line = None    # orange dashed line for trim

        # Tool overlay state (hovering tool buttons on top of plots)
        self._tool_overlay = None
        self._tool_sub_overlay = None
        self._sub_overlay_grid_idx = None
        self._active_tool = None       # None, 'split', 'trim'
        self._active_tool_grid_idx = None  # grid idx where tool was activated
        self._trim_sample = None
        self._trim_line_qx = None      # pixel x of trim line for button placement

        # Undo / redo stacks — each entry is (regions_copy, labels_copy)
        self._undo_stack = []
        self._redo_stack = []

        # Mpl event IDs
        self._mpl_cids = []

        self._build_ui()
        self._split_line = _LineOverlay(self.canvas, QColor(220, 0, 0))
        self._split_line.resize(self.canvas.size())
        self._trim_line = _LineOverlay(self.canvas, QColor(204, 100, 0))
        self._trim_line.resize(self.canvas.size())
        self._render_page()

    def closeEvent(self, event):
        self._hide_class_overlay()
        self._hide_tool_overlay()
        self._hide_tool_sub_overlay()
        for overlay in (self._split_line, self._trim_line):
            if overlay is not None:
                overlay.hide()
        self._split_line = None
        self._trim_line = None
        for canvas, cid in self._mpl_cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids.clear()
        try:
            self.fig.clear()
        except Exception:
            pass
        super().closeEvent(event)

    # ============================= UI Setup =============================

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left: canvas for grid
        self._left_widget = QWidget()
        left_layout = QVBoxLayout(self._left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(14, 9), tight_layout=False)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(self._left_widget, stretch=4)

        self._mpl_cids.append(
            (self.canvas, self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)))
        self._mpl_cids.append(
            (self.canvas, self.canvas.mpl_connect('button_press_event', self._on_click)))
        self._mpl_cids.append(
            (self.canvas, self.canvas.mpl_connect('axes_leave_event', self._on_axes_leave)))

        # Right: side panel
        panel = QWidget()
        panel.setFixedWidth(220)
        self._panel_layout = QVBoxLayout(panel)
        self._panel_layout.setContentsMargins(5, 5, 5, 5)

        # Page info
        self._lbl_page = QLabel("")
        self._lbl_page.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._panel_layout.addWidget(self._lbl_page)

        # Classification counts
        self._lbl_counts = QLabel("")
        self._lbl_counts.setStyleSheet(
            f"font-size: 12px; color: {theme.palette.text};")
        self._panel_layout.addWidget(self._lbl_counts)

        self._panel_layout.addSpacing(5)

        # Batch controls. Regular block: one "Quick Classify" group with 2x2
        # per-page bulk buttons. mz block: a "Batch Processing" group (the
        # automated actions + their parameters) plus a separate "Go to next"
        # navigation group.
        from PySide6.QtWidgets import QSizePolicy
        if not self._show_zone_roles:
            bulk_group = QGroupBox("Quick Classify (this page)")
            bulk_layout = QGridLayout(bulk_group)
            bulk_layout.setSpacing(4)
            bulk_layout.setColumnStretch(0, 1)
            bulk_layout.setColumnStretch(1, 1)
            for pos, cls in enumerate(_CLASS_OPTIONS):
                btn = QPushButton(f"All {cls.capitalize()}")
                color = _CLASS_COLORS[cls]
                btn.setStyleSheet(
                    f"QPushButton {{ border: 2px solid {color}; color: {color}; "
                    f"background: transparent; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}"
                    f"QPushButton:hover {{ background: {color}; color: white; }}"
                )
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.clicked.connect(lambda checked, c=cls: self._bulk(c))
                bulk_layout.addWidget(btn, pos // 2, pos % 2)
            self._panel_layout.addWidget(bulk_group)
        else:
            from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox

            # ---- Batch Processing: Auto Process + the three step actions and
            # their parameters. ----
            proc_group = QGroupBox("Batch Processing")
            proc_layout = QVBoxLayout(proc_group)
            proc_layout.setSpacing(5)

            self._btn_auto_process = QPushButton("Auto Process")
            self._btn_auto_process.setToolTip(
                "Run Classify then Split repeatedly until nothing changes, then\n"
                "jump to the next page holding an uncertain event.")
            self._btn_auto_process.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._btn_auto_process.clicked.connect(self._auto_process)
            proc_layout.addWidget(self._btn_auto_process)

            # The three step actions, one row, all the same width. The default
            # 12px side padding makes "Classify" too wide to share evenly; a
            # compact padding lets equal stretch give all three the same size.
            _step_style = "QPushButton { padding-left: 4px; padding-right: 4px; }"
            steps_row = QHBoxLayout()
            steps_row.setSpacing(4)
            self._btn_auto = QPushButton("Classify")
            self._btn_auto.setToolTip(
                "Classify every pulse by spike count in the Start and End zones:\n"
                "matched count → single (1) or coincident (>1); no spikes → noise; "
                "mismatched → uncertain")
            self._btn_auto.clicked.connect(self._auto_classify)
            self._btn_auto_split = QPushButton("Split")
            self._btn_auto_split.setToolTip(
                "Split coincident events into separable sub-events by matching\n"
                "Start/End zone spikes in time order. A piece is cut wherever every\n"
                "particle that entered has exited; pieces are relabelled single or\n"
                "coincident. Overlapping events that never separate are left as-is.")
            self._btn_auto_split.setEnabled(False)
            self._btn_auto_split.clicked.connect(self._auto_split)
            self._btn_auto_trim = QPushButton("Trim")
            self._btn_auto_trim.setToolTip(
                "Tighten every event window to its signal: span from the first to\n"
                "the last spike across the Start/End zones, padded by the margin.")
            self._btn_auto_trim.clicked.connect(self._auto_trim)
            for b in (self._btn_auto, self._btn_auto_split, self._btn_auto_trim):
                b.setStyleSheet(_step_style)
                b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                steps_row.addWidget(b, 1)  # equal stretch → equal width
            proc_layout.addLayout(steps_row)

            # Parameters on their own rows, labels/fields aligned (QFormLayout).
            from PySide6.QtWidgets import QFormLayout
            param_form = QFormLayout()
            param_form.setContentsMargins(0, 0, 0, 0)
            param_form.setHorizontalSpacing(6)
            param_form.setVerticalSpacing(4)
            self._split_gap_spin = QDoubleSpinBox()
            self._split_gap_spin.setRange(0.0, 20.0)
            self._split_gap_spin.setSingleStep(0.5)
            self._split_gap_spin.setDecimals(1)
            self._split_gap_spin.setValue(_SPLIT_MIN_GAP_WIDTHS)
            self._split_gap_spin.setFixedWidth(60)
            self._split_gap_spin.setToolTip(
                "Minimum gap between two events for Split to separate them,\n"
                "in multiples of the peak width (FWHM). Larger = fewer splits.")
            self._trim_pad_spin = QSpinBox()
            self._trim_pad_spin.setRange(0, 200)
            self._trim_pad_spin.setValue(10)
            self._trim_pad_spin.setSuffix(" %")
            self._trim_pad_spin.setFixedWidth(60)
            self._trim_pad_spin.setToolTip("Margin added to each end by Trim,\n"
                                           "as a percentage of the event width")
            param_form.addRow("split gap", self._split_gap_spin)
            param_form.addRow("margin", self._trim_pad_spin)
            proc_layout.addLayout(param_form)
            self._panel_layout.addWidget(proc_group)

            # ---- Go to next: jump to the next unreviewed Noise / Uncertain. ----
            nav_group = QGroupBox("Go to next")
            nav_layout = QHBoxLayout(nav_group)
            nav_layout.setSpacing(4)
            self._btn_next_noise = QPushButton("Noise ▸")
            self._btn_next_noise.setToolTip(
                "Jump to the next page holding a noise event")
            self._btn_next_noise.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._btn_next_noise.setEnabled(False)
            self._btn_next_noise.clicked.connect(
                lambda: self._goto_next_with_label('noise'))
            nav_layout.addWidget(self._btn_next_noise)
            self._btn_next_uncertain = QPushButton("Uncertain ▸")
            self._btn_next_uncertain.setToolTip(
                "Jump to the next page holding an uncertain event")
            self._btn_next_uncertain.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._btn_next_uncertain.setEnabled(False)
            self._btn_next_uncertain.clicked.connect(
                lambda: self._goto_next_with_label('uncertain'))
            nav_layout.addWidget(self._btn_next_uncertain)
            self._panel_layout.addWidget(nav_group)

        self._panel_layout.addSpacing(5)

        # Display mode toggle
        display_group = QGroupBox("Display Data")
        display_layout = QVBoxLayout(display_group)
        self._display_btn_group = QButtonGroup(self)
        for i, (label, mode) in enumerate([
            ("Both (Raw + Filtered)", "both"),
            ("Raw Data Only", "raw"),
            ("Filtered Data Only", "filtered"),
        ]):
            rb = QRadioButton(label)
            if mode == "filtered":
                rb.setChecked(True)
            rb.mode = mode
            self._display_btn_group.addButton(rb, i)
            display_layout.addWidget(rb)
        self._display_mode = "filtered"
        self._display_btn_group.buttonClicked.connect(self._on_display_mode_changed)
        self._panel_layout.addWidget(display_group)

        self._panel_layout.addSpacing(5)

        # LPF cutoff — parameterizes the Filtered / Both overlay. Own group:
        # label on the first line, spinbox + Apply on the second. Recomputes
        # only when Apply is pressed (filtering all zones can be non-trivial).
        from PySide6.QtWidgets import QDoubleSpinBox
        lpf_group = QGroupBox("Display Filter")
        lpf_layout = QVBoxLayout(lpf_group)
        lpf_layout.addWidget(QLabel("LPF cutoff (Hz):"))
        lpf_row = QHBoxLayout()
        lpf_row.setSpacing(4)
        self._lpf_spin = QDoubleSpinBox()
        self._lpf_spin.setDecimals(0)
        nyq = self._fs / 2.0
        self._lpf_spin.setRange(1.0, nyq if nyq > 2 else 1e6)
        self._lpf_spin.setSingleStep(50.0)
        self._lpf_spin.setKeyboardTracking(False)
        self._lpf_spin.setValue(min(self._lpf_cutoff, self._lpf_spin.maximum()))
        lpf_row.addWidget(self._lpf_spin, 1)
        self._btn_lpf_apply = QPushButton("Apply")
        self._btn_lpf_apply.setFixedWidth(60)
        self._btn_lpf_apply.clicked.connect(self._on_apply_lpf)
        lpf_row.addWidget(self._btn_lpf_apply)
        lpf_layout.addLayout(lpf_row)
        self._panel_layout.addWidget(lpf_group)

        # Zone selector + overlay toggle (multi-zone mzNPS only)
        self._zone_btn_group = None
        if self._n_zones > 1:
            self._panel_layout.addSpacing(5)
            zone_group = QGroupBox("Zone")
            zone_layout = QVBoxLayout(zone_group)
            self._zone_btn_group = QButtonGroup(self)
            for zi in range(self._n_zones):
                rb = QRadioButton(f"Zone {zi + 1}")
                if zi == self._active_zone:
                    rb.setChecked(True)
                self._zone_btn_group.addButton(rb, zi)
                zone_layout.addWidget(rb)
            self._zone_btn_group.idClicked.connect(self._on_zone_selected)
            self._panel_layout.addWidget(zone_group)

            overlay_group = QGroupBox("Plot Zones")
            overlay_layout = QVBoxLayout(overlay_group)
            from PySide6.QtWidgets import QCheckBox
            self._overlay_chk = QCheckBox("All Zones")
            # Default selected (overlay all zones); set the mode + checkbox
            # before connecting so it doesn't fire a render before the dialog
            # finishes building.
            self._overlay_mode = 'all'
            self._overlay_chk.setChecked(True)
            self._overlay_chk.toggled.connect(self._on_overlay_toggled)
            overlay_layout.addWidget(self._overlay_chk)
            self._panel_layout.addWidget(overlay_group)

            # Zone-role selectors (Start / Measurement / End) — mz block only.
            if self._show_zone_roles:
                from PySide6.QtWidgets import QComboBox, QFormLayout
                self._panel_layout.addSpacing(5)
                roles_group = QGroupBox("Zone Roles")
                roles_layout = QFormLayout(roles_group)
                roles_layout.setContentsMargins(8, 6, 8, 6)
                roles_layout.setSpacing(4)
                self._role_combos = {}
                for key, label, cur in (
                        ('start', "Start", self._role_start),
                        ('measurement', "Measurement", self._role_measurement),
                        ('end', "End", self._role_end)):
                    combo = QComboBox()
                    for zi in range(self._n_zones):
                        combo.addItem(f"Zone {zi + 1}", zi)
                    combo.setCurrentIndex(cur)
                    combo.currentIndexChanged.connect(
                        lambda idx, k=key: self._on_zone_role_changed(k, idx))
                    roles_layout.addRow(label, combo)
                    self._role_combos[key] = combo
                self._panel_layout.addWidget(roles_group)

        self._panel_layout.addStretch()

        # Export figure + Restart (side-by-side, equal width)
        from utils.export_figure_ui import make_export_button
        from PySide6.QtWidgets import QSizePolicy
        action_row = QHBoxLayout()
        action_row.setSpacing(4)
        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        self._btn_export.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        action_row.addWidget(self._btn_export, 1)
        self._btn_restart = QPushButton("Restart")
        self._btn_restart.setToolTip(
            "Clear all classifications and edits; revert to original detected regions")
        self._btn_restart.setFixedHeight(self._btn_export.minimumHeight() or self._btn_restart.sizeHint().height())
        self._btn_restart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_restart.clicked.connect(self._on_restart)
        action_row.addWidget(self._btn_restart, 1)
        self._panel_layout.addLayout(action_row)

        # Save classification settings
        self._btn_save = QPushButton("Save Settings")
        self._btn_save.clicked.connect(self._on_save_settings)
        self._panel_layout.addWidget(self._btn_save)

        self._panel_layout.addSpacing(5)

        # Jump to the next page that still has an unclassified pulse.
        self._btn_next_unclassified = QPushButton("Next Unclassified ▸")
        self._btn_next_unclassified.setToolTip(
            "Go to the next page with an unclassified pulse")
        self._btn_next_unclassified.clicked.connect(self._on_next_unclassified)
        self._panel_layout.addWidget(self._btn_next_unclassified)

        self._panel_layout.addSpacing(5)

        # Navigation: << first | < prev | > next | >> last
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(4)

        self._btn_first = QPushButton("<<")
        self._btn_first.setToolTip("First page")
        self._btn_first.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_first.clicked.connect(self._on_first)
        nav_layout.addWidget(self._btn_first, 1)

        self._btn_prev = QPushButton("<")
        self._btn_prev.setToolTip("Previous page")
        self._btn_prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_prev.clicked.connect(self._on_prev)
        nav_layout.addWidget(self._btn_prev, 1)

        self._btn_next = QPushButton(">")
        self._btn_next.setToolTip("Next page")
        self._btn_next.setObjectName("confirmBtn")
        self._btn_next.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self._btn_next, 1)

        self._btn_last = QPushButton(">>")
        self._btn_last.setToolTip("Last page")
        self._btn_last.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_last.clicked.connect(self._on_last)
        nav_layout.addWidget(self._btn_last, 1)

        self._panel_layout.addLayout(nav_layout)

        main_layout.addWidget(panel)

    # ============================= Rendering =============================

    def _plot_zone_segment(self, ax_sub, zi, s_idx, e_idx, seg_t, *, active,
                           shift=0.0):
        """Plot zone *zi*'s raw/filtered segment into *ax_sub* per display mode.

        Active zone uses normal colors/linewidth at its real values. Non-active
        overlay zones are muted gray, thin, low alpha, drawn below; *shift* is
        added to their data so callers can detrend/re-center/stack them (the
        active zone is always plotted at real values with shift=0). Returns the
        actual plotted y-data so overlay y-limits span every drawn trace.
        """
        data = self._zones[zi]
        data_lp = self._zones_lp[zi]
        mode = self._display_mode
        if active:
            if mode in ('both', 'raw'):
                raw_color = '#aaaaaa' if mode == 'both' else '#1f77b4'
                raw_lw = 0.4 if mode == 'both' else 0.8
                ax_sub.plot(seg_t, data[s_idx:e_idx], color=raw_color,
                            linewidth=raw_lw, zorder=1)
            if mode in ('both', 'filtered'):
                ax_sub.plot(seg_t, data_lp[s_idx:e_idx], color='#1f77b4',
                            linewidth=0.8, zorder=2)
            if mode == 'filtered':
                return data_lp[s_idx:e_idx]
            return data[s_idx:e_idx]
        # Non-active overlay: muted gray, thin, low alpha, below the active zone.
        muted = '#999999'
        seg_raw = data[s_idx:e_idx] + shift
        seg_lp = data_lp[s_idx:e_idx] + shift
        if mode in ('both', 'raw'):
            ax_sub.plot(seg_t, seg_raw, color=muted,
                        linewidth=0.4, alpha=0.35, zorder=0)
        if mode in ('both', 'filtered'):
            ax_sub.plot(seg_t, seg_lp, color=muted,
                        linewidth=0.5, alpha=0.35, zorder=0)
        if mode == 'filtered':
            return seg_lp
        return seg_raw

    def _render_page(self):
        self.fig.clear()
        self._axes_grid = []
        self._hide_class_overlay()
        self._hide_tool_overlay()
        self._hide_tool_sub_overlay()
        self._active_tool = None
        self._active_tool_grid_idx = None
        self._hover_idx = None

        self._n_total = len(self._pulse_regions)
        self._total_pages = max(1, int(np.ceil(self._n_total / self._page_size)))
        if self._page >= self._total_pages:
            self._page = max(0, self._total_pages - 1)

        p = self._page
        start = p * self._page_size
        end = min(start + self._page_size, self._n_total)
        n_show = end - start

        self._lbl_page.setText(f"Page {p + 1} / {self._total_pages}")
        self._update_counts()

        is_first = (p == 0)
        is_last = (p == self._total_pages - 1)
        if self._total_pages == 1:
            self._btn_next.setText("✓")
        elif is_last:
            self._btn_next.setText("✓")
        else:
            self._btn_next.setText(">")

        # Disable Finish/Confirm if any pulses are still unclassified
        if is_last:
            has_unclassified = any(l == 'unclassified' for l in self._labels)
            self._btn_next.setEnabled(not has_unclassified)
            if has_unclassified:
                self._btn_next.setToolTip("Classify all pulses before finishing")
            else:
                self._btn_next.setToolTip("Finish review")
        else:
            self._btn_next.setEnabled(True)
            self._btn_next.setToolTip("Next page")

        self._btn_first.setEnabled(not is_first)
        self._btn_prev.setEnabled(not is_first)
        self._btn_last.setEnabled(not is_last)

        rows, cols = 3, 3
        plot_w = 0.28
        plot_h = 0.26
        x_gap = 0.04
        y_gap = 0.06
        x_offset = 0.06
        y_top = 0.70

        for i in range(n_show):
            pulse_idx = start + i
            r = i // cols
            c = i % cols
            ax_sub = self.fig.add_axes([
                x_offset + c * (plot_w + x_gap),
                y_top - r * (plot_h + y_gap),
                plot_w, plot_h
            ])
            self._axes_grid.append(ax_sub)

            region = self._pulse_regions[pulse_idx]
            s_idx = max(0, int(region[0]))
            e_idx = min(len(self._data), int(region[1]) + 1)
            if s_idx < e_idx and len(self._data) > 0:
                seg_t = np.arange(s_idx, e_idx) / self._fs
                if self._overlay_mode == 'all' and self._n_zones > 1:
                    # Baseline-aware stacked overlay. The active zone is the
                    # reference: plotted at real values, sets the y-scale. Other
                    # zones are detrended in-window (subtract own median), re-
                    # centered on the active zone's median, and stacked by a
                    # per-zone offset so they don't overlap. Real amplitudes are
                    # preserved across zones (no per-zone normalization).
                    a = self._active_zone
                    act_win = (self._zones_lp[a][s_idx:e_idx]
                               if self._display_mode == 'filtered'
                               else self._zones[a][s_idx:e_idx])
                    act_med = float(np.median(act_win))
                    arng = float(np.max(act_win) - np.min(act_win))
                    if arng <= 0:
                        arng = 1.0
                    step = arng * 1.5

                    y_all = []
                    for zi in range(self._n_zones):
                        if zi == a:
                            continue
                        own_win = (self._zones_lp[zi][s_idx:e_idx]
                                   if self._display_mode == 'filtered'
                                   else self._zones[zi][s_idx:e_idx])
                        own_med = float(np.median(own_win))
                        # Smaller index sits higher (offset positive).
                        offset = (a - zi) * step
                        shift = (act_med - own_med) + offset
                        yd = self._plot_zone_segment(ax_sub, zi, s_idx, e_idx,
                                                     seg_t, active=False,
                                                     shift=shift)
                        if yd is not None:
                            y_all.append(yd)
                    yd = self._plot_zone_segment(ax_sub, a, s_idx, e_idx,
                                                 seg_t, active=True)
                    if yd is not None:
                        y_all.append(yd)
                    ax_sub.set_xlim(seg_t[0], seg_t[-1])
                    if y_all:
                        y_cat = np.concatenate(y_all)
                        y_min, y_max = float(np.min(y_cat)), float(np.max(y_cat))
                        y_rng = y_max - y_min
                        if y_rng == 0:
                            y_rng = 1
                        ax_sub.set_ylim(y_min - 0.1 * y_rng, y_max + 0.1 * y_rng)
                else:
                    yd = self._plot_zone_segment(ax_sub, self._active_zone,
                                                 s_idx, e_idx, seg_t, active=True)
                    ax_sub.set_xlim(seg_t[0], seg_t[-1])
                    if yd is not None:
                        y_min, y_max = float(np.min(yd)), float(np.max(yd))
                        y_rng = y_max - y_min
                        if y_rng == 0:
                            y_rng = 1
                        ax_sub.set_ylim(y_min - 0.1 * y_rng, y_max + 0.1 * y_rng)

            lbl = self._labels[pulse_idx]
            color = _CLASS_COLORS.get(lbl, 'black')
            ax_sub.set_title(f'#{pulse_idx + 1}: {lbl}', fontsize=8,
                             color=color, fontweight='bold')
            ax_sub.tick_params(labelsize=6)
            # Pin the axis offset/scale text ("1e6", "+2.4e6") to the tick size
            # so it doesn't render larger — and look different between the Raw
            # and Filtered modes, which produce different y-ranges/offsets.
            ax_sub.yaxis.get_offset_text().set_fontsize(6)
            ax_sub.xaxis.get_offset_text().set_fontsize(6)
            ax_sub.grid(True, alpha=0.2)
            # Tinted background makes uncertain cells easy to find in the grid.
            if lbl == 'uncertain':
                ax_sub.set_facecolor(self._uncertain_bg)
            # Colored border to make classification state obvious
            border_lw = 2.5 if lbl != 'unclassified' else 1.0
            border_alpha = 1.0 if lbl != 'unclassified' else 0.3
            for spine in ax_sub.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(border_lw)
                spine.set_alpha(border_alpha)

        for i in range(n_show, self._page_size):
            r = i // cols
            c = i % cols
            ax_empty = self.fig.add_axes([
                x_offset + c * (plot_w + x_gap),
                y_top - r * (plot_h + y_gap),
                plot_w, plot_h
            ])
            ax_empty.set_axis_off()
            self._axes_grid.append(ax_empty)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*tight_layout.*")
            self.canvas.draw()
        self._bg_cache = self.canvas.copy_from_bbox(self.fig.bbox)

    # ============================= Helpers ===============================

    def _get_grid_index(self, event):
        """Return grid index (0..page_size-1) for the axes under event."""
        if event.inaxes is None:
            return None
        for i, ax in enumerate(self._axes_grid):
            if event.inaxes == ax:
                p = self._page
                start = p * self._page_size
                end = min(start + self._page_size, self._n_total)
                if i < (end - start):
                    return i
                return None
        return None

    def _axes_widget_rect(self, grid_idx):
        """Return (ax_x, ax_y_top, ax_w, ax_h) in canvas widget coords."""
        ax = self._axes_grid[grid_idx]
        bbox = ax.get_position()
        cw = self.canvas.width()
        ch = self.canvas.height()
        ax_x = int(bbox.x0 * cw)
        ax_y_top = int((1 - bbox.y1) * ch)
        ax_w = int(bbox.width * cw)
        ax_h = int(bbox.height * ch)
        return ax_x, ax_y_top, ax_w, ax_h

    def _data_x_to_widget_x(self, grid_idx, x_data):
        """Convert a data x coordinate to widget pixel x."""
        ax = self._axes_grid[grid_idx]
        pixel = ax.transData.transform((x_data, 0))
        dpr = self.canvas.devicePixelRatioF()
        return int(pixel[0] / dpr)

    def _show_line_on_axes(self, overlay, grid_idx, event):
        """Position and show a _LineOverlay at event.xdata on the given axes."""
        ax = self._axes_grid[grid_idx]
        pixel = ax.transData.transform((event.xdata, 0))
        bbox = ax.get_window_extent()
        px = int(pixel[0])
        canvas_h = self.canvas.height()
        dpr = self.canvas.devicePixelRatioF()
        qx = int(px / dpr)
        qy0 = int((canvas_h * dpr - bbox.y1) / dpr)
        qy1 = int((canvas_h * dpr - bbox.y0) / dpr)
        overlay.resize(self.canvas.size())
        overlay.set_line(qx, qy0, qy1)
        overlay.show()
        return qx

    # ============================= Mouse Events ==========================

    def _on_mouse_move(self, event):
        grid_idx = self._get_grid_index(event)
        line_mode = self._active_tool in ('split', 'trim')

        # Handle hover change for overlays
        if grid_idx != self._hover_idx:
            self._hover_idx = grid_idx

            if grid_idx is not None and not line_mode and self._tool_sub_overlay is None:
                self._show_class_overlay(grid_idx)
                self._show_tool_overlay(grid_idx)
            elif self._tool_sub_overlay is None and not line_mode:
                self._hide_class_overlay()
                self._hide_tool_overlay()

            # If moved to different plot while sub-overlay is showing, cancel
            if (self._tool_sub_overlay is not None
                    and grid_idx is not None
                    and grid_idx != self._sub_overlay_grid_idx):
                self._cancel_active_tool()
                self._show_class_overlay(grid_idx)
                self._show_tool_overlay(grid_idx)

        # Show vertical line only on the target plot
        if line_mode:
            on_target = (grid_idx is not None
                         and grid_idx == self._active_tool_grid_idx
                         and event.xdata is not None)
            if self._active_tool == 'split':
                if on_target and self._split_line is not None:
                    self._show_line_on_axes(self._split_line, grid_idx, event)
                elif self._split_line is not None:
                    self._split_line.hide()
            elif self._active_tool == 'trim':
                if on_target and self._trim_line is not None:
                    self._show_line_on_axes(self._trim_line, grid_idx, event)
                elif self._trim_line is not None:
                    self._trim_line.hide()

    def _on_axes_leave(self, event):
        self._hover_idx = None
        if self._tool_sub_overlay is None and self._active_tool is None:
            self._hide_class_overlay()
            self._hide_tool_overlay()

    def _on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return

        grid_idx = self._get_grid_index(event)
        if grid_idx is None:
            return

        # Split mode: only on the target plot
        if (self._active_tool == 'split'
                and grid_idx == self._active_tool_grid_idx
                and event.xdata is not None):
            pulse_idx = self._page * self._page_size + grid_idx
            self._do_split(pulse_idx, event.xdata)
            return

        # Trim mode: select location, only on the target plot
        if (self._active_tool == 'trim'
                and grid_idx == self._active_tool_grid_idx
                and event.xdata is not None):
            pulse_idx = self._page * self._page_size + grid_idx
            trim_sample = int(round(event.xdata * self._fs))
            region = self._pulse_regions[pulse_idx]
            s_idx = int(region[0])
            e_idx = int(region[1])
            if trim_sample <= s_idx or trim_sample >= e_idx:
                return  # Invalid location
            self._trim_sample = trim_sample
            # Keep the trim line visible at clicked position
            self._trim_line_qx = self._data_x_to_widget_x(grid_idx, event.xdata)
            self._active_tool = None  # Exit line-follow mode
            self._show_trim_options(grid_idx, pulse_idx)
            return

    # ============================= Hover Classification ==================

    def _show_class_overlay(self, grid_idx):
        """Show classification buttons overlaid on the hovered plot."""
        self._hide_class_overlay()

        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(grid_idx)
        ax_y_bottom = ax_y_top + ax_h

        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(2)

        pulse_idx = self._page * self._page_size + grid_idx

        for cls in _CLASS_OPTIONS:
            btn = QPushButton(cls.capitalize())
            btn.setFixedHeight(32)
            color = _CLASS_COLORS[cls]
            btn.setStyleSheet(
                f"QPushButton {{ background: white; color: {color}; "
                f"font-weight: bold; font-size: 12px; border-radius: 4px; border: 2px solid {color}; "
                f"padding: 4px 10px; }}"
                f"QPushButton:hover {{ background: {color}; color: white; }}"
            )
            btn.clicked.connect(lambda checked, c=cls, pi=pulse_idx, gi=grid_idx: self._classify(pi, c, gi))
            overlay_layout.addWidget(btn)

        overlay.adjustSize()
        ox = ax_x + (ax_w - overlay.width()) // 2
        oy = ax_y_bottom - overlay.height() - 2
        overlay.move(ox, oy)
        overlay.show()
        self._class_overlay = overlay

    def _hide_class_overlay(self):
        if self._class_overlay is not None:
            self._class_overlay.hide()
            self._class_overlay.deleteLater()
            self._class_overlay = None

    def _on_display_mode_changed(self, btn):
        self._display_mode = btn.mode
        self._render_page()

    def _on_apply_lpf(self):
        """Re-filter all zones at the new cutoff and redraw."""
        new_fc = float(self._lpf_spin.value())
        if new_fc == self._lpf_cutoff:
            return
        self._lpf_cutoff = new_fc
        self._zones_lp = [_lp_filter(z, self._fs, cutoff=self._lpf_cutoff)
                          for z in self._zones]
        self._data_lp = self._zones_lp[self._active_zone]
        self._render_page()
        self._autosave()

    def _on_zone_selected(self, idx):
        if not (0 <= idx < self._n_zones):
            return
        self._active_zone = idx
        self._data = self._zones[idx]
        self._data_lp = self._zones_lp[idx]
        self._render_page()

    def _on_overlay_toggled(self, checked):
        self._overlay_mode = 'all' if checked else 'active'
        self._render_page()

    def _on_zone_role_changed(self, key, idx):
        if key == 'start':
            self._role_start = idx
        elif key == 'measurement':
            self._role_measurement = idx
        elif key == 'end':
            self._role_end = idx
        self._autosave()

    def _zone_roles(self):
        """Current Start/Measurement/End zone assignments (0-based indices)."""
        return {
            'start': int(self._role_start),
            'measurement': int(self._role_measurement),
            'end': int(self._role_end),
        }

    def _classify(self, pulse_idx, cls, grid_idx=None):
        if 0 <= pulse_idx < self._n_total:
            self._save_state()
            self._labels[pulse_idx] = cls
            # In-plot floating-button labels are manual — protect from Auto Classify.
            if pulse_idx < len(self._manual):
                self._manual[pulse_idx] = True
            if grid_idx is not None and grid_idx < len(self._axes_grid):
                ax = self._axes_grid[grid_idx]
                color = _CLASS_COLORS.get(cls, 'black')
                ax.set_title(f'#{pulse_idx + 1}: {cls}', fontsize=8,
                             color=color, fontweight='bold')
                ax.set_facecolor(self._uncertain_bg if cls == 'uncertain'
                                 else theme.palette.base_bg)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                    spine.set_alpha(1.0)
                self.canvas.draw_idle()
                self._update_counts()
                # Update Finish/Confirm button state on last page
                is_last = (self._page == self._total_pages - 1)
                if is_last:
                    has_unclassified = any(
                        l == 'unclassified' for l in self._labels)
                    self._btn_next.setEnabled(not has_unclassified)
                    self._btn_next.setToolTip(
                        "Classify all pulses before finishing"
                        if has_unclassified else "")
            else:
                self._render_page()

    # ============================= Tool Overlay ==========================

    def _show_tool_overlay(self, grid_idx):
        """Show tool buttons (Undo, Split, Expand, Trim, Redo) at the top of the hovered plot."""
        self._hide_tool_overlay()

        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(grid_idx)

        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(3)

        btn_undo = QPushButton("\u21A9")
        btn_undo.setFixedHeight(20)
        btn_undo.setStyleSheet(_TOOL_BTN_STYLE)
        btn_undo.setToolTip("Undo")
        btn_undo.setEnabled(len(self._undo_stack) > 0)
        btn_undo.clicked.connect(self._undo)
        overlay_layout.addWidget(btn_undo)

        btn_split = QPushButton("\u2702 Split")
        btn_split.setFixedHeight(20)
        btn_split.setStyleSheet(_TOOL_BTN_STYLE)
        btn_split.setToolTip("Split this pulse at a clicked point")
        btn_split.clicked.connect(lambda: self._activate_tool_mode('split', grid_idx))
        overlay_layout.addWidget(btn_split)

        btn_expand = QPushButton("\u2194 Expand")
        btn_expand.setFixedHeight(20)
        btn_expand.setStyleSheet(_TOOL_BTN_STYLE)
        btn_expand.setToolTip("Expand pulse range by 25% before or after")
        btn_expand.clicked.connect(lambda: self._show_expand_options(grid_idx))
        overlay_layout.addWidget(btn_expand)

        btn_trim = QPushButton("\u2704 Trim")
        btn_trim.setFixedHeight(20)
        btn_trim.setStyleSheet(_TOOL_BTN_STYLE)
        btn_trim.setToolTip("Trim pulse before or after a clicked point")
        btn_trim.clicked.connect(lambda: self._activate_tool_mode('trim', grid_idx))
        overlay_layout.addWidget(btn_trim)

        btn_redo = QPushButton("\u21AA")
        btn_redo.setFixedHeight(20)
        btn_redo.setStyleSheet(_TOOL_BTN_STYLE)
        btn_redo.setToolTip("Redo")
        btn_redo.setEnabled(len(self._redo_stack) > 0)
        btn_redo.clicked.connect(self._redo)
        overlay_layout.addWidget(btn_redo)

        overlay.adjustSize()
        ox = ax_x + (ax_w - overlay.width()) // 2
        oy = ax_y_top + 4
        overlay.move(ox, oy)
        overlay.show()
        self._tool_overlay = overlay

        # Delete: icon-only button pinned to the cell's top-right corner.
        del_btn = QPushButton("\U0001F5D1", self.canvas)
        del_btn.setFixedHeight(20)
        del_btn.setStyleSheet(_TOOL_BTN_STYLE)
        del_btn.setToolTip("Delete this pulse region")
        del_btn.clicked.connect(lambda: self._delete_pulse(grid_idx))
        del_btn.adjustSize()
        del_btn.move(ax_x + ax_w - del_btn.width() - 4, ax_y_top + 4)
        del_btn.show()
        self._delete_overlay = del_btn

    def _hide_tool_overlay(self):
        if self._tool_overlay is not None:
            self._tool_overlay.hide()
            self._tool_overlay.deleteLater()
            self._tool_overlay = None
        if getattr(self, '_delete_overlay', None) is not None:
            self._delete_overlay.hide()
            self._delete_overlay.deleteLater()
            self._delete_overlay = None

    def _hide_tool_sub_overlay(self):
        if self._tool_sub_overlay is not None:
            self._tool_sub_overlay.hide()
            self._tool_sub_overlay.deleteLater()
            self._tool_sub_overlay = None
            self._sub_overlay_grid_idx = None

    def _cancel_active_tool(self):
        """Cancel any active tool mode and hide overlays."""
        self._active_tool = None
        self._active_tool_grid_idx = None
        self._trim_sample = None
        self._trim_line_qx = None
        self._hide_tool_overlay()
        self._hide_tool_sub_overlay()
        if self._split_line is not None:
            self._split_line.hide()
        if self._trim_line is not None:
            self._trim_line.hide()

    def _delete_pulse(self, grid_idx):
        """Remove the pulse region shown in *grid_idx* on the current page."""
        pulse_idx = self._page * self._page_size + grid_idx
        if pulse_idx < 0 or pulse_idx >= len(self._pulse_regions):
            return
        self._save_state()  # undoable
        del self._pulse_regions[pulse_idx]
        del self._labels[pulse_idx]
        if pulse_idx < len(self._manual):
            del self._manual[pulse_idx]
        self._n_total = len(self._pulse_regions)
        self._cancel_active_tool()
        self._render_page()

    def _save_state(self):
        """Push current regions/labels/manual flags onto the undo stack."""
        import copy
        self._undo_stack.append(
            (copy.deepcopy(self._pulse_regions), list(self._labels),
             list(self._manual)))
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        import copy
        self._redo_stack.append(
            (copy.deepcopy(self._pulse_regions), list(self._labels),
             list(self._manual)))
        regions, labels, manual = self._undo_stack.pop()
        self._pulse_regions = regions
        self._labels = labels
        self._manual = manual
        self._n_total = len(self._pulse_regions)
        self._cancel_active_tool()
        self._render_page()

    def _redo(self):
        if not self._redo_stack:
            return
        import copy
        self._undo_stack.append(
            (copy.deepcopy(self._pulse_regions), list(self._labels),
             list(self._manual)))
        regions, labels, manual = self._redo_stack.pop()
        self._pulse_regions = regions
        self._labels = labels
        self._manual = manual
        self._n_total = len(self._pulse_regions)
        self._cancel_active_tool()
        self._render_page()

    def _activate_tool_mode(self, mode, grid_idx):
        """Enter split or trim mode, locked to the given plot."""
        self._active_tool = mode
        self._active_tool_grid_idx = grid_idx
        self._hide_tool_overlay()
        self._hide_tool_sub_overlay()
        self._hide_class_overlay()
        # Show a cancel button at the top of the target plot
        self._show_cancel_overlay(grid_idx)

    def _show_cancel_overlay(self, grid_idx):
        """Show a cancel button at the top-center of the target plot."""
        self._hide_tool_sub_overlay()

        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(grid_idx)

        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(0)

        btn_cancel = QPushButton("\u2715 Cancel")
        btn_cancel.setFixedHeight(20)
        btn_cancel.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_cancel.clicked.connect(self._cancel_active_tool)
        overlay_layout.addWidget(btn_cancel)

        overlay.adjustSize()
        ox = ax_x + (ax_w - overlay.width()) // 2
        overlay.move(ox, ax_y_top + 4)
        overlay.show()
        self._tool_sub_overlay = overlay
        self._sub_overlay_grid_idx = grid_idx

    # ============================= Expand ================================

    def _show_expand_options(self, grid_idx):
        """Show Before/Both/After expand buttons with cancel below."""
        self._hide_tool_overlay()
        self._hide_tool_sub_overlay()
        self._hide_class_overlay()

        pulse_idx = self._page * self._page_size + grid_idx
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(grid_idx)

        # Container spanning the full axes width
        overlay = QWidget(self.canvas)
        overlay.setFixedWidth(ax_w)
        outer_layout = QVBoxLayout(overlay)
        outer_layout.setContentsMargins(4, 2, 4, 2)
        outer_layout.setSpacing(2)

        # Top row: Before | Both | After
        row = QHBoxLayout()
        row.setSpacing(0)

        btn_before = QPushButton("\u2190 Before")
        btn_before.setFixedHeight(20)
        btn_before.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_before.setToolTip("Expand 25% before the pulse start")
        btn_before.clicked.connect(lambda: self._do_expand(pulse_idx, 'before'))
        row.addWidget(btn_before, 0, Qt.AlignmentFlag.AlignLeft)

        row.addStretch()

        btn_both = QPushButton("\u2194 Both")
        btn_both.setFixedHeight(20)
        btn_both.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_both.setToolTip("Expand 25% on both ends")
        btn_both.clicked.connect(lambda: self._do_expand(pulse_idx, 'both'))
        row.addWidget(btn_both)

        row.addStretch()

        btn_after = QPushButton("After \u2192")
        btn_after.setFixedHeight(20)
        btn_after.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_after.setToolTip("Expand 25% after the pulse end")
        btn_after.clicked.connect(lambda: self._do_expand(pulse_idx, 'after'))
        row.addWidget(btn_after, 0, Qt.AlignmentFlag.AlignRight)

        outer_layout.addLayout(row)

        # Bottom row: Cancel centered
        btn_cancel = QPushButton("\u2715 Cancel")
        btn_cancel.setFixedHeight(20)
        btn_cancel.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_cancel.clicked.connect(self._cancel_active_tool)
        outer_layout.addWidget(btn_cancel, 0, Qt.AlignmentFlag.AlignCenter)

        overlay.adjustSize()
        overlay.move(ax_x, ax_y_top + 4)
        overlay.show()
        self._tool_sub_overlay = overlay
        self._sub_overlay_grid_idx = grid_idx

    def _do_expand(self, pulse_idx, direction):
        """Expand pulse range by 25% of current width in the given direction."""
        if pulse_idx >= len(self._pulse_regions):
            return

        self._save_state()
        region = self._pulse_regions[pulse_idx]
        s_idx = int(region[0])
        e_idx = int(region[1])
        width = e_idx - s_idx
        expand_amount = max(1, int(width * 0.25))

        if direction == 'before':
            new_start = max(0, s_idx - expand_amount)
            self._pulse_regions[pulse_idx] = [new_start, e_idx]
        elif direction == 'after':
            new_end = min(len(self._data) - 1, e_idx + expand_amount)
            self._pulse_regions[pulse_idx] = [s_idx, new_end]
        else:  # both
            new_start = max(0, s_idx - expand_amount)
            new_end = min(len(self._data) - 1, e_idx + expand_amount)
            self._pulse_regions[pulse_idx] = [new_start, new_end]

        self._hide_tool_sub_overlay()
        self._render_page()

    # ============================= Trim ==================================

    def _show_trim_options(self, grid_idx, pulse_idx):
        """Show Remove Before / Remove After buttons on left/right of the trim line."""
        self._hide_tool_sub_overlay()
        # Trim line stays visible at clicked position (already set)

        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(grid_idx)
        line_qx = self._trim_line_qx if self._trim_line_qx is not None else ax_x + ax_w // 2

        # Container spanning the full axes width
        overlay = QWidget(self.canvas)
        overlay.setFixedWidth(ax_w)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(4, 2, 4, 2)
        overlay_layout.setSpacing(0)

        btn_before = QPushButton("Remove \u2190")
        btn_before.setFixedHeight(20)
        btn_before.setStyleSheet(_TRIM_BTN_STYLE)
        btn_before.setToolTip("Remove data before the selected point")
        btn_before.clicked.connect(lambda: self._do_trim(pulse_idx, 'before'))

        btn_after = QPushButton("\u2192 Remove")
        btn_after.setFixedHeight(20)
        btn_after.setStyleSheet(_TRIM_BTN_STYLE)
        btn_after.setToolTip("Remove data after the selected point")
        btn_after.clicked.connect(lambda: self._do_trim(pulse_idx, 'after'))

        btn_cancel = QPushButton("\u2715")
        btn_cancel.setFixedHeight(20)
        btn_cancel.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_cancel.setToolTip("Cancel")
        btn_cancel.clicked.connect(self._cancel_active_tool)

        # Position buttons relative to the trim line within the overlay
        # "Remove ←" on the left side of the line, ✕ at center, "→ Remove" on the right
        overlay_layout.addWidget(btn_before, 0, Qt.AlignmentFlag.AlignLeft)
        overlay_layout.addStretch()
        overlay_layout.addWidget(btn_cancel)
        overlay_layout.addStretch()
        overlay_layout.addWidget(btn_after, 0, Qt.AlignmentFlag.AlignRight)

        overlay.adjustSize()

        # Shift the overlay so buttons sit on either side of the line
        # The line_qx is in canvas coords; overlay starts at ax_x
        btn_before_w = btn_before.sizeHint().width() + 8
        btn_after_w = btn_after.sizeHint().width() + 8

        # Compute overlay region: left button right-edge at line, right button left-edge at line
        overlay_left = line_qx - btn_before_w - 2
        overlay_right = line_qx + btn_after_w + 2
        overlay_width = overlay_right - overlay_left

        # Clamp to axes bounds
        if overlay_left < ax_x:
            overlay_left = ax_x
        if overlay_left + overlay_width > ax_x + ax_w:
            overlay_width = ax_x + ax_w - overlay_left

        overlay.setFixedWidth(max(overlay_width, btn_before_w + btn_after_w + 8))
        overlay.adjustSize()
        overlay.move(overlay_left, ax_y_top + 4)
        overlay.show()
        self._tool_sub_overlay = overlay
        self._sub_overlay_grid_idx = grid_idx

    def _do_trim(self, pulse_idx, direction):
        """Trim pulse at the previously selected sample location."""
        if pulse_idx >= len(self._pulse_regions) or self._trim_sample is None:
            self._cancel_active_tool()
            return

        self._save_state()
        region = self._pulse_regions[pulse_idx]
        s_idx = int(region[0])
        e_idx = int(region[1])

        if direction == 'before':
            # Remove before: keep from trim point onward
            self._pulse_regions[pulse_idx] = [self._trim_sample, e_idx]
        else:
            # Remove after: keep up to trim point
            self._pulse_regions[pulse_idx] = [s_idx, self._trim_sample]

        self._active_tool = None
        self._active_tool_grid_idx = None
        self._trim_sample = None
        self._trim_line_qx = None
        self._hide_tool_sub_overlay()
        if self._trim_line is not None:
            self._trim_line.hide()
        self._render_page()

    # ============================= Split =================================

    def _do_split(self, pulse_idx, x_time):
        """Split the pulse at the clicked time point."""
        if pulse_idx >= self._n_total:
            return

        self._save_state()
        split_sample = int(round(x_time * self._fs))
        region = self._pulse_regions[pulse_idx]
        s_idx = int(region[0])
        e_idx = int(region[1])

        if split_sample <= s_idx or split_sample >= e_idx:
            return

        region1 = [s_idx, split_sample]
        region2 = [split_sample + 1, e_idx]

        self._pulse_regions[pulse_idx] = region1
        self._pulse_regions.insert(pulse_idx + 1, region2)

        self._labels.insert(pulse_idx + 1, 'unclassified')
        # New piece is unclassified, so leave it auto-classifiable.
        self._manual.insert(pulse_idx + 1, False)
        self._n_total = len(self._pulse_regions)

        self._active_tool = None
        self._active_tool_grid_idx = None
        if self._split_line is not None:
            self._split_line.hide()

        self._render_page()

    # ============================= Bulk / Nav ============================

    def _bulk(self, cls):
        self._save_state()
        p = self._page
        start = p * self._page_size
        end = min(start + self._page_size, self._n_total)
        for i in range(start, end):
            self._labels[i] = cls
        self._render_page()

    def _classify_all(self):
        """Classify every non-manual pulse by comparing spike counts in the
        Start and End zones. Matched counts → single (N==1) or coincident (N>1);
        no spikes in either → noise; mismatched counts → uncertain. Uses the
        LP-filtered zone traces. Mutates ``self._labels`` in place; returns True
        if any label changed. Events labelled manually via the in-plot floating
        buttons are left untouched."""
        changed = False
        z_start = self._zones_lp[self._role_start]
        z_end = self._zones_lp[self._role_end]
        n = len(z_start)
        for i, region in enumerate(self._pulse_regions):
            if i < len(self._manual) and self._manual[i]:
                continue  # preserve manual classification
            s_idx = max(0, int(region[0]))
            e_idx = min(n, int(region[1]) + 1)
            if s_idx >= e_idx:
                new = 'uncertain'
            else:
                n_s = _count_spikes(z_start[s_idx:e_idx], self._fs)
                n_e = _count_spikes(z_end[s_idx:e_idx], self._fs)
                if n_s == n_e:
                    new = ('noise' if n_s == 0
                           else 'single' if n_s == 1 else 'coincident')
                else:
                    new = 'uncertain'
            if self._labels[i] != new:
                self._labels[i] = new
                changed = True
        return changed

    def _compute_split(self):
        """Compute the coincident-split result WITHOUT mutating state.

        Each Start-zone spike opens a particle (+1), each End-zone spike closes
        one (-1). Walking the spikes in time order, wherever the running depth
        returns to 0 (every particle that entered has exited) with events still
        to come, the region is cut at the midpoint between that End spike and
        the next Start spike. Each resulting piece is relabelled single (1
        particle) or coincident (>1). Overlapping events whose depth never
        returns to 0 mid-way are left unsplit. Returns
        ``(changed, regions, labels, manual)``."""
        z_start = self._zones_lp[self._role_start]
        z_end = self._zones_lp[self._role_end]
        n = len(z_start)
        try:
            gap_widths = float(self._split_gap_spin.value())
        except (AttributeError, RuntimeError):
            gap_widths = _SPLIT_MIN_GAP_WIDTHS
        new_regions, new_labels, new_manual = [], [], []
        changed = False
        for region, label, man in zip(self._pulse_regions, self._labels,
                                      self._manual):
            s_idx = max(0, int(region[0]))
            e_idx = min(n, int(region[1]) + 1)
            if label != 'coincident' or s_idx >= e_idx:
                new_regions.append(list(region))
                new_labels.append(label)
                new_manual.append(man)
                continue
            sp, sp_w = _detect_spikes(z_start[s_idx:e_idx], self._fs)
            ep, ep_w = _detect_spikes(z_end[s_idx:e_idx], self._fs)
            sp = s_idx + sp
            ep = s_idx + ep
            if len(sp) != len(ep) or len(sp) < 2:
                new_regions.append(list(region))
                new_labels.append(label)
                new_manual.append(man)
                continue
            # Time-ordered events: (sample, +1 Start / -1 End, peak width).
            events = sorted(
                [(int(sp[j]), 1, float(sp_w[j])) for j in range(len(sp))] +
                [(int(ep[j]), -1, float(ep_w[j])) for j in range(len(ep))])
            depth = 0
            cuts = []
            for i, (pos, d, w) in enumerate(events):
                depth += d
                if depth == 0 and i < len(events) - 1:
                    next_pos, _, next_w = events[i + 1]
                    # Only split if the two pulses are far enough apart: the gap
                    # between the closing End peak and the next Start peak must
                    # span at least `gap_widths` peak widths (the "split min gap"
                    # control).
                    min_gap = gap_widths * 0.5 * (w + next_w)
                    if (next_pos - pos) >= min_gap:
                        cuts.append((pos + next_pos) // 2)
            if not cuts:
                new_regions.append(list(region))
                new_labels.append(label)
                new_manual.append(man)
                continue
            r0, r1 = int(region[0]), int(region[1])
            starts = [r0] + [c + 1 for c in cuts]
            ends = [c for c in cuts] + [r1]
            for st, en in zip(starts, ends):
                npart = int(np.count_nonzero((sp >= st) & (sp <= en)))
                new_regions.append([st, en])
                new_labels.append('single' if npart == 1 else 'coincident')
                new_manual.append(False)  # auto-generated pieces
            changed = True
        return changed, new_regions, new_labels, new_manual

    def _apply_split(self, nr, nl, nm):
        self._pulse_regions = nr
        self._labels = nl
        self._manual = nm
        self._n_total = len(nr)

    def _auto_classify(self):
        if self._n_total == 0:
            return
        self._save_state()
        self._classify_all()
        self._render_page()
        self._autosave()

    def _auto_split(self):
        if self._n_total == 0:
            return
        changed, nr, nl, nm = self._compute_split()
        if not changed:
            return
        self._save_state()
        self._apply_split(nr, nl, nm)
        self._render_page()
        self._autosave()

    def _auto_process(self):
        """Loop Classify → Split until a full cycle changes nothing, then jump
        to the next page holding an uncertain event. One undo point covers the
        whole operation."""
        if self._n_total == 0:
            return
        self._save_state()
        for _ in range(20):  # convergence guard against pathological cycling
            classified = self._classify_all()
            split, nr, nl, nm = self._compute_split()
            if split:
                self._apply_split(nr, nl, nm)
            if not (classified or split):
                break
        self._render_page()
        self._autosave()
        self._goto_next_with_label('uncertain')

    def _auto_trim(self):
        """Resize every event window to match its actual signal.

        The event spans from the first Start-zone spike (particle entry) to the
        last End-zone spike (exit); both ends are padded by
        ``self._trim_pad_spin`` percent of that span, and the region is set to
        the padded window — which may extend past the original detection, up to
        the signal bounds.

        Only the Start and End zones are used (the Measurement zone's coded
        resistive pulse is not a clean spike and its window-size-dependent
        features would make repeated trims unstable; it is interior to the
        entry/exit span anyway). A region is left unchanged if it is noise, or
        if either the Start or End zone has no detectable spike — this keeps the
        operation idempotent (re-running never erodes a window to nothing)."""
        if self._n_total == 0:
            return
        pad_frac = self._trim_pad_spin.value() / 100.0
        z_start = self._zones_lp[self._role_start]
        z_end = self._zones_lp[self._role_end]
        n = min(len(z_start), len(z_end))
        new_regions = []
        changed = False
        for region, label in zip(self._pulse_regions, self._labels):
            r0, r1 = int(region[0]), int(region[1])
            s_idx = max(0, r0)
            e_idx = min(n, r1 + 1)
            sp = ep = np.empty(0, dtype=int)
            sw = ew = np.empty(0)
            if label != 'noise' and s_idx < e_idx:
                sp, sw = _detect_spikes(z_start[s_idx:e_idx], self._fs)
                ep, ew = _detect_spikes(z_end[s_idx:e_idx], self._fs)
            # Need a spike in both the Start and End zones to bound the event.
            if len(sp) == 0 or len(ep) == 0:
                new_regions.append([r0, r1])
                continue
            allp = np.concatenate([s_idx + sp, s_idx + ep])
            first, last = int(allp.min()), int(allp.max())
            span = last - first
            # Fall back to peak width if entry/exit spikes coincide.
            base = span if span > 0 else float(np.median(np.concatenate([sw, ew])))
            pad = int(round(pad_frac * base))
            # Extend/shrink to the padded window, clamped to the signal bounds.
            new_s = max(0, first - pad)
            new_e = min(n - 1, last + pad)
            if new_s >= new_e:
                new_regions.append([r0, r1])
                continue
            new_regions.append([new_s, new_e])
            if new_s != r0 or new_e != r1:
                changed = True
        if not changed:
            return
        self._save_state()
        self._pulse_regions = new_regions
        self._render_page()
        self._autosave()

    def _on_first(self):
        if self._page > 0:
            self._page = 0
            self._render_page()

    def _on_prev(self):
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def _on_next(self):
        if self._page < self._total_pages - 1:
            self._page += 1
            self._render_page()
        else:
            self.confirmed = True
            self._autosave()
            self.accept()

    def _on_last(self):
        if self._page < self._total_pages - 1:
            self._page = self._total_pages - 1
            self._render_page()

    def _on_next_unclassified(self):
        self._goto_next_with_label('unclassified')

    def _goto_next_with_label(self, target):
        """Jump to the next page (forward, wrapping) holding a *target*-labelled
        pulse. No-op if the only such page is the current one."""
        n = self._n_total
        ps = self._page_size
        cur = self._page
        order = (list(range(cur + 1, self._total_pages)) +
                 list(range(0, cur + 1)))
        for pg in order:
            start = pg * ps
            end = min(start + ps, n)
            if any(self._labels[i] == target for i in range(start, end)):
                if pg != self._page:
                    self._page = pg
                    self._render_page()
                return

    def _autosave(self):
        """Auto-save current settings to SavedTemplates/Review/temp.json."""
        import json, os
        from utils.paths import saved_templates_dir
        folder = saved_templates_dir("Review")
        settings = {
            "labels": list(self._labels),
            "regions": [[int(v) for v in r] for r in self._pulse_regions],
            "lpfCutoff": self._lpf_cutoff,
        }
        if self._show_zone_roles:
            settings["zoneRoles"] = self._zone_roles()
            settings["manual"] = [bool(v) for v in self._manual]
        try:
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _update_counts(self):
        classified = sum(1 for l in self._labels if l != 'unclassified')
        total = len(self._labels)
        # Disable the jump button once everything is classified.
        if hasattr(self, '_btn_next_unclassified'):
            self._btn_next_unclassified.setEnabled(classified < total)
        # Auto Split is usable only when there are coincident events to split.
        if getattr(self, '_btn_auto_split', None) is not None:
            self._btn_auto_split.setEnabled(
                any(l == 'coincident' for l in self._labels))
        # The Noise / Uncertain nav buttons light up only when such events exist.
        if getattr(self, '_btn_next_noise', None) is not None:
            self._btn_next_noise.setEnabled(
                any(l == 'noise' for l in self._labels))
        if getattr(self, '_btn_next_uncertain', None) is not None:
            self._btn_next_uncertain.setEnabled(
                any(l == 'uncertain' for l in self._labels))
        lines = [f"Classified: {classified} / {total}"]
        for cls in _CLASS_OPTIONS:
            n = sum(1 for l in self._labels if l == cls)
            if n > 0:
                color = _CLASS_COLORS.get(cls, "#444")
                lines.append(f"<span style='color:{color}'>{cls}: {n}</span>")
        self._lbl_counts.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_counts.setText("<br>".join(lines))

    def _default_folder(self):
        from utils.paths import saved_templates_dir
        return saved_templates_dir("Review")

    def _on_restart(self):
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Restart Review",
            "Clear all classifications and revert any splits/trims/expands "
            "back to the original detected regions?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._save_state()
        self._redo_stack.clear()
        self._pulse_regions = [list(r) for r in self._orig_pulse_regions]
        self._n_total = len(self._pulse_regions)
        self._labels = ['unclassified'] * self._n_total
        self._manual = [False] * self._n_total
        self._page = 0
        self._hide_class_overlay()
        self._hide_tool_overlay()
        self._hide_tool_sub_overlay()
        self._active_tool = None
        self._active_tool_grid_idx = None
        self._render_page()
        self._autosave()

    def _on_save_settings(self):
        import json
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Classification Settings", self._default_folder(),
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        settings = {
            "labels": list(self._labels),
            "regions": [[int(v) for v in r] for r in self._pulse_regions],
            "lpfCutoff": self._lpf_cutoff,
        }
        if self._show_zone_roles:
            settings["zoneRoles"] = self._zone_roles()
            settings["manual"] = [bool(v) for v in self._manual]
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)


def pulse_region_review(data, detected_pulses, sample_rate=1,
                        init_labels=None, init_regions=None, init_cutoff=None):
    """Interactive UI to review and classify detected pulse regions.

    Parameters
    ----------
    data : np.ndarray
        Input signal (1-D).
    detected_pulses : np.ndarray
        Nx2 array of [startIdx, endIdx] for detected pulses.
    sample_rate : float, optional
        Sampling rate in Hz.
    init_labels : list[str], optional
        Pre-loaded classification labels (one per pulse).

    Returns
    -------
    result : dict
        Keyed by class label ('single', 'coincident', 'noise',
        'uncertain') → Nx2 array of [start, end] indices.
    """
    data = np.asarray(data) if data is not None else np.array([])
    # 1-D stays 1-D; 2-D (N×Z) is kept and split per-zone inside the dialog.
    if data.ndim == 1:
        data = data.ravel()
    pulse_regions = detected_pulses if detected_pulses is not None else np.empty((0, 2), dtype=int)

    pulse_regions = np.atleast_2d(pulse_regions)
    n_total = len(pulse_regions)

    if n_total == 0:
        return {
            'single': np.empty((0, 2), dtype=int),
            'coincident': np.empty((0, 2), dtype=int),
            'noise': np.empty((0, 2), dtype=int),
            'uncertain': np.empty((0, 2), dtype=int),
        }

    dlg = _PulseReviewDialog(data, pulse_regions, sample_rate,
                             init_labels=init_labels, init_regions=init_regions,
                             init_cutoff=init_cutoff)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Detection Review without confirming.")

    regions = np.array(dlg._pulse_regions, dtype=int)
    labels = dlg._labels

    # Split indices by class label
    result = {}
    for cls in ('single', 'coincident', 'noise', 'uncertain'):
        mask = [i for i, lbl in enumerate(labels) if lbl == cls]
        result[cls] = regions[mask] if mask else np.empty((0, 2), dtype=int)

    return result
