"""Coincident Processing — decompose overlapping pulse regions into individual
single-particle pulses, then reuse the single-block per-pulse engine
(_compute_zone) + an interactive review dialog. Standalone flat output.
"""
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QPushButton, QLabel, QSpinBox, QWidget, QGroupBox,
    QScrollArea, QGridLayout, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FuncFormatter

from utils.dialog_style import apply_dialog_style, style_mpl_figure


# Per-particle-within-event colour palette (indexed by within-event order).
_EVENT_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                  "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]

# Resistance display unit: values are ohms; plots read megaohms so the
# "1e6" axis offset text stops overlapping the plot titles.
_MOHM = 1e6


def _mohm_axis(ax):
    """Format an ohms-valued y-axis as megaohms (no 1e6 offset text)."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / _MOHM:.4g}"))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_ylabel("MΩ", fontsize=8)


def guess_measurement_zone(cols):
    """Index of the highest-variance column (the measurement carries the
    largest edge excursions)."""
    if len(cols) == 1:
        return 0
    return int(np.argmax([float(np.var(np.asarray(c, dtype=float))) for c in cols]))


class SavedForLater(ValueError):
    """User saved the review's progress and closed the dialog. Stops the
    pipeline like any unconfirmed close, but lets the runner capture the
    saved state into the block's lastSettings first."""


def recompute(cols, measurement_zone, start_zone, end_zone, tp, fs,
              pulse_indices, prominence=None, trendline=None):
    """Run decomposition + the single-instance engine for the chosen roles.

    Returns (decomp, compute) where compute is a _compute_zone result dict over
    the recovered-particle spans, and decomp is the decomposition provenance.
    """
    from processing.extract_single_pulse_features import _compute_zone, _unpack_tp_zone
    from processing.coincident_decomposition import decompose_coincident_regions
    from utils.tpsettings_io import get_zone

    meas = np.asarray(cols[measurement_zone], dtype=float)
    start_ind = np.asarray(cols[start_zone], dtype=float) if start_zone is not None else None
    end_ind = np.asarray(cols[end_zone], dtype=float) if end_zone is not None else None
    tparams = _unpack_tp_zone(get_zone(tp, measurement_zone))

    decomp = decompose_coincident_regions(
        meas, fs, pulse_indices, tparams,
        start_indicator=start_ind, end_indicator=end_ind, prominence=prominence)

    rec = decomp["recovered_indices"]
    if len(rec) == 0:
        rec = np.zeros((0, 2), dtype=int)
    compute = _compute_zone(meas, fs, trendline, rec, **tparams)
    compute["pulse_indices"] = rec
    compute["trendline"] = trendline
    return decomp, compute


def extract_coincident_pulse_features(
    data_in, tp, sample_rate, pulse_indices,
    measurement_zone=None, start_zone=None, end_zone=None,
    trendline=None, pulse_features=None, settings_file=None, prominence=None,
):
    """Entry point. See module docstring. Returns flat (bare) output dict."""
    from utils.zones import zone_columns
    fs = float(sample_rate)
    cols = zone_columns(np.asarray(data_in))
    if measurement_zone is None:
        measurement_zone = guess_measurement_zone(cols)
    # Standard 3-zone capture layout: start indicator | measurement | end.
    if len(cols) >= 3:
        if start_zone is None and measurement_zone != 0:
            start_zone = 0
        if end_zone is None and measurement_zone != 2:
            end_zone = 2
    _pi = (None if pulse_indices is None
           else np.atleast_2d(np.asarray(pulse_indices, dtype=int)))
    if _pi is None or _pi.size == 0 or _pi.shape[1] != 2:
        return {"pulse_peak_locations": [], "rectangularized_pulses": [],
                "acceptance_id": np.array([], bool),
                "pulse_start_indices": np.array([], int), "coincidence_map": []}
    pulse_indices = _pi

    decomp, compute = recompute(
        cols, measurement_zone, start_zone, end_zone, tp, fs, pulse_indices,
        prominence=prominence, trendline=trendline)

    dlg = _CoincidentReviewDialog(
        compute=compute, decomp=decomp, cols=cols,
        measurement_zone=measurement_zone, start_zone=start_zone,
        end_zone=end_zone, tp=tp, fs=fs, pulse_indices=pulse_indices)
    if settings_file:
        dlg.load_state_from_file(settings_file)
    dlg.exec()
    if not dlg.confirmed:
        if getattr(dlg, "saved_for_later", False):
            raise SavedForLater(
                "Coincident Processing progress saved — re-run the block and "
                "choose Continue to resume the review.")
        raise ValueError("User closed Coincident Processing without confirming.")
    return dlg.results()


class _CoincidentReviewDialog(QDialog):
    """Per-event coincident review dialog.

    Each page shows ONE source event (coincident region) as a scrollable
    2-column plot grid. LEFT column: the un-trimmed source region with all
    zones stacked (filtered over togglable raw), then per-zone filtered
    diffs — start (if assigned), measurement (draggable thresholds), end (if
    assigned). RIGHT column: one plot per decomposed particle (raw with
    rectangularized span + occupancy). All controls live in a grouped right-hand panel; the
    Decomposition ``Particles (N)`` spinbox re-splits the current event.
    Per-particle accept checkboxes write into ``self.acceptance`` and colour
    each particle's raw-axis frame (green=accepted, red=rejected). Confirming
    returns a flat results dict.
    """

    # Fixed pixel height per plot ROW (original row + each particle row). The
    # figure grows to rows * _ROW_PX and the scroll area shows _MIN_ROWS_VISIBLE
    # rows before scrolling (original + 3 particles == the "2x4 minimum").
    _ROW_PX = 175
    _MIN_ROWS_VISIBLE = 4
    _LPF_DEFAULT = 300  # Hz

    def __init__(self, *, compute, decomp, cols, measurement_zone, start_zone,
                 end_zone, tp, fs, pulse_indices, parent=None):
        super().__init__(parent)
        self._compute = compute
        self._decomp = decomp
        self._cols = cols
        self._measurement_zone = measurement_zone
        self._start_zone = start_zone
        self._end_zone = end_zone
        self._fs = fs
        # held for the role-change recompute + split editing added in later tasks
        self._tp = tp
        self._pulse_indices = pulse_indices
        self._init_roles = (measurement_zone, start_zone, end_zone)
        self._prominence = None
        self._trendline = None
        self._rect_method = "median"
        # Per-role draggable diff-threshold pairs: role -> [upper, lower] or
        # None (= seed at half the extremes on next render).
        self._thr = {"start": None, "meas": None, "end": None}
        self._thr_lines = {}     # role -> (upper Line2D, lower Line2D)
        self._diff_ax_role = {}  # diff axis -> role
        self._zone_shifts = {}   # zone idx -> display offset in the signal plot
        self._drag_thr = None    # (role, "upper"|"lower") while dragging
        self._peak_marks = []
        self._ax_event_raw = None
        self._ax_event_diff = None  # the measurement diff axis
        self._ax_warped = None      # the raw-measurement + templates axis
        self._ev_window = None
        self._decomp_waves = {}   # pid -> decomposed waveform (current page)
        self._decomp_models = {}  # pid -> fitted model overlay
        self._decomp_note = ""
        self._decomp_fit = None
        self._peak_roles = {}     # global peak idx -> manual role override
        # Editable rectangular templates on plot #4 (see _draw_warped_templates):
        # committed per-pulse edits that override the exported rect.
        self._tpl_edit = {}       # pid -> {"edges":[x...], "levels":[y...]}
        self._active_tpl = None   # pid whose template is editable, or None
        # Constraint fitting: the auto-fit entity (a pid or "sum") recomputed to
        # hold sum = ΣPᵢ (sum locked to the raw rect) as the active pulse is
        # dragged; None = free per-pulse editing. When set, all pulses share the
        # raw-rect peak grid and only levels move.
        self._autofit_tpl = None
        self._drag_seg = None     # (pid, "level"|"edge", i, was_new) mid-drag
        self._drag_moved = False
        self._tpl_seed_cache = {} # pid -> display model this render (uncommitted)
        self._tpl_active_artists = []
        self._tpl_handle_ms = 3.5  # drag-handle marker size (small = less clutter)
        self._raw_rect = None     # cached (bounds, levels) whole-event raw rect
        self._event_pids = []     # current page's particle ids (colour order)
        # Locked template segments: pid -> {level idx}. A locked level holds
        # during auto-fit and refuses manual drags (toggled by double-clicking
        # the segment's move handle on the plot-#4 adjust panel).
        self._tpl_locks = {}
        # Enlarged plot-#4 adjustment panel (opened from the floating button).
        self._enlarge_panel = None
        self._enlarge_fig = None
        self._enlarge_canvas = None
        self._enlarge_ax = None
        self._enlarge_home = None  # (xlim, ylim) auto-scaled view for nav Home
        self._active_rbs = {}     # pid -> active-pulse radio (adjust panel)
        self._autofit_rbs = {}    # None/pid -> auto-fit radio (adjust panel)
        self._btn_enlarge_snap = None
        # Enlarged Measurement-diff adjustment panel (thresholds + peaks).
        self._meas_panel = None
        self._meas_fig = None
        self._meas_canvas = None
        self._meas_ax = None
        self._peak_palette = None
        self._peak_cursor = None  # cursor-following selection circle (Peaks)
        self._editing = False    # True while the Original Data editor is open
        self._editor = None
        # Indicator Thresholds panel (start/end diffs; editor-only, modeless)
        self._ind_panel = None
        self._ind_figure = None
        self._ind_canvas = None
        self._page = 0
        self._page_size = 1  # one coincident event (group) per page
        self.confirmed = False
        self.saved_for_later = False  # Save for later: state written, UI closed

        self._n_particles = len(compute["pulse_peak_locs"])
        self.acceptance = (np.asarray(compute["sequence_matches"]) >= 1).astype(int)
        # Events the user marked Complete (ids); Finish unlocks when every
        # event is in here.
        self._event_done = set()

        # Source-event grouping: ordered unique event ids and per-event particle
        # index lists (particles appear in compute order).
        self._events = self._group_events()
        self._total_pages = max(1, int(np.ceil(len(self._events) / self._page_size)))

        apply_dialog_style(self)
        self.setWindowTitle("Coincident Processing")
        self.resize(1500, 900)

        # Per-particle accept/reject checkboxes for the current page.
        self._accept_boxes = []

        self._build_ui()
        self._refresh_peaks()  # threshold-based; replaces the template-driven set
        self._render_page()

    # ---- setup helpers ------------------------------------------------------

    def _group_events(self):
        """Return list of (source_event_id, [particle_idx, ...]) in event order."""
        se = list(self._decomp.get("source_event", []))
        groups = {}
        order = []
        for pi, ev in enumerate(se):
            ev = int(ev)
            if ev not in groups:
                groups[ev] = []
                order.append(ev)
            groups[ev].append(pi)
        return [(ev, groups[ev]) for ev in order]

    def _event_mode(self, event_id):
        modes = self._decomp.get("event_mode", [])
        for pi, ev in enumerate(self._decomp.get("source_event", [])):
            if int(ev) == int(event_id) and pi < len(modes):
                return str(modes[pi])
        return ""

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- LEFT: scrollable raw/diff plot grid ---
        self._figure = Figure(figsize=(9, 8), tight_layout=False)
        style_mpl_figure(self._figure)
        self._canvas = FigureCanvas(self._figure)
        # Window-edit: click near a particle's start/end edge on its RAW axis to
        # move that edge to the click x (see _on_canvas_click). Motion/release
        # drive the draggable diff-threshold lines.
        self._connect_canvas(self._canvas)
        self._left_axes_map = {}   # raw axis -> [particle_id]
        self._raw_axes_map = {}    # particle_id -> raw axis (frame recolor)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._canvas)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setMinimumHeight(self._MIN_ROWS_VISIBLE * self._ROW_PX)
        root.addWidget(self._scroll, stretch=1)

        # --- RIGHT: all controls, grouped ---
        panel = QWidget()
        panel.setFixedWidth(300)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)
        # Zones + Filters live in the on-demand Original Data editor, not the
        # main panel — built here (kept alive, held detached) and reparented
        # into the editor when it opens.
        self._grp_zones = self._build_zones_group()
        self._grp_filters = self._build_filters_group()
        self._grp_display = self._build_display_group()
        # Floating "Adjust" buttons (see _place_meas_button /
        # _place_enlarge_button), floated over the measurement-diff and
        # warped-template rows; each opens that plot's adjustment panel.
        float_qss = ("QPushButton{font-size:10px;padding:1px 6px;"
                     "background:rgba(255,255,255,235);border:1px solid #9aa;"
                     "border-radius:4px;}")
        self._btn_meas_adjust = QPushButton("⤢ Adjust")
        self._btn_meas_adjust.setToolTip(
            "Open the Measurement diff in a larger panel to adjust the "
            "thresholds and peaks")
        self._btn_meas_adjust.setStyleSheet(float_qss)
        self._btn_meas_adjust.clicked.connect(self._open_meas_adjust_panel)
        self._btn_meas_adjust.hide()
        self._btn_enlarge = QPushButton("⤢ Adjust")
        self._btn_enlarge.setToolTip(
            "Open this plot in a larger panel to adjust the templates")
        self._btn_enlarge.setStyleSheet(float_qss)
        self._btn_enlarge.clicked.connect(self._open_enlarge_panel)
        self._btn_enlarge.hide()
        pv.addWidget(self._build_template_group())
        pv.addWidget(self._build_decomp_group())
        pv.addWidget(self._build_accept_group())
        pv.addWidget(self._build_detrend_group())
        pv.addStretch()
        pv.addWidget(self._build_nav_group())
        pv.addWidget(self._build_actions_group())
        root.addWidget(panel)

    def _build_template_group(self):
        g = QGroupBox("Template")
        lay = QVBoxLayout(g)
        self._lbl_template = QLabel("")
        self._lbl_template.setTextFormat(Qt.RichText)
        self._lbl_template.setWordWrap(True)
        lay.addWidget(self._lbl_template)
        return g

    def _format_template_seq(self):
        """Template sequence read-out (▲ key max, ▼ key min, ★ key regular,
        ● non-key; green positive / red negative), as in Pulse Processing."""
        seq = self._compute.get("template_seq")
        if seq is None or len(seq) == 0:
            return "(unavailable)"
        pol_map = dict(zip(self._compute.get("template_key_seq_indices") or [],
                           self._compute.get("template_key_polarities") or []))
        parts = []
        for i, v in enumerate(seq):
            color = "green" if int(v) == 1 else "red"
            pol = pol_map.get(i)
            sym = {"max": "▲", "min": "▼",
                   None: "●"}.get(pol, "★")
            parts.append(f'<span style="color:{color}; font-size:13px;">{sym}</span>')
        return " ".join(parts)

    def _build_nav_group(self):
        g = QGroupBox("Navigation")
        lay = QVBoxLayout(g)
        self._lbl_page = QLabel("")
        self._lbl_page.setAlignment(Qt.AlignCenter)
        self._lbl_page.setStyleSheet("font-weight: bold;")
        lay.addWidget(self._lbl_page)
        row = QHBoxLayout()
        self._btn_prev = QPushButton("‹ Prev")
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next = QPushButton("Next ›")
        self._btn_next.clicked.connect(self._on_next)
        row.addWidget(self._btn_prev)
        row.addWidget(self._btn_next)
        lay.addLayout(row)
        return g

    def _build_zones_group(self):
        g = QGroupBox("Zones")
        lay = QGridLayout(g)
        lay.setVerticalSpacing(6)
        n_cols = len(self._cols)

        self._cmb_measurement = QComboBox()
        for i in range(n_cols):
            self._cmb_measurement.addItem(f"Zone {i + 1}", i)
        idx = self._cmb_measurement.findData(self._measurement_zone)
        if idx >= 0:
            self._cmb_measurement.setCurrentIndex(idx)
        self._cmb_measurement.currentIndexChanged.connect(self._on_roles_changed)

        self._cmb_start = QComboBox()
        self._cmb_start.addItem("none", None)
        for i in range(n_cols):
            self._cmb_start.addItem(f"Zone {i + 1}", i)
        idx = self._cmb_start.findData(self._start_zone)
        if idx >= 0:
            self._cmb_start.setCurrentIndex(idx)
        self._cmb_start.currentIndexChanged.connect(self._on_roles_changed)

        self._cmb_end = QComboBox()
        self._cmb_end.addItem("none", None)
        for i in range(n_cols):
            self._cmb_end.addItem(f"Zone {i + 1}", i)
        idx = self._cmb_end.findData(self._end_zone)
        if idx >= 0:
            self._cmb_end.setCurrentIndex(idx)
        self._cmb_end.currentIndexChanged.connect(self._on_roles_changed)

        lay.addWidget(QLabel("Measurement"), 0, 0)
        lay.addWidget(self._cmb_measurement, 0, 1)
        lay.addWidget(QLabel("Start indicator"), 1, 0)
        lay.addWidget(self._cmb_start, 1, 1)
        lay.addWidget(QLabel("End indicator"), 2, 0)
        lay.addWidget(self._cmb_end, 2, 1)
        self._btn_indicators = QPushButton("Indicator Thresholds…")
        self._btn_indicators.setToolTip(
            "Start/End indicator diffs in a separate panel; drag the dashed "
            "lines to adjust their thresholds")
        self._btn_indicators.clicked.connect(self._open_indicator_panel)
        self._btn_indicators.setEnabled(
            self._start_zone is not None or self._end_zone is not None)
        lay.addWidget(self._btn_indicators, 3, 0, 1, 2)
        return g

    def _build_filters_group(self):
        g = QGroupBox("Filters")
        lay = QGridLayout(g)
        lay.setVerticalSpacing(6)
        # Display LPF (measurement diff; zone traces pick it up on re-render).
        # Cap at Nyquist so a no-op cutoff can't be selected.
        self._spin_lpf = QSpinBox()
        self._spin_lpf.setRange(1, max(1, int(self._fs / 2) - 1)
                                if self._fs > 4 else 1_000_000)
        self._spin_lpf.setValue(self._LPF_DEFAULT)
        self._spin_lpf.setKeyboardTracking(False)
        # The cutoff actually in effect; only Apply (or restore/restart)
        # commits the spinbox — a page turn must not silently re-filter.
        self._lpf_applied = float(self._spin_lpf.value())
        self._btn_lpf_apply = QPushButton("Apply")
        self._btn_lpf_apply.clicked.connect(self._on_lpf_apply)
        # Min peak gap: merge detections closer than this so each physical
        # edge gets one mark. Auto derives the gap from the LPF (1/cutoff).
        self._cmb_peak_gap = QComboBox()
        self._cmb_peak_gap.addItem("Off", "none")
        self._cmb_peak_gap.addItem("Auto — 1/cutoff", "auto")
        self._cmb_peak_gap.addItem("Manual", "manual")
        self._cmb_peak_gap.setCurrentIndex(1)
        self._cmb_peak_gap.currentIndexChanged.connect(self._on_peak_gap_changed)
        self._spin_gap_ms = QSpinBox()
        self._spin_gap_ms.setRange(1, 1000)
        self._spin_gap_ms.setValue(10)
        self._spin_gap_ms.setKeyboardTracking(False)
        self._spin_gap_ms.setEnabled(False)
        self._spin_gap_ms.valueChanged.connect(
            lambda *a: (self._refresh_peaks(), self._render_page()))
        lay.addWidget(QLabel("LPF cutoff (Hz)"), 0, 0)
        lay.addWidget(self._spin_lpf, 0, 1)
        lay.addWidget(QLabel("Min peak gap"), 1, 0)
        lay.addWidget(self._cmb_peak_gap, 1, 1)
        # Peaks toggle floats over the enlarged Measurement-diff panel (see
        # _place_adjust_button); it is not placed in this panel.
        self._btn_peak_adjust = QPushButton("✦ Peaks")
        self._btn_peak_adjust.setCheckable(True)
        self._btn_peak_adjust.setToolTip(
            "Click a Measurement-diff peak to reassign its pulse, set its "
            "role, or remove it; click empty space to add one")
        self._btn_peak_adjust.setStyleSheet(
            "QPushButton{font-size:10px;padding:1px 6px;"
            "background:rgba(255,255,255,235);border:1px solid #9aa;"
            "border-radius:4px;}"
            "QPushButton:checked{background:#1f77b4;color:white;"
            "border-color:#1f77b4;}")
        # Thres toggle floats just left of Peaks; the two are mutually
        # exclusive and threshold lines only move while it is on.
        self._btn_thr_adjust = QPushButton("⇅ Thres")
        self._btn_thr_adjust.setCheckable(True)
        self._btn_thr_adjust.setToolTip(
            "Drag the Measurement-diff threshold lines while this is on")
        self._btn_thr_adjust.setStyleSheet(
            "QPushButton{font-size:10px;padding:1px 6px;"
            "background:rgba(255,255,255,235);border:1px solid #9aa;"
            "border-radius:4px;}"
            "QPushButton:checked{background:#1f77b4;color:white;"
            "border-color:#1f77b4;}")
        self._btn_peak_adjust.toggled.connect(
            lambda on: (self._btn_thr_adjust.setChecked(False) if on else
                        (self._close_peak_palette(), self._hide_peak_cursor())))
        self._btn_thr_adjust.toggled.connect(
            lambda on: self._btn_peak_adjust.setChecked(False) if on else None)
        lay.addWidget(QLabel("Gap (ms)"), 2, 0)
        lay.addWidget(self._spin_gap_ms, 2, 1)
        lay.addWidget(self._btn_lpf_apply, 3, 0, 1, 2)
        return g

    @staticmethod
    def _float_at_axis_top_right(ax, canvas, buttons, pad=6):
        """Float `buttons` (rightmost first) at the top-right of `ax` on
        `canvas`. The figures carry a tight-layout engine, so the axis
        position is only final after a draw — placement re-runs on every
        canvas draw/resize (see _place_float_buttons)."""
        pos = ax.get_position()   # figure fraction (origin bottom-left)
        cw, ch = canvas.width(), canvas.height()
        gap = 4
        y = int((1.0 - pos.y1) * ch) + pad
        xr = int(pos.x1 * cw) - pad
        for b in buttons:
            if b.parent() is not canvas:
                b.setParent(canvas)
            b.adjustSize()
            xr -= b.width()
            b.move(max(0, xr), max(0, y))
            b.show()
            b.raise_()
            xr -= gap

    def _place_float_buttons(self):
        """Re-place every floating button on its current canvas/axis."""
        self._place_meas_button()
        self._place_adjust_button()
        self._place_enlarge_button()

    def _place_adjust_button(self):
        """Float the Thres + Peaks toggles at the top-right of the enlarged
        Measurement-diff panel (Peaks rightmost, Thres just left of it)."""
        peak, thr = self._btn_peak_adjust, self._btn_thr_adjust
        if self._meas_canvas is None or self._meas_ax is None:
            peak.hide()
            thr.hide()
            return
        self._float_at_axis_top_right(self._meas_ax, self._meas_canvas,
                                      (peak, thr))

    def _place_meas_button(self):
        """Float the Adjust button at the top-right of the Measurement diff
        on the active editor canvas (re-placed each render)."""
        btn = self._btn_meas_adjust
        ax = self._ax_event_diff
        # Threshold/peak editing happens in the adjust panel this opens; the
        # main view just opens the editor.
        if ax is None or not self._editing or self._meas_panel is not None:
            btn.hide()
            return
        self._float_at_axis_top_right(ax, self._canvas, (btn,))

    def _place_enlarge_button(self):
        """Float the Adjust button at the top-right of the warped-template row
        on the active editor canvas (re-placed each render)."""
        btn = self._btn_enlarge
        ax = self._ax_warped
        if ax is None or not self._editing or self._enlarge_panel is not None:
            btn.hide()
            return
        self._float_at_axis_top_right(ax, self._canvas, (btn,))

    _DISPLAY_MODES = (("Raw data", "raw"),
                      ("Filtered data", "filtered"),
                      ("Raw + Filtered data", "both"))

    def _build_display_group(self):
        g = QGroupBox("Display")
        lay = QVBoxLayout(g)
        self._display_group = QButtonGroup(self)
        self._display_rbs = {}
        for i, (label, mode) in enumerate(self._DISPLAY_MODES):
            rb = QRadioButton(label)
            if mode == "both":
                rb.setChecked(True)
            self._display_group.addButton(rb, i)
            self._display_rbs[mode] = rb
            lay.addWidget(rb)
        self._display_group.buttonClicked.connect(lambda *a: self._render_page())
        return g

    def _display_mode(self):
        for mode, rb in self._display_rbs.items():
            if rb.isChecked():
                return mode
        return "both"

    def _snap_active_to_raw(self):
        """One-click align: move each active-pulse INTERIOR edge (vertical
        transition) to the nearest raw-rect edge (peak boundary), keeping the
        edges ordered, then snap each segment level onto the raw-rect
        staircase (the overlap-dominant step). Outer window edges and locked
        levels are left untouched."""
        pid = self._active_tpl
        if not isinstance(pid, int) or self._raw_rect is None:
            return
        self._ensure_tpl_edit(pid)
        ed = self._tpl_edit.get(pid)
        if ed is None or len(ed.get("edges", [])) < 3:  # need interior edges
            return
        B = np.asarray(self._raw_rect[0], float)  # raw-rect edges (peaks + ends)
        L = np.asarray(self._raw_rect[1], float)  # raw-rect levels
        edges = [float(x) for x in ed["edges"]]
        for i in range(1, len(edges) - 1):
            nearest = float(B[int(np.argmin(np.abs(B - edges[i])))])
            lo = edges[i - 1] + 1.0   # stay strictly ordered
            hi = edges[i + 1] - 1.0
            edges[i] = min(max(nearest, lo), hi)
        ed["edges"] = edges
        # Snap each segment level to the raw-rect level it mostly overlaps
        # (a segment spanning several staircase steps takes the dominant
        # one). Pinned (locked) levels hold.
        locked = self._tpl_locks.get(pid, set())
        levels = [float(v) for v in ed["levels"]]
        for i in range(min(len(levels), len(edges) - 1)):
            if i in locked:
                continue
            ov = np.minimum(edges[i + 1], B[1:]) - np.maximum(edges[i], B[:-1])
            j = int(np.argmax(ov))
            if ov[j] > 0:
                levels[i] = float(L[j])
        ed["levels"] = levels
        if self._enlarge_panel is not None:
            self._render_enlarge()
        else:
            self._render_page()

    def _regenerate_templates(self):
        """Reset the current event's templates to fresh warped seeds — drops
        committed edits and locks so the anchors realign with the key peaks."""
        for pid in self._event_pids:
            self._tpl_edit.pop(pid, None)
            self._tpl_locks.pop(pid, None)
        self._tpl_seed_cache = {}
        if self._autofit_tpl is not None:
            self._enter_constraint_mode()  # recommit fresh own-segment seeds
        if self._enlarge_panel is not None:
            self._render_enlarge()
        else:
            self._render_page()

    def _update_tpl_buttons(self):
        """Enable the Snap-to-raw button only when a pulse is active."""
        if self._btn_enlarge_snap is not None:
            self._btn_enlarge_snap.setEnabled(isinstance(self._active_tpl, int))

    def _on_active_rb(self, pid):
        """Active-pulse radio toggled on in the adjust panel."""
        self._active_tpl = int(pid)
        if self._autofit_tpl == self._active_tpl:  # can't auto-fit the active
            rb = self._autofit_rbs.get(None)
            if rb is not None and not rb.isChecked():
                rb.setChecked(True)   # fires _on_autofit_rb(None)
            else:
                self._autofit_tpl = None
        for val, rb in self._autofit_rbs.items():
            if isinstance(val, int):
                rb.setEnabled(val != self._active_tpl)
        if self._autofit_tpl is not None:
            self._enter_constraint_mode()
        self._update_tpl_buttons()
        self._render_enlarge()

    def _on_autofit_rb(self, val):
        """Auto-fit radio toggled on in the adjust panel."""
        self._autofit_tpl = val
        if val is not None:
            self._enter_constraint_mode()
        self._render_enlarge()

    def _on_peak_gap_changed(self, *a):
        self._spin_gap_ms.setEnabled(self._cmb_peak_gap.currentData() == "manual")
        self._refresh_peaks()
        self._render_page()

    def _peak_distance(self):
        """Min peak separation in samples per the gap mode (None = no merge)."""
        mode = self._cmb_peak_gap.currentData()
        if mode == "auto":
            return max(1, int(round(self._fs / float(self._lpf_applied))))
        if mode == "manual":
            return max(1, int(round(self._spin_gap_ms.value() * 1e-3 * self._fs)))
        return None

    def _build_decomp_group(self):
        g = QGroupBox("Decomposition")
        lay = QGridLayout(g)
        btn_edit = QPushButton("Manual Edit")
        btn_edit.setToolTip("Zones, filters, thresholds and peaks in a larger view")
        btn_edit.clicked.connect(self._open_original_editor)
        lay.addWidget(btn_edit, 0, 0, 1, 3)
        lay.addWidget(QLabel("Particles (N)"), 1, 0)
        self._spin_n = QSpinBox()
        self._spin_n.setRange(1, 99)  # display must not clamp the true count
        lay.addWidget(self._spin_n, 1, 1)
        # Re-split only on Update (not while typing N), so the plots don't churn.
        self._btn_nsplit = QPushButton("Update")
        self._btn_nsplit.setToolTip("Re-split this event into N pulses")
        self._btn_nsplit.clicked.connect(self._on_nsplit_changed)
        lay.addWidget(self._btn_nsplit, 1, 2)
        hint = QLabel("Set N, then click Update to re-split this event")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint, 2, 0, 1, 3)
        return g

    def _current_event_id(self):
        if not self._events:
            return None
        idx = min(self._page, len(self._events) - 1)
        return self._events[idx][0]

    def _on_nsplit_changed(self):
        ev = self._current_event_id()
        if ev is None:
            return
        # editingFinished can fire repeatedly (Enter, then focus-out). Skip when
        # N already matches the current split: it avoids a redundant full
        # recompute+redraw and preserves any manual span edits (re-splitting
        # into the same N would reset them to even sub-spans).
        idx = min(self._page, len(self._events) - 1)
        if self._spin_n.value() == len(self._events[idx][1]):
            return
        self.set_event_n(event=ev, n=self._spin_n.value())

    def _build_accept_group(self):
        # Checkbox list scrolls beyond ~7 rows: an unbounded list forces the
        # whole dialog taller than the screen on many-particle events (and Qt
        # never shrinks the window back).
        g = QGroupBox("Acceptance")
        outer = QVBoxLayout(g)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        self._accept_lay = QVBoxLayout(inner)
        self._accept_lay.setSpacing(4)
        self._accept_lay.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(inner)
        self._accept_scroll = scroll
        outer.addWidget(scroll)
        self._btn_complete = QPushButton("Mark Complete")
        self._btn_complete.setCheckable(True)
        self._btn_complete.setToolTip(
            "Mark this event's review as complete; Finish unlocks once every "
            "event is completed")
        self._btn_complete.setStyleSheet(
            "QPushButton:checked{background:#1f77b4;color:white;"
            "border-color:#1f77b4;}")
        self._btn_complete.toggled.connect(self._on_complete_toggled)
        outer.addWidget(self._btn_complete)
        return g

    def _on_complete_toggled(self, checked):
        ev = self._current_event_id()
        if ev is None:
            return
        (self._event_done.add if checked else self._event_done.discard)(int(ev))
        self._update_progress_ui()

    def _update_progress_ui(self):
        """Sync the Complete toggle, page label and Finish gate to the current
        event's completion state."""
        ev = self._current_event_id()
        done = ev is not None and int(ev) in self._event_done
        self._lbl_page.setText(
            f"Event {self._page + 1} / {max(1, len(self._events))}"
            + (" ✓" if done else ""))
        self._btn_complete.blockSignals(True)
        self._btn_complete.setChecked(done)
        self._btn_complete.setText("✓ Completed" if done else "Mark Complete")
        self._btn_complete.setEnabled(ev is not None)
        self._btn_complete.blockSignals(False)
        missing = len({int(e) for e, _ in self._events} - self._event_done)
        self._btn_ok.setEnabled(missing == 0)
        self._btn_ok.setToolTip(
            "" if missing == 0 else
            f"{missing} event(s) not yet marked Complete")

    def _build_detrend_group(self):
        g = QGroupBox("Detrending")
        lay = QVBoxLayout(g)

        self._chk_detrend = QCheckBox("Linear baseline")
        self._chk_detrend.setChecked(False)
        self._chk_detrend.toggled.connect(lambda *a: self._render_page())
        lay.addWidget(self._chk_detrend)
        hint = QLabel("Fits a line through the flat regions before the first "
                      "pulse and after the last, and subtracts it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)

        self._rect_group = QButtonGroup(self)
        self._rb_rect_median = QRadioButton("Median")
        self._rb_rect_median.setChecked(True)
        self._rb_rect_mean = QRadioButton("Mean")
        self._rect_group.addButton(self._rb_rect_median, 0)
        self._rect_group.addButton(self._rb_rect_mean, 1)
        self._rect_group.buttonClicked.connect(self._on_rect_method_changed)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Rect:"))
        row3.addWidget(self._rb_rect_median)
        row3.addWidget(self._rb_rect_mean)
        lay.addLayout(row3)
        return g

    def _build_actions_group(self):
        g = QGroupBox("Actions")
        lay = QVBoxLayout(g)
        try:
            from utils.export_figure_ui import make_export_button
            btn_export = make_export_button(lambda: self._figure, parent=self)
        except Exception:
            btn_export = QPushButton("Export Figure")
            btn_export.clicked.connect(self._on_export_fallback)
        self._btn_restart = QPushButton("Restart")
        self._btn_restart.clicked.connect(self._on_restart)
        self._btn_save = QPushButton("Save for later")
        self._btn_save.setToolTip(
            "Save the current progress and close; re-running the block "
            "offers to continue from it")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_ok = QPushButton("Finish")
        self._btn_ok.setObjectName("confirmBtn")
        self._btn_ok.clicked.connect(self._on_ok)
        lay.addWidget(self._btn_save)
        # One row, equal sizes: widths stretch-driven, heights pinned to the
        # tallest (the confirmBtn style makes Finish taller by default).
        row = QHBoxLayout()
        row.setSpacing(6)
        btns = (btn_export, self._btn_restart, self._btn_ok)
        h = max(b.sizeHint().height() for b in btns)
        for b in btns:
            # QToolButton (the Export split button) defaults to a Fixed
            # horizontal policy and would ignore the stretch.
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setFixedHeight(h)
            row.addWidget(b, 1)
        lay.addLayout(row)
        return g

    def _on_save(self):
        """Write the current progress to the autosave file and close — a
        re-run of the block offers to continue from the saved state. On a
        write failure the dialog stays open so no work is lost."""
        if not self._autosave():
            from PySide6.QtCore import QTimer
            btn = self._btn_save
            btn.setText("Save failed")
            QTimer.singleShot(1500, btn, lambda: btn.setText("Save for later"))
            return
        self.saved_for_later = True
        self.reject()

    def _on_restart(self):
        """Reset every parameter to its default (zones back to the initial
        roles, LPF/gap/thresholds/detrending/rect) and recompute from scratch,
        discarding manual splits and acceptance edits."""
        m0, s0, e0 = self._init_roles
        widgets = (self._cmb_measurement, self._cmb_start, self._cmb_end,
                   self._spin_lpf, self._cmb_peak_gap, self._spin_gap_ms,
                   self._chk_detrend, self._display_rbs["both"])
        for w in widgets:
            w.blockSignals(True)
        self._cmb_measurement.setCurrentIndex(self._cmb_measurement.findData(m0))
        self._cmb_start.setCurrentIndex(self._cmb_start.findData(s0))
        self._cmb_end.setCurrentIndex(self._cmb_end.findData(e0))
        self._spin_lpf.setValue(self._LPF_DEFAULT)
        self._lpf_applied = float(self._spin_lpf.value())
        self._cmb_peak_gap.setCurrentIndex(self._cmb_peak_gap.findData("auto"))
        self._spin_gap_ms.setValue(10)
        self._chk_detrend.setChecked(False)
        self._display_rbs["both"].setChecked(True)
        for w in widgets:
            w.blockSignals(False)
        self._rb_rect_median.setChecked(True)
        self._spin_gap_ms.setEnabled(False)
        self._rect_method = "median"
        # re-seeded from the fresh diffs on render
        self._thr = {"start": None, "meas": None, "end": None}
        self._peak_roles = {}
        self._close_peak_palette()
        self._btn_peak_adjust.setChecked(False)
        self._on_roles_changed()

    # ---- slot stubs / small handlers ---------------------------------------

    def _refresh_events_and_pages(self):
        """Re-derive event grouping + page count from the current decomp, and
        clamp the current page so it stays in range. Shared by role changes and
        split editing so the two paths stay consistent."""
        self._events = self._group_events()
        self._total_pages = max(1, int(np.ceil(len(self._events) / self._page_size)))
        self._page = min(self._page, self._total_pages - 1)

    def _on_roles_changed(self, *args):
        # A role's zone change means a new diff scale for that role; stale
        # thresholds rarely make sense — cleared pairs re-seed on render.
        new_m = self._cmb_measurement.currentData()
        new_s = self._cmb_start.currentData()
        new_e = self._cmb_end.currentData()
        if new_m != self._measurement_zone:
            self._thr["meas"] = None
        if new_s != self._start_zone:
            self._thr["start"] = None
        if new_e != self._end_zone:
            self._thr["end"] = None
        self._measurement_zone = new_m
        self._start_zone = new_s
        self._end_zone = new_e
        self._event_done.clear()  # events rebuild from scratch below
        self._btn_indicators.setEnabled(new_s is not None or new_e is not None)
        if self._ind_panel is not None and new_s is None and new_e is None:
            self._ind_panel.close()
        decomp, compute = recompute(
            self._cols, self._measurement_zone, self._start_zone,
            self._end_zone, self._tp, self._fs, self._pulse_indices,
            prominence=self._prominence, trendline=self._trendline)
        self._decomp = decomp
        self._compute = compute
        self._n_particles = len(compute["pulse_peak_locs"])
        self.acceptance = (np.asarray(compute["sequence_matches"]) >= 1).astype(int)
        self._page = 0
        self._refresh_events_and_pages()
        self._refresh_peaks()
        self._render_page()

    def _on_rect_method_changed(self, *a):
        self._rect_method = "median" if self._rb_rect_median.isChecked() else "mean"
        self._render_page()

    def _on_lpf_apply(self, *a):
        """Commit the spinbox cutoff, re-seed every threshold pair at half
        its new diff extremes (a cutoff change rescales the diffs), re-detect
        the particle peaks, and re-render — detection drives the rects, so
        the particle plots must refresh too."""
        if self._ev_window is None:
            return
        self._lpf_applied = float(self._spin_lpf.value())
        self._thr = {r: None for r in self._thr}
        self._refresh_peaks()
        self._render_page()

    def _zone_lp(self, zi, s, e):
        """LPF copy of zone `zi` over [s, e] (display only). Filters a margin-
        padded window rather than the whole column — long recordings would make
        full-column filtfilt slow/memory-hungry — and uses SOS, which stays
        numerically stable at low cutoff/fs ratios where ba-form blows up."""
        from scipy import signal as sig
        arr = np.asarray(self._cols[zi], dtype=float)
        seg = arr[s:e + 1]
        cutoff = float(self._lpf_applied)
        nyq = self._fs / 2.0
        if cutoff >= nyq:
            return seg
        # Margin for edge transients: a few filter time constants.
        pad = int(3 * self._fs / cutoff)
        lo = max(0, s - pad)
        hi = min(len(arr), e + 1 + pad)
        sos = sig.butter(4, cutoff / nyq, btype="low", output="sos")
        try:
            f = sig.sosfiltfilt(sos, arr[lo:hi])
        except Exception:
            return seg
        return f[s - lo:s - lo + len(seg)]

    def _on_export_fallback(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "coincident.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self._figure.savefig(path, dpi=150)

    def _on_prev(self):
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def _on_next(self):
        if self._page < len(self._events) - 1:
            self._page += 1
            self._render_page()

    def _on_ok(self):
        self.confirmed = True
        self._autosave()
        self.accept()

    # ---- settings save / load ---------------------------------------------

    def _get_state(self):
        return {
            "measurement_zone": self._measurement_zone,
            "start_zone": self._start_zone,
            "end_zone": self._end_zone,
            "recovered_indices": np.asarray(self._decomp["recovered_indices"], int).tolist(),
            "source_event": list(self._decomp["source_event"]),
            "acceptance": [int(x) for x in self.acceptance],
            "detrend": bool(self._chk_detrend.isChecked()),
            "rect_method": self._rect_method,
            "lpf_cutoff": int(self._spin_lpf.value()),
            "thresholds": {r: (None if v is None else [float(v[0]), float(v[1])])
                           for r, v in self._thr.items()},
            "peak_locs": [np.asarray(p, dtype=int).tolist()
                          for p in self._compute.get("pulse_peak_locs", [])],
            "peak_roles": {str(k): v for k, v in self._peak_roles.items()},
            "tpl_edit": {str(pid): {"edges": [float(x) for x in ed["edges"]],
                                    "levels": [float(l) for l in ed["levels"]]}
                         for pid, ed in self._tpl_edit.items()},
            "tpl_locks": {str(pid): sorted(int(i) for i in lk)
                          for pid, lk in self._tpl_locks.items() if lk},
            "event_mode": [str(m) for m in self._decomp.get("event_mode", [])],
            "peak_gap_mode": self._cmb_peak_gap.currentData(),
            "peak_gap_ms": int(self._spin_gap_ms.value()),
            "display_mode": self._display_mode(),
            "completed_events": sorted(int(e) for e in self._event_done),
            "page": int(self._page),
        }

    def _apply_state(self, st):
        # Sanitize saved zone roles against THIS capture's column count: a
        # settings file from a different dataset can reference zones that
        # don't exist here (would IndexError in _zone_lp / recompute).
        n_cols = len(self._cols)
        zone_ok = lambda z: z if isinstance(z, int) and 0 <= z < n_cols else None
        mz = zone_ok(st.get("measurement_zone"))
        st = dict(st)
        st["measurement_zone"] = mz if mz is not None else self._measurement_zone
        st["start_zone"] = zone_ok(st.get("start_zone"))
        st["end_zone"] = zone_ok(st.get("end_zone"))
        # Guard the role combos: setCurrentIndex fires currentIndexChanged ->
        # _on_roles_changed, which recomputes and resets acceptance. We restore
        # acceptance explicitly after the rebuild below, so suppress it here.
        for cmb, key in ((self._cmb_measurement, "measurement_zone"),
                         (self._cmb_start, "start_zone"),
                         (self._cmb_end, "end_zone")):
            cmb.blockSignals(True)
            cmb.setCurrentIndex(cmb.findData(st.get(key)))
            cmb.blockSignals(False)
        self._measurement_zone = st["measurement_zone"]
        self._start_zone = st.get("start_zone")
        self._end_zone = st.get("end_zone")
        self._btn_indicators.setEnabled(
            self._start_zone is not None or self._end_zone is not None)
        if (self._ind_panel is not None and self._start_zone is None
                and self._end_zone is None):
            self._ind_panel.close()
        self._decomp["recovered_indices"] = np.asarray(st["recovered_indices"], int).reshape(-1, 2)
        self._decomp["source_event"] = list(st.get("source_event", []))
        self._decomp["event_n"] = [self._decomp["source_event"].count(s)
                                   for s in self._decomp["source_event"]]
        em = st.get("event_mode")
        if em is not None and len(em) == len(self._decomp["source_event"]):
            self._decomp["event_mode"] = [str(m) for m in em]
        else:  # legacy files without event_mode
            self._decomp.setdefault("event_mode", ["manual"] * len(self._decomp["source_event"]))
            if len(self._decomp["event_mode"]) != len(self._decomp["source_event"]):
                self._decomp["event_mode"] = ["manual"] * len(self._decomp["source_event"])
        # Block the detrend checkbox: a mid-restore render would draw the
        # already-replaced decomp against the not-yet-rebuilt compute.
        self._chk_detrend.blockSignals(True)
        self._chk_detrend.setChecked(st.get("detrend", False))
        self._chk_detrend.blockSignals(False)
        self._rect_method = st.get("rect_method", "median")
        (self._rb_rect_median if self._rect_method == "median"
         else self._rb_rect_mean).setChecked(True)
        self._spin_lpf.blockSignals(True)
        self._spin_lpf.setValue(int(st.get("lpf_cutoff", self._LPF_DEFAULT)))
        self._spin_lpf.blockSignals(False)
        self._lpf_applied = float(self._spin_lpf.value())
        thr = st.get("thresholds")
        if thr is None:  # legacy single-pair settings files
            u, l = st.get("thr_upper"), st.get("thr_lower")
            thr = {"meas": [u, l] if u is not None and l is not None else None}
        self._thr = {r: (list(thr.get(r)) if thr.get(r) else None)
                     for r in ("start", "meas", "end")}
        self._cmb_peak_gap.blockSignals(True)
        idx = self._cmb_peak_gap.findData(st.get("peak_gap_mode", "auto"))
        self._cmb_peak_gap.setCurrentIndex(max(0, idx))
        self._cmb_peak_gap.blockSignals(False)
        self._spin_gap_ms.blockSignals(True)
        self._spin_gap_ms.setValue(int(st.get("peak_gap_ms", 10)))
        self._spin_gap_ms.blockSignals(False)
        self._spin_gap_ms.setEnabled(self._cmb_peak_gap.currentData() == "manual")
        # Back-compat: legacy files carry show_raw (bool) instead of a mode.
        mode = st.get("display_mode")
        if mode is None:
            mode = "both" if st.get("show_raw", True) else "filtered"
        rb = self._display_rbs.get(mode, self._display_rbs["both"])
        rb.blockSignals(True)
        rb.setChecked(True)
        rb.blockSignals(False)
        self._rebuild_compute_from_spans(render=False)  # rendered once below
        # Restore manual peak edits over the recomputed peaks (count must
        # match: a stale file for different spans keeps the fresh compute).
        pl = st.get("peak_locs")
        if pl is not None and len(pl) == self._n_particles:
            self._compute["pulse_peak_locs"] = [np.asarray(p, dtype=int)
                                                for p in pl]
            for pid in range(self._n_particles):
                self._recompute_rect(pid)
                self._check_sequence(pid)
            self._peak_roles = {int(k): v
                                for k, v in st.get("peak_roles", {}).items()}
        # Restore committed template edits (drop any pid outside this capture).
        self._tpl_edit = {}
        for k, ed in (st.get("tpl_edit") or {}).items():
            pid = int(k)
            if 0 <= pid < self._n_particles and ed.get("edges") \
                    and len(ed["edges"]) == len(ed.get("levels", [])) + 1:
                self._tpl_edit[pid] = {
                    "edges": [float(x) for x in ed["edges"]],
                    "levels": [float(l) for l in ed["levels"]]}
        self._tpl_locks = {}
        for k, lv in (st.get("tpl_locks") or {}).items():
            pid = int(k)
            if 0 <= pid < self._n_particles:
                self._tpl_locks[pid] = {int(i) for i in lv}
        acc = st.get("acceptance")
        if acc is not None and len(acc) == self._n_particles:
            self.acceptance = np.asarray(acc, dtype=int)
        ids = {int(e) for e, _ in self._events}
        self._event_done = {int(e) for e in st.get("completed_events", [])
                            if int(e) in ids}
        self._page = min(max(0, int(st.get("page", 0))), self._total_pages - 1)
        self._render_page()

    def load_state_from_file(self, path):
        import json, os
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self._apply_state(json.load(f))
            except Exception:
                pass

    def _autosave(self):
        """Write the state to the autosave file; True on success."""
        import json, os
        from utils.paths import saved_templates_dir
        p = os.path.join(saved_templates_dir("Coincident"), "temp.json")
        try:
            with open(p, "w") as f:
                json.dump(self._get_state(), f, indent=2)
            return True
        except Exception:
            return False

    # ---- split editing (mutate spans + recompute) --------------------------

    def _rebuild_compute_from_spans(self, render=True):
        from processing.extract_single_pulse_features import _compute_zone, _unpack_tp_zone
        from utils.tpsettings_io import get_zone
        # Snapshot old span -> acceptance so unchanged particles keep the user's
        # accept/reject decision when a single edit re-runs the whole compute.
        _old_rec = np.asarray(self._compute.get("pulse_indices",
                              np.zeros((0, 2), int))).reshape(-1, 2)
        _prev_acc = {(int(a), int(b)): int(self.acceptance[i])
                     for i, (a, b) in enumerate(_old_rec)
                     if i < len(self.acceptance)}
        meas = np.asarray(self._cols[self._measurement_zone], dtype=float)
        tparams = _unpack_tp_zone(get_zone(self._tp, self._measurement_zone))
        rec = np.asarray(self._decomp["recovered_indices"], dtype=int).reshape(-1, 2)
        self._compute = _compute_zone(meas, self._fs, self._trendline, rec, **tparams)
        self._compute["pulse_indices"] = rec
        self._compute["trendline"] = self._trendline
        self._n_particles = len(self._compute["pulse_peak_locs"])
        seq = np.asarray(self._compute["sequence_matches"])
        # Unchanged spans keep the prior value; new/edited spans seed fresh.
        self.acceptance = np.array(
            [_prev_acc.get((int(a), int(b)), int(1 if seq[i] >= 1 else 0))
             for i, (a, b) in enumerate(rec)], dtype=int)
        self._refresh_events_and_pages()
        self._refresh_peaks()
        if render:
            self._render_page()

    # Per-pulse compute arrays (length == n_particles) spliced by
    # _rebuild_compute_incremental. Global/template keys are inherited from the
    # partial recompute unchanged.
    _PER_PULSE_KEYS = ("filtered_pulses", "rect_pulses", "rect_pulses_mean",
                       "pulse_peak_locs", "pulse_diffs", "thresholds_used",
                       "counts_used", "pulse_excl_ranges_list",
                       "pulse_key_peaks_list", "pulse_key_peak_pols_list",
                       "pulse_key_peak_seqs_list")

    def _rebuild_compute_incremental(self, new_spans, origins, only_events=None):
        """Recompute ONLY ``new_spans`` (the edited event's particles) and reuse
        cached per-pulse results for every unchanged particle, instead of
        re-running _compute_zone over all events. ``only_events`` scopes the
        peak re-detection the same way (see _refresh_peaks), so unchanged
        events keep their manual peaks/roles/template edits.

        ``origins[p]`` is ``("old", old_idx)`` to reuse ``self._compute[key]
        [old_idx]`` or ``("new", k)`` to take the k-th freshly computed span.
        The per-pulse features are exact: _compute_zone computes each pulse from
        only its own segment + shared template params (deterministic), so a kept
        span recomputes to its cached value. ``old_idx`` indexes ``self._compute``
        per-pulse arrays, which are in the same order as
        ``self._decomp["recovered_indices"]`` (the array set just above in
        ``set_event_n``). The one CROSS-pulse quantity — the shape-outlier
        downgrade of sequence_matches (1 -> 2) — is re-derived on the merged
        population below, so the result is identical to a full recompute over
        the same reordered recovered_indices.
        """
        from processing.extract_single_pulse_features import (
            _compute_zone, _unpack_tp_zone, _flag_shape_outliers)
        from utils.tpsettings_io import get_zone
        old = self._compute
        _old_rec = np.asarray(old.get("pulse_indices",
                              np.zeros((0, 2), int))).reshape(-1, 2)
        _prev_acc = {(int(a), int(b)): int(self.acceptance[i])
                     for i, (a, b) in enumerate(_old_rec)
                     if i < len(self.acceptance)}
        meas = np.asarray(self._cols[self._measurement_zone], dtype=float)
        tparams = _unpack_tp_zone(get_zone(self._tp, self._measurement_zone))
        partial = _compute_zone(
            meas, self._fs, self._trendline,
            np.asarray(new_spans, dtype=int).reshape(-1, 2), **tparams)

        new = dict(partial)  # inherit global/template keys (data, template_seq…)
        for key in self._PER_PULSE_KEYS:
            o, p = old.get(key), partial.get(key)
            new[key] = [(o[i] if kind == "old" else p[i]) for kind, i in origins]
        so = np.asarray(old.get("sequence_matches", []))
        sp = np.asarray(partial.get("sequence_matches", []))
        new["sequence_matches"] = np.array(
            [(so[i] if kind == "old" else sp[i]) for kind, i in origins], dtype=int)

        rec = np.asarray(self._decomp["recovered_indices"], dtype=int).reshape(-1, 2)
        new["pulse_indices"] = rec
        # pulse_start_indices is derived from rec, not spliced — keep it out of
        # _PER_PULSE_KEYS and set it explicitly (partial only has the new spans').
        new["pulse_start_indices"] = (rec[:, 0].copy() if len(rec)
                                      else np.array([], dtype=int))
        new["trendline"] = self._trendline
        # Re-derive the cross-pulse shape-outlier warnings on the MERGED
        # population: spliced values reflect the old/partial populations, which
        # can differ. Clear stale warnings (2 -> 1), then flag against all pulses.
        seq = new["sequence_matches"]
        seq[seq == 2] = 1
        _flag_shape_outliers(seq, new["pulse_peak_locs"], rec, new.get("template_seq"))

        self._compute = new
        self._n_particles = len(new["pulse_peak_locs"])
        self.acceptance = np.array(
            [_prev_acc.get((int(a), int(b)), int(1 if seq[i] >= 1 else 0))
             for i, (a, b) in enumerate(rec)], dtype=int)
        self._refresh_events_and_pages()
        self._refresh_peaks(only_events)
        self._render_page()

    def set_particle_window(self, particle, start, end):
        rec = np.asarray(self._decomp["recovered_indices"], dtype=int).reshape(-1, 2)
        particle = int(particle)
        if not (0 <= particle < len(rec)):
            return
        start, end = int(start), int(end)
        if end <= start:  # keep width >= 1 so _compute_zone doesn't choke
            end = start + 1
        rec[particle] = [start, end]
        self._decomp["recovered_indices"] = rec
        # Only this particle's compute is fresh; peak re-detection is scoped
        # to its event so the other events keep their manual edits.
        src = self._decomp.get("source_event", [])
        ev = int(src[particle]) if particle < len(src) else None
        origins = [("new", 0) if i == particle else ("old", i)
                   for i in range(len(rec))]
        self._rebuild_compute_incremental(
            [[start, end]], origins,
            only_events=None if ev is None else {ev})

    def set_event_n(self, event, n):
        # Re-split ONE source event into n even sub-spans across its current extent.
        n = max(1, int(n))
        rec = np.asarray(self._decomp["recovered_indices"], dtype=int).reshape(-1, 2)
        src = list(self._decomp["source_event"])
        mode = list(self._decomp["event_mode"])
        keep = [i for i in range(len(src)) if src[i] != event]
        ev_rows = [i for i in range(len(src)) if src[i] == event]
        if not ev_rows:
            return
        lo = int(rec[ev_rows, 0].min()); hi = int(rec[ev_rows, 1].max())
        # Clamp N so sub-spans stay distinct (no overlap/dupes past extent width).
        n = max(1, min(int(n), max(1, hi - lo)))
        if hi <= lo:  # degenerate extent -> guarantee n distinct width-1 spans
            hi = lo + n
        edges = np.linspace(lo, hi, n + 1).astype(int)
        new_spans = [[int(edges[k]),
                      int(max(edges[k + 1], edges[k] + 1))] for k in range(n)]
        # Pre-order layout: kept particles (unchanged) then this event's N new
        # spans. `origin` tags each slot so the compute can reuse cached
        # per-pulse results for kept particles and only run the new spans.
        pre_rec = [list(rec[i]) for i in keep] + new_spans
        pre_src = [src[i] for i in keep] + [event] * n
        pre_mode = [mode[i] for i in keep] + ["manual"] * n
        pre_origin = [("old", i) for i in keep] + [("new", k) for k in range(n)]
        # re-sort by (source_event, start) so ordering stays stable
        order = sorted(range(len(pre_src)),
                       key=lambda i: (pre_src[i], pre_rec[i][0]))
        self._decomp["recovered_indices"] = np.asarray([pre_rec[i] for i in order], int).reshape(-1, 2)
        self._decomp["source_event"] = [pre_src[i] for i in order]
        self._decomp["event_n"] = [self._decomp["source_event"].count(s)
                                   for s in self._decomp["source_event"]]
        self._decomp["event_mode"] = [pre_mode[i] for i in order]
        self._event_done.discard(int(event))  # its particles changed: re-review
        origins = [pre_origin[i] for i in order]
        # Kept particles change index in the re-sorted layout: remap the
        # pid-keyed edit state so it stays with its particle (the edited
        # event's entries drop — its rows are fresh).
        remap = {i: new_i for new_i, (kind, i) in enumerate(origins)
                 if kind == "old"}
        self._tpl_edit = {remap[p]: ed for p, ed in self._tpl_edit.items()
                          if p in remap}
        self._tpl_locks = {remap[p]: lk for p, lk in self._tpl_locks.items()
                           if p in remap}
        self._active_tpl = remap.get(self._active_tpl)
        if self._autofit_tpl != "sum":
            self._autofit_tpl = remap.get(self._autofit_tpl)
        self._rebuild_compute_incremental(new_spans, origins,
                                          only_events={int(event)})

    # ---- acceptance controls ------------------------------------------------

    def _rebuild_accept_boxes(self, particle_ids):
        """Rebuild the per-particle accept checkboxes (vertical, right panel).
        Labels use within-event numbering (each event has its own P0, P1…)."""
        while self._accept_lay.count():
            item = self._accept_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._accept_boxes = []
        for k, pid in enumerate(particle_ids):
            chk = QCheckBox(f"P{k}   accept")
            chk.setChecked(pid < len(self.acceptance) and int(self.acceptance[pid]) >= 1)
            chk.toggled.connect(
                lambda checked, p=pid: self._on_accept_toggled(p, checked))
            self._accept_lay.addWidget(chk)
            self._accept_boxes.append(chk)
        # Hug the checkbox list (the scroll area otherwise reserves its full
        # cap even for 1-2 particles); scroll only beyond ~7 rows. Summed from
        # the boxes' own hints: the layout's sizeHint reads 0 here because
        # widgets added to a visible parent stay pending-show until the event
        # loop runs, and pending-show items count as empty.
        n = len(self._accept_boxes)
        row_h = max((c.sizeHint().height() for c in self._accept_boxes),
                    default=0)
        content = n * row_h + max(0, n - 1) * self._accept_lay.spacing()
        self._accept_scroll.setFixedHeight(min(content, 180))

    def _on_accept_toggled(self, pid, checked):
        self.acceptance[pid] = 1 if checked else 0
        ax = self._raw_axes_map.get(pid)
        if ax is not None:
            edge = "#2ca02c" if checked else "#d62728"
            for spine in ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(2.0)
            self._canvas.draw_idle()

    # ---- diff thresholds + peak marks ---------------------------------------

    def _draw_thresholds(self, ax, role):
        """Draggable upper/lower threshold lines on one diff axis."""
        u, l = self._thr[role]
        self._thr_lines[role] = (
            ax.axhline(u, color="#2ca02c", linewidth=1.0, linestyle="--",
                       alpha=0.8),
            ax.axhline(l, color="#d62728", linewidth=1.0, linestyle="--",
                       alpha=0.8))

    def _set_diff_ylim(self, ax, d, thr=None):
        """Diff-data y range with a 5% margin, spanning the threshold pair
        (both values, both sides: a line dragged past the data range must
        stay visible/grabbable)."""
        lo, hi = float(np.min(d)), float(np.max(d))
        if thr is not None:
            lo = min(lo, float(thr[0]), float(thr[1]))
            hi = max(hi, float(thr[0]), float(thr[1]))
        rng = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.05 * rng, hi + 0.05 * rng)

    def _update_event_peaks(self):
        """Mark peak locations on the signal plot as short notches crossing each
        role zone's displayed trace (green = rising edge, red = falling). The
        MEASUREMENT notches follow the actual (editable) peak set so manual
        add/remove in the diff plot shows here too; start/end use their
        threshold-detected diff peaks."""
        from scipy.signal import find_peaks
        ax = self._ax_event_raw
        if ax is None or self._ev_window is None or not self._events:
            return
        _, meas_pids = self._events[min(self._page, len(self._events) - 1)]
        pk_all = self._compute.get("pulse_peak_locs", [])
        for art in self._peak_marks:
            try:
                art.remove()
            except Exception:
                pass
        self._peak_marks = []
        ev_s, ev_e = self._ev_window
        y0, y1 = ax.get_ylim()
        h = 0.06 * (y1 - y0)  # notch half-height: crosses just the trace
        dist = self._peak_distance()
        for role, zi in (("start", self._start_zone),
                         ("meas", self._measurement_zone),
                         ("end", self._end_zone)):
            shift = self._zone_shifts.get(zi)
            if zi is None or shift is None:
                continue
            act = self._zone_lp(zi, ev_s, ev_e)
            if len(act) <= 2:
                continue
            d = np.diff(act)
            pair = self._thr.get(role)
            if pair is None:  # e.g. an LPF change cleared it; seed so the marks
                pair = [0.5 * float(np.max(d)), 0.5 * float(np.min(d))]
                self._thr[role] = pair  # (and the diff plot picks up the same)
            t = (np.arange(len(act)) + ev_s) / self._fs
            if role == "meas":
                # The editable peak set, split by diff sign (rise/fall).
                locs = set()
                for pid in meas_pids:
                    if pid < len(pk_all) and pk_all[pid] is not None:
                        for g in np.asarray(pk_all[pid], dtype=int):
                            lo = int(g) - ev_s
                            if 0 <= lo < len(d):
                                locs.add(lo)
                locs = np.array(sorted(locs), dtype=int)
                if len(locs):
                    pos, neg = locs[d[locs] > 0], locs[d[locs] <= 0]
                else:
                    pos = neg = np.array([], dtype=int)
            else:
                pos, _ = find_peaks(d, height=pair[0], distance=dist)
                neg, _ = find_peaks(-d, height=-pair[1], distance=dist)
            y = act + shift
            for idx, color in ((pos, "#2ca02c"), (neg, "#d62728")):
                if len(idx):
                    self._peak_marks.append(
                        ax.vlines(t[idx], y[idx] - h, y[idx] + h,
                                  colors=color, linewidth=1.2, alpha=0.9,
                                  zorder=5))
        # The signal axis may live on the meas adjust panel's canvas.
        ax.figure.canvas.draw_idle()

    def _thr_hit(self, event):
        """(role, which) of a threshold line within grab range of a press on
        a diff axis; None otherwise."""
        role = self._diff_ax_role.get(event.inaxes)
        if role is None or role not in self._thr_lines:
            return None
        trans = event.inaxes.transData
        best = None
        for name, val in zip(("upper", "lower"), self._thr[role]):
            py = trans.transform((0.0, float(val)))[1]
            dpx = abs(event.y - py)
            if dpx <= 6 and (best is None or dpx < best[0]):
                best = (dpx, name)
        return (role, best[1]) if best else None

    def _on_canvas_motion(self, event):
        if self._drag_seg is not None:
            self._drag_template(event)
            return
        if self._drag_thr is not None:
            if event.ydata is None:
                return
            role, which = self._drag_thr
            if (self._diff_ax_role.get(event.inaxes) != role
                    or role not in self._thr_lines):
                return
            v = float(event.ydata)
            idx = 0 if which == "upper" else 1
            self._thr[role][idx] = v
            self._thr_lines[role][idx].set_ydata([v, v])
            # The dragged line may live on the Indicator Thresholds panel's
            # canvas rather than the active main/editor one.
            event.canvas.draw_idle()
            return
        # Peaks mode: a selection circle follows the cursor over the meas diff.
        if (self._editing and self._btn_peak_adjust.isChecked()
                and event.inaxes is self._ax_event_diff
                and event.xdata is not None and event.ydata is not None):
            self._update_peak_cursor(event)
        else:
            self._hide_peak_cursor()

    def _on_canvas_release(self, event):
        if self._drag_seg is not None:
            pid, _, _, was_new = self._drag_seg
            self._drag_seg = None
            # A click without an actual drag shouldn't commit an override.
            if was_new and not self._drag_moved:
                self._tpl_edit.pop(pid, None)
            # Redraw the surface the drag happened on.
            if self._enlarge_panel is not None:
                self._render_enlarge()
            else:
                self._render_page()
            return
        if self._drag_thr is not None:
            role, _ = self._drag_thr
            self._drag_thr = None
            if role == "meas":  # measurement pair drives the particle peaks
                self._refresh_peaks()
            self._render_meas_surface()

    # ---- per-particle peak editing (see Pulse Processing) -------------------

    _PEAK_DOT_MS = 8.0   # representative peak-marker size (points)

    _PEAK_CURSOR_SCALE = 1.1  # circle diameter as a multiple of the peak dot

    def _peak_cursor_radius_px(self):
        """Selection/circle radius in display (physical) pixels = 1.1× the peak
        dot size. points_to_pixels matches the event/transData space, so it is
        correct on HiDPI too."""
        r_pts = 0.5 * self._PEAK_CURSOR_SCALE * self._PEAK_DOT_MS  # radius, pts
        try:
            return float(self._figure.canvas.get_renderer()
                         .points_to_pixels(r_pts))
        except Exception:
            return r_pts * self._figure.dpi / 72.0

    def _nearest_meas_peak(self, event, r_px):
        """(pid, global_idx, dist_px) of the single measurement-diff peak
        nearest the cursor within `r_px`, or None."""
        if self._ev_window is None or not self._events:
            return None
        ev_s, ev_e = self._ev_window
        act = self._zone_lp(self._measurement_zone, ev_s, ev_e)
        if len(act) < 2:
            return None
        d = np.diff(act)
        t = (np.arange(len(d)) + ev_s) / self._fs
        ax = event.inaxes
        cursor = np.asarray(ax.transData.transform((event.xdata, event.ydata)))
        _, particle_ids = self._events[min(self._page, len(self._events) - 1)]
        best = None  # (pid, global idx, dist_px)
        for pid in particle_ids:
            for g in np.asarray(self._compute["pulse_peak_locs"][pid], int):
                loc = int(g) - ev_s
                if not (0 <= loc < len(d)):
                    continue
                px = np.asarray(ax.transData.transform((t[loc], d[loc])))
                dist = float(np.hypot(*(cursor - px)))
                if dist <= r_px and (best is None or dist < best[2]):
                    best = (pid, int(g), dist)
        return best

    def _ensure_peak_cursor(self, canvas):
        from processing.extract_single_pulse_features import _CircleOverlay
        if self._peak_cursor is None:
            self._peak_cursor = _CircleOverlay(canvas)
        elif self._peak_cursor.parent() is not canvas:
            self._peak_cursor.setParent(canvas)
        return self._peak_cursor

    def _update_peak_cursor(self, event):
        """Move the cursor circle over the Measurement diff; red when a single
        peak is inside its radius."""
        canvas = event.canvas
        circ = self._ensure_peak_cursor(canvas)
        r_px = self._peak_cursor_radius_px()
        ratio = canvas.devicePixelRatioF() or 1.0
        qx = int(event.x / ratio)
        qy = int(canvas.height() - event.y / ratio)
        has_peak = self._nearest_meas_peak(event, r_px) is not None
        circ.set_pos(qx, qy, has_peak, draw_radius=int(round(r_px / ratio)))
        circ.show()
        circ.raise_()

    def _hide_peak_cursor(self):
        if self._peak_cursor is not None:
            self._peak_cursor.hide()

    def _handle_peak_click(self, event):
        """Peak-adjust click on the measurement diff: open the peak palette
        for the single peak inside the cursor radius; on empty space, add the
        strongest-|diff| sample as a regular peak (● of its containing pulse)
        and open the palette for it."""
        self._close_peak_palette()
        ax = event.inaxes
        r = self._peak_cursor_radius_px()
        ev_s, ev_e = self._ev_window
        act = self._zone_lp(self._measurement_zone, ev_s, ev_e)
        if len(act) < 2 or not self._events:
            return
        d = np.diff(act)
        t = (np.arange(len(d)) + ev_s) / self._fs
        _, particle_ids = self._events[min(self._page, len(self._events) - 1)]
        pidx = np.atleast_2d(np.asarray(self._compute["pulse_indices"], int))
        cursor = np.asarray(ax.transData.transform((event.xdata, event.ydata)))
        # matplotlib event coords are physical pixels; Qt move() wants logical.
        canvas = event.canvas
        ratio = canvas.devicePixelRatioF() or 1.0
        qx = int(event.x / ratio)
        qy = int(canvas.height() - event.y / ratio)

        # Single existing peak inside the radius -> palette for it.
        best = self._nearest_meas_peak(event, r)
        if best is not None:
            self._open_peak_palette(best[0], best[1], qx, qy, canvas)
            return

        # Empty space -> add the strongest |diff| sample within the radius.
        inv = ax.transData.inverted()
        lo_x = inv.transform((cursor[0] - r, cursor[1]))[0]
        hi_x = inv.transform((cursor[0] + r, cursor[1]))[0]
        lo = max(0, int(round(lo_x * self._fs)) - ev_s)
        hi = min(len(d), int(round(hi_x * self._fs)) - ev_s + 1)
        best_loc, best_abs = None, -1.0
        for ci in range(lo, hi):
            px = np.asarray(ax.transData.transform((t[ci], d[ci])))
            if np.hypot(*(cursor - px)) <= r and abs(float(d[ci])) > best_abs:
                best_abs = abs(float(d[ci]))
                best_loc = ci
        if best_loc is None:
            return
        g = best_loc + ev_s
        # Containing pulse, else the nearest by span midpoint.
        pid = None
        for p in particle_ids:
            if int(pidx[p, 0]) <= g <= int(pidx[p, 1]):
                pid = p
                break
        if pid is None:
            pid = min(particle_ids,
                      key=lambda p: abs(g - int(pidx[p].mean())))
        peaks = np.asarray(self._compute["pulse_peak_locs"][pid], int)
        if g not in peaks:
            self._compute["pulse_peak_locs"][pid] = np.sort(np.append(peaks, g))
            self._recompute_rect(pid)
            self._check_sequence(pid)
            self._render_meas_surface()
        self._open_peak_palette(pid, g, qx, qy, canvas)

    def _after_peak_edit(self, pid):
        self._recompute_rect(pid)
        self._check_sequence(pid)
        self._render_meas_surface()

    # ---- peak palette (per-peak pulse / role / remove) ---------------------

    def _close_peak_palette(self):
        if self._peak_palette is not None:
            self._peak_palette.setParent(None)
            self._peak_palette.deleteLater()
            self._peak_palette = None

    _PALETTE_ROLES = (("start", "▶"), ("end", "◀"), ("max", "▲"),
                      ("min", "▼"), ("regular", "●"))

    def _open_peak_palette(self, pid, gidx, qx, qy, canvas=None):
        """Floating palette over the click: one row per particle (coloured by
        pulse), each with the 5 role labels — clicking sets the peak's pulse
        AND role in one action. A Remove row spans the bottom."""
        self._close_peak_palette()
        canvas = canvas or self._canvas
        _, particle_ids = self._events[min(self._page, len(self._events) - 1)]
        fr = QFrame(canvas)
        fr.setObjectName("peakPalette")
        fr.setStyleSheet(
            "#peakPalette{background:rgba(255,255,255,244);"
            "border:1px solid #9aa;border-radius:6px;}")
        lay = QGridLayout(fr)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setHorizontalSpacing(3)
        lay.setVerticalSpacing(3)
        ncol = len(self._PALETTE_ROLES)
        for k, p in enumerate(particle_ids):
            color = _EVENT_PALETTE[k % len(_EVENT_PALETTE)]
            for c, (role, sym) in enumerate(self._PALETTE_ROLES):
                b = QPushButton(sym)
                b.setFixedSize(38, 30)
                # Explicit font-size so the ▶◀▲▼● glyphs aren't clipped; no
                # bold (these symbols lack a bold face and render ragged).
                b.setStyleSheet(
                    f"QPushButton{{color:{color};border:1.4px solid {color};"
                    f"border-radius:3px;background:white;font-size:16px;"
                    f"padding:0px;}}"
                    f"QPushButton:hover{{background:{color};color:white;}}")
                b.clicked.connect(
                    lambda _=0, tp=p, rr=role: self._palette_set(gidx, tp, rr))
                lay.addWidget(b, k, c)
        b = QPushButton("Remove")
        b.setFixedHeight(24)
        b.clicked.connect(lambda: self._palette_remove(pid, gidx))
        lay.addWidget(b, len(particle_ids), 0, 1, ncol)
        fr.adjustSize()
        # Centered just below the click; keep fully on the canvas.
        cw, ch = canvas.width(), canvas.height()
        x = min(max(0, qx - fr.width() // 2), max(0, cw - fr.width()))
        y = qy + 8
        if y + fr.height() > ch:      # flip above the click if it would clip
            y = qy - fr.height() - 8
        y = min(max(0, y), max(0, ch - fr.height()))
        fr.move(x, y)
        fr.show()
        self._peak_palette = fr

    def _palette_set(self, gidx, target_pid, role):
        """Assign the peak to `target_pid` (reassigning + recolouring if it
        currently belongs to another pulse) and set its role, enforcing one
        start/end/max/min per pulse."""
        self._close_peak_palette()
        gidx, target_pid = int(gidx), int(target_pid)
        cur = None
        for p in range(self._n_particles):
            if gidx in np.asarray(self._compute["pulse_peak_locs"][p], int):
                cur = p
                break
        if cur is not None and cur != target_pid:
            self._reassign_peak(gidx, cur, target_pid)  # pops role, re-checks
        elif cur is None:
            arr = np.asarray(self._compute["pulse_peak_locs"][target_pid], int)
            self._compute["pulse_peak_locs"][target_pid] = np.sort(
                np.append(arr, gidx))
            self._recompute_rect(target_pid)
            self._check_sequence(target_pid)
        if role == "regular":
            self._peak_roles.pop(gidx, None)
        else:
            pulse = {int(x) for x in
                     np.asarray(self._compute["pulse_peak_locs"][target_pid], int)}
            for g, r in list(self._peak_roles.items()):
                if r == role and g in pulse and g != gidx:
                    self._peak_roles.pop(g)
            self._peak_roles[gidx] = role
        self._render_meas_surface()

    def _palette_remove(self, pid, gidx):
        self._close_peak_palette()
        a = np.asarray(self._compute["pulse_peak_locs"][pid], int)
        self._compute["pulse_peak_locs"][pid] = a[a != int(gidx)]
        self._peak_roles.pop(int(gidx), None)
        kp = self._compute.get("pulse_key_peaks_list")
        if kp is not None and pid < len(kp) and kp[pid]:
            pidx = np.atleast_2d(np.asarray(self._compute["pulse_indices"], int))
            local = int(gidx) - int(pidx[pid, 0])
            kp[pid] = [int(x) for x in kp[pid] if int(x) != local]
        self._after_peak_edit(pid)

    def _reassign_peak(self, gidx, from_pid, to_pid):
        """Move a peak from one pulse to another (recolors + re-checks both).
        The moved peak drops its role (becomes a dot in its new pulse)."""
        a = np.asarray(self._compute["pulse_peak_locs"][from_pid], int)
        self._compute["pulse_peak_locs"][from_pid] = a[a != gidx]
        b = np.asarray(self._compute["pulse_peak_locs"][to_pid], int)
        if gidx not in b:
            self._compute["pulse_peak_locs"][to_pid] = np.sort(np.append(b, gidx))
        self._peak_roles.pop(int(gidx), None)
        for p in (from_pid, to_pid):
            self._recompute_rect(p)
            self._check_sequence(p)

    def _recompute_rect(self, pid):
        """Rebuild the particle's rect (median + mean per inter-peak segment)
        from the raw signal at the current peaks, as in Pulse Processing."""
        s, e = (int(x) for x in self._compute["pulse_indices"][pid])
        seg = np.asarray(self._compute["data"], dtype=float)[s:e + 1]
        if not len(seg):
            return
        peaks = np.asarray(self._compute["pulse_peak_locs"][pid], int) - s
        peaks = peaks[(peaks > 0) & (peaks < len(seg))]
        bounds = [0] + sorted(int(p) for p in peaks) + [len(seg)]
        rect = np.empty(len(seg))
        rect_m = np.empty(len(seg))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                rect[a:b] = float(np.median(seg[a:b]))
                rect_m[a:b] = float(np.mean(seg[a:b]))
        self._compute["rect_pulses"][pid] = rect
        self._compute["rect_pulses_mean"][pid] = rect_m

    def _check_sequence(self, pid):
        """Sign sequence at the current peaks vs the template: 1 = match,
        2 = warning (mirrors Pulse Processing's _check_sequence)."""
        tseq = self._compute.get("template_seq")
        seqm = self._compute.get("sequence_matches")
        dseg = self._compute.get("pulse_diffs", [None] * (pid + 1))[pid]
        if tseq is None or seqm is None or dseg is None or pid >= len(seqm):
            return
        s = int(self._compute["pulse_indices"][pid, 0])
        peaks = np.asarray(self._compute["pulse_peak_locs"][pid], int) - s
        peaks = peaks[(peaks >= 0) & (peaks < len(dseg))]
        signs = [1 if float(dseg[p]) >= 0 else 0 for p in peaks]
        seqm[pid] = 1 if signs == [int(v) for v in tseq] else 2

    def _on_canvas_click(self, event):
        """Editor: press near a threshold line to drag it, or (in adjust mode)
        click the measurement diff to edit peaks. Main view: clicking any left
        plot opens the Original Data editor."""
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        # Main view — any left plot (signal, a diff, or the warped-template
        # row) opens the editor.
        if not self._editing and (event.inaxes is self._ax_event_raw
                                  or event.inaxes is self._ax_warped
                                  or event.inaxes in self._diff_ax_role):
            self._open_original_editor()
            return
        # Editor — grab the active template's level/edge on the warped row.
        # A double-click on a segment's move handle toggles that segment's
        # lock instead of dragging.
        if self._editing and event.inaxes is self._ax_warped:
            if event.dblclick:
                i = self._lock_hit(event)
                if i is not None:
                    locks = self._tpl_locks.setdefault(self._active_tpl, set())
                    locks.symmetric_difference_update({i})
                    self._drag_seg = None  # cancel the grab from the first press
                    self._redraw_warped_axis()
                return
            hit = self._tpl_hit(event)
            if hit is not None:
                pid, kind, i = hit
                was_new = self._ensure_tpl_edit(pid)
                self._drag_seg = (pid, kind, i, was_new)
                self._drag_moved = False
            return
        if event.inaxes in self._diff_ax_role:
            role = self._diff_ax_role[event.inaxes]
            if (self._btn_peak_adjust.isChecked() and role == "meas"
                    and event.ydata is not None):
                self._handle_peak_click(event)
                return
            # The meas-diff threshold moves only while the Thres toggle is on;
            # the indicator panel's start/end diffs are always adjustable.
            if role != "meas" or self._btn_thr_adjust.isChecked():
                self._drag_thr = self._thr_hit(event)
            return
        particle_ids = self._left_axes_map.get(event.inaxes)
        if not particle_ids:
            return
        rec = np.asarray(self._decomp["recovered_indices"], dtype=int).reshape(-1, 2)
        click_sample = int(round(event.xdata * self._fs))
        # Tolerance ~2% of the visible x-range, in samples (min a few samples).
        x0, x1 = event.inaxes.get_xlim()
        tol = max(3, int(round(abs(x1 - x0) * self._fs * 0.02)))
        best = None  # (dist, particle_id, which_edge)
        for pid in particle_ids:
            if pid >= len(rec):
                continue
            ps, pe = int(rec[pid, 0]), int(rec[pid, 1])
            for edge, val in (("start", ps), ("end", pe)):
                d = abs(click_sample - val)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, pid, edge)
        if best is None:
            return
        _, pid, edge = best
        ps, pe = int(rec[pid, 0]), int(rec[pid, 1])
        if edge == "start":
            self.set_particle_window(particle=pid, start=click_sample, end=pe)
        else:
            self.set_particle_window(particle=pid, start=ps, end=click_sample)

    # ---- rendering ----------------------------------------------------------

    def _render_page(self):
        self._figure.clear()
        self._left_axes_map = {}
        self._raw_axes_map = {}
        self._ax_event_raw = None
        self._ax_event_diff = None
        self._ax_warped = None
        self._tpl_seed_cache = {}
        self._tpl_active_artists = []
        self._raw_rect = None
        self._ev_window = None
        self._peak_marks = []
        self._diff_axes = []
        self._thr_lines = {}     # dead after figure.clear(); redrawn (or not,
        self._diff_ax_role = {}  # for degenerate windows) by the diff render
        self._zone_shifts = {}
        events = self._events
        n_events = len(events)

        self._update_progress_ui()  # page label + Complete toggle + Finish gate
        self._lbl_template.setText(self._format_template_seq())
        self._btn_prev.setEnabled(self._page > 0)
        self._btn_next.setEnabled(self._page < n_events - 1)

        if not events:
            self._canvas.draw_idle()
            return

        self._page = min(self._page, n_events - 1)
        event_id, particle_ids = events[self._page]

        # Sync the N spinbox to this event without firing set_event_n.
        self._spin_n.blockSignals(True)
        self._spin_n.setValue(max(1, len(particle_ids)))
        self._spin_n.blockSignals(False)
        if self._editing:
            # Active/auto-fit pulses are picked in the adjust panel; drop
            # selections that left the current event.
            if self._active_tpl not in particle_ids:
                self._active_tpl = None
            if self._autofit_tpl not in particle_ids:
                self._autofit_tpl = None
            self._update_tpl_buttons()

        data = np.asarray(self._compute.get("data", []), dtype=float)
        rec = np.asarray(self._decomp.get("recovered_indices", []), dtype=int)
        rec = rec.reshape(-1, 2) if rec.size else np.zeros((0, 2), int)

        # LEFT column, top to bottom: [rectangularized template — editor only]
        # · signal (all zones) · per-zone diffs · raw-measurement + warped
        # templates. Editor hides the start/end diffs (they live in the
        # Indicator Thresholds panel) so it carries the measurement diff only.
        # RIGHT column (main only): one decomposed-pulse plot per particle.
        n_specs = 1 + (0 if self._editing else
                       int(self._start_zone is not None)
                       + int(self._end_zone is not None))
        rows_left = int(self._editing) + 2 + n_specs
        # Editor shows the 4 left plots only; main also shows the decomposed
        # right column (one per particle).
        ncols = 1 if self._editing else 2
        rows = rows_left if self._editing else max(rows_left, len(particle_ids))
        # Fixed (not minimum) height so each row is exactly _ROW_PX tall: the
        # canvas never stretches to fill the viewport (which spread rows apart),
        # and the top signal row keeps a constant size as N changes.
        self._canvas.setFixedHeight(rows * self._ROW_PX)
        # Fixed-pixel top margin: a fraction would clip the row-0 titles at
        # the minimum canvas height.
        top = 1.0 - 30.0 / (rows * self._ROW_PX)
        gs = self._figure.add_gridspec(
            rows, ncols, hspace=0.45, wspace=0.22,
            left=0.09, right=0.975, top=top, bottom=0.04)

        self._render_left_column(gs, event_id, particle_ids, data, rec)
        if not self._editing:
            for k, pid in enumerate(particle_ids):
                self._render_particle_plot(gs, k, pid, data, rec)
            self._rebuild_accept_boxes(particle_ids)
        style_mpl_figure(self._figure)
        for ax in self._figure.axes:
            ax.tick_params(labelsize=6)
            ax.grid(False)  # style_mpl_figure enables it; keep these plots clean
            # Cap tick density: matplotlib text layout over the 2*(1+N) axes is
            # the dominant redraw cost, and tick labels are the bulk of it.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        # Apply acceptance frames AFTER style_mpl_figure, which resets spine
        # colors to the theme border — otherwise the green/red frame is lost.
        if not self._editing:
            self._apply_accept_frames()
        self._canvas.draw_idle()
        self._place_float_buttons()  # re-placed again after the actual draw
        # Re-register + redraw the panel diffs: the resets above dropped
        # their _diff_ax_role/_thr_lines entries.
        self._render_indicator_panel()

    # ---- Original Data editor (progressive disclosure) ---------------------

    def _connect_canvas(self, canvas):
        canvas.mpl_connect("button_press_event", self._on_canvas_click)
        canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        canvas.mpl_connect("button_release_event", self._on_canvas_release)
        # The dialog/editor is sized after the first render, and the tight-
        # layout engine moves the axes during each draw — reposition the
        # floating buttons on both signals so they track the true axis corner.
        canvas.mpl_connect("resize_event",
                           lambda e: self._place_float_buttons())
        canvas.mpl_connect("draw_event",
                           lambda e: self._place_float_buttons())

    def _open_original_editor(self):
        """Larger editing surface for the template, signal, measurement diff
        and raw-measurement + warped-template rows plus the Zones and Filters
        controls (start/end diffs open from the Indicator Thresholds button);
        Update commits changes to the main view, Cancel reverts. Rendering is
        retargeted to the editor's own canvas while it is open (the main canvas
        is hidden behind this modal)."""
        if self._editor is not None:
            return
        snapshot = self._get_state()

        dlg = QDialog(self)
        apply_dialog_style(dlg)
        dlg.setWindowTitle("Manual Edit")
        # Fit the left plots exactly (template + signal + meas-diff +
        # raw-measurement/templates = the editor's row count), capped at screen.
        h = 4 * self._ROW_PX + 48
        screen = dlg.screen() or self.screen()
        if screen is not None:
            h = min(h, screen.availableGeometry().height() - 40)
        dlg.resize(1200, h)
        root = QHBoxLayout(dlg)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        fig = Figure(figsize=(8, 8), tight_layout=False)
        style_mpl_figure(fig)
        canvas = FigureCanvas(fig)
        self._connect_canvas(canvas)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(canvas)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(self._MIN_ROWS_VISIBLE * self._ROW_PX)
        root.addWidget(scroll, stretch=1)

        panel = QWidget()
        panel.setFixedWidth(300)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)
        pv.addWidget(self._grp_zones)
        pv.addWidget(self._grp_filters)
        pv.addWidget(self._grp_display)
        pv.addStretch()
        row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_update = QPushButton("Update")
        btn_update.setObjectName("confirmBtn")
        btn_update.clicked.connect(dlg.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_update)
        pv.addLayout(row)
        root.addWidget(panel)

        # Retarget rendering to the editor canvas.
        self._main_figure, self._main_canvas = self._figure, self._canvas
        self._figure, self._canvas = fig, canvas
        self._editor = dlg
        self._editing = True
        # finished fires exactly once (Update=Accepted, Cancel / window-X =
        # Rejected) -> single, re-entrancy-free teardown.
        dlg.finished.connect(lambda result: self._close_editor(snapshot, result))
        self._render_page()
        dlg.exec()
        dlg.deleteLater()  # groups already detached in _close_editor

    def _close_editor(self, snapshot, result):
        if not self._editing:
            return
        self._close_peak_palette()  # child of the editor canvas being destroyed
        if self._ind_panel is not None:  # child of the editor dialog
            self._ind_panel.close()
        if self._enlarge_panel is not None:  # child of the editor dialog
            self._enlarge_panel.close()
        if self._meas_panel is not None:  # child of the editor dialog
            self._meas_panel.close()
        # Detach shared widgets parented to the editor canvas so they survive
        # its destruction (else their C++ objects are deleted with it).
        self._grp_zones.setParent(None)
        self._grp_filters.setParent(None)
        self._grp_display.setParent(None)
        self._btn_peak_adjust.setParent(None)
        self._btn_thr_adjust.setParent(None)
        self._btn_meas_adjust.setParent(None)
        self._btn_enlarge.setParent(None)
        # The cursor circle is a child of the editor canvas being destroyed;
        # drop it so it's recreated fresh next time.
        self._peak_cursor = None
        # Restore rendering to the main canvas before any re-render.
        self._figure, self._canvas = self._main_figure, self._main_canvas
        self._editing = False
        self._editor = None
        if result == QDialog.Accepted:
            self._render_page()          # commit: main reflects the live edits
        else:
            self._apply_state(snapshot)  # revert + re-render the main view

    # ---- Indicator Thresholds panel -----------------------------------------

    def _indicator_specs(self):
        """(title, zone, role) per assigned indicator zone."""
        specs = []
        if self._start_zone is not None:
            specs.append(("Start — Diff (filtered)", self._start_zone,
                          "start"))
        if self._end_zone is not None:
            specs.append(("End — Diff (filtered)", self._end_zone, "end"))
        return specs

    def _open_indicator_panel(self):
        """Modeless child panel with the start/end indicator diffs and their
        draggable threshold pairs; releases feed the signal-plot notches
        live. Closes with the editor, and its thresholds commit/revert with
        the editor's Update/Cancel."""
        if self._ind_panel is not None:
            self._ind_panel.raise_()
            self._ind_panel.activateWindow()
            return
        specs = self._indicator_specs()
        if not specs:
            return
        dlg = QDialog(self._editor or self)
        apply_dialog_style(dlg)
        dlg.setWindowTitle("Indicator Thresholds")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(8, 8, 8, 8)
        fig = Figure(figsize=(7, 4), tight_layout=False)
        style_mpl_figure(fig)
        canvas = FigureCanvas(fig)
        self._connect_canvas(canvas)
        lay.addWidget(canvas)
        self._ind_panel = dlg
        self._ind_figure = fig
        self._ind_canvas = canvas
        dlg.finished.connect(self._on_indicator_panel_closed)
        dlg.resize(900, len(specs) * self._ROW_PX + 32)
        self._render_indicator_panel()
        dlg.show()

    def _render_indicator_panel(self):
        """(Re)draw the panel's diff axes. The _diff_ax_role/_thr_lines
        registrations are per-render — _render_page resets both, so it
        re-invokes this after every main/editor redraw."""
        if self._ind_panel is None:
            return
        fig = self._ind_figure
        fig.clear()
        specs = self._indicator_specs()
        if not specs:
            self._ind_canvas.draw_idle()
            return
        data = np.asarray(self._compute.get("data", []), dtype=float)
        gs = fig.add_gridspec(
            len(specs), 1, hspace=0.45, left=0.09, right=0.975,
            top=1.0 - 30.0 / (len(specs) * self._ROW_PX), bottom=0.08)
        for row, (title, zi, role) in enumerate(specs):
            ax = fig.add_subplot(gs[row, 0])
            if len(data) and self._ev_window is not None:
                self._draw_zone_diff(ax, zi, title, role)
            else:
                ax.set_title(title, fontsize=9, loc="left")
            self._diff_ax_role[ax] = role
        style_mpl_figure(fig)
        for ax in fig.axes:
            ax.tick_params(labelsize=6)
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        self._ind_canvas.draw_idle()

    def _on_indicator_panel_closed(self, *a):
        """Drop the panel's axis/line registrations so a later render can't
        route threshold drags to dead artists."""
        if self._ind_panel is None:
            return
        for ax in self._ind_figure.axes:
            self._diff_ax_role.pop(ax, None)
        for role in ("start", "end"):
            self._thr_lines.pop(role, None)
        dlg = self._ind_panel
        self._ind_panel = self._ind_figure = self._ind_canvas = None
        dlg.deleteLater()

    # ---- Enlarged plot-#4 adjustment panel ---------------------------------

    def _open_enlarge_panel(self):
        """Modal child panel showing the Raw-Measurement + Templates plot on a
        large canvas for comfortable dragging, with the active/auto-fit pulse
        radio groups to its right and a segment-lock toggle floated over the
        plot. Drag targeting is retargeted to this panel's axis while it is
        open; the editor re-renders (reflecting edits) on close."""
        if self._enlarge_panel is not None or not self._editing:
            return
        particle_ids = (self._events[min(self._page, len(self._events) - 1)][1]
                        if self._events else [])
        # Radios have no "none": default the active pulse to the first one.
        if self._active_tpl not in particle_ids and particle_ids:
            self._active_tpl = int(particle_ids[0])
        if self._autofit_tpl == self._active_tpl:
            self._autofit_tpl = None

        dlg = QDialog(self._editor or self)
        apply_dialog_style(dlg)
        dlg.setWindowTitle("Raw Measurement + Templates")
        dlg.resize(1250, 520)
        root = QHBoxLayout(dlg)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        fig = Figure(figsize=(10, 4), tight_layout=False)
        style_mpl_figure(fig)
        canvas = FigureCanvas(fig)
        self._connect_canvas(canvas)
        root.addWidget(canvas, stretch=1)

        panel = QWidget()
        panel.setFixedWidth(150)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)
        # Radio labels use within-event numbering (each event has its own
        # P0, P1, …, matching the palette order); values stay global pids.
        grp_a = QGroupBox("Active")
        va = QVBoxLayout(grp_a)
        grp_active = QButtonGroup(dlg)
        self._active_rbs = {}
        for k, pid in enumerate(particle_ids):
            rb = QRadioButton(f"P{k}")
            rb.setChecked(int(pid) == self._active_tpl)
            rb.toggled.connect(
                lambda on, p=int(pid): self._on_active_rb(p) if on else None)
            grp_active.addButton(rb)
            va.addWidget(rb)
            self._active_rbs[int(pid)] = rb
        pv.addWidget(grp_a)
        grp_f = QGroupBox("Auto-fit")
        vf = QVBoxLayout(grp_f)
        grp_autofit = QButtonGroup(dlg)
        self._autofit_rbs = {}
        for val, label in ([(None, "none")]
                           + [(int(p), f"P{k}")
                              for k, p in enumerate(particle_ids)]):
            rb = QRadioButton(label)
            rb.setChecked(val == self._autofit_tpl)
            rb.setEnabled(val is None or val != self._active_tpl)
            rb.toggled.connect(
                lambda on, v=val: self._on_autofit_rb(v) if on else None)
            grp_autofit.addButton(rb)
            vf.addWidget(rb)
            self._autofit_rbs[val] = rb
        pv.addWidget(grp_f)
        btn_regen = QPushButton("Regenerate")
        btn_regen.setToolTip(
            "Reset this event's templates to freshly warped seeds (anchors "
            "realigned to the key peaks); discards this event's manual "
            "template edits and locks")
        btn_regen.clicked.connect(self._regenerate_templates)
        pv.addWidget(btn_regen)
        self._btn_enlarge_snap = QPushButton("Snap Edges")
        self._btn_enlarge_snap.setToolTip(
            "Move each of the active pulse's transitions (vertical edges) to "
            "the nearest raw-rect edge, and each segment level onto the "
            "raw-rect step it mostly overlaps")
        self._btn_enlarge_snap.clicked.connect(self._snap_active_to_raw)
        self._btn_enlarge_snap.setEnabled(isinstance(self._active_tpl, int))
        pv.addWidget(self._btn_enlarge_snap)
        pv.addWidget(self._build_zoom_group())
        hint = QLabel("Drag the active pulse's segments (snap to the raw "
                      "rect). With Auto-fit set, that pulse absorbs the change "
                      "so sum = ΣPᵢ (locked to the raw rect); edges lock to "
                      "the peaks. Double-click one of the active pulse's "
                      "segment handles to pin its level (◆): pinned segments "
                      "hold during auto-fit and can't be dragged.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        pv.addWidget(hint)
        pv.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        pv.addWidget(btn_close)
        root.addWidget(panel)

        self._enlarge_panel = dlg
        self._enlarge_fig = fig
        self._enlarge_canvas = canvas
        self._btn_enlarge.hide()
        dlg.finished.connect(self._on_enlarge_closed)
        self._render_enlarge()
        dlg.exec()
        dlg.deleteLater()

    def _render_enlarge(self):
        """Draw plot #4 onto the enlarge panel's canvas; retarget drag hit-
        testing/redraw to its axis."""
        if self._enlarge_panel is None:
            return
        fig = self._enlarge_fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._enlarge_ax = ax
        self._ax_warped = ax  # drag hit-testing + active redraw target
        if not self._events:
            self._enlarge_canvas.draw_idle()
            return
        event_id, particle_ids = self._events[
            min(self._page, len(self._events) - 1)]
        data = np.asarray(self._compute.get("data", []), dtype=float)
        rec = np.asarray(self._decomp.get("recovered_indices", []), dtype=int)
        rec = rec.reshape(-1, 2) if rec.size else np.zeros((0, 2), int)
        ev_s, ev_e = self._event_window(event_id, particle_ids, data, rec)
        self._ev_window = (ev_s, ev_e)
        if len(data):
            self._draw_warped_templates(ax, particle_ids)
        else:
            ax.set_title("Raw Measurement + Templates", fontsize=9, loc="left")
        style_mpl_figure(fig)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        # Remember the auto-scaled view as "home" so navigation can restore it.
        self._enlarge_home = (ax.get_xlim(), ax.get_ylim())
        self._enlarge_canvas.draw_idle()

    def _build_zoom_group(self):
        """Zoom in / out / home controls for the enlarged warped-template
        axis (a compact row in the panel)."""
        g = QGroupBox("Navigation")
        lay = QHBoxLayout(g)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        for label, tip, fn in (
                ("−", "Zoom out", lambda: self._nav_zoom(1.4)),
                ("+", "Zoom in", lambda: self._nav_zoom(1 / 1.4)),
                ("⌂", "Reset view", self._nav_home)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setFixedWidth(36)
            b.clicked.connect(fn)
            lay.addWidget(b)
        return g

    def _nav_zoom(self, factor):
        """Scale the enlarged axis' x/y span by `factor` about its centre
        (factor < 1 zooms in)."""
        ax = self._enlarge_ax
        if ax is None:
            return
        for get, set_ in ((ax.get_xlim, ax.set_xlim),
                          (ax.get_ylim, ax.set_ylim)):
            lo, hi = get()
            c = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            set_(c - half, c + half)
        self._enlarge_canvas.draw_idle()

    def _nav_home(self):
        """Restore the enlarged axis to its auto-scaled home view."""
        ax = self._enlarge_ax
        home = getattr(self, "_enlarge_home", None)
        if ax is None or home is None:
            return
        ax.set_xlim(*home[0])
        ax.set_ylim(*home[1])
        self._enlarge_canvas.draw_idle()

    def _on_enlarge_closed(self, *a):
        self._enlarge_panel = None
        self._enlarge_fig = None
        self._enlarge_canvas = None
        self._enlarge_ax = None
        self._enlarge_home = None
        self._active_rbs = {}
        self._autofit_rbs = {}
        self._btn_enlarge_snap = None
        # Restore the inline plot #4 as the drag target and reflect edits.
        self._render_page()

    # ---- Enlarged Measurement-diff adjustment panel -------------------------

    def _open_meas_adjust_panel(self):
        """Modal child panel with the event signal plot over the Measurement
        diff on a large canvas; the Thres + Peaks toggles float over the diff's
        top-right corner. Threshold/peak interactions (and the signal plot's
        peak notches) are retargeted to this panel's axes while it is open;
        the editor re-renders (reflecting edits) on close."""
        if self._meas_panel is not None or not self._editing:
            return
        dlg = QDialog(self._editor or self)
        apply_dialog_style(dlg)
        dlg.setWindowTitle("Measurement — Diff (filtered)")
        dlg.resize(1100, 720)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        fig = Figure(figsize=(10, 4), tight_layout=False)
        style_mpl_figure(fig)
        canvas = FigureCanvas(fig)
        self._connect_canvas(canvas)
        lay.addWidget(canvas, stretch=1)
        row = QHBoxLayout()
        row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        row.addWidget(btn_close)
        lay.addLayout(row)
        self._meas_panel = dlg
        self._meas_fig = fig
        self._meas_canvas = canvas
        self._btn_meas_adjust.hide()
        dlg.finished.connect(self._on_meas_panel_closed)
        self._render_meas_adjust()
        dlg.exec()
        dlg.deleteLater()

    def _render_meas_adjust(self):
        """Draw the event signal + measurement diff onto the adjust panel's
        canvas; retarget threshold/peak hit-testing and the signal plot's
        peak notches to its axes while the panel is open, so edits in the
        diff show on the event plot immediately."""
        if self._meas_panel is None:
            return
        fig = self._meas_fig
        fig.clear()
        self._diff_ax_role.pop(self._meas_ax, None)  # this render's dead axis
        self._peak_marks = []                        # died with fig.clear()
        gs = fig.add_gridspec(2, 1, hspace=0.3, height_ratios=(1.0, 1.5),
                              left=0.07, right=0.975, top=0.95, bottom=0.07)
        ax_sig = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        self._meas_ax = ax
        self._ax_event_diff = ax   # threshold drags + peak edits target here
        self._ax_event_raw = ax_sig  # peak notches follow the edits live
        data = np.asarray(self._compute.get("data", []), dtype=float)
        title = "Measurement — Diff (filtered)"
        drawable = len(data) and self._ev_window is not None and self._events
        if drawable:
            event_id, particle_ids = self._events[
                min(self._page, len(self._events) - 1)]
            rec = np.asarray(self._decomp.get("recovered_indices", []),
                             dtype=int)
            rec = rec.reshape(-1, 2) if rec.size else np.zeros((0, 2), int)
            self._draw_signal(ax_sig, event_id, particle_ids, data, rec)
            self._draw_zone_diff(ax, self._measurement_zone, title, "meas")
        else:
            ax.set_title(title, fontsize=9, loc="left")
        self._diff_ax_role[ax] = "meas"
        style_mpl_figure(fig)
        for a in fig.axes:
            a.tick_params(labelsize=8)
            a.grid(False)
            a.xaxis.set_major_locator(MaxNLocator(nbins=8))
            a.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if drawable:
            self._update_event_peaks()
        self._meas_canvas.draw_idle()
        self._place_adjust_button()

    def _render_meas_surface(self):
        """Redraw whichever surface currently hosts the measurement diff."""
        if self._meas_panel is not None:
            self._render_meas_adjust()
        else:
            self._render_page()

    def _on_meas_panel_closed(self, *a):
        self._close_peak_palette()  # child of the panel canvas being destroyed
        # Adjustment modes end with the panel (their toggles live on it).
        for b in (self._btn_peak_adjust, self._btn_thr_adjust):
            b.setChecked(False)
            b.setParent(None)
        self._diff_ax_role.pop(self._meas_ax, None)
        self._meas_panel = None
        self._meas_fig = None
        self._meas_canvas = None
        self._meas_ax = None
        self._peak_cursor = None  # child of the panel canvas; recreate fresh
        self._drag_thr = None
        # Restore the editor's diff as the interaction target + reflect edits.
        self._render_page()

    def _event_window(self, event_id, particle_ids, data, rec):
        """Un-trimmed window of the source coincident region (as detected).
        Falls back to the recovered-particle extent when the id is stale or
        the region no longer overlaps this event's particles."""
        n = len(data)
        clamp = lambda v: min(n - 1, max(0, int(v))) if n else 0
        spans = [rec[p] for p in particle_ids if p < len(rec)]
        regions = np.atleast_2d(np.asarray(self._pulse_indices, dtype=int))
        if regions.size and 0 <= int(event_id) < len(regions) and n:
            ev_s = clamp(regions[int(event_id), 0])
            ev_e = max(clamp(regions[int(event_id), 1]), ev_s)
            if not spans or (min(s[0] for s in spans) <= ev_e
                             and max(s[1] for s in spans) >= ev_s):
                return ev_s, ev_e
        if spans and n:
            ev_s = clamp(min(s[0] for s in spans))
            ev_e = clamp(max(s[1] for s in spans))
        else:
            ev_s, ev_e = 0, max(0, n - 1)
        return ev_s, max(ev_e, ev_s)

    def _render_left_column(self, gs, event_id, particle_ids, data, rec):
        """Left column: signal (all zones stacked over the un-trimmed source
        region, with per-pulse regions shaded), then per-zone filtered diffs
        (start/meas/end). The editor prepends the rectangularized-template row
        on top."""
        off = 1 if self._editing else 0
        if self._editing:
            self._render_template_plot(self._figure.add_subplot(gs[0, 0]))

        ax_sig = self._figure.add_subplot(gs[off, 0])
        ev_s, ev_e = self._event_window(event_id, particle_ids, data, rec)
        self._ax_event_raw = ax_sig
        self._ev_window = (ev_s, ev_e)
        self._draw_signal(ax_sig, event_id, particle_ids, data, rec)

        # Editor shows the measurement diff only — start/end diffs live in
        # the on-demand Indicator Thresholds panel.
        specs = []
        if self._start_zone is not None and not self._editing:
            specs.append(("Start — Diff (filtered)", self._start_zone, "start"))
        specs.append(("Measurement — Diff (filtered)",
                      self._measurement_zone, "meas"))
        if self._end_zone is not None and not self._editing:
            specs.append(("End — Diff (filtered)", self._end_zone, "end"))
        for row, (title, zi, role) in enumerate(specs, start=off + 1):
            ax = self._figure.add_subplot(gs[row, 0])
            if len(data):
                self._draw_zone_diff(ax, zi, title, role)
            else:
                ax.set_title(title, fontsize=9, loc="left")
            self._diff_axes.append((ax, zi, title, role))
            self._diff_ax_role[ax] = role
            if role == "meas":
                self._ax_event_diff = ax
        if len(data):
            self._update_event_peaks()

        # Last left row: raw measurement + peak-location marks + warped
        # templates. Clicking it (main view) opens the editor like the others.
        ax_w = self._figure.add_subplot(gs[off + 1 + len(specs), 0])
        self._ax_warped = ax_w
        if len(data):
            self._draw_warped_templates(ax_w, particle_ids)
        else:
            ax_w.set_title("Raw Measurement + Templates", fontsize=9,
                           loc="left")

    def _pulse_bounds(self, pid):
        """(start, end) global sample indices bounding a pulse — its start/end
        role peaks, falling back to its first/last peak."""
        peaks = np.sort(np.asarray(
            self._compute["pulse_peak_locs"][pid], dtype=int))
        if not len(peaks):
            return None
        start = next((int(g) for g in peaks
                      if self._peak_roles.get(int(g)) == "start"), int(peaks[0]))
        end = next((int(g) for g in reversed(peaks)
                    if self._peak_roles.get(int(g)) == "end"), int(peaks[-1]))
        return start, max(end, start)

    def _overlap_regions(self, particle_ids):
        """[(lo, hi, prev_color, new_color)] for each consecutive-pulse overlap
        (region_{k+1}.start .. region_k.end), ordered by pulse start."""
        regs = []
        for k, pid in enumerate(particle_ids):
            b = self._pulse_bounds(pid)
            if b is not None:
                regs.append((b[0], b[1], k))
        regs.sort(key=lambda r: r[0])
        out = []
        for (lo0, hi0, k0), (lo1, hi1, k1) in zip(regs, regs[1:]):
            if lo1 <= hi0:   # they overlap
                out.append((lo1, hi0,
                            _EVENT_PALETTE[k0 % len(_EVENT_PALETTE)],
                            _EVENT_PALETTE[k1 % len(_EVENT_PALETTE)]))
        return out

    @staticmethod
    def _overlap_colors(g, overlaps):
        """(prev_color, new_color) if g falls in an overlap, else None."""
        for lo, hi, prev_c, new_c in overlaps:
            if lo <= g <= hi:
                return prev_c, new_c
        return None

    def _render_template_plot(self, ax):
        """Editor row 0: the measurement zone's rectangularized pulse template
        (median-per-segment staircase at the template's own peak boundaries),
        with its peaks marked by polarity — green positive / red negative,
        ▲ key max, ▼ key min, ★ key regular, ● non-key (Pulse Processing)."""
        from processing.extract_single_pulse_features import _unpack_tp_zone
        from utils.tpsettings_io import get_zone
        tparams = _unpack_tp_zone(get_zone(self._tp, self._measurement_zone))
        tpl = tparams.get("template_original")
        tpl = np.asarray(tpl, dtype=float).ravel() if tpl is not None \
            else np.array([])
        if len(tpl) > 1:
            x = np.arange(len(tpl))
            rect = self._rectangularize_template(tpl, tparams)
            ax.plot(x, rect, color="#1f77b4", linewidth=1.4, zorder=2)
            ax.set_xlim(0, len(tpl) - 1)
            self._mark_template_peaks(ax, tparams, rect)
            _mohm_axis(ax)
        else:
            ax.text(0.5, 0.5, "no template", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
        ax.set_title("Rectangularized Template", fontsize=9, loc="left")

    def _mark_template_peaks(self, ax, tparams, rect):
        """Peak markers on the rectangularized template, riding the staircase.
        Color = sign (green +/red −). The first peak is the start triangle (▶)
        and the last the end triangle (◀); the rest by key polarity."""
        pk = tparams.get("peak_locations")
        pk = (np.asarray(pk, dtype=int).ravel() if pk is not None
              else np.array([], dtype=int))
        seq = self._compute.get("template_seq")
        if seq is None or not len(pk):
            return
        last = len(pk) - 1
        pol_map = dict(zip(self._compute.get("template_key_seq_indices") or [],
                           self._compute.get("template_key_polarities") or []))
        for i, loc in enumerate(pk):
            if not (0 <= int(loc) < len(rect)) or i >= len(seq):
                continue
            color = "#2ca02c" if int(seq[i]) == 1 else "#d62728"
            if i == 0:
                marker, ms = ">", 9      # start triangle
            elif i == last:
                marker, ms = "<", 9      # end triangle
            elif i in pol_map:
                marker, ms = {"max": ("^", 9), "min": ("v", 9)}.get(
                    pol_map[i], ("*", 11))
            else:
                marker, ms = "o", 5
            ax.plot(int(loc), rect[int(loc)], marker, color=color,
                    markersize=ms, markeredgecolor="none", zorder=4)

    def _rectangularize_template(self, tpl, tparams):
        """Median-per-segment staircase of `tpl`, segmented at the template's
        stored peak_locations (mirrors Pulse Template Processing)."""
        n = len(tpl)
        pk = tparams.get("peak_locations")
        pk = (np.asarray(pk, dtype=int).ravel() if pk is not None
              else np.array([], dtype=int))
        pk = np.sort(pk[(pk > 0) & (pk < n)])
        bounds = np.unique(np.concatenate([[0], pk, [n]]))
        rect = np.empty(n)
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                rect[a:b] = float(np.median(tpl[a:b]))
        return rect

    def _measurement_template_rect(self):
        """(rect_staircase, sorted_peak_locations) for the measurement zone's
        rectangularized template, or None when it has fewer than 2 peaks
        (nothing to stretch between)."""
        from processing.extract_single_pulse_features import _unpack_tp_zone
        from utils.tpsettings_io import get_zone
        tparams = _unpack_tp_zone(get_zone(self._tp, self._measurement_zone))
        tpl = tparams.get("template_original")
        tpl = np.asarray(tpl, dtype=float).ravel() if tpl is not None \
            else np.array([])
        if len(tpl) <= 1:
            return None
        rect = self._rectangularize_template(tpl, tparams)
        pk = tparams.get("peak_locations")
        pk = (np.asarray(pk, dtype=int).ravel() if pk is not None
              else np.array([], dtype=int))
        pk = np.sort(pk[(pk >= 0) & (pk < len(rect))])
        return (rect, pk) if len(pk) >= 2 else None

    def _warp_template(self, rect_t, tpk, s, e, raw, line=None):
        """Warp the rectangularized template onto the pulse span whose start/
        end peaks are global samples `s`/`e`. Horizontal: the template's first
        and last peaks map linearly onto `s` and `e`. Vertical: the dip is
        depth-scaled so its depth equals the pulse's measured dip. When `line`
        (slope, intercept) is given the raw is linearly detrended, so ALL
        templates share the one flat baseline at 0 — overlap flanks can't tilt
        it; otherwise the baseline is taken locally from the pulse's own flanks.
        Returns (global_sample_x, y) or (None, None)."""
        x0, x1 = int(tpk[0]), int(tpk[-1])
        if x1 <= x0 or e <= s:
            return None, None
        x = np.arange(len(rect_t))
        gx = s + (x - x0) * (e - s) / float(x1 - x0)

        def detr(i0, i1):  # detrended raw[i0:i1]
            a = raw[i0:i1]
            if line is None or not len(a):
                return a
            return a - (line[0] * np.arange(i0, i1) + line[1])

        if line is not None:
            # Detrended: one flat baseline (0) shared by every template.
            base_at = np.zeros_like(gx)
            b_ref = 0.0
        else:
            # Local raw baseline from the flanks just outside [s, e].
            pad = max(3, (e - s) // 4)
            pre = raw[max(0, s - pad):s]
            post = raw[e + 1:min(len(raw), e + 1 + pad)]
            b_pre = float(np.median(pre)) if len(pre) else float(raw[s])
            b_post = float(np.median(post)) if len(post) else float(raw[e])
            base_at = b_pre + (b_post - b_pre) * np.clip(
                (gx - s) / (e - s), 0.0, 1.0)
            b_ref = 0.5 * (b_pre + b_post)
        tpl_base = 0.5 * (float(rect_t[0]) + float(rect_t[-1]))
        tpl_depth = tpl_base - float(np.min(rect_t))  # >= 0 (template dips)
        reg = detr(s, e + 1)
        # Signed pulse depth: baseline - trough (positive for a dip); an
        # upward pulse yields a negative depth and flips the overlay to match.
        raw_depth = (b_ref - float(np.min(reg))) if len(reg) else 0.0
        scale = raw_depth / tpl_depth if abs(tpl_depth) > 1e-30 else 0.0
        return gx, base_at - scale * (tpl_base - rect_t)

    def _detrend_line(self):
        """Linear baseline (slope, intercept) for plot #4's display frame. The
        raw-rect, per-pulse template overlays and Σ Pᵢ are only meaningful with
        the baseline removed (otherwise Σ double-counts each pulse's baseline),
        so plot #4 is ALWAYS detrended regardless of the output-detrend
        checkbox. Falls back to a flat line at the window-flank median when a
        linear fit isn't available."""
        line = self._event_baseline_line()
        if line is not None:
            return line
        if self._ev_window is None:
            return None
        ev_s, ev_e = self._ev_window
        raw = np.asarray(self._cols[self._measurement_zone], dtype=float)
        seg = raw[ev_s:ev_e + 1]
        if not len(seg):
            return None
        k = max(1, len(seg) // 10)
        flanks = np.concatenate([seg[:k], seg[-k:]])
        return (0.0, float(np.median(flanks)))

    def _detrend_seg(self, i0, i1, line):
        """Detrended raw[i0:i1] (or raw as-is when `line` is None)."""
        raw = np.asarray(self._cols[self._measurement_zone], dtype=float)
        a = raw[i0:i1].astype(float)
        if line is None or not len(a):
            return a
        return a - (line[0] * np.arange(i0, i1) + line[1])

    def _event_raw_rect(self):
        """(bounds_global, levels) staircase of the (detrended) raw over the
        event, segmented at EVERY peak of the page's pulses; each level = MEAN
        of the raw within the segment. The snap target for template edits.
        None when unavailable."""
        if self._ev_window is None or not self._events:
            return None
        ev_s, ev_e = self._ev_window
        line = self._detrend_line()
        _, pids = self._events[min(self._page, len(self._events) - 1)]
        pk_all = self._compute.get("pulse_peak_locs", [])
        peaks = set()
        for pid in pids:
            if pid < len(pk_all) and pk_all[pid] is not None:
                peaks.update(int(p) for p in np.asarray(pk_all[pid], int)
                             if ev_s < p < ev_e)
        bounds = np.array([ev_s] + sorted(peaks) + [ev_e], dtype=int)
        if len(bounds) < 2:
            return None
        levels = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = self._detrend_seg(a, max(b, a + 1), line)
            levels.append(float(np.mean(seg)) if len(seg) else 0.0)
        return bounds, np.asarray(levels)

    def _pid_color(self, pid):
        """Palette colour for pulse `pid` by its within-event order."""
        pids = self._event_pids
        k = pids.index(pid) if pid in pids else 0
        return _EVENT_PALETTE[k % len(_EVENT_PALETTE)]

    @staticmethod
    def _piecewise_x(bp, t_anchors, g_anchors):
        """Map template x-coords `bp` through the piecewise-linear anchor map
        (t_anchors -> g_anchors, both strictly increasing), extrapolating the
        first/last segment slopes beyond the outer anchors."""
        bp = np.asarray(bp, dtype=float)
        t = np.asarray(t_anchors, dtype=float)
        g = np.asarray(g_anchors, dtype=float)
        out = np.interp(bp, t, g)
        lo, hi = bp < t[0], bp > t[-1]
        out[lo] = g[0] + (bp[lo] - t[0]) * (g[1] - g[0]) / (t[1] - t[0])
        out[hi] = g[-1] + (bp[hi] - t[-1]) * (g[-1] - g[-2]) / (t[-1] - t[-2])
        return out

    def _tpl_warp_anchors(self, pid, tpk, s, e):
        """Anchor pairs [(template_x, global_x), ...] tying the template's key
        peaks to pulse `pid`'s same-role peaks: start/end always, plus the key
        max/min when marked on both sides. Interior anchors are kept only when
        they stay strictly ordered on both axes, so the map is monotonic."""
        first = (float(tpk[0]), float(s))
        last = (float(tpk[-1]), float(e))
        interior = []
        tseq = self._compute.get("template_seq")
        kidx = self._compute.get("template_key_seq_indices") or []
        kpol = self._compute.get("template_key_polarities") or []
        pk_all = self._compute.get("pulse_peak_locs", [])
        if (tseq is not None and len(tpk) == len(tseq)
                and pid < len(pk_all) and pk_all[pid] is not None):
            t_role = {}
            for i, pol in zip(kidx, kpol):
                i = int(i)
                if 0 < i < len(tpk) - 1 and pol in ("max", "min"):
                    t_role[pol] = float(tpk[i])
            g_role = {}
            for gpk in np.asarray(pk_all[pid], dtype=int):
                role = self._peak_roles.get(int(gpk))
                if role in ("max", "min") and s < gpk < e:
                    g_role[role] = float(gpk)
            interior = sorted(
                ((t_role[r], g_role[r]) for r in ("max", "min")
                 if r in t_role and r in g_role))
        anchors = [first]
        for tx, gx in interior:
            if (anchors[-1][0] < tx < last[0]
                    and anchors[-1][1] < gx < last[1]):
                anchors.append((tx, gx))
        anchors.append(last)
        return anchors

    def _seed_tpl_edit(self, pid, s, e, line):
        """Editable rectangular model {edges(global x), levels(y)} seeded from
        the warped template for pulse `pid`, or None. Horizontal placement is
        anchored at the key peaks (start/max/min/end map onto the pulse's
        same-role peaks); the segments between anchors stretch proportionally."""
        tpl = self._measurement_template_rect()
        if tpl is None:
            return None
        rect_t, tpk = tpl
        raw = np.asarray(self._cols[self._measurement_zone], dtype=float)
        gx, wy = self._warp_template(rect_t, tpk, s, e, raw, line)
        if gx is None:
            return None
        n = len(rect_t)
        bp = np.unique(np.concatenate([[0], np.asarray(tpk, int), [n]]))
        t_a, g_a = zip(*self._tpl_warp_anchors(pid, tpk, s, e))
        edges = self._piecewise_x(bp, t_a, g_a)
        levels = []
        for a, b in zip(bp[:-1], bp[1:]):
            mid = min(max(int((a + b) // 2), 0), len(wy) - 1)
            levels.append(float(wy[mid]))
        return {"edges": [float(x) for x in edges], "levels": levels}

    def _tpl_model(self, pid):
        """The rectangular model shown for pulse `pid`: the committed edit if
        any, else this render's uncommitted seed."""
        return self._tpl_edit.get(pid) or self._tpl_seed_cache.get(pid)

    def _ensure_tpl_edit(self, pid):
        """Commit pid's seed into `_tpl_edit` (so a drag mutates a stored copy)
        and report whether it was freshly created."""
        if pid in self._tpl_edit:
            return False
        m = self._tpl_seed_cache.get(pid)
        if m is None:
            return False
        self._tpl_edit[pid] = {"edges": list(m["edges"]),
                               "levels": list(m["levels"])}
        return True

    @staticmethod
    def _step_xy(edges, levels):
        """Staircase polyline for (edges, levels): repeated edges give the
        vertical risers automatically."""
        xs, ys = [], []
        for i in range(len(levels)):
            xs += [edges[i], edges[i + 1]]
            ys += [levels[i], levels[i]]
        return np.asarray(xs), np.asarray(ys)

    def _draw_edit_template(self, ax, pid, color, active, is_autofit=False):
        """Draw pulse `pid`'s editable rectangular template; when `active`, add
        draggable level (○, midpoints) and edge (□, interior) handles. The
        auto-fit pulse is drawn a touch heavier so its derived shape stands out.
        Locked levels render as ◆ (dark edge) on every pulse; only the active
        pulse's segments are lock-toggle click targets.
        Returns the created artists."""
        ed = self._tpl_model(pid)
        if ed is None:
            return []
        edges = np.asarray(ed["edges"], float)
        levels = np.asarray(ed["levels"], float)
        xs, ys = self._step_xy(edges, levels)
        lw = 1.7 if active else (1.3 if is_autofit else 1.1)
        ls = "-" if active else "--"
        if active:
            alpha = 0.95
        elif is_autofit:
            alpha = 0.75
        else:
            alpha = 0.5 if not self._editing else 0.4
        z = 6 if active else (4 if is_autofit else 3)
        arts = list(ax.plot(xs / self._fs, ys, color=color, linewidth=lw,
                            linestyle=ls, alpha=alpha, zorder=z))
        ms = self._tpl_handle_ms
        locked = self._tpl_locks.get(pid, set()) if self._editing else set()
        if active:
            for i in range(len(levels)):
                if i in locked:
                    continue
                xm = 0.5 * (edges[i] + edges[i + 1]) / self._fs
                arts += ax.plot([xm], [levels[i]], "o", color=color,
                                markersize=ms, markeredgecolor="white",
                                markeredgewidth=0.4, zorder=7)
        for i in sorted(locked):
            if i >= len(levels):
                continue
            xm = 0.5 * (edges[i] + edges[i + 1]) / self._fs
            arts += ax.plot([xm], [levels[i]], "D", color=color,
                            markersize=ms + 0.5, markeredgecolor="#333333",
                            markeredgewidth=0.8, zorder=8)
        if active:
            for i in range(1, len(edges) - 1):
                ym = 0.5 * (levels[i - 1] + levels[i])
                arts += ax.plot([edges[i] / self._fs], [ym], "s", color=color,
                                markersize=ms, markeredgecolor="white",
                                markeredgewidth=0.4, zorder=7)
        return arts

    def _redraw_active_template(self):
        """Lightweight redraw of just the active template (during a drag)."""
        for art in self._tpl_active_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._tpl_active_artists = []
        pid = self._active_tpl
        if pid is None or self._ax_warped is None:
            return
        self._tpl_active_artists = self._draw_edit_template(
            self._ax_warped, pid, self._pid_color(pid), active=True)
        self._ax_warped.figure.canvas.draw_idle()

    def _draw_warped_templates(self, ax, particle_ids):
        """4th plot: the RAW measurement zone over the event window (that zone
        only, light grey), linearly detrended when the Detrending checkbox is
        on. In the editor it also draws the whole-event rectangularized raw
        staircase (mean per inter-peak segment) as the snap target. Each pulse
        peak is a vertical line; each pulse's rectangular template is drawn from
        its editable model (committed edit or warped seed) — the active one
        prominent + draggable."""
        ev_s, ev_e = self._ev_window
        line = self._detrend_line()
        idx = np.arange(ev_s, ev_e + 1)
        seg = self._detrend_seg(ev_s, ev_e + 1, line)
        t = idx / self._fs
        ax.plot(t, seg, color="#b8b8b8", linewidth=0.8, zorder=1)
        self._event_pids = list(particle_ids)
        self._tpl_seed_cache = {}
        self._tpl_active_artists = []
        self._raw_rect = self._event_raw_rect() if self._editing else None
        y_all = [seg]
        # Whole-event rectangularized raw staircase (editor only).
        if self._raw_rect is not None:
            bnds, lvls = self._raw_rect
            rxs, rys = self._step_xy(bnds.astype(float), lvls)
            ax.plot(rxs / self._fs, rys, color="#6b6b6b", linewidth=1.0,
                    zorder=2)
            y_all.append(lvls)
        pk_all = self._compute.get("pulse_peak_locs", [])
        # Cache each pulse's own-segment seed (for those not committed) before
        # drawing / auto-fit, so Σ and the auto-fit derivation see every pulse.
        for pid in particle_ids:
            if pid not in self._tpl_edit and pid not in self._tpl_seed_cache:
                b = self._pulse_bounds(pid)
                if b is not None:
                    seed = self._seed_tpl_edit(pid, b[0], b[1], line)
                    if seed is not None:
                        self._tpl_seed_cache[pid] = seed
        # Extend every pulse's outer edges to the window so each staircase fills
        # the full x range (its baseline flats run edge to edge). Interior edges
        # are untouched; the outer edges aren't draggable.
        for pid in particle_ids:
            m = self._tpl_model(pid)
            if m is not None and len(m.get("edges", [])) >= 2:
                m["edges"][0] = min(float(ev_s), m["edges"][1] - 1.0)
                m["edges"][-1] = max(float(ev_e), m["edges"][-2] + 1.0)
        # In constraint mode, derive the auto-fit pulse on its own segments so
        # Σ Pᵢ matches the raw rect.
        if self._editing:
            self._recompute_autofit()
        for k, pid in enumerate(particle_ids):
            b = self._pulse_bounds(pid)
            if b is None:
                continue
            color = _EVENT_PALETTE[k % len(_EVENT_PALETTE)]
            if pid < len(pk_all) and pk_all[pid] is not None:
                for loc in np.asarray(pk_all[pid], dtype=int):
                    if ev_s <= loc <= ev_e:
                        ax.axvline(loc / self._fs, color=color, linewidth=0.7,
                                   alpha=0.5, zorder=2)
            active = self._editing and self._active_tpl == pid
            is_autofit = self._editing and self._autofit_tpl == pid
            arts = self._draw_edit_template(ax, pid, color, active, is_autofit)
            if active:
                self._tpl_active_artists = arts
            model = self._tpl_model(pid)
            if model is not None:
                y_all.append(np.asarray(model["levels"], float))
        # Sum overlay (black) = the original rect pulse (raw-rect staircase).
        # The measured signal IS the sum, so it's always pinned to the raw rect
        # (fixed); the pulses decompose it (Σ Pᵢ ≈ this).
        if self._editing and len(particle_ids) >= 2 and self._raw_rect is not None:
            B, L = self._raw_rect
            sx, sy = self._step_xy(B.astype(float), L)
            ax.plot(sx / self._fs, sy, color="#111111", linewidth=1.4,
                    alpha=0.85, zorder=5)
            y_all.append(L)
        if len(t) > 1:
            ax.set_xlim(t[0], t[-1])
        y_cat = np.concatenate([np.atleast_1d(a) for a in y_all])
        y_min, y_max = float(np.min(y_cat)), float(np.max(y_cat))
        rng = (y_max - y_min) or 1.0
        ax.set_ylim(y_min - 0.1 * rng, y_max + 0.1 * rng)
        _mohm_axis(ax)
        ax.set_title("Raw Measurement + Templates", fontsize=9, loc="left")

    def _tpl_hit(self, event):
        """(active_pid, "edge"|"level", i) of the template handle within grab
        range on the warped axis; None otherwise. Only the active template is
        hit-tested; locked levels refuse the grab."""
        pid = self._active_tpl
        if pid is None or event.inaxes is not self._ax_warped:
            return None
        ed = self._tpl_model(pid)
        if ed is None:
            return None
        edges = np.asarray(ed["edges"], float)
        levels = np.asarray(ed["levels"], float)
        locked = self._tpl_locks.get(pid, set())
        trans = event.inaxes.transData
        tol = 8.0
        best = None  # (dpx, kind, i)
        # Edges lock to the peaks in constraint mode (levels-only fitting).
        if self._autofit_tpl is None:
            for i in range(1, len(edges) - 1):  # interior edges (x position)
                px = trans.transform((edges[i] / self._fs, 0.0))[0]
                d = abs(event.x - px)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, "edge", i)
        for i in range(len(levels)):  # segment levels (y within the x-range)
            if i in locked:
                continue
            x0p = trans.transform((edges[i] / self._fs, 0.0))[0]
            x1p = trans.transform((edges[i + 1] / self._fs, 0.0))[0]
            if min(x0p, x1p) - 4 <= event.x <= max(x0p, x1p) + 4:
                py = trans.transform((0.0, levels[i]))[1]
                d = abs(event.y - py)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, "level", i)
        return (pid, best[1], best[2]) if best else None

    def _lock_hit(self, event):
        """Level index of the ACTIVE pulse's segment nearest the click (within
        grab range); None otherwise. The double-click lock-toggle target —
        locked levels stay hittable so they can be unlocked."""
        pid = self._active_tpl
        if pid is None:
            return None
        ed = self._tpl_model(pid)
        if ed is None:
            return None
        edges = np.asarray(ed["edges"], float)
        levels = np.asarray(ed["levels"], float)
        trans = event.inaxes.transData
        tol = 8.0
        best = None  # (dpx, i)
        for i in range(len(levels)):
            x0p = trans.transform((edges[i] / self._fs, 0.0))[0]
            x1p = trans.transform((edges[i + 1] / self._fs, 0.0))[0]
            if not (min(x0p, x1p) - 4 <= event.x <= max(x0p, x1p) + 4):
                continue
            py = trans.transform((0.0, levels[i]))[1]
            d = abs(event.y - py)
            if d <= tol and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    def _snap_level(self, pid, i, v, event):
        """Snap a dragged level y to a nearby raw-rect level or another segment
        level in the same template (within ~8 px), else return v."""
        trans = event.inaxes.transData
        py = trans.transform((0.0, v))[1]
        cands = []
        if self._raw_rect is not None:
            cands.extend(float(l) for l in self._raw_rect[1])
        ed = self._tpl_edit.get(pid)
        if ed:
            cands.extend(float(l) for j, l in enumerate(ed["levels"]) if j != i)
        best = None
        for c in cands:
            d = abs(py - trans.transform((0.0, c))[1])
            if d <= 8.0 and (best is None or d < best[1]):
                best = (c, d)
        return best[0] if best else v

    def _snap_edge(self, gx, event):
        """Snap a dragged edge x (global samples) to the nearest raw-rect step
        (within ~8 px), else return gx."""
        if self._raw_rect is None:
            return gx
        trans = event.inaxes.transData
        px = trans.transform((gx / self._fs, 0.0))[0]
        best = None
        for bx in self._raw_rect[0]:
            d = abs(px - trans.transform((float(bx) / self._fs, 0.0))[0])
            if d <= 8.0 and (best is None or d < best[1]):
                best = (float(bx), d)
        return best[0] if best else gx

    def _drag_template(self, event):
        """Apply one motion of a template level/edge drag, with snapping."""
        pid, kind, i, _ = self._drag_seg
        ed = self._tpl_edit.get(pid)
        if ed is None or event.inaxes is not self._ax_warped:
            return
        if kind == "level":
            if event.ydata is None:
                return
            ed["levels"][i] = self._snap_level(pid, i, float(event.ydata), event)
            # The auto-fit pulse (if any) is re-derived in the redraw below.
        else:  # edge — clamp between its neighbours so edges stay ordered
            if event.xdata is None:
                return
            gx = self._snap_edge(float(event.xdata) * self._fs, event)
            lo = ed["edges"][i - 1] + 1.0
            hi = ed["edges"][i + 1] - 1.0
            ed["edges"][i] = float(min(max(gx, lo), hi))
        self._drag_moved = True
        # In constraint mode the auto-fit entity + sum also move: redraw the
        # whole axis. Otherwise the lightweight active-only redraw suffices.
        if self._autofit_tpl is not None:
            self._redraw_warped_axis()
        else:
            self._redraw_active_template()

    # ---- constraint fitting (each pulse keeps its own segments) ------------

    @staticmethod
    def _eval_staircase(edges, levels, x, outside=0.0):
        """Sample a staircase (edges, levels) at global samples `x`. Values
        outside [edges[0], edges[-1]] are `outside` (None = clamp to the ends)."""
        edges = np.asarray(edges, float)
        levels = np.asarray(levels, float)
        si = np.clip(np.searchsorted(edges, x, side="right") - 1,
                     0, len(levels) - 1)
        vals = levels[si]
        if outside is not None:
            vals = np.where((x >= edges[0]) & (x <= edges[-1]), vals, outside)
        return vals

    def _eval_staircase_ramped(self, edges, levels, x, w, outside=0.0):
        """_eval_staircase with each step ramped linearly over `w` samples
        centred on the edge (clamped to half the gap to neighbouring edges).
        Subtracting an ideal zero-width step from band-limited data leaves a
        derivative spike at every edge; a transition-width-matched ramp
        suppresses them (used for the decomposed-residual display)."""
        edges = np.asarray(edges, dtype=float)
        levels = np.asarray(levels, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.array(self._eval_staircase(edges, levels, x, outside),
                     dtype=float)
        if w <= 1 or not len(levels):
            return y
        out = outside if outside is not None else None
        for j, e in enumerate(edges):
            left = levels[j - 1] if j >= 1 else (
                out if out is not None else levels[0])
            right = levels[j] if j < len(levels) else (
                out if out is not None else levels[-1])
            if left == right:
                continue
            half = w / 2.0
            if j > 0:
                half = min(half, (e - edges[j - 1]) / 2.0)
            if j < len(edges) - 1:
                half = min(half, (edges[j + 1] - e) / 2.0)
            if half <= 0:
                continue
            mask = (x >= e - half) & (x <= e + half)
            if mask.any():
                t = (x[mask] - (e - half)) / (2.0 * half)
                y[mask] = left + (right - left) * t
        return y

    def _edge_ramp_samples(self):
        """Step transition width (samples) for the residual subtraction —
        matched to the detection LPF's rise time (~1/(2·cutoff))."""
        return max(3, int(round(self._fs / (2.0 * max(
            float(self._lpf_applied), 1.0)))))

    def _enter_constraint_mode(self):
        """Commit every pulse's OWN-segment model (no re-gridding) so drags and
        the auto-fit derivation mutate stored copies. Each pulse keeps the
        number of segments its template has; only the sum spans all peaks."""
        line = self._detrend_line()
        for pid in self._event_pids:
            if pid in self._tpl_edit:
                continue
            m = self._tpl_seed_cache.get(pid)
            if m is None:
                b = self._pulse_bounds(pid)
                if b is not None:
                    m = self._seed_tpl_edit(pid, b[0], b[1], line)
            if m is not None:
                self._tpl_edit[pid] = {"edges": list(m["edges"]),
                                       "levels": list(m["levels"])}

    def _recompute_autofit(self):
        """Derive the auto-fit pulse so it complements the others toward the raw
        rect, WITHOUT changing its segmentation: it keeps its own template
        segments and each segment level = mean over that segment of
        (raw_rect − Σ other pulses). Every pulse thus stays at the template's
        segment count; only the sum (raw rect) uses all peak edges, so Σ Pᵢ
        approximates the raw rect rather than matching it exactly. No-op unless
        the auto-fit target is a pulse."""
        af = self._autofit_tpl
        if af is None or af == "sum" or self._raw_rect is None:
            return
        self._ensure_tpl_edit(af)
        ed = self._tpl_edit.get(af)
        if ed is None or not ed.get("levels"):
            return
        ev_s, ev_e = self._ev_window
        x = np.arange(ev_s, ev_e + 1)
        B, L = self._raw_rect
        target = self._eval_staircase(B, L, x, outside=None)  # raw rect per sample
        for pid in self._event_pids:
            if pid == af:
                continue
            m = self._tpl_model(pid)
            if m is not None and m.get("levels"):
                target = target - self._eval_staircase(
                    m["edges"], m["levels"], x, 0.0)
        edges = np.asarray(ed["edges"], float)
        locked = self._tpl_locks.get(af, set())
        for s in range(len(ed["levels"])):
            if s in locked:  # pinned level: auto-fit must not move it
                continue
            mask = (x >= edges[s]) & (x <= edges[s + 1])
            if mask.any():
                ed["levels"][s] = float(np.mean(target[mask]))

    def _redraw_warped_axis(self):
        """Full redraw of just the warped axis (used mid-drag when the auto-fit
        entity and sum also move)."""
        ax = self._ax_warped
        if ax is None or not self._event_pids:
            return
        ax.clear()
        self._draw_warped_templates(ax, self._event_pids)
        big = self._enlarge_panel is not None
        ax.tick_params(labelsize=8 if big else 6)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8 if big else 4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6 if big else 4))
        ax.figure.canvas.draw_idle()

    def _rect_from_edit(self, pid):
        """Sample pulse `pid`'s committed edited template over its output region
        [s,e] (baseline 0 outside the edited span). None when uncommitted."""
        ed = self._tpl_edit.get(pid)
        if not ed or not ed.get("levels"):
            return None
        s, e = (int(x) for x in self._compute["pulse_indices"][pid])
        if e < s:
            return np.array([])
        edges = np.asarray(ed["edges"], float)
        levels = np.asarray(ed["levels"], float)
        x = np.arange(s, e + 1)
        inside = (x >= edges[0]) & (x <= edges[-1])
        seg_idx = np.clip(np.searchsorted(edges, x, side="right") - 1,
                          0, len(levels) - 1)
        out = np.where(inside, levels[seg_idx], 0.0)
        return out

    def _draw_signal(self, ax_sig, event_id, particle_ids, data, rec):
        """The signal plot's content: all zones stacked over the un-trimmed
        source region. The Display mode selects raw only, filtered only, or
        both (raw faint underlay + filtered overlay). Peak notches always ride
        the filtered trace, so its display shift is recorded either way."""
        ev_s, ev_e = self._ev_window
        self._zone_shifts = {}
        if len(data):
            mode = self._display_mode()
            show_raw = mode in ("raw", "both")
            show_filt = mode in ("filtered", "both")
            act = self._zone_lp(self._measurement_zone, ev_s, ev_e)
            act_raw = np.asarray(self._cols[self._measurement_zone],
                                 dtype=float)[ev_s:ev_e + 1]
            t = (np.arange(len(act)) + ev_s) / self._fs
            act_med = float(np.median(act))
            arng = float(np.max(act) - np.min(act)) or 1.0
            step = arng * 1.5
            y_all = []
            self._zone_shifts[self._measurement_zone] = 0.0
            # Stack every other zone around the measurement (Detection Review
            # style overlays: re-centered on the measurement's median). An
            # assigned start/end role pins the outermost top/bottom slot;
            # unassigned zones sit by column order (earlier above, later below).
            m = self._measurement_zone
            roles = {self._start_zone, self._end_zone}
            above = sorted((zi for zi in range(len(self._cols))
                            if zi < m and zi not in roles), reverse=True)
            below = sorted(zi for zi in range(len(self._cols))
                           if zi > m and zi not in roles)
            if self._start_zone is not None and self._start_zone != m:
                above.append(self._start_zone)
            if (self._end_zone is not None and self._end_zone != m
                    and self._end_zone != self._start_zone):
                below.append(self._end_zone)
            for zs, sign in ((above, 1.0), (below, -1.0)):
                for i, zi in enumerate(zs):  # inner -> outer
                    seg = self._zone_lp(zi, ev_s, ev_e)
                    shift = (act_med - float(np.median(seg))
                             + sign * (i + 1) * step)
                    self._zone_shifts[zi] = shift
                    if show_raw:
                        raw = np.asarray(self._cols[zi],
                                         dtype=float)[ev_s:ev_e + 1] + shift
                        ax_sig.plot(t, raw, color="#999999", linewidth=0.4,
                                    alpha=0.2, zorder=0)
                        y_all.append(raw)
                    if show_filt:
                        segp = seg + shift
                        ax_sig.plot(t, segp, color="#999999", linewidth=0.5,
                                    alpha=0.35, zorder=1)
                        y_all.append(segp)
            # Measurement (active) zone: raw as faint underlay in 'both', or
            # promoted to the primary blue trace in 'raw' mode.
            if show_raw:
                raw_c = "#aaaaaa" if show_filt else "#1f77b4"
                raw_lw = 0.4 if show_filt else 0.8
                ax_sig.plot(t, act_raw, color=raw_c, linewidth=raw_lw, zorder=1)
                y_all.append(act_raw)
            if show_filt:
                ax_sig.plot(t, act, color="#1f77b4", linewidth=0.8, zorder=2)
                y_all.append(act)
            if len(t) > 1:
                ax_sig.set_xlim(t[0], t[-1])
            y_cat = np.concatenate(y_all) if y_all else act
            y_min, y_max = float(np.min(y_cat)), float(np.max(y_cat))
            y_rng = (y_max - y_min) or 1.0
            ax_sig.set_ylim(y_min - 0.1 * y_rng, y_max + 0.1 * y_rng)
        _mohm_axis(ax_sig)
        mode = self._event_mode(event_id)
        ax_sig.set_title(
            f"Event {event_id}" + (f" ({mode})" if mode else ""),
            fontsize=9, loc="left")

    def _draw_zone_diff(self, ax, zi, title, role):
        """One zone's filtered diff into `ax`, with that role's draggable
        threshold pair (seeded at half the extremes when unset) and its
        threshold-detected peaks labeled."""
        ev_s, ev_e = self._ev_window
        act = self._zone_lp(zi, ev_s, ev_e)
        if len(act) > 1:
            t = (np.arange(len(act)) + ev_s) / self._fs
            d = np.diff(act)
            ax.plot(t[:-1], d, color="#7f8c8d", linewidth=0.7)
            ax.set_xlim(t[0], t[-1])
            if self._thr.get(role) is None:
                self._thr[role] = [0.5 * float(np.max(d)),
                                   0.5 * float(np.min(d))]
            self._draw_thresholds(ax, role)
            self._set_diff_ylim(ax, d, thr=self._thr[role])
            if role == "meas":
                self._draw_meas_peaks(ax, ev_s, d)
            else:
                self._draw_role_peaks(ax, ev_s, d, role)
        ax.set_title(title, fontsize=9, loc="left")

    def _detect_peaks(self, d, pair):
        """All diff peaks above pair[0] / below pair[1], min-gap deduped."""
        from scipy.signal import find_peaks
        dist = self._peak_distance()
        pos, _ = find_peaks(d, height=pair[0], distance=dist)
        neg, _ = find_peaks(-d, height=-pair[1], distance=dist)
        return pos, neg

    def _draw_role_peaks(self, ax, ev_s, d, role):
        """Threshold-detected peaks labeled on a start/end zone diff (green
        positive / red negative)."""
        pair = self._thr.get(role)
        if pair is None:
            return
        t = (np.arange(len(d)) + ev_s) / self._fs
        for idx, color in zip(self._detect_peaks(d, pair),
                              ("#2ca02c", "#d62728")):
            if len(idx):
                ax.plot(t[idx], d[idx], "o", color=color, markersize=5,
                        markeredgecolor="none", zorder=5)

    # Per-pulse peak role markers: start "▶", end "◀", max "▲", min "▼",
    # regular "●". Priority max/min > start/end > regular.
    _ROLE_MARKER = {"start": (">", 8), "end": ("<", 8),
                    "max": ("^", 9), "min": ("v", 9), "regular": ("o", 5)}

    def _draw_meas_peaks(self, ax, ev_s, d):
        """Peaks on the measurement diff, colored per pulse and role-marked
        (▶ start, ◀ end, ▲ max, ▼ min, ● regular); gray ● for detections
        outside every particle span. One marker per peak; roles come straight
        from self._peak_roles (seeded per pulse, editable via the palette)."""
        if not self._events:
            return
        _, particle_ids = self._events[min(self._page, len(self._events) - 1)]
        pk_all = self._compute.get("pulse_peak_locs", [])
        t = (np.arange(len(d)) + ev_s) / self._fs
        overlaps = self._overlap_regions(particle_ids)
        owned = set()
        for k, pid in enumerate(particle_ids):
            if pid >= len(pk_all) or pk_all[pid] is None:
                continue
            color = _EVENT_PALETTE[k % len(_EVENT_PALETTE)]
            # Per-pulse region, bounded by this pulse's start/end peaks.
            b = self._pulse_bounds(pid)
            if b is not None:
                ax.axvspan(b[0] / self._fs, b[1] / self._fs,
                           color=color, alpha=0.08, zorder=0)
            locs = np.asarray(pk_all[pid], dtype=int) - ev_s
            locs = np.sort(locs[(locs >= 0) & (locs < len(d))])
            owned.update(int(x) for x in locs)
            for loc in locs:
                g = int(loc) + ev_s
                role = self._peak_roles.get(g, "regular")
                marker, ms = self._ROLE_MARKER[role]
                # Dots in a pulse-overlap are two-tone: outline = previous
                # pulse, fill = new pulse. Triangles are left single-colour.
                oc = self._overlap_colors(g, overlaps) if marker == "o" else None
                if oc is not None:
                    ax.plot(t[loc], d[loc], "o", markerfacecolor=oc[1],
                            markeredgecolor=oc[0], markeredgewidth=1.3,
                            markersize=ms, zorder=5)
                else:
                    ax.plot(t[loc], d[loc], marker, color=color, markersize=ms,
                            markeredgecolor="none", zorder=5)
        pair = self._thr.get("meas")
        if pair is not None:
            pos, neg = self._detect_peaks(d, pair)
            stray = np.asarray([int(i) for i in np.concatenate([pos, neg])
                                if int(i) not in owned], dtype=int)
            if len(stray):
                ax.plot(t[stray], d[stray], "o", color="#999999",
                        markersize=4, alpha=0.7, zorder=4)

    def _refresh_peaks(self, only_events=None):
        """Threshold-based peak detection on every event's measurement diff,
        assigned to particles by span containment. Replaces template-driven
        detection — the template only guides decomposition and the sequence
        read-out. Overwrites manual edits (runs only when detection inputs
        change: thresholds, cutoff, gap, roles, splits). `only_events` (set of
        source-event ids) scopes re-detection to the edited events so every
        OTHER event's manual peaks, roles and committed template edits survive
        a per-event split edit."""
        data = np.asarray(self._compute.get("data", []), dtype=float)
        rec = np.asarray(self._decomp.get("recovered_indices", []),
                         dtype=int).reshape(-1, 2)
        if only_events is None:
            # Global re-detection reassigns/renumbers peaks -> manual role
            # overrides (keyed by global index) and template edits/locks
            # (keyed to the segment structure) no longer apply.
            self._peak_roles = {}
            self._tpl_edit = {}
            self._tpl_locks = {}
            self._autofit_tpl = None
        else:
            only_events = {int(e) for e in only_events}
            # Drop only the edited events' manual state: their particles'
            # template edits/locks and role overrides inside their windows.
            for ev_id, pids in self._events:
                if int(ev_id) not in only_events:
                    continue
                for pid in pids:
                    self._tpl_edit.pop(pid, None)
                    self._tpl_locks.pop(pid, None)
                    if self._autofit_tpl == pid:
                        self._autofit_tpl = None
                    if self._active_tpl == pid:
                        self._active_tpl = None
                if len(data):
                    ev_s, ev_e = self._event_window(ev_id, pids, data, rec)
                    self._peak_roles = {g: r
                                        for g, r in self._peak_roles.items()
                                        if not ev_s <= g <= ev_e}
        if not len(data) or not self._events:
            return
        if self._thr.get("meas") is None:
            ev_id, pids = self._events[min(self._page, len(self._events) - 1)]
            ev_s, ev_e = self._event_window(ev_id, pids, data, rec)
            act = self._zone_lp(self._measurement_zone, ev_s, ev_e)
            if len(act) < 2:
                return
            d = np.diff(act)
            self._thr["meas"] = [0.5 * float(np.max(d)),
                                 0.5 * float(np.min(d))]
        pair = self._thr["meas"]
        r_max, r_min = self._template_key_ranks()  # template-guided max/min
        for ev_id, pids in self._events:
            if only_events is not None and int(ev_id) not in only_events:
                continue
            ev_s, ev_e = self._event_window(ev_id, pids, data, rec)
            act = self._zone_lp(self._measurement_zone, ev_s, ev_e)
            if len(act) < 3:
                continue
            d_ev = np.diff(act)
            pos, neg = self._detect_peaks(d_ev, pair)
            pos_g = np.sort(pos.astype(int) + ev_s)   # rising (green) edges
            neg_g = np.sort(neg.astype(int) + ev_s)   # falling (red) edges
            locs = np.sort(np.concatenate([pos_g, neg_g]))
            usable = sorted((p for p in pids
                             if p < len(rec) and p < self._n_particles),
                            key=lambda p: int(rec[p, 0]))
            if not usable:
                continue
            # Indicator-referenced start/end peak per pulse: the measurement
            # falling edge just before that pulse's start-zone rising edge, and
            # the rising edge just after its end-zone falling edge. The first
            # pulse's start and last pulse's end are pinned to the event's
            # first/last peak.
            se = self._indicator_start_end(usable, ev_s, ev_e, pos_g, neg_g)
            if len(locs):
                se[usable[0]][0] = int(locs[0])
                se[usable[-1]][1] = int(locs[-1])
            # Region per pulse: [start_peak, end_peak], falling back to the
            # recovered span when an indicator edge wasn't found.
            region = {}
            for pid in usable:
                lo = se[pid][0] if se[pid][0] is not None else int(rec[pid, 0])
                hi = se[pid][1] if se[pid][1] is not None else int(rec[pid, 1])
                region[pid] = (min(lo, hi), max(lo, hi))
            # Assign each peak to exactly one pulse by region (nearest centre on
            # overlap); then force each start/end peak into its own pulse.
            assign = {p: [] for p in usable}
            for g in locs:
                inreg = [p for p in usable
                         if region[p][0] <= int(g) <= region[p][1]]
                cands = inreg if inreg else usable
                p = min(cands, key=lambda p: abs(int(g) - sum(region[p]) / 2))
                assign[p].append(int(g))
            for pid in usable:
                for role_g in se[pid]:
                    if role_g is None:
                        continue
                    for q in usable:
                        if q != pid and role_g in assign[q]:
                            assign[q].remove(role_g)
                    if role_g not in assign[pid]:
                        assign[pid].append(int(role_g))
            for pid in usable:
                self._compute["pulse_peak_locs"][pid] = np.array(
                    sorted(set(assign[pid])), dtype=int)
            # Seed roles: start/end from the indicator-referenced peaks (or the
            # pulse's first/last peak), interior argmax/argmin = max/min.
            for pid in usable:
                pk = np.sort(np.asarray(assign[pid], dtype=int))
                pk = pk[(pk - ev_s >= 0) & (pk - ev_s < len(d_ev))]
                if not len(pk):
                    continue
                sg = se[pid][0] if se[pid][0] in pk else int(pk[0])
                eg = se[pid][1] if (se[pid][1] in pk
                                    and se[pid][1] != sg) else None
                if eg is None and pk[-1] != sg:
                    eg = int(pk[-1])
                self._peak_roles[int(sg)] = "start"
                if eg is not None:
                    self._peak_roles[int(eg)] = "end"
                # Max = the r_max-th positive edge, min = the r_min-th negative
                # edge of the pulse — the template's key-peak positions. Falls
                # back to the strongest interior edge when the template rank
                # isn't available or collides with start/end.
                dv = d_ev[pk - ev_s]
                pos_pk, neg_pk = pk[dv >= 0], pk[dv < 0]
                max_g = (int(pos_pk[r_max]) if r_max is not None
                         and 0 <= r_max < len(pos_pk) else None)
                min_g = (int(neg_pk[r_min]) if r_min is not None
                         and 0 <= r_min < len(neg_pk) else None)
                interior = (pk != sg) & (pk != eg)
                if max_g is None or max_g in (sg, eg):
                    cand = pk[interior & (dv >= 0)]
                    max_g = (int(cand[np.argmax(d_ev[cand - ev_s])])
                             if len(cand) else None)
                if min_g is None or min_g in (sg, eg) or min_g == max_g:
                    cand = pk[interior & (dv < 0)]
                    min_g = (int(cand[np.argmin(d_ev[cand - ev_s])])
                             if len(cand) else None)
                if max_g is not None:
                    self._peak_roles[int(max_g)] = "max"
                if min_g is not None and min_g != max_g:
                    self._peak_roles[int(min_g)] = "min"
            for pid in usable:
                self._recompute_rect(pid)
                self._check_sequence(pid)

    def _template_key_ranks(self):
        """(r_max, r_min): in the template peak sequence the key max is the
        r_max-th positive edge and the key min is the r_min-th negative edge.
        A pulse's max/min are then the same-ranked edges of its own sequence,
        so the labels follow the template's structure rather than raw
        magnitude. (None, None) when the template lacks key polarities."""
        tseq = self._compute.get("template_seq")
        kidx = self._compute.get("template_key_seq_indices") or []
        kpol = self._compute.get("template_key_polarities") or []
        if tseq is None or not len(tseq) or not len(kidx):
            return None, None
        tseq = [int(v) for v in tseq]
        r_max = r_min = None
        for i, pol in zip(kidx, kpol):
            i = int(i)
            if not (0 <= i < len(tseq)):
                continue
            if pol == "max":
                r_max = sum(1 for j in range(i) if tseq[j] == 1)
            elif pol == "min":
                r_min = sum(1 for j in range(i) if tseq[j] == 0)
        return r_max, r_min

    def _indicator_start_end(self, usable_sorted, ev_s, ev_e, pos_g, neg_g):
        """{pid: [start_g, end_g]} from the indicator edges: pulse start = the
        measurement falling edge just before that pulse's start-zone rising
        edge; end = the rising edge just after its end-zone falling edge
        (FIFO-paired). None where the indicator edge isn't found."""
        from processing.coincident_decomposition import indicator_edge_peaks
        from processing.extract_single_pulse_features import _unpack_tp_zone
        from utils.tpsettings_io import get_zone
        fcfg = _unpack_tp_zone(
            get_zone(self._tp, self._measurement_zone)).get("filter_config")
        n = len(usable_sorted)
        sr = ef = np.array([], dtype=int)
        if self._start_zone is not None:
            sr = indicator_edge_peaks(
                np.asarray(self._cols[self._start_zone], dtype=float),
                ev_s, ev_e + 1, +1, fs=self._fs, filter_config=fcfg, top=n)
        if self._end_zone is not None:
            ef = indicator_edge_peaks(
                np.asarray(self._cols[self._end_zone], dtype=float),
                ev_s, ev_e + 1, -1, fs=self._fs, filter_config=fcfg, top=n)
        out = {}
        for k, pid in enumerate(usable_sorted):
            s = e = None
            if k < len(sr) and len(neg_g):
                before = neg_g[neg_g < sr[k]]
                if len(before):
                    s = int(before[-1])
            if k < len(ef) and len(pos_g):
                after = pos_g[pos_g > ef[k]]
                if len(after):
                    e = int(after[0])
            out[pid] = [s, e]
        return out

    def _run_decomposition(self, event_id, particle_ids, data, rec):
        """Additive decomposition for the current event: residual-based
        single-pulse waveforms per particle (fit-subtract-refine against the
        template unit shape). 1 particle = trivial; 2 = two-pulse overlap,
        valid only on landmark full-extent spans (the unit shape spans
        entry->exit; indicator-window spans would misalign it); 3+ = not
        implemented yet."""
        from processing.coincident_decomposition import (
            unit_shape_from_template, decompose_two_pulses)
        from processing.extract_single_pulse_features import _unpack_tp_zone
        from utils.tpsettings_io import get_zone
        self._decomp_waves = {}   # pid -> waveform over the particle span
        self._decomp_models = {}  # pid -> fitted model (QA overlay)
        self._decomp_note = ""
        if not len(data):
            return
        spans = [(int(rec[p, 0]), int(rec[p, 1]))
                 for p in particle_ids if p < len(rec)]
        if len(spans) == 1:
            p = particle_ids[0]
            self._decomp_waves[p] = data[spans[0][0]:spans[0][1] + 1]
            return
        if len(spans) != 2:
            self._decomp_note = "N > 2"
            return
        mode = self._event_mode(event_id)
        if mode != "landmark":
            self._decomp_note = f"needs landmark timing (mode: {mode or '?'})"
            return
        tparams = _unpack_tp_zone(get_zone(self._tp, self._measurement_zone))
        shape = unit_shape_from_template(tparams.get("template_original"))
        res = decompose_two_pulses(data, spans, shape)
        if res is None:
            self._decomp_note = "fit failed"
            return
        order = np.argsort([s[0] for s in spans])
        for rank, pi_ in enumerate(order):
            pid = particle_ids[int(pi_)]
            self._decomp_waves[pid] = res["outputs"][rank]
            self._decomp_models[pid] = res["models"][rank]
        self._decomp_fit = res

    def _render_particle_plot(self, gs, k, pid, data, rec):
        """Right column, row k: particle pid's isolated raw — the (detrended)
        raw measurement minus every OTHER pulse's rect template (P0 = raw − P1
        rect, P1 = raw − P0 rect, …), with pid's own rect overlaid in colour.
        Carries the acceptance frame."""
        color = _EVENT_PALETTE[k % len(_EVENT_PALETTE)]
        ax = self._figure.add_subplot(gs[k, 1])
        self._raw_axes_map[pid] = ax
        ev = self._ev_window
        if ev is not None and len(data):
            ev_s, ev_e = ev
            x = np.arange(ev_s, ev_e + 1)
            line = self._detrend_line()
            resid = self._detrend_seg(ev_s, ev_e + 1, line).astype(float)
            ramp = self._edge_ramp_samples()
            for j in self._event_pids:
                if j == pid:
                    continue
                m = self._tpl_model(j)
                if m is not None and m.get("levels"):
                    resid = resid - self._eval_staircase_ramped(
                        m["edges"], m["levels"], x, ramp, 0.0)
            t = x / self._fs
            ax.plot(t, resid, color="#b8b8b8", linewidth=0.6, zorder=1)
            own = self._tpl_model(pid)
            if own is not None and own.get("levels"):
                ax.plot(t, self._eval_staircase(own["edges"], own["levels"],
                                                x, 0.0),
                        color=color, linewidth=1.1, zorder=3)
            if ev_e > ev_s:
                ax.set_xlim(ev_s / self._fs, ev_e / self._fs)
            _mohm_axis(ax)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
        ax.set_title(f"P{k} — Decomposed", fontsize=9, loc="left",
                     color=color)

    def _apply_accept_frames(self):
        """Colour each particle's RAW-axis frame by acceptance (green=accepted,
        red=rejected). Called after style_mpl_figure so the colour survives."""
        for pid, ax in self._raw_axes_map.items():
            accepted = (int(self.acceptance[pid]) >= 1
                        if pid < len(self.acceptance) else True)
            edge = "#2ca02c" if accepted else "#d62728"
            for spine in ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(2.0)

    # ---- results ------------------------------------------------------------

    def _event_baseline_line(self):
        """The displayed page's baseline line (render/preview path)."""
        if not self._events or self._ev_window is None:
            return None
        ev_id, pids = self._events[min(self._page, len(self._events) - 1)]
        return self._baseline_line_for(ev_id, pids)

    def _baseline_line_for(self, event_id, pids):
        """(slope, intercept) of a line fit through the event's baseline
        samples — the flat regions before the first pulse's start and after
        the last pulse's end. None when there aren't enough baseline samples."""
        data = np.asarray(self._compute.get("data", []), dtype=float)
        if not len(data):
            return None
        rec = np.asarray(self._decomp.get("recovered_indices", []),
                         dtype=int).reshape(-1, 2)
        ev_s, ev_e = self._event_window(event_id, pids, data, rec)
        bounds = [b for b in (self._pulse_bounds(p) for p in pids)
                  if b is not None]
        if not bounds:
            return None
        first_start = min(b[0] for b in bounds)
        last_end = max(b[1] for b in bounds)
        idx = np.concatenate([np.arange(ev_s, first_start),
                              np.arange(last_end + 1, ev_e + 1)])
        idx = idx[(idx >= 0) & (idx < len(data))].astype(int)
        if len(idx) < 2:
            return None
        slope, intercept = np.polyfit(idx, data[idx], 1)
        return float(slope), float(intercept)

    def _rect_for_particle(self, pid, baseline="page"):
        """Display/output rectangularized span for particle `pid`. A committed
        template edit (plot #4) wins; otherwise honour the rect method and the
        linear-baseline detrend (identical logic for the output and any
        preview). `baseline` is the detrend line to use — the default takes
        the displayed page's; _rects_out passes each particle's own event's."""
        edited = self._rect_from_edit(pid)
        if edited is not None:
            return edited
        use_mean = (self._rect_method == "mean")
        if self._chk_detrend.isChecked():
            line = (self._event_baseline_line() if baseline == "page"
                    else baseline)
            if line is not None:
                return self._detrended_rect(pid, use_mean, line)
        rect = (self._compute["rect_pulses_mean"][pid] if use_mean
                else self._compute["rect_pulses"][pid])
        # _compute_zone leaves rect as None for sub-3-sample spans (reachable
        # via the split editor); emit an empty array so the output list never
        # holds None.
        return rect if rect is not None else np.array([])

    def _detrended_rect(self, pid, use_mean, line):
        """Rect of pulse `pid` after subtracting the event's linear baseline
        `line` (slope, intercept), rectangularized per inter-peak segment."""
        slope, intercept = line
        s, e = (int(x) for x in self._compute["pulse_indices"][pid])
        seg = np.asarray(self._compute.get("data", []), dtype=float)[s:e + 1]
        if not len(seg):
            return np.array([])
        x = np.arange(s, e + 1)
        seg = seg - (slope * x + intercept)
        peaks = np.asarray(self._compute["pulse_peak_locs"][pid], int) - s
        peaks = peaks[(peaks > 0) & (peaks < len(seg))]
        bounds = [0] + sorted(int(p) for p in peaks) + [len(seg)]
        rect = np.empty(len(seg))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                rect[a:b] = (float(np.mean(seg[a:b])) if use_mean
                             else float(np.median(seg[a:b])))
        return rect

    def _rects_out(self):
        """Per-particle rectangularized output (rect-method + detrend aware).
        Detrended rects subtract each particle's OWN event's baseline, not the
        displayed page's."""
        line_of = {}
        for ev, pids in self._events:
            line = (self._baseline_line_for(ev, pids)
                    if self._chk_detrend.isChecked() else None)
            for p in pids:
                line_of[p] = line
        return [self._rect_for_particle(i, baseline=line_of.get(i))
                for i in range(self._n_particles)]

    def results(self):
        rec = np.asarray(self._decomp.get("recovered_indices", []), dtype=int)
        if self._n_particles:
            start_indices = rec.reshape(-1, 2)[:, 0]
        else:
            start_indices = np.array([], int)
        # Peaks/rects are returned for ALL recovered particles; rejected ones
        # are filtered downstream (e.g. Pulse Slicing) via acceptance_id.
        return {
            "pulse_peak_locations": list(self._compute["pulse_peak_locs"]),
            "rectangularized_pulses": self._rects_out(),
            "acceptance_id": np.asarray(self.acceptance) >= 1,
            "pulse_start_indices": start_indices,
            "coincidence_map": [
                {"source_event": int(se), "N": int(nn)}
                for se, nn in zip(self._decomp["source_event"],
                                  self._decomp["event_n"])],
        }
