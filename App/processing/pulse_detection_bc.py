"""Interactive pulse detection with baseline correction.

Two-tab interactive UI: (1) Detection — lowpass filter calibration on the
derivative plus threshold-based pulse detection; (2) Baseline — baseline
estimation via interpolation through the detected pulse regions. Outputs
both trendline and detected pulses.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGroupBox, QWidget, QDoubleSpinBox, QSpinBox,
    QTabWidget, QRadioButton, QStackedWidget,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from utils.decimate import minmax_decimate


_MAX_PLOT_PTS = 10000


def _ds(x, y, x0=None, x1=None, max_pts=_MAX_PLOT_PTS):
    """View-adaptive min-max decimation.

    Slices data to [x0, x1] visible range, then decimates to max_pts
    via ``minmax_decimate`` (preserves peaks). Returns (x_out, y_out).
    """
    if x0 is not None or x1 is not None:
        mask = np.ones(len(x), dtype=bool)
        if x0 is not None:
            mask &= x >= x0
        if x1 is not None:
            mask &= x <= x1
        x = x[mask]
        y = y[mask]
    return minmax_decimate(x, y, max_pts)


class _PerZoneState:
    """Mutable per-zone working set for the Detection BC dialog.

    Holds everything that differs between zones: the signal column, derived
    arrays, thresholds, detected regions, and baseline products. UI control
    *values* (cutoffs, margins) are also cached here so switching zones
    restores them.
    """
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float).ravel()
        self.n = len(self.data)
        # Stage 1 (detection)
        self.data_diff = None
        self.data_diff_filtered = None
        self.data_display = None
        self.pulse_regions = np.empty((0, 2), dtype=int)
        self.pos_thresh = None
        self.neg_thresh = None
        self.applied_cutoff = None
        self.applied_display_cutoff = None
        # Merge factors (per-zone; default on first visit).
        # region_margin = leading (pre-pulse) pad; region_margin_post = trailing.
        self.region_margin = None
        self.region_margin_post = None
        self.merge_distance = None
        # No-template pulse width in samples (per-zone; only used when no
        # template is connected — with one, the width is the template length).
        self.template_width = None
        # Stage 2 (baseline)
        self.filtered_signal = None
        self.baseline_smooth = None
        self.detrended_data = None
        self.use_auto_tangent = False  # False = linear (matches default radio)
        self.smooth_window = None
        # outputs
        self.trendline = None
        self.detected_pulses = None
        self.confirmed = False
        self.finished = False  # zone's Baseline stage confirmed (multi-zone)
        # True when thresholds/cutoffs were restored from saved settings but
        # the zone hasn't been visited yet — its first visit must use the
        # restored values (and recompute regions from them) instead of
        # re-deriving defaults. Consumed on first visit.
        self.restored = False


class _PulseDetectionBCDialog(QDialog):
    """Two-tab interactive pulse detection + baseline correction dialog.

    Tab 1 (Detection): lowpass filter calibration on the derivative,
        threshold-based pulse detection and region extraction
    Tab 2 (Baseline): baseline estimation and detrending
    """

    def __init__(self, zone_signals, sample_rate, pulse_template=None,
                 filter_config=None, downsample_factor=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pulse Detection (BC)")
        self.resize(1400, 800)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)
        self.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)

        # One working-set per zone (column). Single-zone → one entry, and the
        # zone tab bar stays hidden so behavior is identical to before.
        self._zones = [_PerZoneState(s) for s in zone_signals]
        self._active_zone = 0

        # Sample-rate info (from the Global Sample Rate block via injected
        # globals): sample_rate is the *effective* rate, downsample_factor is N,
        # base = effective × N. Shown in the Sample Rate info group.
        self._fs = sample_rate
        self._nyq = sample_rate / 2.0
        try:
            self._downsample_factor = max(1, int(round(downsample_factor or 1)))
        except (TypeError, ValueError):
            self._downsample_factor = 1
        self._effective_fs = float(sample_rate)
        self._base_fs = self._effective_fs * self._downsample_factor

        # Filter config for display (top plot)
        self._filter_config = filter_config

        # Default cutoff for derivative filter
        self._default_cutoff = min(100.0, self._nyq - 1.0)

        # Display lowpass cutoff for top plot
        self._display_cutoff = min(1000.0, self._nyq - 1.0)

        # Pulse template info (shared across zones). With a real template the
        # width is its length; otherwise the Margin / Merge Distance factors
        # multiply an adjustable sample count (default 500), editable via the
        # "Template width" spinbox shown only when no template is connected.
        self._has_template = (pulse_template is not None
                              and len(pulse_template) > 0)
        if self._has_template:
            self._pulse_template_width = len(pulse_template)
        else:
            self._pulse_template_width = int(
                getattr(self, '_live_template_width', 500))
        # Baseline for zones without their own width yet (updated on settings
        # load) — NOT tracked by spinbox edits, so zones stay independent.
        self._default_template_width = int(self._pulse_template_width)
        self._endpoint_window_width = max(1, round(0.1 * self._pulse_template_width))

        self._smooth_window_default = max(1, round(0.2 * self._fs))

        # Plain working-set attributes mirror the ACTIVE zone; compute zone 0.
        self._compute_zone_derived(initial=True)

        # Zoom box / pan / threshold drag state
        self._zb_mode = False
        self._zb_start = None
        self._zb_span = None
        self._pan_mode = False
        self._pan_start = None
        self._thresh_drag = None  # 'pos' or 'neg' when dragging threshold
        self._thresh_hl_pos = None  # axhline artist for pos threshold
        self._thresh_hl_neg = None  # axhline artist for neg threshold
        self._thresh_bg = None  # cached background for blitting the drag
        self._drag_line = None  # the threshold line being dragged
        self._syncing_thresh = False  # guards threshold spinbox<->line sync

        # Mpl event connection IDs for cleanup
        self._mpl_cids = []

        # Track highest initialized stage
        self._current_stage = 0  # 0-based
        self._tab_figs = [None, None]
        self._tab_canvases = [None, None]
        self._tab_axes = [None, None]
        self._tab_lines = {}  # {tab_idx: (lines_top, lines_bot)}

        self._build_ui()
        self._apply_filter()
        self._init_detection()
        self._update_thresholds()

    def closeEvent(self, event):
        """Disconnect mpl callbacks to avoid weakref errors on teardown."""
        for canvas, cid in self._mpl_cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids.clear()
        super().closeEvent(event)

    # ============================= Zone snapshot ========================

    def _on_zone_changed(self, idx):
        """Switch zones: snapshot the current zone, restore the new one.

        A real zone index shows the per-zone stage page; the All-Zones index
        shows the merge page. Leaving a real zone snapshots it first.
        """
        self._deactivate_tools()

        if self._all_zones_idx is not None and idx == self._all_zones_idx:
            # Entering the All-Zones page: snapshot whatever real zone is live.
            if self._active_zone != self._all_zones_idx:
                self._save_active_zone()
            self._active_zone = self._all_zones_idx
            self._content_stack.setCurrentWidget(self._all_zones_page)
            self._recompute_all_zones_merge()
            return

        if idx == self._active_zone:
            return
        # Switching to a real zone (possibly back from All Zones).
        self._content_stack.setCurrentIndex(0)
        prev = self._active_zone
        if prev != self._all_zones_idx:
            self._save_active_zone()
        self._load_active_zone(idx)

    def _refresh_zone_tab_labels(self):
        """Set each real zone's tab text to show its finished status."""
        for i, z in enumerate(self._zones):
            label = f"Zone {i + 1} ✓" if z.finished else f"Zone {i + 1}"
            self._zone_bar.setTabText(i, label)

    def _update_all_zones_enabled(self):
        """Enable the All-Zones tab only once every zone is finished."""
        if self._all_zones_idx is None:
            return
        self._zone_bar.setTabEnabled(
            self._all_zones_idx, all(z.finished for z in self._zones))

    def _save_active_zone(self):
        """Copy the plain working-set attributes into the active zone state."""
        z = self._zones[self._active_zone]
        z.data = self._data
        z.n = self._n
        # pulse_regions and the baseline products below are copied so a later
        # in-place edit of self._* can't corrupt the snapshot. data/data_diff*/
        # data_display are stored as references: _compute_zone_derived rebuilds
        # them from z.data on restore (z.data_diff only feeds the "fresh zone"
        # check), and self._data is itself a copy of z.data, so no aliasing.
        z.data_diff = self._data_diff
        z.data_diff_filtered = self._data_diff_filtered
        z.data_display = self._data_display
        z.pulse_regions = (self._pulse_regions.copy()
                           if self._pulse_regions is not None else None)
        z.pos_thresh = self._pos_thresh
        z.neg_thresh = self._neg_thresh
        z.applied_cutoff = self._applied_cutoff
        z.applied_display_cutoff = self._applied_display_cutoff
        # Merge factors mirror the spinboxes via _live_* (kept in sync on
        # valueChanged), so they survive the detection-panel teardown too.
        z.region_margin = getattr(self, '_live_region_margin', 0.2)
        z.region_margin_post = getattr(
            self, '_live_region_margin_post', z.region_margin)
        z.merge_distance = getattr(self, '_live_merge_distance', 0.5)
        if not self._has_template:
            z.template_width = int(getattr(
                self, '_live_template_width', self._pulse_template_width))
        z.filtered_signal = (self._filtered_signal.copy()
                             if self._filtered_signal is not None else None)
        z.baseline_smooth = (self._baseline_smooth.copy()
                             if self._baseline_smooth is not None else None)
        z.detrended_data = (self._detrended_data.copy()
                            if self._detrended_data is not None else None)
        z.use_auto_tangent = self._use_auto_tangent
        try:
            z.smooth_window = int(self._spin_smooth.value())
        except (AttributeError, RuntimeError):
            pass
        # Finish outputs, captured when available
        z.trendline = getattr(self, 'trendline', None)
        z.detected_pulses = getattr(self, 'detected_pulses', None)
        z.confirmed = getattr(self, 'confirmed', False)

    def _sync_cutoff_spin(self, attr, value):
        """Set a detection-panel spinbox to *value* if it is a live widget.

        The widget's underlying C++ object may already be deleted when the
        detection panel is not the active stage (e.g. on Baseline). The Python
        attribute lingers, so a plain access raises RuntimeError; swallow it.
        The value is re-applied from self._applied_* when the detection panel
        is rebuilt, so skipping here is safe.
        """
        spin = getattr(self, attr, None)
        if spin is None:
            return
        try:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        except RuntimeError:
            pass  # widget deleted; detection panel not currently shown

    def _spin_value(self, spin_attr, fallback_attr, default=None):
        """Read a spinbox's value, or the mirrored fallback attribute when
        the widget is absent or its C++ object was deleted (the detection
        panel is torn down on the Baseline tab; the Python attr lingers)."""
        try:
            return getattr(self, spin_attr).value()
        except (AttributeError, RuntimeError):
            return getattr(self, fallback_attr, default)

    def _load_active_zone(self, idx):
        """Make zone *idx* active: restore its working set into self._*."""
        self._active_zone = idx
        z = self._zones[idx]
        # A zone whose settings were restored from file is not "fresh": its
        # first visit must keep the restored thresholds (and compute regions
        # from them) rather than re-deriving defaults.
        restored_first = getattr(z, 'restored', False)
        fresh = z.data_diff is None and not restored_first

        self._compute_zone_derived()

        # Filter with THIS zone's cutoff (default for fresh, restored
        # otherwise) — not the prior zone's stale spinbox value. Also sync the
        # detection spinboxes IF they're currently shown. When the active stage
        # is Baseline the detection panel has been torn down, so those widgets'
        # C++ objects are deleted (the Python attr lingers) — accessing them
        # raises RuntimeError. Guard it: the values are already held in
        # self._applied_*, and _build_detection_panel re-applies them when the
        # detection panel is next shown, so skipping here is safe.
        self._sync_cutoff_spin('_spin_cutoff', self._applied_cutoff)
        self._sync_cutoff_spin('_spin_display_cutoff', self._applied_display_cutoff)
        # Restore this zone's merge factors into the live spinboxes (guarded
        # the same way — the detection panel may be torn down on Baseline).
        self._sync_cutoff_spin('_spin_padding', self._live_region_margin)
        self._sync_cutoff_spin('_spin_padding_post', self._live_region_margin_post)
        self._sync_cutoff_spin('_spin_spacing', self._live_merge_distance)
        if not self._has_template:
            self._sync_cutoff_spin('_spin_tpl_width',
                                   int(self._pulse_template_width))
        self._apply_filter(from_zone=True)

        if fresh:
            # First visit: derive default thresholds like a new zone.
            self._init_detection()
        else:
            # Restore previously computed smooth window.
            if z.smooth_window is not None:
                self._smooth_window_default = z.smooth_window

        # Re-apply per-zone Baseline-tab gating: enabled only if this zone
        # has detected pulse regions.
        has_regions = (self._pulse_regions is not None
                       and len(self._pulse_regions) > 0)
        self._current_stage = 1 if has_regions else 0
        if hasattr(self, '_tabs'):
            self._tabs.blockSignals(True)
            self._tabs.setTabEnabled(1, has_regions)
            if not has_regions and self._tabs.currentIndex() == 1:
                self._tabs.setCurrentIndex(0)
            self._tabs.blockSignals(False)
            cur = self._tabs.currentIndex()
            if cur == 1 and has_regions:
                self._init_baseline_tab()
            self._on_tab_changed(cur)
            if cur == 0:
                # Keep restored regions for a previously-saved zone; recompute
                # from thresholds for a fresh zone OR a zone whose thresholds
                # were just restored from file (first visit).
                self._update_thresholds(recompute=fresh or restored_first)
        # Consume the restore flag: subsequent visits behave like a normal
        # previously-visited zone.
        z.restored = False

    def _compute_zone_derived(self, initial=False):
        """Derive display/detection working arrays for the ACTIVE zone.

        Mirrors the active ``_PerZoneState`` into the plain ``self._*``
        attributes and fills in any derived arrays that aren't computed yet.
        Run for zone 0 in ``__init__`` and on first visit of any other zone.
        """
        z = self._zones[self._active_zone]

        self._data = z.data.copy()
        self._n = z.n
        self._t = np.arange(self._n) / self._fs

        # Derivative
        self._data_diff = np.diff(self._data)
        self._t_diff = np.arange(len(self._data_diff)) / self._fs

        # Cutoffs (per-zone; default on first visit)
        self._applied_cutoff = (z.applied_cutoff if z.applied_cutoff is not None
                                else self._default_cutoff)
        self._applied_display_cutoff = (z.applied_display_cutoff
                                        if z.applied_display_cutoff is not None
                                        else self._display_cutoff)

        # Merge factors (per-zone; default on first visit). _live_* is the
        # single source of truth the detection spinboxes read/write.
        self._live_region_margin = (z.region_margin
                                      if z.region_margin is not None else 0.2)
        self._live_region_margin_post = (
            z.region_margin_post if z.region_margin_post is not None
            else self._live_region_margin)
        self._live_merge_distance = (z.merge_distance
                                       if z.merge_distance is not None else 0.5)

        # No-template pulse width (per-zone; a fresh zone starts at the
        # default/loaded baseline, not the last-edited value, so zones stay
        # independent). With a template the width is its length, shared.
        if not self._has_template:
            self._pulse_template_width = int(
                z.template_width if z.template_width is not None
                else getattr(self, '_default_template_width', 500))
            self._live_template_width = int(self._pulse_template_width)
            self._endpoint_window_width = max(
                1, round(0.1 * self._pulse_template_width))

        # Display filter for top plot
        self._data_display = self._apply_display_lp(self._data)

        # Filtered derivative (recomputed by _apply_filter; init unfiltered)
        self._data_diff_filtered = self._data_diff.copy()

        # Pulse regions (Nx2 [start, end] indices) — copy so the working
        # array doesn't alias the saved zone snapshot.
        self._pulse_regions = (z.pulse_regions.copy()
                               if z.pulse_regions is not None
                               else np.empty((0, 2), dtype=int))

        # Baseline-stage state
        self._filtered_signal = z.filtered_signal
        self._baseline_smooth = z.baseline_smooth
        self._detrended_data = z.detrended_data
        self._use_auto_tangent = z.use_auto_tangent

        # Threshold state (per-zone; default on first visit)
        self._pos_thresh = z.pos_thresh if z.pos_thresh is not None else 0.01
        self._neg_thresh = z.neg_thresh if z.neg_thresh is not None else -0.01

        # Output (per-zone)
        self.trendline = (z.trendline.copy() if z.trendline is not None
                          else np.zeros(self._n))
        self.detected_pulses = z.detected_pulses
        self.confirmed = z.confirmed

    # ============================= All-Zones page =======================

    def _build_all_zones_page(self):
        """Build the multi-zone overlay page + merge controls."""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: overlay figure, compact at the top with empty space below
        # (fixed pixel height, user-adjustable via the Plot Height spinbox).
        fig = Figure(figsize=(13, 4.5), tight_layout=False)
        style_mpl_figure(fig)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
        self._all_zones_fig = fig
        self._all_zones_canvas = FigureCanvas(fig)
        self._all_zones_ax = fig.add_subplot(1, 1, 1)
        self._az_default_plot_height = min(420, 120 * len(self._zones) + 90)
        self._all_zones_canvas.setFixedHeight(self._az_default_plot_height)

        canvas_col = QVBoxLayout()
        canvas_col.setContentsMargins(0, 0, 0, 0)
        canvas_col.addWidget(self._all_zones_canvas)
        canvas_col.addStretch()
        layout.addLayout(canvas_col, stretch=4)

        # Standard mpl toolbar bound to this canvas backs the nav buttons
        # below (the stage-tab nav handlers target _tab_axes, not this ax).
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _NavTB
        self._all_zones_toolbar = _NavTB(self._all_zones_canvas, page)
        self._all_zones_toolbar.hide()  # buttons drive it; toolbar stays out of view

        # Right: merge controls panel
        panel = QWidget()
        panel.setFixedWidth(220)
        panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(5, 5, 5, 5)

        grp = QGroupBox("Merge Zones")
        gl = QVBoxLayout(grp)
        note = QLabel("Factor of pulse template width")
        note.setStyleSheet("color: gray; font-size: 11px;")
        note.setWordWrap(True)
        gl.addWidget(note)

        gl.addWidget(QLabel("Merge Margin:"))
        self._spin_merge_margin = QDoubleSpinBox()
        self._spin_merge_margin.setRange(0.0, 5.0)
        self._spin_merge_margin.setSingleStep(0.1)
        self._spin_merge_margin.setDecimals(2)
        self._spin_merge_margin.setValue(getattr(self, '_live_region_margin', 0.2))
        self._spin_merge_margin.valueChanged.connect(self._recompute_all_zones_merge)
        gl.addWidget(self._spin_merge_margin)

        gl.addWidget(QLabel("Merge Distance:"))
        self._spin_merge_distance = QDoubleSpinBox()
        self._spin_merge_distance.setRange(0.0, 10.0)
        self._spin_merge_distance.setSingleStep(0.1)
        self._spin_merge_distance.setDecimals(2)
        self._spin_merge_distance.setValue(getattr(self, '_live_merge_distance', 0.5))
        self._spin_merge_distance.valueChanged.connect(self._recompute_all_zones_merge)
        gl.addWidget(self._spin_merge_distance)

        gl.addWidget(QLabel("Plot Height:"))
        self._spin_az_height = QSpinBox()
        self._spin_az_height.setRange(200, 1500)
        self._spin_az_height.setSingleStep(50)
        self._spin_az_height.setSuffix(" px")
        self._spin_az_height.setValue(self._az_default_plot_height)
        self._spin_az_height.valueChanged.connect(self._on_az_height_changed)
        gl.addWidget(self._spin_az_height)

        pl.addWidget(grp)
        pl.addSpacing(8)

        # Navigation for the All-Zones canvas (drives the mpl toolbar).
        nav_grp = QGroupBox("Navigation")
        nl = QVBoxLayout(nav_grp)
        btn_home = QPushButton("Home")
        btn_home.clicked.connect(self._all_zones_toolbar.home)
        nl.addWidget(btn_home)
        mode_row = QHBoxLayout()
        self._az_btn_zoom = QPushButton("Zoom Box")
        self._az_btn_zoom.setCheckable(True)
        self._az_btn_zoom.clicked.connect(self._on_az_toggle_zoom)
        mode_row.addWidget(self._az_btn_zoom)
        self._az_btn_pan = QPushButton("Pan")
        self._az_btn_pan.setCheckable(True)
        self._az_btn_pan.clicked.connect(self._on_az_toggle_pan)
        mode_row.addWidget(self._az_btn_pan)
        nl.addLayout(mode_row)
        from utils.export_figure_ui import make_export_button
        btn_export = make_export_button(lambda: self._all_zones_fig, parent=self)
        nl.addWidget(btn_export)
        pl.addWidget(nav_grp)
        pl.addSpacing(8)

        self._lbl_merged_count = QLabel("Merged regions: 0")
        pl.addWidget(self._lbl_merged_count)

        pl.addStretch()

        btn_finish = QPushButton("Finish")
        btn_finish.setObjectName("confirmBtn")
        btn_finish.clicked.connect(self._on_all_zones_finish)
        pl.addWidget(btn_finish)

        layout.addWidget(panel)
        self._all_zones_page = page
        self._final_merged = None

    def _recompute_all_zones_merge(self, *args):
        """Recompute the merged regions across all zones and redraw."""
        if self._all_zones_idx is None:
            return
        n = max((z.n for z in self._zones), default=0)
        # Widths are per-zone when no template is connected; the cross-zone
        # merge uses the widest so its margins cover every zone's pulses.
        if self._has_template:
            w = self._pulse_template_width
        else:
            w = max((int(z.template_width) for z in self._zones
                     if z.template_width is not None),
                    default=int(self._pulse_template_width))
        try:
            margin_factor = float(self._spin_merge_margin.value())
            distance_factor = float(self._spin_merge_distance.value())
        except (AttributeError, RuntimeError):
            margin_factor, distance_factor = 0.2, 0.5
        margin_samples = round(margin_factor * w)
        distance_samples = round(distance_factor * w)
        per_zone_regions = [z.pulse_regions for z in self._zones]
        self._final_merged = _merge_zone_regions(
            per_zone_regions, margin_samples, distance_samples, n)
        n_merged = len(self._final_merged)
        if hasattr(self, '_lbl_merged_count'):
            self._lbl_merged_count.setText(f"Merged regions: {n_merged}")
        self._draw_all_zones()

    def _zone_detrended(self, z):
        """Return a zone's detrended signal (filtered - baseline) or None."""
        if z.detrended_data is not None:
            return np.asarray(z.detrended_data)
        if z.filtered_signal is not None and z.baseline_smooth is not None:
            return np.asarray(z.filtered_signal) - np.asarray(z.baseline_smooth)
        return None

    def _draw_all_zones(self):
        """Compact waterfall: each zone's detrended trace normalized into a
        fixed band on its own row (uniform step) so spike size no longer
        stretches rows.

        Draw order: merged bands (background) -> per-zone region bands ->
        zone traces on top. Per-zone regions use the zone's own color clipped
        to its row band; merged regions are a light neutral full-height band.
        """
        ax = self._all_zones_ax
        ax.clear()
        cmap = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E',
                '#EDB120', '#4DBEEE', '#A2142F']

        dets = [self._zone_detrended(z) for z in self._zones]

        # Uniform row spacing; each trace normalized to a compact ±0.4 band.
        step = 1.0
        half_band = 0.45 * step

        n_zones = len(self._zones)
        ylo = -0.6 * step
        yhi = (n_zones - 1) * step + 0.6 * step if n_zones > 0 else 0.6 * step
        yspan = (yhi - ylo) or 1.0

        # 1) Merged-region bands FIRST as background (full height).
        merged = getattr(self, '_final_merged', None)
        if merged is not None:
            for s, e in np.asarray(merged).reshape(-1, 2):
                ax.axvspan(s / self._fs, e / self._fs,
                           alpha=0.08, color='#777777', zorder=0)

        # 2) Per-zone region bands, color-coded, clipped to the zone's row.
        # Zone 1 sits at the TOP; zones descend top -> bottom.
        y_ticks, y_labels = [], []
        for zi, z in enumerate(self._zones):
            color = cmap[zi % len(cmap)]
            offset = (n_zones - 1 - zi) * step
            y_ticks.append(offset)
            y_labels.append(f"Zone {zi + 1}")
            frac_lo = max(0.0, (offset - half_band - ylo) / yspan)
            frac_hi = min(1.0, (offset + half_band - ylo) / yspan)
            regs_arr = np.asarray(z.pulse_regions)
            regs = regs_arr.reshape(-1, 2) if regs_arr.size \
                else np.empty((0, 2))
            for s, e in regs:
                ax.axvspan(s / self._fs, e / self._fs,
                           ymin=frac_lo, ymax=frac_hi,
                           alpha=0.12, color=color, zorder=1)

        # 3) Zone traces on top (normalized into the band).
        for zi, z in enumerate(self._zones):
            color = cmap[zi % len(cmap)]
            offset = (n_zones - 1 - zi) * step
            det = dets[zi]
            if det is not None and len(det) > 0:
                # Robust, always finite & positive scale.
                p99 = float(np.percentile(np.abs(det), 99)) if det.size else 0.0
                if np.isfinite(p99) and p99 > 0:
                    scale = p99
                else:
                    m = float(np.max(np.abs(det))) if det.size else 0.0
                    scale = m if np.isfinite(m) and m > 0 else 1.0
                t = np.arange(len(det)) / self._fs
                dx, dy = _ds(t, det)
                # Clip the normalized deviation so a big artifact spike can't
                # bleed into the adjacent row (|deviation| <= 0.48 < 0.5*step).
                y = np.clip(dy / scale, -1.2, 1.2) * 0.4 + offset
                ax.plot(dx, y, color=color,
                        linewidth=1.0, zorder=3, label=f"Zone {zi + 1}")
            else:
                ax.axhline(offset, color=color, linewidth=0.8, alpha=0.4,
                           zorder=3, label=f"Zone {zi + 1}")

        ax.set_ylim(ylo, yhi)
        # Tight x-range: drop matplotlib's default 5% padding so the axes fit
        # the data (no empty margins before 0 / after the end).
        n_max = max((z.n for z in self._zones), default=0)
        t_max = (n_max - 1) / self._fs if n_max > 1 else 1.0
        ax.set_xlim(0, t_max)
        ax.margins(x=0)
        # Re-base the nav history so Home returns to this tight view.
        tb = getattr(self, '_all_zones_toolbar', None)
        if tb is not None:
            try:
                tb.update()
            except (RuntimeError, AttributeError):
                pass
        if y_ticks:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Zone (detrended, normalized)')
        n_merged = len(merged) if merged is not None else 0
        ax.set_title(f'All Zones (detrended waterfall) — '
                     f'{n_merged} merged regions')
        ax.grid(False)
        if n_zones > 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
        self._all_zones_canvas.draw_idle()

    def _on_az_toggle_zoom(self, checked):
        """All-Zones Zoom Box: mutually exclusive with Pan."""
        tb = self._all_zones_toolbar
        if checked:
            self._az_btn_pan.setChecked(False)
            if tb.mode.name == 'PAN':
                tb.pan()      # turn pan off first
            if tb.mode.name != 'ZOOM':
                tb.zoom()
        elif tb.mode.name == 'ZOOM':
            tb.zoom()         # toggle zoom off

    def _on_az_toggle_pan(self, checked):
        """All-Zones Pan: mutually exclusive with Zoom Box."""
        tb = self._all_zones_toolbar
        if checked:
            self._az_btn_zoom.setChecked(False)
            if tb.mode.name == 'ZOOM':
                tb.zoom()
            if tb.mode.name != 'PAN':
                tb.pan()
        elif tb.mode.name == 'PAN':
            tb.pan()

    def _on_az_height_changed(self, value):
        """Resize the All-Zones canvas to the requested pixel height."""
        try:
            self._all_zones_canvas.setFixedHeight(int(value))
            self._all_zones_canvas.draw_idle()
        except RuntimeError:
            pass

    def _on_all_zones_finish(self):
        """The real finish: lock in the merged result and close."""
        self._recompute_all_zones_merge()
        self.confirmed = True
        self._cached_settings = self.get_settings()
        self._autosave()
        self.accept()

    def _apply_fc(self, data):
        """Apply filter_config (list of filter dicts) to data for display."""
        if not self._filter_config or not isinstance(self._filter_config, list):
            return data.copy()
        try:
            from processing.filter_ui import _apply_single_filter
            result = data.copy()
            for fi in self._filter_config:
                result = _apply_single_filter(
                    result, self._fs,
                    fi['type'], fi['cutoff1'], fi.get('cutoff2', 0),
                    fi.get('notch_mode', 'freqBW'), fi.get('notch_bw', 2.0),
                )
            return result
        except Exception:
            return data.copy()

    # ============================= UI Setup =============================

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        # Zone tab bar (multi-zone only) above everything, narrow tabs.
        from PySide6.QtWidgets import QTabBar
        self._multi_zone = len(self._zones) > 1
        self._zone_bar = QTabBar()
        self._zone_bar.setExpanding(False)
        for i in range(len(self._zones)):
            self._zone_bar.addTab(f"Zone {i + 1}")
        self._all_zones_idx = None
        if self._multi_zone:
            self._all_zones_idx = self._zone_bar.addTab("All Zones")
        self._zone_bar.setVisible(self._multi_zone)
        self._zone_bar.setCurrentIndex(self._active_zone)
        self._zone_bar.currentChanged.connect(self._on_zone_changed)
        outer_layout.addWidget(self._zone_bar)

        # Top-level stack: page 0 per-zone stage content, page 1 all-zones.
        self._content_stack = QStackedWidget()
        outer_layout.addWidget(self._content_stack, stretch=1)

        # --- Per-zone page (the existing stage UI) ---
        zone_page = QWidget()
        main_layout = QHBoxLayout(zone_page)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: tab widget with matplotlib canvases (stage bar hidden; stages
        # driven only by the Next/Back/Finish wizard buttons).
        self._tabs = QTabWidget()
        self._tabs.tabBar().hide()

        # Block signals while adding tabs to prevent premature currentChanged
        self._tabs.blockSignals(True)

        self._tab_pages = []
        tab_titles = ["1. Detection", "2. Baseline"]
        for title in tab_titles:
            page = QWidget()
            self._tab_pages.append(page)
            self._tabs.addTab(page, title)

        # Disable baseline tab until detection is done
        self._tabs.setTabEnabled(1, False)

        self._tabs.blockSignals(False)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        main_layout.addWidget(self._tabs, stretch=4)

        # Right: side panel (changes per tab)
        self._panel = QWidget()
        self._panel.setFixedWidth(220)
        self._panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._panel.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self._panel_layout = QVBoxLayout(self._panel)
        self._panel_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self._panel)

        self._content_stack.addWidget(zone_page)

        # --- All-Zones page (multi-zone only) ---
        self._all_zones_page = None
        if self._multi_zone:
            self._build_all_zones_page()
            self._content_stack.addWidget(self._all_zones_page)
            self._update_all_zones_enabled()

        # Initialize Detection tab
        self._make_tab_figure(0)
        self._build_detection_panel()

    def _make_tab_figure(self, tab_index):
        """Create a matplotlib Figure + Canvas for a tab page."""
        fig = Figure(figsize=(13, 8), tight_layout=False)
        style_mpl_figure(fig)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.95,
                            bottom=0.08, hspace=0.30)
        canvas = FigureCanvas(fig)
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2)

        layout = QVBoxLayout(self._tab_pages[tab_index])
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)

        self._tab_figs[tab_index] = fig
        self._tab_canvases[tab_index] = canvas
        self._tab_axes[tab_index] = (ax_top, ax_bot)

        # Connect mouse events for zoom box / pan on every canvas
        self._mpl_cids.append((canvas, canvas.mpl_connect('button_press_event', self._mpl_press)))
        self._mpl_cids.append((canvas, canvas.mpl_connect('motion_notify_event', self._mpl_motion)))
        self._mpl_cids.append((canvas, canvas.mpl_connect('button_release_event', self._mpl_release)))

        return fig, canvas, ax_top, ax_bot

    def _clear_panel(self):
        self._clear_layout(self._panel_layout)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    # ============================= Stage 1: Detection ===================

    def _noise_thresholds(self, k=4.0):
        """Thresholds at ±k·sigma from the noise floor (robust MAD estimate).

        Keys detection off the *noise*, not the largest pulse: scaling by
        max(diff) puts the thresholds at 50% of the biggest pulse, so small
        pulses fall under them and go undetected. MAD ignores the pulses
        (a minority of samples), so k·sigma sits just above the noise and
        catches the small ones too.
        """
        ds = self._data_diff_filtered
        if len(ds) == 0:
            return 0.01, -0.01
        med = float(np.median(ds))
        mad = float(np.median(np.abs(ds - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(ds))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.01
        return med + k * sigma, med - k * sigma

    def _init_detection(self):
        """Set initial thresholds from the noise floor of the filtered derivative."""
        self._pos_thresh, self._neg_thresh = self._noise_thresholds()

    def _build_detection_panel(self):
        self._clear_panel()
        layout = self._panel_layout

        grp = QGroupBox("Detection Controls")
        grp_layout = QVBoxLayout(grp)

        # Sample-rate info (from the Global Sample Rate block)
        grp_info = QGroupBox("Sample Rate")
        gi = QVBoxLayout(grp_info)

        def _sr(v):
            return f"{v:,.0f} Hz" if v and v >= 1 else f"{v:g} Hz"
        gi.addWidget(QLabel(f"Base: {_sr(self._base_fs)}"))
        gi.addWidget(QLabel(f"Downsample: ×{self._downsample_factor}"))
        gi.addWidget(QLabel(f"Effective: {_sr(self._effective_fs)}"))
        grp_layout.addWidget(grp_info)

        grp_layout.addSpacing(5)

        # Lowpass filter + Apply
        fc_lo = min(1.0, self._nyq)  # guard against a degenerate (tiny) Nyquist
        grp_filt = QGroupBox("Lowpass Filter")
        gf = QVBoxLayout(grp_filt)
        gf.addWidget(QLabel("Data (Hz):"))
        self._spin_display_cutoff = QDoubleSpinBox()
        self._spin_display_cutoff.setRange(fc_lo, self._nyq)
        self._spin_display_cutoff.setSingleStep(10.0)
        self._spin_display_cutoff.setDecimals(1)
        self._spin_display_cutoff.setValue(min(self._applied_display_cutoff, self._nyq))
        gf.addWidget(self._spin_display_cutoff)
        gf.addWidget(QLabel("Detection (Hz):"))
        self._spin_cutoff = QDoubleSpinBox()
        self._spin_cutoff.setRange(fc_lo, self._nyq)
        self._spin_cutoff.setSingleStep(10.0)
        self._spin_cutoff.setDecimals(1)
        self._spin_cutoff.setValue(min(self._applied_cutoff, self._nyq))
        gf.addWidget(self._spin_cutoff)
        btn_filt_apply = QPushButton("Apply")
        btn_filt_apply.clicked.connect(self._on_apply_detection)
        gf.addWidget(btn_filt_apply)
        grp_layout.addWidget(grp_filt)

        grp_layout.addSpacing(5)

        # Merge Regions + Apply
        grp2 = QGroupBox("Merge Regions")
        grp2_layout = QVBoxLayout(grp2)
        note = QLabel("Factor of pulse template width")
        note.setStyleSheet("color: gray; font-size: 11px;")
        note.setWordWrap(True)
        grp2_layout.addWidget(note)
        grp2_layout.addWidget(QLabel("Margin:"))
        # Leading (pre-pulse) and trailing (post-pulse) pads, side by side.
        margin_row = QHBoxLayout()
        margin_row.setContentsMargins(0, 0, 0, 0)
        self._spin_padding = QDoubleSpinBox()
        self._spin_padding.setRange(0.01, 5.0)
        self._spin_padding.setSingleStep(0.1)
        self._spin_padding.setDecimals(2)
        self._spin_padding.setValue(getattr(self, '_live_region_margin', 0.2))
        self._spin_padding.setToolTip("Leading margin (before pulse)")
        # Mirror the spinboxes into `_live_*` so the values survive panel
        # rebuilds (switching to Baseline tab destroys these spinboxes,
        # and get_settings()'s spinbox read would otherwise raise and
        # fall back to defaults — losing the user's input).
        self._spin_padding.valueChanged.connect(
            lambda v: setattr(self, '_live_region_margin', float(v)))
        self._live_region_margin = float(self._spin_padding.value())
        self._spin_padding_post = QDoubleSpinBox()
        self._spin_padding_post.setRange(0.01, 5.0)
        self._spin_padding_post.setSingleStep(0.1)
        self._spin_padding_post.setDecimals(2)
        self._spin_padding_post.setValue(
            getattr(self, '_live_region_margin_post',
                    getattr(self, '_live_region_margin', 0.2)))
        self._spin_padding_post.setToolTip("Trailing margin (after pulse)")
        self._spin_padding_post.valueChanged.connect(
            lambda v: setattr(self, '_live_region_margin_post', float(v)))
        self._live_region_margin_post = float(self._spin_padding_post.value())
        margin_row.addWidget(self._spin_padding, 1)
        margin_row.addWidget(self._spin_padding_post, 1)
        grp2_layout.addLayout(margin_row)
        grp2_layout.addWidget(QLabel("Merge Distance:"))
        self._spin_spacing = QDoubleSpinBox()
        self._spin_spacing.setRange(0.01, 10.0)
        self._spin_spacing.setSingleStep(0.1)
        self._spin_spacing.setDecimals(2)
        self._spin_spacing.setValue(getattr(self, '_live_merge_distance', 0.5))
        self._spin_spacing.valueChanged.connect(
            lambda v: setattr(self, '_live_merge_distance', float(v)))
        self._live_merge_distance = float(self._spin_spacing.value())
        grp2_layout.addWidget(self._spin_spacing)

        # No template connected → the factors above have no template width to
        # multiply, so expose the assumed pulse width (in samples) directly.
        if not self._has_template:
            grp2_layout.addWidget(QLabel("Template Width (samples):"))
            self._spin_tpl_width = QSpinBox()
            self._spin_tpl_width.setRange(10, 1000000)
            self._spin_tpl_width.setSingleStep(50)
            self._spin_tpl_width.setValue(int(self._pulse_template_width))
            self._spin_tpl_width.setToolTip(
                "Assumed pulse width in samples (no template connected).\n"
                "The Margin and Merge Distance factors multiply this.")
            self._spin_tpl_width.valueChanged.connect(self._on_tpl_width_changed)
            grp2_layout.addWidget(self._spin_tpl_width)

        btn_merge_apply = QPushButton("Apply")
        btn_merge_apply.clicked.connect(self._on_apply_merge)
        grp2_layout.addWidget(btn_merge_apply)
        grp_layout.addWidget(grp2)

        grp_layout.addSpacing(5)

        # Threshold (upper/lower spinboxes + Auto)
        grp_th = QGroupBox("Threshold")
        gt = QVBoxLayout(grp_th)
        gt.addWidget(QLabel("Upper:"))
        self._spin_pos_thresh = QDoubleSpinBox()
        self._spin_pos_thresh.setDecimals(8)
        self._spin_pos_thresh.setRange(0.0, 1.0)
        self._spin_pos_thresh.setSingleStep(1e-7)
        gt.addWidget(self._spin_pos_thresh)
        gt.addWidget(QLabel("Lower:"))
        self._spin_neg_thresh = QDoubleSpinBox()
        self._spin_neg_thresh.setDecimals(8)
        self._spin_neg_thresh.setRange(-1.0, 0.0)
        self._spin_neg_thresh.setSingleStep(1e-7)
        gt.addWidget(self._spin_neg_thresh)
        self._sync_thresh_spins()  # seed from current thresholds
        self._spin_pos_thresh.valueChanged.connect(self._on_thresh_spin_changed)
        self._spin_neg_thresh.valueChanged.connect(self._on_thresh_spin_changed)
        btn_auto = QPushButton("Auto Thresholds")
        btn_auto.clicked.connect(self._on_auto_thresholds)
        gt.addWidget(btn_auto)
        grp_layout.addWidget(grp_th)

        layout.addWidget(grp)
        layout.addSpacing(10)

        self._add_nav_buttons(layout)

        layout.addStretch()

        # Save / Load settings
        io_layout = QHBoxLayout()
        btn_save = QPushButton("Save Settings")
        btn_save.clicked.connect(self._save_settings_file)
        io_layout.addWidget(btn_save)
        btn_load = QPushButton("Load Settings")
        btn_load.clicked.connect(self._load_settings_file)
        io_layout.addWidget(btn_load)
        layout.addLayout(io_layout)

        bottom = QHBoxLayout()
        btn_back = QPushButton("\u2190 Back")
        btn_back.clicked.connect(self._on_detection_back)
        bottom.addWidget(btn_back)
        btn_next = QPushButton("Next")
        btn_next.setObjectName("confirmBtn")
        btn_next.clicked.connect(self._on_next_detection)
        bottom.addWidget(btn_next)
        layout.addLayout(bottom)

    # ---------- view-adaptive refresh (shared by all tabs) ----------

    def _store_lines(self, tab_index, lines_top, lines_bot):
        """Store (line_artist, full_x, full_y) tuples for a tab."""
        self._tab_lines[tab_index] = (lines_top, lines_bot)

    def _refresh_view(self, tab_index=None):
        """Re-decimate visible data for current xlim and auto-fit Y."""
        if tab_index is None:
            tab_index = self._tabs.currentIndex()
        if tab_index not in self._tab_lines:
            return
        lines_top, lines_bot = self._tab_lines[tab_index]
        ax_top, ax_bot = self._tab_axes[tab_index]

        # Use same xlim for both axes (synced)
        x0, x1 = ax_bot.get_xlim()
        ax_top.set_xlim(x0, x1)

        # Update bottom lines and auto-fit Y
        y_mins, y_maxs = [], []
        for line, full_x, full_y in lines_bot:
            dx, dy = _ds(full_x, full_y, x0, x1)
            line.set_data(dx, dy)
            if len(dy) > 0:
                y_mins.append(np.min(dy))
                y_maxs.append(np.max(dy))
        if y_mins:
            ymin, ymax = min(y_mins), max(y_maxs)
            margin = 0.05 * (ymax - ymin) if ymax > ymin else abs(ymax) * 0.05 or 1
            ax_bot.set_ylim(ymin - margin, ymax + margin)

        # Update top lines and auto-fit Y
        if lines_top:
            yt_mins, yt_maxs = [], []
            for line, full_x, full_y in lines_top:
                dx, dy = _ds(full_x, full_y, x0, x1)
                line.set_data(dx, dy)
                if len(dy) > 0:
                    yt_mins.append(np.min(dy))
                    yt_maxs.append(np.max(dy))
            if yt_mins:
                ymin, ymax = min(yt_mins), max(yt_maxs)
                margin = 0.05 * (ymax - ymin) if ymax > ymin else abs(ymax) * 0.05 or 1
                ax_top.set_ylim(ymin - margin, ymax + margin)

        self._tab_canvases[tab_index].draw_idle()

    # ============================= Detection helpers ====================

    def _butter_lp_padded(self, data, wn, order=2):
        """Zero-phase Butterworth lowpass with reflect padding.

        filtfilt's default padding (~a few samples) is far too short for low
        cutoffs and leaves large end-effect transients (spikes) at the signal
        edges. Reflect-pad by 0.5 s first — matching the external Filter UI
        (filter_ui._apply_single_filter) — then trim back.
        """
        b, a = butter(order, wn, btype='low')
        data = np.asarray(data)
        pad = min(int(round(self._fs * 0.5)), max(0, len(data) - 1))
        if pad <= 0:
            return filtfilt(b, a, data)
        from processing.filter_ui import _reflect_pad
        out = filtfilt(b, a, _reflect_pad(data, pad))
        return out[pad:pad + len(data)]

    def _apply_filter(self, from_zone=False):
        # from_zone=True: filter with the active zone's restored cutoff
        # (self._applied_cutoff), ignoring the previous zone's stale spinbox.
        if from_zone:
            cutoff = self._applied_cutoff
        else:
            cutoff = self._spin_cutoff.value() if hasattr(self, '_spin_cutoff') else self._default_cutoff
        if cutoff <= 0 or cutoff >= self._nyq:
            cutoff = self._default_cutoff
        self._applied_cutoff = cutoff
        wn = cutoff / self._nyq
        wn = np.clip(wn, 0.01, 0.99)
        self._data_diff_filtered = self._butter_lp_padded(self._data_diff, wn)

    def _apply_display_lp(self, data):
        """Apply display lowpass + external filter config to data."""
        cutoff = self._applied_display_cutoff
        if cutoff > 0 and cutoff < self._nyq:
            wn = np.clip(cutoff / self._nyq, 0.01, 0.99)
            data = self._butter_lp_padded(data, wn)
        return self._apply_fc(data)

    def _apply_display_filter(self):
        """Apply display lowpass filter to data for top plot."""
        if hasattr(self, '_spin_display_cutoff'):
            self._applied_display_cutoff = self._spin_display_cutoff.value()
        self._data_display = self._apply_display_lp(self._data)

    def _on_apply_detection(self):
        """Apply filter and recompute thresholds (Lowpass Filter group Apply)."""
        self._apply_filter()
        self._apply_display_filter()
        self._update_thresholds()

    def _on_tpl_width_changed(self, v):
        """No-template pulse width changed — update the width the Margin /
        Merge Distance factors multiply. Click Apply to recompute regions."""
        self._pulse_template_width = int(v)
        self._live_template_width = int(v)
        self._endpoint_window_width = max(
            1, round(0.1 * self._pulse_template_width))
        # Stamp the active zone right away — the edit belongs to it alone.
        if self._active_zone < len(self._zones):
            self._zones[self._active_zone].template_width = int(v)

    def _on_apply_merge(self):
        """Merge Regions group Apply — recompute regions with the current
        margin / distance (both feed _compute_pulse_regions)."""
        self._update_thresholds()

    def _sync_thresh_spins(self):
        """Push the current thresholds into the spinboxes (guarded so it doesn't
        re-fire valueChanged)."""
        if not hasattr(self, '_spin_pos_thresh'):
            return
        self._syncing_thresh = True
        self._spin_pos_thresh.setValue(float(self._pos_thresh))
        self._spin_neg_thresh.setValue(float(self._neg_thresh))
        self._syncing_thresh = False

    def _on_thresh_spin_changed(self, _v=None):
        if getattr(self, '_syncing_thresh', False):
            return
        self._pos_thresh = max(0.0, float(self._spin_pos_thresh.value()))
        self._neg_thresh = min(0.0, float(self._spin_neg_thresh.value()))
        self._update_thresholds()

    def _on_auto_thresholds(self):
        if len(self._data_diff_filtered) == 0:
            return
        # Noise-floor based (see _noise_thresholds) — sits just above the noise
        # so small pulses are caught, instead of 50% of the biggest pulse.
        self._pos_thresh, self._neg_thresh = self._noise_thresholds()
        self._update_thresholds()

    def _on_detection_back(self):
        """Detection-stage Back.

        Multi-zone, not on the first zone → step back to the PREVIOUS zone
        (the wizard's prior step) instead of cancelling — this preserves every
        zone's work. On the first zone (or single zone) Back cancels the dialog
        as before.
        """
        if (self._multi_zone and self._all_zones_idx is not None
                and 0 < self._active_zone < self._all_zones_idx):
            self._on_zone_changed(self._active_zone - 1)
        else:
            self.reject()

    def _on_next_detection(self):
        self._deactivate_tools()
        self._on_apply_detection()
        self._current_stage = max(self._current_stage, 1)
        self._tabs.setTabEnabled(1, True)
        self._init_baseline_tab()
        self._build_baseline_panel()
        self._tabs.setCurrentIndex(1)

    def _update_thresholds(self, recompute=True):
        ds = self._data_diff_filtered

        # Find peaks above positive threshold
        pks_pos_idx, _ = find_peaks(ds, height=self._pos_thresh)
        # Find peaks below negative threshold
        pks_neg_idx, _ = find_peaks(-ds, height=abs(self._neg_thresh))

        # Recompute pulse regions from thresholds, or keep restored ones
        # (recompute=False is used when re-displaying a saved zone).
        if recompute:
            self._compute_pulse_regions(pks_pos_idx, pks_neg_idx)

        # Redraw — preserve zoom (save from one axis, apply to both)
        ax_top, ax_bot = self._tab_axes[0]
        t_max = self._t[-1] if self._n > 1 else 1
        prev_xlim = self._save_xlim(ax_bot, t_max)
        prev_ylim_bot = ax_bot.get_ylim()
        prev_ylim_top = ax_top.get_ylim()
        has_prev = prev_xlim is not None

        # Bottom plot: filtered diff with threshold lines and peaks
        ax_bot.clear()
        dx, dy = _ds(self._t_diff, ds)
        l_ds, = ax_bot.plot(dx, dy, color='#0066cc', linewidth=1.0)
        self._thresh_hl_pos = ax_bot.axhline(
            self._pos_thresh, color='#009900', linewidth=1.5, linestyle='--')
        self._thresh_hl_neg = ax_bot.axhline(
            self._neg_thresh, color='#cc0000', linewidth=1.5, linestyle='--')

        if len(pks_pos_idx) > 0:
            ax_bot.scatter(self._t_diff[pks_pos_idx], ds[pks_pos_idx],
                           c='#009900', s=20, zorder=5)
        if len(pks_neg_idx) > 0:
            ax_bot.scatter(self._t_diff[pks_neg_idx], ds[pks_neg_idx],
                           c='#cc0000', s=20, zorder=5)

        ax_bot.set_ylabel('Difference')
        ax_bot.set_xlabel('Time (s)')
        ax_bot.set_title('Threshold Selection')
        ax_bot.grid(False)
        self._restore_xlim(ax_bot, prev_xlim, 0, t_max)
        if has_prev:
            ax_bot.set_ylim(prev_ylim_bot)
        else:
            # Initial draw: make y-limits symmetric around y=0
            ylo, yhi = ax_bot.get_ylim()
            yabs = max(abs(ylo), abs(yhi))
            ax_bot.set_ylim(-yabs, yabs)

        # Top plot: filtered data with pulse region patches
        ax_top.clear()
        tx, ty = _ds(self._t, self._data_display)
        l_top, = ax_top.plot(tx, ty, linewidth=1.2)
        for row in self._pulse_regions:
            start_t = row[0] / self._fs
            end_t = row[1] / self._fs
            ax_top.axvspan(start_t, end_t, alpha=0.15, color='green')
        ax_top.set_ylabel('Amplitude')
        ax_top.set_title(f'Filtered Input Data with Pulse Regions '
                         f'({len(self._pulse_regions)} regions)')
        ax_top.grid(False)
        self._restore_xlim(ax_top, prev_xlim, 0, t_max)
        if has_prev:
            ax_top.set_ylim(prev_ylim_top)

        self._store_lines(0,
            [(l_top, self._t, self._data_display)],
            [(l_ds, self._t_diff, ds)])
        self._refresh_view(0)
        self._sync_thresh_spins()  # keep the Threshold group spinboxes in sync

    def _compute_pulse_regions(self, peak_locs_pos, peak_locs_neg):
        """Group detected peaks into pulse regions with padding and merging."""
        all_peak_locs = np.sort(np.concatenate([peak_locs_pos, peak_locs_neg]))
        padding_factor = self._spin_value(
            '_spin_padding', '_live_region_margin', 0.2)
        padding_factor_post = self._spin_value(
            '_spin_padding_post', '_live_region_margin_post', padding_factor)
        spacing_factor = self._spin_value(
            '_spin_spacing', '_live_merge_distance', 0.5)

        if len(all_peak_locs) == 0:
            self._pulse_regions = np.empty((0, 2), dtype=int)
            return

        regions = []
        i = 0
        while i < len(all_peak_locs):
            first_peak = all_peak_locs[i]
            pre_margin = round(padding_factor * self._pulse_template_width)
            region_start = max(0, first_peak - pre_margin)
            last_peak = first_peak
            j = i + 1
            while j < len(all_peak_locs):
                if all_peak_locs[j] - last_peak <= spacing_factor * self._pulse_template_width:
                    last_peak = all_peak_locs[j]
                    j += 1
                else:
                    break
            post_margin = round(padding_factor_post * self._pulse_template_width)
            region_end = min(self._n - 1, last_peak + post_margin)

            # Merge overlapping or nearly-touching regions
            merge_tol = max(1, round(spacing_factor * self._pulse_template_width))
            if len(regions) > 0 and region_start <= regions[-1][1] + merge_tol:
                regions[-1][1] = max(regions[-1][1], region_end)
            else:
                regions.append([region_start, region_end])
            i = j

        self._pulse_regions = np.array(regions, dtype=int) if regions else np.empty((0, 2), dtype=int)

    # ============================= Stage 2: Baseline ====================

    def _init_baseline_tab(self):
        if self._tab_figs[1] is None:
            self._make_tab_figure(1)

        # The input signal arrives pre-filtered (Filter UI upstream); apply
        # only the external filter config, never an extra lowpass — this
        # signal anchors the baseline and feeds the detrended output.
        self._filtered_signal = self._apply_fc(self._data)

        self._baseline_recompute_and_draw()

    def _build_baseline_panel(self):
        self._clear_panel()
        layout = self._panel_layout

        grp = QGroupBox("Baseline Controls")
        grp_layout = QVBoxLayout(grp)

        grp_layout.addWidget(QLabel("Interpolation Method:"))
        self._radio_auto_tangent = QRadioButton("Auto tangent poly2")
        self._radio_linear = QRadioButton("Linear")
        self._radio_auto_tangent.toggled.connect(self._on_toggle_interp_mode)
        # Reflect the active zone's flag so the radio and flag never diverge
        # (covers zone switches, restart, and load_settings).
        self._radio_auto_tangent.setChecked(self._use_auto_tangent)
        self._radio_linear.setChecked(not self._use_auto_tangent)
        grp_layout.addWidget(self._radio_auto_tangent)
        grp_layout.addWidget(self._radio_linear)

        grp_layout.addSpacing(5)

        grp_layout.addWidget(QLabel("Smooth Window:"))
        self._spin_smooth = QSpinBox()
        self._spin_smooth.setRange(1, self._n)
        self._spin_smooth.setValue(self._smooth_window_default)
        grp_layout.addWidget(self._spin_smooth)

        grp_layout.addSpacing(5)

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._baseline_recompute_and_draw)
        grp_layout.addWidget(btn_apply)

        layout.addWidget(grp)
        layout.addSpacing(10)

        self._add_nav_buttons(layout)

        layout.addStretch()

        # Restart button (full width)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self._on_restart)
        layout.addWidget(btn_restart)

        # Back / Finish buttons
        bottom_layout = QHBoxLayout()
        btn_back = QPushButton("\u2190 Back")
        btn_back.clicked.connect(lambda: self._tabs.setCurrentIndex(0))
        bottom_layout.addWidget(btn_back)
        finish_label = "Finish Zone" if self._multi_zone else "Finish"
        btn_finish = QPushButton(finish_label)
        btn_finish.setObjectName("confirmBtn")
        btn_finish.clicked.connect(self._on_finish)
        bottom_layout.addWidget(btn_finish)
        layout.addLayout(bottom_layout)

    def _on_toggle_interp_mode(self, checked):
        self._use_auto_tangent = checked

    def _baseline_build(self):
        """Build baseline by interpolating through pulse regions."""
        baseline = self._filtered_signal.copy()
        sig = self._filtered_signal

        for row in self._pulse_regions:
            idx_start = max(0, int(row[0]))
            idx_end = min(len(sig) - 1, int(row[1]))
            n_samples = idx_end - idx_start + 1
            if n_samples < 2:
                continue

            half_win = round(self._endpoint_window_width / 2)

            # Compute start endpoint value
            s0 = max(0, idx_start - half_win)
            s1 = min(len(sig), idx_start + half_win + 1)
            y_start = np.mean(sig[s0:s1])

            # Compute end endpoint value
            e0 = max(0, idx_end - half_win)
            e1 = min(len(sig), idx_end + half_win + 1)
            y_end = np.mean(sig[e0:e1])

            tau = np.linspace(0, 1, n_samples)

            if self._use_auto_tangent:
                # Hermite spline interpolation with tangent estimation
                x_factor = 0.2
                seg_len = max(1, round(x_factor * self._pulse_template_width))
                pre_start = max(0, idx_start - seg_len + 1)
                post_end = min(len(sig), idx_end + seg_len)

                slope_start = np.mean(np.diff(sig[pre_start:idx_start + 1])) if idx_start > pre_start else 0.0
                slope_end = np.mean(np.diff(sig[idx_end:post_end])) if post_end > idx_end + 1 else 0.0

                delta_x = n_samples
                t2 = tau * tau
                t3 = t2 * tau
                y_segment = ((2 * t3 - 3 * t2 + 1) * y_start +
                             (-2 * t3 + 3 * t2) * y_end +
                             (t3 - 2 * t2 + tau) * slope_start * delta_x +
                             (t3 - t2) * slope_end * delta_x)
            else:
                # Linear interpolation
                y_segment = (1 - tau) * y_start + tau * y_end

            baseline[idx_start:idx_end + 1] = y_segment

        return baseline

    def _baseline_recompute_and_draw(self):
        baseline_raw = self._baseline_build()

        smooth_window = self._spin_value('_spin_smooth',
                                         '_smooth_window_default')
        if smooth_window > 1 and smooth_window < len(baseline_raw):
            baseline_smoothed = uniform_filter1d(baseline_raw, size=smooth_window)
        else:
            baseline_smoothed = baseline_raw

        self._baseline_smooth = baseline_smoothed
        self._detrended_data = self._filtered_signal - baseline_raw

        ax_top, ax_bot = self._tab_axes[1]
        t_max = self._t[-1]

        ax_top.clear()
        tx1, ty1 = _ds(self._t, self._filtered_signal)
        l_filt, = ax_top.plot(tx1, ty1, color='#0072BD', linewidth=1.0)
        tx2, ty2 = _ds(self._t, baseline_raw)
        l_bl, = ax_top.plot(tx2, ty2, color='#D95319', linewidth=1.2)
        ax_top.set_title('Filtered Data with Calculated Baseline')
        ax_top.set_ylabel('Amplitude')
        ax_top.legend(['Filtered', 'Baseline'], loc='best')
        ax_top.grid(False)
        ax_top.set_xlim(0, t_max)

        ax_bot.clear()
        bx1, by1 = _ds(self._t, self._data)
        l_orig, = ax_bot.plot(bx1, by1, color='#b0b0b0', linewidth=0.8)
        bx2, by2 = _ds(self._t, baseline_smoothed)
        l_sm, = ax_bot.plot(bx2, by2, color='#D95319', linewidth=1.5)
        ax_bot.set_xlabel('Time (s)')
        ax_bot.set_ylabel('Amplitude')
        ax_bot.set_title(f'Smoothed Trendline (Window = {smooth_window} samples)')
        ax_bot.legend(['Original Data', 'Smoothed Trendline'], loc='best')
        ax_bot.grid(False)
        ax_bot.set_xlim(0, t_max)

        self._store_lines(1,
            [(l_filt, self._t, self._filtered_signal),
             (l_bl, self._t, baseline_raw)],
            [(l_orig, self._t, self._data),
             (l_sm, self._t, baseline_smoothed)])
        self._refresh_view(1)

    def _on_restart(self):
        # Reset to Detection stage
        self._current_stage = 0
        self._data_diff_filtered = self._data_diff.copy()
        self.trendline = np.zeros(self._n)
        self._pulse_regions = np.empty((0, 2), dtype=int)
        # Clear stage-2 state so a stale baseline can't outlive the restart.
        self._baseline_smooth = None
        self._detrended_data = None

        self._tabs.setTabEnabled(1, False)

        self._apply_filter()
        self._init_detection()
        self._update_thresholds()
        self._build_detection_panel()
        self._tabs.setCurrentIndex(0)

    def _on_finish(self):
        if self._baseline_smooth is None:
            # Finish reached without a computed baseline (skipped/failed
            # Baseline pass) — compute it now so the block never emits a
            # zero trendline for a zone the user finished.
            from utils.app_logger import logger
            logger.warning("[PulseDetectionBC] no baseline at Finish; "
                           "recomputing from current regions.")
            try:
                self._init_baseline_tab()
            except Exception as e:
                logger.error(f"[PulseDetectionBC] baseline recompute "
                             f"failed: {e}")
        if self._baseline_smooth is not None:
            self.trendline = self._baseline_smooth.copy()
        elif self._detrended_data is not None:
            self.trendline = self._data - self._detrended_data
        self.detected_pulses = self._pulse_regions.copy()

        if self._multi_zone:
            # Multi-zone: this is "Finish Zone" — mark the active zone done,
            # capture its outputs, then advance instead of closing.
            active = self._active_zone
            self._zones[active].finished = True
            self._save_active_zone()  # captures trendline/detected_pulses
            self._refresh_zone_tab_labels()
            self._update_all_zones_enabled()
            # Defer the zone switch: setCurrentIndex fires currentChanged ->
            # _on_zone_changed -> panel rebuild, which deletes this very
            # "Finish Zone" button mid-click and aborts the tab change in the
            # live event loop. Run it after the click handler returns.
            from PySide6.QtCore import QTimer
            if all(z.finished for z in self._zones):
                target = self._all_zones_idx
            else:
                target = next(i for i, z in enumerate(self._zones)
                              if not z.finished)
            def _go(t=target):
                try:
                    # Move the tab highlight without re-emitting currentChanged,
                    # then run the switch explicitly so it is deterministic in
                    # the live modal loop (relying on the signal alone proved
                    # unreliable). Force a synchronous repaint afterwards —
                    # draw_idle can sit unprocessed while the dialog is busy.
                    self._zone_bar.blockSignals(True)
                    self._zone_bar.setCurrentIndex(t)
                    self._zone_bar.blockSignals(False)
                    self._on_zone_changed(t)
                    canvases = list(getattr(self, '_tab_canvases', []))
                    azc = getattr(self, '_all_zones_canvas', None)
                    if azc is not None:
                        canvases.append(azc)
                    for c in canvases:
                        if c is not None:  # a tab not yet built has no canvas
                            c.draw()
                except RuntimeError:
                    pass  # bar/dialog destroyed before the tick fired
            QTimer.singleShot(0, _go)
            return

        self.confirmed = True
        # Capture the active zone's outputs into its state.
        self._save_active_zone()
        # Default any zone the user never processed (no baseline computed).
        from utils.app_logger import logger
        for i, z in enumerate(self._zones):
            if z.trendline is None or z.baseline_smooth is None:
                z.trendline = np.zeros(z.n)
                z.detected_pulses = np.empty((0, 2), dtype=int)
                logger.warning(
                    f"[PulseDetectionBC] zone {i + 1} left unprocessed; "
                    f"defaulting to zero trendline and no pulses.")
        # Cache widget values before dialog closes (widgets may be deleted)
        self._cached_settings = self.get_settings()
        self._autosave()
        self.accept()

    def _autosave(self):
        """Auto-save current settings to SavedTemplates/Detection/temp.json."""
        import json, os
        from utils.paths import saved_templates_dir
        folder = saved_templates_dir("Detection")
        try:
            settings = getattr(self, '_cached_settings', None) or self.get_settings()
            clean = {k: float(v) if hasattr(v, 'item') else v
                     for k, v in settings.items()}
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(clean, f, indent=2)
        except Exception as e:
            from utils.app_logger import logger
            logger.warning(f"[PulseDetectionBC] autosave failed: {e}")

    def get_settings(self):
        """Export current settings as a dict for save/load."""
        region_margin = float(self._spin_value(
            '_spin_padding', '_live_region_margin', 0.2))
        region_margin_post = float(self._spin_value(
            '_spin_padding_post', '_live_region_margin_post', region_margin))
        merge_distance = float(self._spin_value(
            '_spin_spacing', '_live_merge_distance', 0.5))
        # Snapshot the active zone so its live edits are in _zones before we
        # read every zone's settings. On the All-Zones page _active_zone is the
        # all-zones tab index (out of range for _zones), and every real zone
        # was already snapshotted on entry — so guard against indexing it.
        if self._active_zone < len(self._zones):
            self._save_active_zone()
        settings = {
            # Flat fields mirror the active zone — kept for single-zone /
            # legacy consumers that don't read the per-zone list.
            "deriv_cutoff": self._applied_cutoff,
            "display_cutoff": self._applied_display_cutoff,
            "pos_thresh": self._pos_thresh,
            "neg_thresh": self._neg_thresh,
            "endpoint_window": self._endpoint_window_width,
            "smooth_window": self._smooth_window_default,
            "use_auto_tangent": self._use_auto_tangent,
            "region_margin": region_margin,
            "region_margin_post": region_margin_post,
            "merge_distance": merge_distance,
            # No-template pulse width (samples); restored only when no template.
            "template_width": int(getattr(self, '_live_template_width', 500)),
        }
        # Per-zone settings so EVERY zone is restored, not just the active one.
        settings["zones"] = [{
            "deriv_cutoff": z.applied_cutoff,
            "display_cutoff": z.applied_display_cutoff,
            "pos_thresh": z.pos_thresh,
            "neg_thresh": z.neg_thresh,
            "use_auto_tangent": z.use_auto_tangent,
            "smooth_window": z.smooth_window,
            "region_margin": (z.region_margin if z.region_margin is not None
                              else region_margin),
            "region_margin_post": (z.region_margin_post
                                   if z.region_margin_post is not None
                                   else region_margin_post),
            "merge_distance": (z.merge_distance if z.merge_distance is not None
                               else merge_distance),
            "template_width": (int(z.template_width)
                               if z.template_width is not None
                               else int(getattr(self, '_live_template_width',
                                                500))),
        } for z in self._zones]
        return settings

    def load_settings(self, settings):
        """Apply previously saved settings.

        New per-zone format (``settings["zones"]``) restores every zone; the
        legacy flat format restores the active zone only (back-compatible).
        """
        if not isinstance(settings, dict):
            return
        # No-template pulse width — applies only when no template is connected
        # (with a template the width is fixed to the template length).
        self._live_template_width = int(settings.get(
            "template_width", getattr(self, '_live_template_width', 500)))
        self._default_template_width = self._live_template_width
        if not getattr(self, '_has_template', False):
            self._pulse_template_width = self._live_template_width
            self._endpoint_window_width = max(
                1, round(0.1 * self._pulse_template_width))
            if hasattr(self, '_spin_tpl_width'):
                try:
                    self._spin_tpl_width.blockSignals(True)
                    self._spin_tpl_width.setValue(self._live_template_width)
                    self._spin_tpl_width.blockSignals(False)
                except (RuntimeError, AttributeError):
                    pass
        zones_list = settings.get("zones")
        if isinstance(zones_list, list) and zones_list:
            self._load_all_zone_settings(settings, zones_list)
            return
        self._applied_cutoff = settings.get("deriv_cutoff", self._applied_cutoff)
        self._applied_display_cutoff = settings.get("display_cutoff", self._applied_display_cutoff)
        self._pos_thresh = settings.get("pos_thresh", self._pos_thresh)
        self._neg_thresh = settings.get("neg_thresh", self._neg_thresh)
        # endpoint_window is derived from the CURRENT pulse-template width, not
        # loaded: a saved absolute sample count from a different sample rate
        # (e.g. a 50 kHz session reused in a ×20-downsampled pipeline) would be
        # ~20× too large and average the bridge endpoints across neighbouring
        # pulses / the baseline trend — the sagging, stepped Linear baseline.
        self._endpoint_window_width = max(
            1, round(0.1 * self._pulse_template_width))
        self._smooth_window_default = settings.get("smooth_window", self._smooth_window_default)
        self._use_auto_tangent = settings.get("use_auto_tangent", self._use_auto_tangent)
        self._live_region_margin = settings.get("region_margin", 0.2)
        self._live_region_margin_post = settings.get(
            "region_margin_post", self._live_region_margin)
        self._live_merge_distance = settings.get("merge_distance", 0.5)
        # Update spinboxes if they exist so _apply_filter reads the right values
        if hasattr(self, '_spin_cutoff'):
            self._spin_cutoff.setValue(self._applied_cutoff)
        if hasattr(self, '_spin_display_cutoff'):
            self._spin_display_cutoff.setValue(self._applied_display_cutoff)
        # Re-apply filter with loaded cutoff
        # Do NOT call _init_detection — it would overwrite loaded thresholds
        self._apply_filter()
        self._apply_display_filter()
        # Rebuild panel first so spinboxes have the loaded merge values,
        # then update thresholds which reads from the new spinboxes
        self._build_detection_panel()
        self._update_thresholds()
        self.setFocus()

    def _load_all_zone_settings(self, settings, zones_list):
        """Restore per-zone thresholds/cutoffs for EVERY zone.

        Each zone's saved values are stashed into its ``_PerZoneState`` and
        the zone is marked ``restored`` so its first visit applies them (and
        recomputes regions from them) instead of re-deriving defaults. Zone 0
        is also applied to the live working set right now.
        """
        # Derived from the current template width, never the stale saved value
        # (see load_settings — avoids a full-rate window used on downsampled data).
        self._endpoint_window_width = max(
            1, round(0.1 * self._pulse_template_width))
        self._live_region_margin = settings.get("region_margin", 0.2)
        self._live_region_margin_post = settings.get(
            "region_margin_post", self._live_region_margin)
        self._live_merge_distance = settings.get("merge_distance", 0.5)

        for k, zs in enumerate(zones_list):
            if k >= len(self._zones) or not isinstance(zs, dict):
                break
            z = self._zones[k]
            z.applied_cutoff = zs.get("deriv_cutoff", z.applied_cutoff)
            z.applied_display_cutoff = zs.get(
                "display_cutoff", z.applied_display_cutoff)
            z.pos_thresh = zs.get("pos_thresh", z.pos_thresh)
            z.neg_thresh = zs.get("neg_thresh", z.neg_thresh)
            z.use_auto_tangent = zs.get("use_auto_tangent", z.use_auto_tangent)
            z.smooth_window = zs.get("smooth_window", z.smooth_window)
            # Fall back to the flat value for legacy saves lacking per-zone keys.
            z.region_margin = zs.get("region_margin", self._live_region_margin)
            z.region_margin_post = zs.get(
                "region_margin_post",
                zs.get("region_margin", self._live_region_margin_post))
            z.merge_distance = zs.get("merge_distance", self._live_merge_distance)
            z.template_width = zs.get("template_width",
                                      self._live_template_width)
            z.restored = True

        # Apply zone 0 to the live working set now (active zone is 0 at load).
        self._active_zone = 0
        z0 = self._zones[0]
        if z0.applied_cutoff is not None:
            self._applied_cutoff = z0.applied_cutoff
        if z0.applied_display_cutoff is not None:
            self._applied_display_cutoff = z0.applied_display_cutoff
        if z0.pos_thresh is not None:
            self._pos_thresh = z0.pos_thresh
        if z0.neg_thresh is not None:
            self._neg_thresh = z0.neg_thresh
        self._use_auto_tangent = z0.use_auto_tangent
        if z0.smooth_window is not None:
            self._smooth_window_default = z0.smooth_window
        if z0.region_margin is not None:
            self._live_region_margin = z0.region_margin
        if z0.region_margin_post is not None:
            self._live_region_margin_post = z0.region_margin_post
        if z0.merge_distance is not None:
            self._live_merge_distance = z0.merge_distance
        # Zone 0's per-zone width overrides the flat value applied by
        # load_settings (which mirrors whichever zone was active at save time).
        if (not getattr(self, '_has_template', False)
                and z0.template_width is not None):
            self._pulse_template_width = int(z0.template_width)
            self._live_template_width = int(z0.template_width)
            self._endpoint_window_width = max(
                1, round(0.1 * self._pulse_template_width))
            self._sync_cutoff_spin('_spin_tpl_width',
                                   self._pulse_template_width)
        z0.restored = False  # applied live now

        if hasattr(self, '_spin_cutoff'):
            self._spin_cutoff.setValue(self._applied_cutoff)
        if hasattr(self, '_spin_display_cutoff'):
            self._spin_display_cutoff.setValue(self._applied_display_cutoff)
        # Do NOT call _init_detection — it would overwrite loaded thresholds.
        self._apply_filter()
        self._apply_display_filter()
        self._build_detection_panel()
        self._update_thresholds()
        self.setFocus()

    def _default_folder(self):
        from utils.paths import saved_templates_dir
        return saved_templates_dir("Detection")

    def _save_settings_file(self):
        """Save settings to a JSON file."""
        import json
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Detection Settings", self._default_folder(),
            "JSON Files (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(self.get_settings(), f, indent=2)

    def _load_settings_file(self):
        """Load settings from a JSON file."""
        import json
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Detection Settings", self._default_folder(),
            "JSON Files (*.json)")
        if path:
            with open(path, "r") as f:
                settings = json.load(f)
            self.load_settings(settings)

    # ============================= Shared Helpers =======================

    def _on_tab_changed(self, index):
        if index > self._current_stage:
            self._tabs.setCurrentIndex(self._current_stage)
            return
        # Rebuild panel for the selected tab
        builders = [
            self._build_detection_panel,
            self._build_baseline_panel,
        ]
        if index <= self._current_stage:
            builders[index]()

    @staticmethod
    def _save_xlim(ax, full_range_max):
        """Return current xlim, or None if it's the matplotlib default (0,1)
        or already covers the full range."""
        xl = ax.get_xlim()
        # matplotlib default is (0, 1) before any data is plotted
        if xl == (0.0, 1.0) and full_range_max > 1.5:
            return None
        span = xl[1] - xl[0]
        if span >= full_range_max * 0.99:
            return None
        return xl

    @staticmethod
    def _restore_xlim(ax, saved, x_min, x_max):
        """Restore a previously saved xlim (clamped), or set to full range."""
        if saved is not None:
            s0, s1 = saved
            span = s1 - s0
            if span >= x_max - x_min:
                ax.set_xlim(x_min, x_max)
            elif s0 < x_min:
                ax.set_xlim(x_min, x_min + span)
            elif s1 > x_max:
                ax.set_xlim(x_max - span, x_max)
            else:
                ax.set_xlim(saved)
        else:
            ax.set_xlim(x_min, x_max)

    def _deactivate_tools(self):
        """Turn off zoom box and pan modes (stage + All-Zones)."""
        self._zb_mode = False
        self._pan_mode = False
        if hasattr(self, '_btn_zoom_box'):
            self._btn_zoom_box.setChecked(False)
        if hasattr(self, '_btn_pan'):
            self._btn_pan.setChecked(False)
        # Reset the All-Zones mpl toolbar mode + its mode buttons too, so
        # leaving the page doesn't strand it in PAN/ZOOM.
        tb = getattr(self, '_all_zones_toolbar', None)
        if tb is not None:
            try:
                if tb.mode.name == 'PAN':
                    tb.pan()
                elif tb.mode.name == 'ZOOM':
                    tb.zoom()
            except (RuntimeError, AttributeError):
                pass
        for name in ('_az_btn_zoom', '_az_btn_pan'):
            btn = getattr(self, name, None)
            if btn is not None:
                try:
                    btn.setChecked(False)
                except RuntimeError:
                    pass

    def _add_nav_buttons(self, layout):
        """Add Home / Zoom / Pan / Export buttons in a Navigation group."""
        nav_grp = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_grp)

        btn_home = QPushButton("Home")
        btn_home.clicked.connect(self._on_home)
        nav_layout.addWidget(btn_home)

        zoom_layout = QHBoxLayout()
        btn_zi = QPushButton("Zoom In")
        btn_zi.clicked.connect(self._on_zoom_in)
        zoom_layout.addWidget(btn_zi)
        btn_zo = QPushButton("Zoom Out")
        btn_zo.clicked.connect(self._on_zoom_out)
        zoom_layout.addWidget(btn_zo)
        nav_layout.addLayout(zoom_layout)

        tool_layout = QHBoxLayout()
        self._btn_zoom_box = QPushButton("Zoom Box")
        self._btn_zoom_box.setCheckable(True)
        self._btn_zoom_box.clicked.connect(self._on_toggle_zoom_box)
        tool_layout.addWidget(self._btn_zoom_box)
        self._btn_pan = QPushButton("Pan")
        self._btn_pan.setCheckable(True)
        self._btn_pan.clicked.connect(self._on_toggle_pan)
        tool_layout.addWidget(self._btn_pan)
        nav_layout.addLayout(tool_layout)

        from utils.export_figure_ui import make_export_button
        btn_export = make_export_button(self._current_fig, parent=self)
        nav_layout.addWidget(btn_export)

        layout.addWidget(nav_grp)

    # ----- interactive zoom box / pan -----

    def _on_toggle_zoom_box(self, checked):
        if checked:
            self._btn_pan.setChecked(False)
            self._pan_mode = False
            self._zb_mode = True
        else:
            self._zb_mode = False

    def _on_toggle_pan(self, checked):
        if checked:
            self._btn_zoom_box.setChecked(False)
            self._zb_mode = False
            self._pan_mode = True
        else:
            self._pan_mode = False

    def _set_synced_xlim(self, idx, xmin, xmax):
        """Set xlim on both top and bottom axes (synced like MATLAB), clamped to data range."""
        ax_top, ax_bot = self._tab_axes[idx]
        t_min = 0
        t_max = self._t[-1] if self._n > 1 else 1
        span = xmax - xmin
        if span > t_max - t_min:
            xmin, xmax = t_min, t_max
        elif xmin < t_min:
            xmin, xmax = t_min, t_min + span
        elif xmax > t_max:
            xmin, xmax = t_max - span, t_max
        ax_bot.set_xlim(xmin, xmax)
        ax_top.set_xlim(xmin, xmax)

    def _begin_thresh_blit(self, idx, ax, line):
        """Cache the static background once so threshold drags blit just the
        moving line instead of redrawing the whole (decimated) figure."""
        canvas = self._tab_canvases[idx]
        self._drag_line = line
        if line is not None:
            line.set_animated(True)
        try:
            canvas.draw()
            self._thresh_bg = canvas.copy_from_bbox(ax.figure.bbox)
            if line is not None:
                ax.draw_artist(line)
            canvas.blit(ax.figure.bbox)
        except Exception:
            self._thresh_bg = None

    def _mpl_press(self, event):
        idx = self._tabs.currentIndex()
        if self._tab_axes[idx] is None:
            return
        ax_top, ax_bot = self._tab_axes[idx]
        if event.button != 1:
            return

        # Only respond to clicks in top or bottom axes
        if event.inaxes not in (ax_top, ax_bot):
            return

        # Threshold drag on tab 0 (Detection stage) — takes priority
        if idx == 0 and event.inaxes == ax_bot and event.ydata is not None \
                and not self._zb_mode and not self._pan_mode:
            ylims = ax_bot.get_ylim()
            tol = 0.03 * (ylims[1] - ylims[0])
            if abs(event.ydata - self._pos_thresh) < tol:
                self._thresh_drag = 'pos'
                self._begin_thresh_blit(idx, ax_bot, self._thresh_hl_pos)
                return
            if abs(event.ydata - self._neg_thresh) < tol:
                self._thresh_drag = 'neg'
                self._begin_thresh_blit(idx, ax_bot, self._thresh_hl_neg)
                return

        if self._zb_mode:
            self._zb_start = event.xdata
            self._zb_span = ax_bot.axvspan(
                event.xdata, event.xdata, alpha=0.25, color='#4488cc')
            self._tab_canvases[idx].draw_idle()
        elif self._pan_mode:
            # Store pixel x and current xlim for stable panning
            self._pan_start = (event.x, ax_bot.get_xlim())

    def _mpl_motion(self, event):
        idx = self._tabs.currentIndex()
        if self._tab_axes[idx] is None:
            return
        ax_top, ax_bot = self._tab_axes[idx]

        # Threshold drag in progress — blit only the moving line (no full redraw)
        if self._thresh_drag is not None:
            if event.inaxes != ax_bot or event.ydata is None:
                return
            if self._thresh_drag == 'pos':
                self._pos_thresh = max(0, event.ydata)
                if self._thresh_hl_pos is not None:
                    self._thresh_hl_pos.set_ydata([self._pos_thresh])
            else:
                self._neg_thresh = min(0, event.ydata)
                if self._thresh_hl_neg is not None:
                    self._thresh_hl_neg.set_ydata([self._neg_thresh])
            canvas = self._tab_canvases[idx]
            if self._thresh_bg is not None and self._drag_line is not None:
                canvas.restore_region(self._thresh_bg)
                ax_bot.draw_artist(self._drag_line)
                canvas.blit(ax_bot.figure.bbox)
            else:
                canvas.draw_idle()
            return

        # Pan must keep tracking even when cursor leaves the axes
        if self._pan_mode and self._pan_start is not None:
            px_start, (xl0, xl1) = self._pan_start
            inv = ax_bot.transData.inverted()
            dx_data = inv.transform((px_start, 0))[0] - inv.transform((event.x, 0))[0]
            self._set_synced_xlim(idx, xl0 + dx_data, xl1 + dx_data)
            self._tab_canvases[idx].draw_idle()
            return

        if event.inaxes not in (ax_top, ax_bot) or event.xdata is None:
            return

        if self._zb_mode and self._zb_start is not None and self._zb_span is not None:
            x0, x1 = self._zb_start, event.xdata
            self._zb_span.set_x(min(x0, x1))
            self._zb_span.set_width(abs(x1 - x0))
            self._tab_canvases[idx].draw_idle()

    def _mpl_release(self, event):
        idx = self._tabs.currentIndex()
        if self._tab_axes[idx] is None:
            return
        ax_top, ax_bot = self._tab_axes[idx]

        # Finish threshold drag — stop blitting, then full recompute + redraw
        if self._thresh_drag is not None:
            if self._drag_line is not None:
                self._drag_line.set_animated(False)
            self._drag_line = None
            self._thresh_bg = None
            self._thresh_drag = None
            self._update_thresholds()
            return

        if self._zb_mode and self._zb_start is not None:
            if self._zb_span is not None:
                try:
                    self._zb_span.remove()
                except ValueError:
                    pass
                self._zb_span = None
            if event.xdata is not None:
                x0 = self._zb_start
                x1 = event.xdata
                if abs(x1 - x0) > 0.001:
                    self._set_synced_xlim(idx, min(x0, x1), max(x0, x1))
            self._zb_start = None
            self._zb_mode = False
            self._btn_zoom_box.setChecked(False)
            self._refresh_view(idx)
        if self._pan_mode and self._pan_start is not None:
            self._pan_start = None
            self._refresh_view(idx)
        self._pan_start = None

    # ----- standard nav buttons -----

    def _on_home(self):
        idx = self._tabs.currentIndex()
        if self._tab_axes[idx] is None:
            return
        ax_top, ax_bot = self._tab_axes[idx]
        t_max = self._t[-1] if self._n > 1 else 1
        ax_bot.set_xlim(0, t_max)
        ax_bot.autoscale(axis='y')
        ax_top.set_xlim(0, t_max)
        ax_top.autoscale(axis='y')
        self._refresh_view(idx)

    def _on_zoom_in(self):
        idx = self._tabs.currentIndex()
        if self._tab_axes[idx] is None:
            return
        _, ax_bot = self._tab_axes[idx]
        xl = ax_bot.get_xlim()
        mid = (xl[0] + xl[1]) / 2
        span = (xl[1] - xl[0]) / 4
        self._set_synced_xlim(idx, mid - span, mid + span)
        self._refresh_view(idx)

    def _on_zoom_out(self):
        idx = self._tabs.currentIndex()
        if self._tab_axes[idx] is None:
            return
        _, ax_bot = self._tab_axes[idx]
        xl = ax_bot.get_xlim()
        mid = (xl[0] + xl[1]) / 2
        span = (xl[1] - xl[0])
        self._set_synced_xlim(idx, mid - span, mid + span)
        self._refresh_view(idx)

    def _current_fig(self):
        """Return the matplotlib Figure on the active tab (or None)."""
        idx = self._tabs.currentIndex()
        return self._tab_figs[idx] if 0 <= idx < len(self._tab_figs) else None


def pulse_detection_bc(data, sample_rate=None, pulse_template=None,
                       filter_config=None, saved_settings=None,
                       downsample_factor=1):
    """Interactive pulse detection + baseline correction UI.

    Two tabs: (1) Detection — filter derivative + threshold-based pulse
    detection; (2) Baseline — interpolation through detected pulse regions.

    Parameters
    ----------
    data : np.ndarray
        1-D signal data.
    sample_rate : float, optional
        Sampling rate in Hz.
    pulse_template : np.ndarray, optional
        Pulse template for reference (used for width estimation).
    filter_config : list, optional
        Filter configuration list for display filtering on top plot.
    saved_settings : dict, optional
        Previously saved detection settings to pre-populate the dialog.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'trendline': np.ndarray — estimated baseline (same size as input)
        - 'detectedPulses': np.ndarray — Nx2 array of [start, end] indices
    """
    from utils.zones import zone_columns
    data = np.atleast_1d(data)
    zone_signals = zone_columns(data)  # list of 1-D columns; len 1 = single zone

    if sample_rate is None:
        sample_rate = 1.0

    dlg = _PulseDetectionBCDialog(zone_signals, sample_rate, pulse_template,
                                   filter_config,
                                   downsample_factor=downsample_factor)
    # Settings come only from the loadFile port (resolved by the runner).
    # Empty port → defaults; applied to the active zone (zone 0) as today.
    if saved_settings:
        dlg.load_settings(saved_settings)
    dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Detection without confirming.")

    per_zone = [{"trendline": z.trendline, "detectedPulses": z.detected_pulses}
                for z in dlg._zones]
    return _assemble_zone_results(
        per_zone, merged=getattr(dlg, "_final_merged", None))


def _merge_zone_regions(per_zone_regions, margin, merge_distance, n):
    """Pool all zones' [start, end] regions, pad each by *margin* samples
    (clamped to [0, n-1]), then merge regions whose gap <= *merge_distance*
    samples into one. Returns a sorted Nx2 int array (empty (0,2) if none).
    margin/merge_distance are in samples.
    """
    regs = []
    for arr in per_zone_regions:
        a = np.asarray(arr, dtype=int).reshape(-1, 2) if np.asarray(arr).size else None
        if a is None:
            continue
        for s, e in a:
            regs.append((max(0, int(s) - margin), min(n - 1, int(e) + margin)))
    if not regs:
        return np.empty((0, 2), dtype=int)
    regs.sort()
    merged = [list(regs[0])]
    for s, e in regs[1:]:
        if s - merged[-1][1] <= merge_distance:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array(merged, dtype=int)


def _assemble_zone_results(per_zone, merged=None):
    """Combine per-zone Detection BC results into the block output.

    Returns ``{trendline, detectedPulses, detectedPerZone}``:
      - trendline: 1-D for one zone, else N×Z column stack.
      - detectedPerZone: per-zone regions via wrap_zones (bare Nx2 for one zone).
      - detectedPulses: the MERGED Nx2 across zones. For one zone this is that
        zone's regions (bare). For >1 zone, *merged* is used if supplied (the
        interactive All-Zones result); otherwise a default union+merge is
        computed (margin=0, merge_distance=0).
    """
    from utils.zones import wrap_zones
    trends = [np.asarray(z["trendline"]).ravel() for z in per_zone]
    pulses = [np.asarray(z["detectedPulses"], dtype=int) for z in per_zone]
    trendline = trends[0] if len(per_zone) == 1 else np.column_stack(trends)
    detected_per_zone = wrap_zones(pulses)
    if len(per_zone) == 1:
        detected = pulses[0]
    elif merged is not None:
        detected = np.asarray(merged, dtype=int)
    else:
        n = len(trends[0])
        detected = _merge_zone_regions(pulses, margin=0, merge_distance=0, n=n)
    return {"trendline": trendline,
            "detectedPulses": detected,
            "detectedPerZone": detected_per_zone}
