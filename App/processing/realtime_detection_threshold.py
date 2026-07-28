"""RT Detection (Threshold): pure helpers and the interactive QDialog.

Causal pulse detection on the running first-difference of the signal
(matches Pulse Detection BC's approach but tuned for real-time use).

Why diff-then-threshold:
- Baseline drift is rejected: a slowly varying baseline produces a near-
  zero diff, so the detector doesn't see drift as elevated signal.
- Sharp pulse edges produce strong positive diffs at the rising edge and
  strong negative diffs at the falling edge — natural rising/falling
  threshold pair.
- Causal IIR smoothing on the diff suppresses sample-to-sample noise
  while preserving rising/falling-edge energy. One-pole IIR is FPGA-
  trivial (single multiplier), so this matches the eventual real-time
  hardware path.

Companion to RT Detection (CWT): same dialog layout, same sliding-window
live-feed simulation.
"""

import json
import os
from time import perf_counter

import numpy as np

from utils.paths import saved_templates_dir


_DEFAULT_PARAMS = {
    "thresholdKPos":     3.0,         # rising threshold = +kPos · σ
    "thresholdKNeg":     3.0,         # falling threshold = -kNeg · σ
    "filterAlpha":       0.3,         # one-pole IIR: y[n] = α·x[n] + (1-α)·y[n-1]
    "mergeGapMs":        100.0,       # peak-to-peak grouping window (peaks closer than this stay in one region)
    "padBeforeFactor":   0.2,         # pad before = padBeforeFactor × core_width (per cluster)
    "padAfterFactor":    0.2,         # pad after  = padAfterFactor  × core_width
    "minEventMs":        5.0,         # reject events shorter than this (applied to peak-cluster span, padding excluded)
    # Pre-filters applied to the signal BEFORE computing the diff
    "lpfEnabled":        False,
    "lpfCutoffHz":       1000.0,
    "lpfOrder":          2,
    "notchEnabled":      False,
    "notchHz":           60.0,
    "notchQ":            30.0,
    "runMode":           "batch",     # "batch" or "live"
    "simSpeed":          "10x",
    "liveWindowSec":     3.0,
}


# --- Pure algorithm helpers ----------------------------------------------

def _causal_filter(b, a, x):
    """Apply a causal IIR filter with steady-state initial conditions
    (so the output doesn't ramp from 0 over the first ~1/(1-pole_radius)
    samples, which would otherwise create a fake startup transient that
    the diff/threshold detector picks up as an event).
    """
    from scipy.signal import lfilter, lfilter_zi
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    zi = lfilter_zi(b, a) * float(arr[0])
    y, _ = lfilter(b, a, arr, zi=zi)
    return y


def _causal_lowpass(x, cutoff_hz, sample_rate, order=2):
    """Causal Butterworth lowpass (uses lfilter, not filtfilt, so it's
    valid for real-time / streaming use). Returns x unchanged if cutoff
    is invalid or out of range.
    """
    from scipy.signal import butter
    sr = float(sample_rate)
    nyq = sr / 2.0
    fc = float(cutoff_hz)
    if fc <= 0 or fc >= nyq:
        return np.asarray(x, dtype=float)
    wn = max(0.001, min(0.999, fc / nyq))
    b, a = butter(int(order), wn, btype="low")
    return _causal_filter(b, a, x)


def _causal_notch(x, freq_hz, q, sample_rate):
    """Causal IIR notch filter (lfilter). Removes a narrow band around
    `freq_hz` (mains interference, etc.). Returns x unchanged if freq
    is invalid.
    """
    from scipy.signal import iirnotch
    sr = float(sample_rate)
    nyq = sr / 2.0
    f0 = float(freq_hz)
    if f0 <= 0 or f0 >= nyq:
        return np.asarray(x, dtype=float)
    w0 = f0 / nyq
    b, a = iirnotch(w0, max(1.0, float(q)))
    return _causal_filter(b, a, x)


def _apply_iir(x, alpha):
    """One-pole causal IIR: y[n] = α·x[n] + (1-α)·y[n-1].

    α ∈ (0, 1). α near 1 = pass-through, α near 0 = heavy smoothing.
    Outside that range or alpha is None → return x unchanged.
    """
    a = float(alpha) if alpha is not None else 1.0
    if a >= 1.0 or a <= 0.0:
        return np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=np.float64)
    y = float(x[0]) if len(x) else 0.0
    one_minus_a = 1.0 - a
    for i in range(len(x)):
        y = a * float(x[i]) + one_minus_a * y
        out[i] = y
    return out


def _group_peaks_into_regions(peaks_sorted, n_signal, max_gap,
                              pad_before_factor, pad_after_factor):
    """Group sorted peak indices into clusters; pad each cluster by a
    factor of its own core width (peak-to-peak span).

    Two peaks separated by ≤ `max_gap` samples join the same cluster.
    For each cluster:
      core_width = last_peak − first_peak + 1
      pad_before = round(pad_before_factor × core_width)
      pad_after  = round(pad_after_factor  × core_width)
      region = [first_peak − pad_before, last_peak + pad_after]

    Wider clusters get wider paddings, so padding scales with the actual
    detected event size — mirrors PulseDetectionBC's
    `padding_factor × pulse_template_width` but without needing a known
    template (uses the cluster span itself as the reference).

    Returns (padded_regions, core_regions). Padded regions are NOT
    merged here — that's a separate post-step.
    """
    if len(peaks_sorted) == 0:
        return [], []
    padded = []
    cores = []
    i = 0
    while i < len(peaks_sorted):
        first = int(peaks_sorted[i])
        last = first
        j = i + 1
        while j < len(peaks_sorted):
            if int(peaks_sorted[j]) - last <= max_gap:
                last = int(peaks_sorted[j])
                j += 1
            else:
                break
        core_width = max(1, last - first + 1)
        pad_before = max(0, int(round(pad_before_factor * core_width)))
        pad_after = max(0, int(round(pad_after_factor * core_width)))
        rs = max(0, first - pad_before)
        re = min(n_signal - 1, last + pad_after)
        padded.append((rs, re))
        cores.append((first, last))
        i = j
    return padded, cores


def detect_threshold(signal, sample_rate, params=None):
    """End-to-end: returns dict with detectedPulses, score (signed
    filtered diff), coincidentHint, thresholdPos / thresholdNeg, sigma.

    score is the filtered diff aligned to signal samples (length matches
    signal; first sample is 0).
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        # Legacy compat: a single `thresholdK` propagates to pos/neg
        # unless caller explicitly supplied them.
        if ("thresholdK" in params
                and "thresholdKPos" not in params
                and "thresholdKNeg" not in params):
            params = dict(params)
            params["thresholdKPos"] = params["thresholdK"]
            params["thresholdKNeg"] = params["thresholdK"]
        p.update(params)

    sig_raw = np.asarray(signal, dtype=float).ravel()
    n = sig_raw.size
    sr = float(sample_rate) if sample_rate else 10000.0

    empty = {
        "detectedPulses": np.zeros((0, 2), dtype=int),
        "score":          np.zeros(n, dtype=float),
        "signalFiltered": sig_raw.copy(),
        "coincidentHint": np.zeros(0, dtype=np.int8),
        "thresholdPos":   0.0,
        "thresholdNeg":   0.0,
        "threshold":      0.0,
        "sigma":          0.0,
    }
    if n < 2:
        return empty

    # Pre-filters: applied to the SIGNAL (not the diff). Both are causal
    # so the same filter stage works in batch and real-time. Notch first
    # (kill mains interference), then lowpass (suppress out-of-band
    # noise above the highest pulse-edge frequency of interest).
    sig = sig_raw
    if bool(p.get("notchEnabled", False)):
        sig = _causal_notch(sig,
                            freq_hz=p.get("notchHz", 60.0),
                            q=p.get("notchQ", 30.0),
                            sample_rate=sr)
    if bool(p.get("lpfEnabled", False)):
        sig = _causal_lowpass(sig,
                              cutoff_hz=p.get("lpfCutoffHz", 1000.0),
                              sample_rate=sr,
                              order=int(p.get("lpfOrder", 2)))
    sig = np.asarray(sig, dtype=float)

    # Causal first difference + one-pole IIR smoothing.
    diff = np.diff(sig)
    alpha = float(p.get("filterAlpha", 0.3))
    diff_filt = _apply_iir(diff, alpha)

    # Robust σ on the filtered diff (zero-mean for stationary noise).
    med = float(np.median(diff_filt))
    sigma = 1.4826 * float(np.median(np.abs(diff_filt - med)))
    if sigma <= 0:
        return empty

    # Independent positive and negative threshold multipliers. Old
    # saved settings with a symmetric `thresholdK` still load via the
    # fallback below.
    legacy_k = float(p.get("thresholdK", 3.0))
    k_pos = float(p.get("thresholdKPos", legacy_k))
    k_neg = float(p.get("thresholdKNeg", legacy_k))
    t_pos = +k_pos * sigma
    t_neg = -k_neg * sigma

    # Find threshold-crossing peaks on the filtered diff. Positive peaks
    # are upward edges (signal rising); negative peaks are downward edges
    # (signal falling). For multi-segment pulses both kinds appear within
    # one event (one rising edge per plateau, one falling edge per gap).
    from scipy.signal import find_peaks
    pos_idx, _ = find_peaks(diff_filt, height=t_pos)
    neg_idx, _ = find_peaks(-diff_filt, height=-t_neg)
    # Convert from diff indices to signal sample indices.
    all_peaks = np.sort(np.concatenate([pos_idx + 1, neg_idx + 1]))

    merge_gap = max(1, int(round(float(p.get("mergeGapMs", 100.0))
                                 * 0.001 * sr)))
    pad_before_factor = max(0.0, float(p.get("padBeforeFactor", 0.2)))
    pad_after_factor = max(0.0, float(p.get("padAfterFactor", 0.2)))
    min_event = max(1, int(round(float(p.get("minEventMs", 5.0))
                                 * 0.001 * sr)))

    if all_peaks.size == 0:
        score = np.zeros(n, dtype=float)
        score[1:] = diff_filt
        return {
            "detectedPulses": np.zeros((0, 2), dtype=int),
            "coreRegions":    np.zeros((0, 2), dtype=int),
            "score":          score,
            "signalFiltered": sig,
            "coincidentHint": np.zeros(0, dtype=np.int8),
            "thresholdPos":   float(t_pos),
            "thresholdNeg":   float(t_neg),
            "threshold":      float(t_pos),
            "sigma":          float(sigma),
        }

    # Peak grouping into clusters (one core per cluster). Padding is a
    # factor of each cluster's own core width — wider events get wider
    # paddings.
    padded_grouped, cores_grouped = _group_peaks_into_regions(
        all_peaks, n_signal=n, max_gap=merge_gap,
        pad_before_factor=pad_before_factor,
        pad_after_factor=pad_after_factor,
    )

    # Drop clusters whose CORE span (peak-to-peak, padding excluded) is
    # shorter than min_event. This keeps padding decoupled from the
    # noise-rejection threshold.
    keep = [
        i for i in range(len(cores_grouped))
        if (cores_grouped[i][1] - cores_grouped[i][0] + 1) >= min_event
    ]
    padded_kept = [padded_grouped[i] for i in keep]
    cores_kept = [cores_grouped[i] for i in keep]

    # Post-padding merge: if two adjacent padded regions are within
    # `merge_gap` samples of each other (or overlap), fuse them into one
    # padded region. The cores from both clusters are preserved (visualized
    # as separate dark spans inside the merged light region).
    merged_padded = []
    merged_cores = []   # flat list of all cores (each core stays as is)
    for pp, cc in zip(padded_kept, cores_kept):
        if merged_padded and pp[0] - merged_padded[-1][1] - 1 <= merge_gap:
            merged_padded[-1] = (merged_padded[-1][0],
                                 max(merged_padded[-1][1], pp[1]))
        else:
            merged_padded.append(pp)
        merged_cores.append(cc)

    regions_arr = (np.asarray(merged_padded, dtype=int)
                   if merged_padded else np.zeros((0, 2), dtype=int))
    cores_arr = (np.asarray(merged_cores, dtype=int)
                 if merged_cores else np.zeros((0, 2), dtype=int))

    # Score for plotting: filtered diff, length matched to signal
    # (first sample is 0 since diff has N-1 samples).
    score = np.zeros(n, dtype=float)
    score[1:] = diff_filt

    return {
        "detectedPulses": regions_arr,
        "coreRegions":    cores_arr,
        "score":          score,
        "signalFiltered": sig,
        "coincidentHint": np.zeros(len(regions_arr), dtype=np.int8),
        "thresholdPos":   float(t_pos),
        "thresholdNeg":   float(t_neg),
        "threshold":      float(t_pos),  # legacy field (positive threshold)
        "sigma":          float(sigma),
    }


# --- Settings persistence -------------------------------------------------

def save_state(params, filepath):
    payload = {"version": 1}
    payload.update({k: params[k] for k in _DEFAULT_PARAMS if k in params})
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, filepath)


def load_state(filepath):
    with open(filepath, "r") as f:
        loaded = json.load(f)
    # Legacy compat: old saved files used a single `thresholdK`.
    if ("thresholdK" in loaded
            and "thresholdKPos" not in loaded
            and "thresholdKNeg" not in loaded):
        loaded["thresholdKPos"] = loaded["thresholdK"]
        loaded["thresholdKNeg"] = loaded["thresholdK"]
    # Older schemas used absolute-ms paddings (`paddingMs` /
    # `padBeforeMs` / `padAfterMs`). Those keys have no clean conversion
    # to the current factor-based padding, so they're ignored — defaults
    # take effect on load.
    out = dict(_DEFAULT_PARAMS)
    for k in _DEFAULT_PARAMS:
        if k in loaded:
            out[k] = loaded[k]
    return out


# --- Dialog ---------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QRadioButton, QButtonGroup, QGroupBox,
    QCheckBox, QFileDialog, QMessageBox, QWidget, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure


_SIM_SPEEDS = [
    ("1x",   1.0),
    ("10x",  10.0),
    ("100x", 100.0),
    ("asap", 0.0),
]


class RealtimeDetectionThresholdDialog(QDialog):
    """Interactive dialog for tuning + previewing threshold-based pulse
    detection. Layout and live-feed simulation match the CWT block.
    """

    def __init__(self, signal, sample_rate, params,
                 current_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RT Detection (Threshold)")
        self.resize(1200, 760)

        try:
            from utils.dialog_style import apply_dialog_style
            apply_dialog_style(self)
        except ImportError:
            pass

        self._signal = np.asarray(signal, dtype=float).ravel()
        self._sample_rate = float(sample_rate)
        self._params = dict(_DEFAULT_PARAMS)
        self._params.update(params or {})
        self._current_path = current_path
        if current_path:
            self.setWindowTitle(
                f"RT Detection (Threshold) — {os.path.basename(current_path)}")

        self._result = None
        self._sim_timer = None
        self._sim_revealed = 0
        self._sim_paused = False
        self._sim_t0 = 0.0
        self._sim_revealed_at_resume = 0
        self._live_view = False
        # Threshold drag state: 'pos' / 'neg' / None
        self._thr_drag = None
        # Artist refs so motion handler can move the lines smoothly.
        self._thr_line_pos = None
        self._thr_line_neg = None
        # Cached score-axis background for blit-based dragging.
        self._drag_bg = None

        self._build_ui()
        self._sync_widgets_from_params()
        self._run_batch_and_redraw()

    # --- UI construction -------------------------------------------------

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        plot_col = QWidget()
        plot_col_layout = QVBoxLayout(plot_col)
        plot_col_layout.setContentsMargins(0, 0, 0, 0)
        plot_col_layout.setSpacing(0)

        self._fig = Figure(figsize=(8, 6))
        style_mpl_figure(self._fig)
        self._ax_sig = self._fig.add_subplot(2, 1, 1)
        self._ax_score = self._fig.add_subplot(2, 1, 2, sharex=self._ax_sig)
        self._fig.subplots_adjust(hspace=0.25, left=0.08, right=0.97,
                                  top=0.96, bottom=0.08)
        self._canvas = FigureCanvas(self._fig)
        self._nav_toolbar = NavigationToolbar(self._canvas, plot_col)
        plot_col_layout.addWidget(self._nav_toolbar)
        plot_col_layout.addWidget(self._canvas, stretch=1)
        main.addWidget(plot_col, stretch=4)

        # Mouse drag for threshold lines (only on the bottom score axis,
        # only when the navigation toolbar isn't in pan/zoom mode).
        self._canvas.mpl_connect("button_press_event",
                                 self._on_thr_press)
        self._canvas.mpl_connect("motion_notify_event",
                                 self._on_thr_motion)
        self._canvas.mpl_connect("button_release_event",
                                 self._on_thr_release)

        panel = QWidget()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        main.addWidget(panel)

        self._sr_label = QLabel(f"Sample rate: {int(self._sample_rate)} Hz")
        self._sr_label.setWordWrap(True)
        panel_layout.addWidget(self._sr_label)

        panel_layout.addSpacing(5)

        det_box = QGroupBox("Detection")
        det_grid = QGridLayout(det_box)
        row = 0

        det_grid.addWidget(QLabel("Threshold k+:"), row, 0)
        self._thr_pos_spin = QDoubleSpinBox()
        self._thr_pos_spin.setRange(0.1, 50.0)
        self._thr_pos_spin.setSingleStep(0.5)
        self._thr_pos_spin.setDecimals(2)
        self._thr_pos_spin.setToolTip(
            "Rising-edge threshold multiplier: rising edge fires when "
            "filtered Δsignal > +k·σ. Drag the upper green line in the "
            "bottom plot to adjust visually.")
        det_grid.addWidget(self._thr_pos_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Threshold k−:"), row, 0)
        self._thr_neg_spin = QDoubleSpinBox()
        self._thr_neg_spin.setRange(0.1, 50.0)
        self._thr_neg_spin.setSingleStep(0.5)
        self._thr_neg_spin.setDecimals(2)
        self._thr_neg_spin.setToolTip(
            "Falling-edge threshold multiplier: falling edge fires when "
            "filtered Δsignal < −k·σ. Drag the lower red line in the "
            "bottom plot to adjust visually.")
        det_grid.addWidget(self._thr_neg_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Smoothing:"), row, 0)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.01, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setToolTip(
            "One-pole causal IIR on the Δ signal: y[n] = α·x[n] + "
            "(1−α)·y[n−1]. Lower = heavier smoothing of sample-to-"
            "sample noise on the diff; α=1 disables smoothing.")
        det_grid.addWidget(self._alpha_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Merge gap (ms):"), row, 0)
        self._merge_spin = QDoubleSpinBox()
        self._merge_spin.setRange(0.0, 1000.0)
        self._merge_spin.setSingleStep(5.0)
        self._merge_spin.setDecimals(1)
        self._merge_spin.setToolTip(
            "Maximum peak-to-peak spacing within one region. Threshold "
            "crossings closer than this are grouped into the same "
            "detected pulse — set this to be wider than the longest "
            "internal gap of your pulse template.")
        det_grid.addWidget(self._merge_spin, row, 1)
        row += 1

        # "Event width" sub-section: padding before and after the peak
        # cluster. The cluster span itself is the core event; these
        # extend the output region without merging adjacent events.
        ew_label = QLabel("Event width:")
        f = ew_label.font()
        f.setBold(True)
        ew_label.setFont(f)
        det_grid.addWidget(ew_label, row, 0, 1, 2)
        row += 1

        det_grid.addWidget(QLabel("  Pad before (×):"), row, 0)
        self._pad_before_spin = QDoubleSpinBox()
        self._pad_before_spin.setRange(0.0, 5.0)
        self._pad_before_spin.setSingleStep(0.1)
        self._pad_before_spin.setDecimals(2)
        self._pad_before_spin.setToolTip(
            "Pre-event padding as a factor of the cluster core width "
            "(peak-to-peak span). Wider events get proportionally wider "
            "padding, so this scales naturally with pulse size.")
        det_grid.addWidget(self._pad_before_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("  Pad after (×):"), row, 0)
        self._pad_after_spin = QDoubleSpinBox()
        self._pad_after_spin.setRange(0.0, 5.0)
        self._pad_after_spin.setSingleStep(0.1)
        self._pad_after_spin.setDecimals(2)
        self._pad_after_spin.setToolTip(
            "Post-event padding as a factor of the cluster core width.")
        det_grid.addWidget(self._pad_after_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Min event (ms):"), row, 0)
        self._minev_spin = QDoubleSpinBox()
        self._minev_spin.setRange(0.0, 5000.0)
        self._minev_spin.setSingleStep(1.0)
        self._minev_spin.setDecimals(1)
        self._minev_spin.setToolTip(
            "Reject events shorter than this duration as noise.")
        det_grid.addWidget(self._minev_spin, row, 1)
        panel_layout.addWidget(det_box)

        # Pre-filter group: causal lowpass + notch applied to the signal
        # before computing the diff. Both can be toggled independently.
        pre_box = QGroupBox("Pre-filter (signal)")
        pre_grid = QGridLayout(pre_box)
        prow = 0

        self._lpf_chk = QCheckBox("Lowpass")
        self._lpf_chk.setToolTip(
            "Causal Butterworth lowpass on the signal before the diff.")
        pre_grid.addWidget(self._lpf_chk, prow, 0)
        self._lpf_cutoff_spin = QDoubleSpinBox()
        self._lpf_cutoff_spin.setRange(1.0, 1e6)
        self._lpf_cutoff_spin.setDecimals(0)
        self._lpf_cutoff_spin.setSuffix(" Hz")
        self._lpf_cutoff_spin.setSingleStep(50.0)
        pre_grid.addWidget(self._lpf_cutoff_spin, prow, 1)
        prow += 1

        self._notch_chk = QCheckBox("Notch")
        self._notch_chk.setToolTip(
            "Causal IIR notch filter — kills mains interference.")
        pre_grid.addWidget(self._notch_chk, prow, 0)
        self._notch_freq_spin = QDoubleSpinBox()
        self._notch_freq_spin.setRange(1.0, 1e6)
        self._notch_freq_spin.setDecimals(1)
        self._notch_freq_spin.setSuffix(" Hz")
        self._notch_freq_spin.setSingleStep(1.0)
        pre_grid.addWidget(self._notch_freq_spin, prow, 1)
        prow += 1

        pre_grid.addWidget(QLabel("Notch Q:"), prow, 0)
        self._notch_q_spin = QDoubleSpinBox()
        self._notch_q_spin.setRange(1.0, 200.0)
        self._notch_q_spin.setSingleStep(1.0)
        self._notch_q_spin.setDecimals(1)
        self._notch_q_spin.setToolTip(
            "Quality factor: Q = freq / bandwidth. Higher Q = narrower "
            "notch (default 30 → ~2 Hz wide at 60 Hz).")
        pre_grid.addWidget(self._notch_q_spin, prow, 1)
        panel_layout.addWidget(pre_box)

        self._apply_btn = QPushButton("Update Detection")
        self._apply_btn.clicked.connect(self._on_apply)
        panel_layout.addWidget(self._apply_btn)

        panel_layout.addSpacing(5)

        run_box = QGroupBox("Run mode")
        run_layout = QVBoxLayout(run_box)
        self._mode_group = QButtonGroup(self)
        self._mode_live = QRadioButton("Live simulation")
        self._mode_batch = QRadioButton("Batch (skip)")
        self._mode_group.addButton(self._mode_live)
        self._mode_group.addButton(self._mode_batch)
        run_layout.addWidget(self._mode_live)
        run_layout.addWidget(self._mode_batch)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Sim speed:"))
        self._speed_combo = QComboBox()
        for label, _ in _SIM_SPEEDS:
            self._speed_combo.addItem(label)
        speed_row.addWidget(self._speed_combo)
        run_layout.addLayout(speed_row)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Window (s):"))
        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(0.1, 600.0)
        self._window_spin.setSingleStep(0.5)
        self._window_spin.setDecimals(2)
        win_row.addWidget(self._window_spin)
        run_layout.addLayout(win_row)

        run_btns = QHBoxLayout()
        self._run_btn = QPushButton("▶ Run")
        self._run_btn.clicked.connect(self._on_run)
        self._pause_btn = QPushButton("❚❚ Pause")
        self._pause_btn.clicked.connect(self._on_pause_toggle)
        self._pause_btn.setEnabled(False)
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        run_btns.addWidget(self._run_btn)
        run_btns.addWidget(self._pause_btn)
        run_btns.addWidget(self._stop_btn)
        run_layout.addLayout(run_btns)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        run_layout.addWidget(self._progress)

        panel_layout.addWidget(run_box)

        panel_layout.addSpacing(5)

        stats_box = QGroupBox("Stats")
        stats_layout = QVBoxLayout(stats_box)
        self._stats_label = QLabel("—")
        self._stats_label.setWordWrap(True)
        stats_layout.addWidget(self._stats_label)
        panel_layout.addWidget(stats_box)

        panel_layout.addSpacing(5)

        save_box = QGroupBox("Saved settings")
        save_layout = QHBoxLayout(save_box)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn = QPushButton("Save as…")
        self._save_as_btn.clicked.connect(self._on_save_as)
        self._load_btn = QPushButton("Load…")
        self._load_btn.clicked.connect(self._on_load)
        save_layout.addWidget(self._save_btn)
        save_layout.addWidget(self._save_as_btn)
        save_layout.addWidget(self._load_btn)
        panel_layout.addWidget(save_box)

        panel_layout.addStretch(1)

        bottom = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(ok_btn)
        bottom.addWidget(cancel_btn)
        panel_layout.addLayout(bottom)

    # --- Param ⇄ widget sync ---------------------------------------------

    def _sync_widgets_from_params(self):
        p = self._params
        # Backward compat: old saved settings used a single thresholdK.
        legacy_k = float(p.get("thresholdK", 3.0))
        self._thr_pos_spin.setValue(float(p.get("thresholdKPos", legacy_k)))
        self._thr_neg_spin.setValue(float(p.get("thresholdKNeg", legacy_k)))
        self._alpha_spin.setValue(float(p.get("filterAlpha", 0.3)))
        self._merge_spin.setValue(float(p.get("mergeGapMs", 100.0)))
        self._pad_before_spin.setValue(float(p.get("padBeforeFactor", 0.2)))
        self._pad_after_spin.setValue(float(p.get("padAfterFactor", 0.2)))
        self._minev_spin.setValue(float(p.get("minEventMs", 5.0)))
        self._lpf_chk.setChecked(bool(p.get("lpfEnabled", False)))
        self._lpf_cutoff_spin.setValue(float(p.get("lpfCutoffHz", 1000.0)))
        self._notch_chk.setChecked(bool(p.get("notchEnabled", False)))
        self._notch_freq_spin.setValue(float(p.get("notchHz", 60.0)))
        self._notch_q_spin.setValue(float(p.get("notchQ", 30.0)))
        if p.get("runMode") == "live":
            self._mode_live.setChecked(True)
        else:
            self._mode_batch.setChecked(True)
        speeds = [s for s, _ in _SIM_SPEEDS]
        if p.get("simSpeed") in speeds:
            self._speed_combo.setCurrentIndex(speeds.index(p["simSpeed"]))
        self._window_spin.setValue(float(p.get("liveWindowSec", 3.0)))

    def _read_params_from_widgets(self):
        return {
            "thresholdKPos":  float(self._thr_pos_spin.value()),
            "thresholdKNeg":  float(self._thr_neg_spin.value()),
            "filterAlpha":    float(self._alpha_spin.value()),
            "mergeGapMs":      float(self._merge_spin.value()),
            "padBeforeFactor": float(self._pad_before_spin.value()),
            "padAfterFactor":  float(self._pad_after_spin.value()),
            "minEventMs":      float(self._minev_spin.value()),
            "lpfEnabled":     bool(self._lpf_chk.isChecked()),
            "lpfCutoffHz":    float(self._lpf_cutoff_spin.value()),
            "notchEnabled":   bool(self._notch_chk.isChecked()),
            "notchHz":        float(self._notch_freq_spin.value()),
            "notchQ":         float(self._notch_q_spin.value()),
            "runMode":        "live" if self._mode_live.isChecked()
                              else "batch",
            "simSpeed":       self._speed_combo.currentText(),
            "liveWindowSec":  float(self._window_spin.value()),
        }

    # --- Compute + draw --------------------------------------------------

    def _run_batch_and_redraw(self):
        try:
            self._result = detect_threshold(self._signal, self._sample_rate,
                                            self._params)
        except Exception as e:
            QMessageBox.critical(self, "Detection failed", str(e))
            self._result = None
            return
        self._sim_revealed = len(self._signal)
        self._redraw()

    def _redraw(self):
        prev_xlim = None
        prev_ylim = None
        if self._ax_sig.has_data():
            prev_xlim = self._ax_sig.get_xlim()
            prev_ylim = self._ax_sig.get_ylim()

        self._ax_sig.clear()
        self._ax_score.clear()

        if self._result is None or self._signal.size == 0:
            self._canvas.draw_idle()
            return

        sr = max(self._sample_rate, 1.0)
        n = len(self._signal)
        t = np.arange(n) / sr
        revealed = max(0, min(n, int(self._sim_revealed)))
        score = self._result["score"]
        regions = self._result["detectedPulses"]
        cores = self._result.get("coreRegions",
                                  np.zeros((0, 2), dtype=int))
        # Two-tone shading: light pad zone for the full padded region,
        # darker shade overlaid on the core (peak cluster) span.
        PAD_COLOR = "#cfe6cf"   # light green
        CORE_COLOR = "#3aa55a"  # solid green

        # When pre-filters (LPF/notch) are enabled, plot the filtered
        # signal so the user sees what the detector actually sees.
        filters_on = bool(self._params.get("lpfEnabled")
                          or self._params.get("notchEnabled"))
        sig_to_plot = (self._result.get("signalFiltered")
                       if filters_on else self._signal)
        if sig_to_plot is None or len(sig_to_plot) != n:
            sig_to_plot = self._signal
        sig_label = "signal (filtered)" if filters_on else "signal"

        if self._live_view and revealed < n:
            window_sec = float(self._params.get("liveWindowSec", 3.0))
            window_samples = max(2, int(round(window_sec * sr)))
            left_idx = max(0, revealed - window_samples)
            right_idx = revealed
            self._ax_sig.plot(t[left_idx:right_idx],
                              sig_to_plot[left_idx:right_idx],
                              color="#3b6fb6", lw=0.8)
            self._ax_score.plot(t[left_idx:right_idx],
                                score[left_idx:right_idx],
                                color="#e08533", lw=0.8)
            # Padded regions (light) — one per merged group
            for (a, b) in regions:
                a_i, b_i = int(a), int(b)
                if b_i < left_idx or a_i >= right_idx:
                    continue
                a_show = max(a_i, left_idx)
                b_show = min(b_i, right_idx - 1)
                if b_show <= a_show:
                    continue
                self._ax_sig.axvspan(a_show / sr, b_show / sr,
                                     color=PAD_COLOR, alpha=0.55)
            # Cores (dark) — possibly multiple per padded region
            for (cs, ce) in cores:
                cs_i, ce_i = int(cs), int(ce)
                if ce_i < left_idx or cs_i >= right_idx:
                    continue
                a_show = max(cs_i, left_idx)
                b_show = min(ce_i, right_idx - 1)
                if b_show > a_show:
                    self._ax_sig.axvspan(a_show / sr, b_show / sr,
                                         color=CORE_COLOR, alpha=0.30)
        else:
            self._ax_sig.plot(t, sig_to_plot, color="#3b6fb6", lw=0.8)
            self._ax_score.plot(t, score, color="#e08533", lw=0.8)
            for (a, b) in regions:
                self._ax_sig.axvspan(int(a) / sr, int(b) / sr,
                                     color=PAD_COLOR, alpha=0.55)
            for (cs, ce) in cores:
                if int(ce) > int(cs):
                    self._ax_sig.axvspan(int(cs) / sr, int(ce) / sr,
                                         color=CORE_COLOR, alpha=0.30)

        self._ax_sig.set_ylabel(sig_label)
        self._ax_sig.set_title("Detection preview")
        t_pos = self._result.get("thresholdPos", self._result.get("threshold", 0))
        t_neg = self._result.get("thresholdNeg", -t_pos)
        # Make threshold lines thicker so they're easier to grab and so
        # we can capture artist refs for drag tracking.
        self._thr_line_pos = self._ax_score.axhline(
            t_pos, color="#3aa55a", ls="--", lw=1.4,
            label=f"+k·σ={t_pos:.3g}")
        self._thr_line_neg = self._ax_score.axhline(
            t_neg, color="#d23f3f", ls="--", lw=1.4,
            label=f"−k·σ={t_neg:.3g}")
        self._ax_score.axhline(0, color="#888", ls=":", lw=0.5)
        self._ax_score.set_ylabel("filtered Δsignal")
        self._ax_score.set_xlabel("time (s)")
        self._ax_score.legend(loc="upper right", fontsize=8)

        full_xlim = (0.0, max(t[-1], 1.0 / sr))
        live_active = self._live_view and revealed < n
        if live_active:
            window_sec = float(self._params.get("liveWindowSec", 3.0))
            window_sec = max(window_sec, 1.0 / sr)
            right = revealed / sr
            left = max(0.0, right - window_sec)
            if right < window_sec:
                left, right = 0.0, window_sec
            self._ax_sig.set_xlim(left, right)
            left_idx = max(0, int(round(left * sr)))
            right_idx = max(left_idx + 1, int(round(right * sr)))
            visible = sig_to_plot[left_idx:min(right_idx, len(sig_to_plot))]
            if visible.size:
                y_lo = float(np.min(visible))
                y_hi = float(np.max(visible))
                rng = y_hi - y_lo
                if rng <= 0:
                    rng = max(abs(y_hi), 1.0) * 0.01
                pad = 0.15 * rng
                self._ax_sig.set_ylim(y_lo - pad, y_hi + pad)
        elif prev_xlim is not None and not np.allclose(prev_xlim, full_xlim,
                                                       atol=1e-6):
            self._ax_sig.set_xlim(prev_xlim)
        else:
            self._ax_sig.set_xlim(full_xlim)

        if prev_ylim is not None and not live_active:
            self._ax_sig.set_ylim(prev_ylim)

        n_reg = len(regions)
        sigma = self._result.get("sigma", 0.0)
        self._stats_label.setText(
            f"Detected: {n_reg}\n"
            f"σ̂ (1.4826·MAD of Δ): {sigma:.3g}\n"
            f"+threshold: {t_pos:.3g}\n"
            f"−threshold: {t_neg:.3g}"
        )

        self._canvas.draw_idle()

    # --- Threshold drag handlers ----------------------------------------

    def _toolbar_active(self):
        """Return True if the navigation toolbar is in pan or zoom mode,
        in which case threshold drags should be ignored.
        """
        try:
            mode = self._nav_toolbar.mode
            if hasattr(mode, "name"):
                mode = mode.name
            return str(mode).strip().lower() not in ("", "none")
        except Exception:
            return False

    def _on_thr_press(self, event):
        if event.inaxes is not self._ax_score:
            return
        if self._toolbar_active() or event.button != 1:
            return
        if self._result is None:
            return
        self._thr_drag = self._threshold_at(event)
        if self._thr_drag is None:
            return
        # Switch threshold lines to animated artists and cache the
        # rest of the score axis as a static background — motion
        # events will blit just the lines instead of repainting the
        # full score trace each time.
        try:
            if self._thr_line_pos is not None:
                self._thr_line_pos.set_animated(True)
            if self._thr_line_neg is not None:
                self._thr_line_neg.set_animated(True)
            self._canvas.draw()
            self._drag_bg = self._canvas.copy_from_bbox(
                self._ax_score.bbox)
        except Exception:
            self._drag_bg = None

    def _threshold_at(self, event):
        """Return 'pos' / 'neg' / None depending on whether the event
        falls within the hit-tolerance of either threshold line.
        """
        if event.inaxes is not self._ax_score or self._result is None:
            return None
        if event.y is None:
            return None
        t_pos = self._result.get("thresholdPos", 0.0)
        t_neg = self._result.get("thresholdNeg", -t_pos)
        try:
            y_pos_disp = self._ax_score.transData.transform((0, t_pos))[1]
            y_neg_disp = self._ax_score.transData.transform((0, t_neg))[1]
        except Exception:
            return None
        tol = 6.0
        d_pos = abs(event.y - y_pos_disp)
        d_neg = abs(event.y - y_neg_disp)
        if d_pos <= tol and d_pos <= d_neg:
            return "pos"
        if d_neg <= tol:
            return "neg"
        return None

    def _set_hover_cursor(self, near_line):
        """Show a vertical-resize cursor when near a draggable threshold
        line; revert otherwise. Skipped when the matplotlib navigation
        toolbar is in pan/zoom mode (it manages its own cursor).
        """
        if self._toolbar_active():
            return
        if near_line:
            self._canvas.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self._canvas.unsetCursor()

    def _on_thr_motion(self, event):
        if self._thr_drag is None or self._result is None:
            self._set_hover_cursor(self._threshold_at(event) is not None)
            return
        if event.inaxes is not self._ax_score or event.ydata is None:
            return
        sigma = float(self._result.get("sigma", 0.0)) or 1e-12
        # Drag dictates a new |k| value (always positive).
        new_k = abs(float(event.ydata) / sigma)
        new_k = max(0.1, min(50.0, new_k))
        if self._thr_drag == "pos":
            self._thr_pos_spin.blockSignals(True)
            self._thr_pos_spin.setValue(new_k)
            self._thr_pos_spin.blockSignals(False)
            if self._thr_line_pos is not None:
                self._thr_line_pos.set_ydata([new_k * sigma, new_k * sigma])
        else:
            self._thr_neg_spin.blockSignals(True)
            self._thr_neg_spin.setValue(new_k)
            self._thr_neg_spin.blockSignals(False)
            if self._thr_line_neg is not None:
                self._thr_line_neg.set_ydata([-new_k * sigma, -new_k * sigma])
        if self._drag_bg is not None:
            self._canvas.restore_region(self._drag_bg)
            if self._thr_line_pos is not None:
                self._ax_score.draw_artist(self._thr_line_pos)
            if self._thr_line_neg is not None:
                self._ax_score.draw_artist(self._thr_line_neg)
            self._canvas.blit(self._ax_score.bbox)
        else:
            self._canvas.draw_idle()

    def _on_thr_release(self, event):
        if self._thr_drag is None:
            return
        self._thr_drag = None
        if self._thr_line_pos is not None:
            self._thr_line_pos.set_animated(False)
        if self._thr_line_neg is not None:
            self._thr_line_neg.set_animated(False)
        self._drag_bg = None
        # Recompute detection with the new threshold (cheap; same σ).
        self._params = self._read_params_from_widgets()
        try:
            self._result = detect_threshold(self._signal, self._sample_rate,
                                            self._params)
        except Exception:
            return
        self._sim_revealed = len(self._signal)
        self._live_view = False
        self._redraw()

    # --- Button handlers -------------------------------------------------

    def _on_apply(self):
        self._params = self._read_params_from_widgets()
        self._stop_sim()
        self._run_batch_and_redraw()

    def _on_run(self):
        self._params = self._read_params_from_widgets()
        try:
            self._result = detect_threshold(self._signal, self._sample_rate,
                                            self._params)
        except Exception as e:
            QMessageBox.critical(self, "Detection failed", str(e))
            return
        if self._params["runMode"] == "batch":
            self._sim_revealed = len(self._signal)
            self._redraw()
            return
        self._start_sim()

    def _on_pause_toggle(self):
        if self._sim_timer is None:
            return
        if self._sim_paused:
            self._sim_paused = False
            self._sim_revealed_at_resume = self._sim_revealed
            self._sim_t0 = perf_counter()
            self._sim_timer.start()
            self._pause_btn.setText("❚❚ Pause")
        else:
            self._sim_paused = True
            self._sim_timer.stop()
            self._pause_btn.setText("▶ Resume")

    def _on_stop(self):
        self._stop_sim()
        if self._result is not None:
            self._sim_revealed = len(self._signal)
            self._live_view = False
            self._progress.setValue(100)
            self._redraw()

    def _on_accept(self):
        self._params = self._read_params_from_widgets()
        self._stop_sim()
        try:
            self._result = detect_threshold(self._signal, self._sample_rate,
                                            self._params)
        except Exception as e:
            QMessageBox.critical(self, "Detection failed", str(e))
            return
        self.accept()

    # --- Live simulation -------------------------------------------------

    def _speed_factor(self):
        label = self._speed_combo.currentText()
        for lab, factor in _SIM_SPEEDS:
            if lab == label:
                return factor
        return 10.0

    def _start_sim(self):
        self._stop_sim()
        self._sim_revealed = 0
        self._sim_revealed_at_resume = 0
        self._sim_paused = False
        self._live_view = True
        self._sim_t0 = perf_counter()

        # Fixed ~30 Hz refresh — sample reveal count is driven by wall time
        # and speed factor, not by tick count, so playback rate stays correct
        # regardless of redraw cost or speed setting.
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(33)
        self._sim_timer.timeout.connect(self._sim_tick)
        self._run_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._pause_btn.setText("❚❚ Pause")
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)
        self._redraw()
        self._sim_timer.start()

    def _sim_tick(self):
        if self._result is None:
            self._stop_sim()
            return
        speed = self._speed_factor()
        if speed <= 0:
            self._sim_revealed = len(self._signal)
        else:
            elapsed = perf_counter() - self._sim_t0
            advance = int(elapsed * self._sample_rate * speed)
            self._sim_revealed = min(
                len(self._signal),
                self._sim_revealed_at_resume + advance,
            )
        pct = int(round(100.0 * self._sim_revealed / max(1, len(self._signal))))
        self._progress.setValue(pct)
        self._redraw()
        if self._sim_revealed >= len(self._signal):
            self._stop_sim()
            self._live_view = False
            self._redraw()

    def _stop_sim(self):
        if self._sim_timer is not None:
            self._sim_timer.stop()
            self._sim_timer = None
        self._sim_paused = False
        self._run_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("❚❚ Pause")
        self._stop_btn.setEnabled(False)

    # --- Save / Load ----------------------------------------------------

    def _settings_folder(self):
        return saved_templates_dir(os.path.join("Sorting",
                                                "RealtimeThreshold"))

    def _on_save(self):
        if self._current_path:
            target = self._current_path
        else:
            self._on_save_as()
            return
        p = self._read_params_from_widgets()
        try:
            save_state(p, target)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_save_as(self):
        p = self._read_params_from_widgets()
        folder = self._settings_folder()
        os.makedirs(folder, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save detection settings", folder, "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_state(p, path)
            self._current_path = path
            self.setWindowTitle(
                f"RT Detection (Threshold) — {os.path.basename(path)}")
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_load(self):
        folder = self._settings_folder()
        os.makedirs(folder, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load detection settings", folder, "JSON (*.json)")
        if not path:
            return
        try:
            self._params = load_state(path)
            self._current_path = path
            self.setWindowTitle(
                f"RT Detection (Threshold) — {os.path.basename(path)}")
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._sync_widgets_from_params()
        self._stop_sim()
        self._run_batch_and_redraw()

    def result_state(self):
        return dict(self._params), self._result
