"""Realtime Detection (CWT): pure helpers and the interactive QDialog.

Detects pulses in a 1-D signal by matched-filtering against a user-provided
mother wavelet (typically the output of the Wavelet Editor), evaluated at
several width scales (CWT-style) and combined into a single detection score.
"""

import json
import os
from time import perf_counter

import numpy as np

from utils.paths import saved_templates_dir


_DEFAULT_PARAMS = {
    "scaleMin":          0.5,         # smallest matched-filter width factor
    "scaleMax":          2.0,         # largest matched-filter width factor
    "scaleCount":        3,           # number of log-spaced scales
    "thresholdK":        5.0,
    "minSeparationFrac": 1.0,
    "scoringMethod":     "max",       # "max" or "rss"
    "autoNormalize":     True,
    "splitCoincidents":  True,        # segment each region by signal-domain runs
    "splitReturnSigma":  3.0,         # signal sample is "in event" when |s - baseline| > Nσ; below that = baseline
    "splitMergeGapFrac": 0.1,         # baseline gaps shorter than this fraction of template_len are not treated as event separators (handles internal template gaps)
    "runMode":           "batch",     # "batch" or "live"
    "simSpeed":          "10x",       # "1x", "10x", "100x", "asap"
    "liveWindowSec":     3.0,         # sliding x-axis window in seconds
}


def autocorr_min_separation(template, threshold_frac=0.3):
    """Largest lag at which the template's autocorrelation magnitude
    exceeds `threshold_frac` of its peak (i.e., the side-lobe extent).

    Two pulses separated by less than this distance can have their
    matched-filter responses confused with template self-correlations,
    producing spurious sub-peaks. Used as a template-aware floor under
    the user's `minSeparationFrac`. For a smooth single-bump template
    this returns ~half the template length; for a multi-plateau template
    it can be a larger fraction of the template length because the
    autocorrelation has prominent side-lobes from inter-plateau spacing.

    Walks from the largest lag inward to find the first lag where the
    autocorrelation magnitude crosses the threshold — that's the
    farthest side-lobe still above tolerance.
    """
    from scipy.signal import fftconvolve
    t = np.asarray(template, dtype=float).ravel()
    if t.size <= 1:
        return 1
    ac = fftconvolve(t, t[::-1], mode="full")
    center = len(t) - 1
    peak = float(np.abs(ac[center]))
    if peak <= 0:
        return 1
    thr = threshold_frac * peak
    n = len(t)
    for i in range(n - 1, 0, -1):
        if abs(float(ac[center + i])) > thr:
            return int(i)
    return 1


def make_scales(scale_min, scale_max, count):
    """Return `count` log-spaced scales from `scale_min` to `scale_max`.

    `count` is clipped to >= 1; when `count == 1` the single scale is the
    geometric mean of min/max.
    """
    smin = float(scale_min)
    smax = float(scale_max)
    n = max(1, int(count))
    if smin <= 0 or smax <= 0:
        raise ValueError("scaleMin and scaleMax must be > 0")
    if smin > smax:
        smin, smax = smax, smin
    if n == 1:
        return [float(np.sqrt(smin * smax))]
    return [float(s) for s in np.geomspace(smin, smax, n)]


# --- Pure algorithm helpers ----------------------------------------------

def normalize_wavelet(wavelet, auto=True):
    """Zero-mean + unit-L2 normalize. Returns (w_normalized, applied_bool).

    If `auto` is False, returns the wavelet unchanged. If the input is
    degenerate (zero norm after demean), returns the input unchanged with
    applied=False.
    """
    w = np.asarray(wavelet, dtype=float).ravel()
    if not auto or w.size == 0:
        return w, False
    w = w - w.mean()
    n = np.linalg.norm(w)
    if n < 1e-12:
        return np.asarray(wavelet, dtype=float).ravel(), False
    return w / n, True


def build_kernels(wavelet, scales):
    """Build matched-filter kernels at the requested width scales.

    For each scale s, the wavelet is resampled to length max(round(len(w)*s), 4)
    and re-normalized to zero-mean / unit-L2 so that the resulting CWT
    coefficient is directly comparable across scales.

    Returns a list of 1-D numpy arrays, one per scale (same order as input).
    """
    from scipy.signal import resample
    w = np.asarray(wavelet, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("wavelet is empty")
    out = []
    for s in scales:
        s = float(s)
        if s <= 0:
            raise ValueError(f"scale must be > 0, got {s}")
        n_new = max(int(round(len(w) * s)), 4)
        if n_new == len(w):
            ws = w.copy()
        else:
            ws = resample(w, n_new)
        ws = ws - ws.mean()
        n = np.linalg.norm(ws)
        if n > 1e-12:
            ws = ws / n
        out.append(ws)
    return out


def compute_score(signal, kernels, method="max"):
    """Convolve signal with each (time-reversed) kernel, combine across scales.

    method="max" -> max(c_s) at each time index — POSITIVE correlations
        only, so detection is direction-aware: only signal features whose
        shape matches the template (not its inverse) register. This is
        what NPS detection wants when the template has signed structure
        (e.g. positive plateaus with negative baseline tails).
    method="rss" -> sqrt(sum(c_s**2)) at each time index — sign-agnostic;
        picks up both correlations and anti-correlations as peaks.

    The signal is edge-padded by the longest kernel's length on each side
    before convolving and trimmed back afterward, so a zero-mean kernel
    integrating against a constant baseline at the boundary returns zero
    (no edge artifact).

    Returns a 1-D numpy array the same length as `signal`.
    """
    from scipy.signal import fftconvolve
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    if n == 0 or not kernels:
        return np.zeros_like(x)
    pad = max(len(k) for k in kernels)
    pad = min(pad, n)
    x_padded = np.pad(x, pad, mode="edge")
    coeffs = []
    for k in kernels:
        c = fftconvolve(x_padded, k[::-1], mode="same")
        coeffs.append(c[pad:pad + n])
    stack = np.vstack(coeffs)
    if method == "rss":
        return np.sqrt(np.sum(stack ** 2, axis=0))
    # Direction-aware: keep only positive correlations. Anti-correlations
    # from template-tail alignment are spurious for pulse-shape matching.
    return np.max(stack, axis=0)


def mad_threshold(score, k=5.0):
    """Robust threshold: k * 1.4826 * MAD(score - median(score)).

    Returns (threshold, sigma_hat).
    """
    s = np.asarray(score, dtype=float).ravel()
    if s.size == 0:
        return 0.0, 0.0
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    sigma = 1.4826 * mad
    return med + float(k) * sigma, sigma


def segment_region_by_signal(signal, region, baseline, sigma,
                             threshold_sigma=3.0, merge_gap_samples=1,
                             min_event_samples=1):
    """Identify discrete events within a region by signal-domain segmentation.

    A signal sample is "in event" when |signal - baseline| > threshold_sigma
    × sigma. Contiguous in-event samples form a "run". Two runs separated
    by fewer than `merge_gap_samples` of below-threshold signal are merged
    (this prevents internal template gaps — short returns to baseline
    within one multi-segment pulse — from being mistaken for event
    separations). Runs shorter than `min_event_samples` are dropped.

    Returns a list of (start, end) sub-regions (inclusive), or [region] if
    fewer than 2 distinct events are found.
    """
    a, b = int(region[0]), int(region[1])
    if b <= a or sigma <= 0:
        return [(a, b)]
    sig = np.asarray(signal, dtype=float).ravel()
    n = len(sig)
    a = max(0, a)
    b = min(n - 1, b)
    if b <= a:
        return [(a, b)]
    seg = sig[a:b + 1]
    above = np.abs(seg - baseline) > threshold_sigma * sigma
    if not above.any():
        return [(a, b)]

    # Find contiguous run starts/ends in the local frame.
    flips = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.where(flips == 1)[0]
    ends = np.where(flips == -1)[0] - 1   # inclusive

    # Merge runs separated by gaps shorter than merge_gap_samples.
    merged_s = [int(starts[0])]
    merged_e = [int(ends[0])]
    for i in range(1, len(starts)):
        if int(starts[i]) - merged_e[-1] - 1 < merge_gap_samples:
            merged_e[-1] = int(ends[i])
        else:
            merged_s.append(int(starts[i]))
            merged_e.append(int(ends[i]))

    # Drop runs shorter than min_event_samples.
    runs = [(s, e) for s, e in zip(merged_s, merged_e)
            if (e - s + 1) >= min_event_samples]
    if len(runs) <= 1:
        return [(a, b)]
    return [(a + s, a + e) for s, e in runs]


def split_coincident_region(score, region, template_len, threshold,
                            signal=None, signal_baseline=None,
                            signal_sigma=None,
                            min_prominence_frac=0.3,
                            return_sigma=3.0,
                            autocorr_floor_samples=0):
    """Split a coincident region into per-pulse sub-regions.

    Two-stage filter:

    1. Score-domain prominence: find local maxima in the score within the
       region whose prominence (peak height above its surrounding valley)
       exceeds `min_prominence_frac` of (region_max − threshold) AND that
       are at least 0.5 * template_len apart (same physical floor as the
       main peak picker, to reject matched-filter side-lobes).

    2. Signal-domain return-to-baseline check (if `signal` is provided):
       between two consecutive candidate peaks, the raw signal must come
       back within `return_sigma` * `signal_sigma` of `signal_baseline` at
       some point — proof that one event ended before the next began.
       Peaks that fail this test get folded into a single event with
       their preceding peak. This distinguishes a stuck/sustained event
       (signal stays away from baseline → one region) from a real
       coincident burst (signal returns between events → multiple regions).

    Returns a list of (start, end) tuples. Falls back to the original
    region if fewer than 2 peaks survive both filters.
    """
    from scipy.signal import find_peaks
    a, b = int(region[0]), int(region[1])
    if b - a < 3:
        return [(a, b)]
    sub = np.asarray(score, dtype=float)[a:b + 1]
    region_max = float(sub.max())
    if region_max <= threshold:
        return [(a, b)]

    # Use the autocorrelation floor (template-aware) when available;
    # otherwise fall back to half the template length.
    floor_default = max(1, int(round(0.5 * template_len)))
    min_dist = max(1, int(autocorr_floor_samples) or floor_default)
    min_prom = max(min_prominence_frac * (region_max - threshold), 1e-12)
    peaks, _ = find_peaks(sub, height=threshold, distance=min_dist,
                          prominence=min_prom)
    if len(peaks) < 2:
        return [(a, b)]

    # Stage 2: signal-domain return-to-baseline filter.
    if signal is not None and signal_baseline is not None and signal_sigma:
        sig_arr = np.asarray(signal, dtype=float).ravel()
        thr_dev = float(return_sigma) * float(signal_sigma)
        kept_local = [int(peaks[0])]
        for i in range(1, len(peaks)):
            p1 = a + int(kept_local[-1])
            p2 = a + int(peaks[i])
            if p2 <= p1 + 1 or p1 < 0 or p2 >= len(sig_arr):
                kept_local.append(int(peaks[i]))
                continue
            seg = sig_arr[p1:p2 + 1]
            min_dev = float(np.min(np.abs(seg - signal_baseline)))
            if min_dev <= thr_dev:
                # Signal touches baseline between these peaks → real
                # separation; keep peaks[i] as its own event.
                kept_local.append(int(peaks[i]))
            # else: peaks[i] gets folded into the prior event.
        peaks = np.asarray(kept_local, dtype=int)
        if len(peaks) < 2:
            return [(a, b)]

    half = max(1, int(round(0.5 * template_len)))
    n = len(score)
    subs = []
    for p in peaks:
        g = a + int(p)
        left = max(0, g - half)
        right = min(n - 1, g + half)
        subs.append((left, right))
    return subs


def detect_regions(score, threshold, template_len, min_separation_frac=1.0,
                   split_coincidents=True,
                   signal=None, signal_baseline=None, signal_sigma=None,
                   split_return_sigma=3.0,
                   split_merge_gap_frac=0.1,
                   autocorr_floor_samples=0):
    """Peak-pick the score, expand each peak by ±0.5*template_len, merge
    overlaps. Return (regions Nx2 int, coincident_hint Nx1 bool).

    `min_separation_frac` sets the minimum peak-to-peak distance as a
    fraction of `template_len`. Default 1.0 = one full template length,
    which suppresses autocorrelation side-lobes from a single pulse so
    each event reduces to one dominant peak.

    A merged region is flagged as a coincident-hint when it contains two or
    more surviving peaks (i.e. two events spaced by ≥ min_separation_frac
    × template_len that landed in adjacent expansion windows).
    """
    from scipy.signal import find_peaks
    s = np.asarray(score, dtype=float).ravel()
    if s.size == 0 or template_len <= 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=bool)

    user_min_dist = int(round(min_separation_frac * template_len))
    min_dist = max(1, user_min_dist, int(autocorr_floor_samples))
    peaks, _ = find_peaks(s, height=threshold, distance=min_dist)
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=bool)

    # For each surviving peak, find the contiguous above-threshold span
    # containing it. Two peaks landing in the same above-threshold span
    # collapse to one region (which is then flagged as coincident).
    above = s >= threshold
    n = len(s)
    spans = []   # list of [start, end] inclusive
    span_peak_counts = []
    last_end = -1
    for p in peaks:
        if p <= last_end:
            span_peak_counts[-1] += 1
            continue
        # Walk outward from p while above threshold.
        left = int(p)
        while left > 0 and above[left - 1]:
            left -= 1
        right = int(p)
        while right < n - 1 and above[right + 1]:
            right += 1
        spans.append([left, right])
        span_peak_counts.append(1)
        last_end = right

    regions = np.asarray(spans, dtype=int)
    widths = regions[:, 1] - regions[:, 0]
    counts = np.asarray(span_peak_counts, dtype=int)

    # Drop sub-template-width regions: a real pulse's matched-filter
    # response has an above-threshold span ≈ 2 × template_len, so anything
    # narrower than 0.5 × template_len is almost certainly an edge crumb
    # or a borderline noise excursion, not a pulse.
    min_width = max(1, int(round(0.5 * template_len)))
    keep = widths >= min_width
    regions = regions[keep]
    widths = widths[keep]
    counts = counts[keep]

    # Initial coincident hint based on score-domain alone (≥2 peaks
    # merged into one span). Refined below by signal-based segmentation.
    hint = (counts >= 2)

    if (not split_coincidents or signal is None
            or signal_baseline is None or not signal_sigma):
        return regions, hint

    # Signal-based segmentation: within each region, split into separate
    # events using runs of |signal − baseline| > Nσ. Internal template
    # gaps (signal briefly returns to baseline within one multi-segment
    # pulse) are bridged by merge_gap.
    merge_gap = max(1, int(round(split_merge_gap_frac * template_len)))
    # Minimum event length: a real event spans at least one template
    # plateau worth of samples; reject tiny noise crossings.
    min_event = max(1, int(round(0.05 * template_len)))
    out_regions = []
    out_hint = []
    for i in range(len(regions)):
        subs = segment_region_by_signal(
            signal, regions[i],
            baseline=signal_baseline, sigma=signal_sigma,
            threshold_sigma=split_return_sigma,
            merge_gap_samples=merge_gap,
            min_event_samples=min_event,
        )
        if len(subs) <= 1:
            out_regions.append(tuple(regions[i]))
            out_hint.append(bool(hint[i]))
        else:
            for sub in subs:
                out_regions.append(sub)
                out_hint.append(True)
    return (np.asarray(out_regions, dtype=int),
            np.asarray(out_hint, dtype=bool))


def detect_cwt(signal, wavelet, params=None):
    """End-to-end: returns dict with detectedPulses, score, coincidentHint,
    threshold, sigma, kernels.
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)
    w, _ = normalize_wavelet(wavelet, auto=p.get("autoNormalize", True))
    # Backward compat: a legacy `scales` list still works if provided.
    if "scales" in p and p.get("scales"):
        scales = list(p["scales"])
    else:
        scales = make_scales(p.get("scaleMin", 0.5),
                             p.get("scaleMax", 2.0),
                             p.get("scaleCount", 3))
    kernels = build_kernels(w, scales)
    score = compute_score(signal, kernels, method=p.get("scoringMethod", "max"))
    thr, sigma = mad_threshold(score, k=p.get("thresholdK", 5.0))

    # Template-aware lower bound on peak separation: the farthest lag at
    # which the template's autocorrelation magnitude still exceeds 10 %
    # of its peak. Multi-plateau templates have prominent side-lobes
    # from inter-plateau spacing that can extend almost to the full
    # template length; this distance is what's required to suppress them.
    # Used as a hard floor under the user's min_sep_frac.
    autocorr_floor = autocorr_min_separation(w, threshold_frac=0.1)

    # Robust signal baseline + sigma for the return-to-baseline check
    # used by the coincident splitter. Global median is a fine baseline
    # estimate as long as detected regions cover a small fraction of the
    # record, which holds for typical NPS data.
    sig_arr = np.asarray(signal, dtype=float).ravel()
    sig_baseline = float(np.median(sig_arr)) if sig_arr.size else 0.0
    sig_sigma = (1.4826 * float(np.median(np.abs(sig_arr - sig_baseline)))
                 if sig_arr.size else 0.0)

    regions, hint = detect_regions(
        score, thr, template_len=len(w),
        min_separation_frac=p.get("minSeparationFrac", 1.0),
        split_coincidents=p.get("splitCoincidents", True),
        signal=sig_arr, signal_baseline=sig_baseline, signal_sigma=sig_sigma,
        split_return_sigma=p.get("splitReturnSigma", 3.0),
        split_merge_gap_frac=p.get("splitMergeGapFrac", 0.1),
        autocorr_floor_samples=autocorr_floor,
    )
    return {
        "detectedPulses": regions,
        "score":          score,
        "coincidentHint": hint.astype(np.int8),
        "threshold":      float(thr),
        "sigma":          float(sigma),
        "kernels":        kernels,
        "wavelet_used":   w,
        "autocorrFloor":  int(autocorr_floor),
    }


# --- Settings persistence -------------------------------------------------

def save_state(params, filepath):
    """Atomic write of detector params to JSON."""
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
    """Load detector params from JSON; returns merged defaults + loaded."""
    with open(filepath, "r") as f:
        loaded = json.load(f)
    out = dict(_DEFAULT_PARAMS)
    for k in _DEFAULT_PARAMS:
        if k in loaded:
            out[k] = loaded[k]
    return out


def parse_scales(text):
    """Parse a comma/space-separated string into a list of positive floats.

    Raises ValueError on any malformed token.
    """
    parts = [t.strip() for t in str(text).replace(",", " ").split()]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("scales list is empty")
    out = []
    for p in parts:
        v = float(p)
        if v <= 0:
            raise ValueError(f"scales must be positive, got {v}")
        out.append(v)
    return out


# --- Dialog ---------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QDoubleSpinBox, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QFileDialog, QMessageBox, QWidget,
    QProgressBar,
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
    ("asap", 0.0),  # 0 == no delay
]


class RealtimeDetectionCWTDialog(QDialog):
    """Interactive dialog for tuning + previewing CWT-based pulse detection.

    Produces detectedPulses + score + coincidentHint on accept, derived from
    a deterministic batch run regardless of which preview mode the user used.
    """

    def __init__(self, signal, wavelet, sample_rate, params,
                 sample_rate_source="default", current_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Realtime Detection (CWT)")
        self.resize(1200, 760)

        try:
            from utils.dialog_style import apply_dialog_style
            apply_dialog_style(self)
        except ImportError:
            pass

        self._signal = np.asarray(signal, dtype=float).ravel()
        self._wavelet_in = np.asarray(wavelet, dtype=float).ravel()
        self._sample_rate = float(sample_rate)
        self._sample_rate_source = sample_rate_source
        self._params = dict(_DEFAULT_PARAMS)
        self._params.update(params or {})
        self._current_path = current_path
        if current_path:
            self.setWindowTitle(
                f"Realtime Detection (CWT) — {os.path.basename(current_path)}")

        self._result = None       # last batch detect_cwt result
        self._sim_timer = None
        self._sim_revealed = 0    # samples revealed so far in live mode
        self._sim_paused = False
        self._sim_t0 = 0.0
        self._sim_revealed_at_resume = 0
        self._live_view = False   # True while a live sim is in progress

        self._build_ui()
        self._sync_widgets_from_params()
        self._run_batch_and_redraw()

    # --- UI construction -------------------------------------------------

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        # Left: matplotlib canvas + navigation toolbar wrapped in a column
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

        # Right: parameter side panel
        panel = QWidget()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        main.addWidget(panel)

        # Sample rate (read-only label)
        self._sr_label = QLabel(f"Sample rate: {int(self._sample_rate)} Hz")
        self._sr_label.setWordWrap(True)
        panel_layout.addWidget(self._sr_label)

        panel_layout.addSpacing(5)

        # Detection params
        det_box = QGroupBox("Detection")
        det_grid = QGridLayout(det_box)
        row = 0

        det_grid.addWidget(QLabel("Scale min:"), row, 0)
        self._scale_min_spin = QDoubleSpinBox()
        self._scale_min_spin.setRange(0.1, 4.0)
        self._scale_min_spin.setSingleStep(0.1)
        self._scale_min_spin.setDecimals(2)
        det_grid.addWidget(self._scale_min_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Scale max:"), row, 0)
        self._scale_max_spin = QDoubleSpinBox()
        self._scale_max_spin.setRange(0.1, 4.0)
        self._scale_max_spin.setSingleStep(0.1)
        self._scale_max_spin.setDecimals(2)
        det_grid.addWidget(self._scale_max_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Scale points:"), row, 0)
        from PySide6.QtWidgets import QSpinBox
        self._scale_count_spin = QSpinBox()
        self._scale_count_spin.setRange(1, 20)
        det_grid.addWidget(self._scale_count_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Threshold k:"), row, 0)
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(1.0, 20.0)
        self._thr_spin.setSingleStep(0.5)
        self._thr_spin.setDecimals(2)
        det_grid.addWidget(self._thr_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Min sep frac:"), row, 0)
        self._minsep_spin = QDoubleSpinBox()
        self._minsep_spin.setRange(0.1, 3.0)
        self._minsep_spin.setSingleStep(0.1)
        self._minsep_spin.setDecimals(2)
        det_grid.addWidget(self._minsep_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Scoring:"), row, 0)
        self._method_combo = QComboBox()
        self._method_combo.addItem("max across scales", "max")
        self._method_combo.addItem("root-sum-of-squares", "rss")
        det_grid.addWidget(self._method_combo, row, 1)
        row += 1

        self._auto_norm_chk = QCheckBox("Auto-normalize wavelet "
                                        "(zero-mean, unit L2)")
        det_grid.addWidget(self._auto_norm_chk, row, 0, 1, 2)
        row += 1

        self._split_chk = QCheckBox("Split coincident regions "
                                    "into per-pulse sub-regions")
        det_grid.addWidget(self._split_chk, row, 0, 1, 2)
        row += 1

        det_grid.addWidget(QLabel("Event σ:"), row, 0)
        self._split_return_spin = QDoubleSpinBox()
        self._split_return_spin.setRange(0.5, 20.0)
        self._split_return_spin.setSingleStep(0.5)
        self._split_return_spin.setDecimals(1)
        self._split_return_spin.setToolTip(
            "A signal sample is 'in-event' when |signal − baseline| > "
            "this many σ. Lower values = more sensitive (catches subtle "
            "events but more noise-prone); higher = stricter.")
        det_grid.addWidget(self._split_return_spin, row, 1)
        row += 1

        det_grid.addWidget(QLabel("Merge gap frac:"), row, 0)
        self._split_merge_spin = QDoubleSpinBox()
        self._split_merge_spin.setRange(0.0, 1.0)
        self._split_merge_spin.setSingleStep(0.05)
        self._split_merge_spin.setDecimals(2)
        self._split_merge_spin.setToolTip(
            "Baseline gaps shorter than this fraction of the template "
            "length are bridged (treated as part of the same event). "
            "Larger = more events get merged into one; smaller = more "
            "splits. Should exceed the template's longest internal gap.")
        det_grid.addWidget(self._split_merge_spin, row, 1)
        panel_layout.addWidget(det_box)

        # Apply button (recomputes batch preview)
        self._apply_btn = QPushButton("Update Detection")
        self._apply_btn.clicked.connect(self._on_apply)
        panel_layout.addWidget(self._apply_btn)

        panel_layout.addSpacing(5)

        # Run mode
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

        # Stats
        stats_box = QGroupBox("Stats")
        stats_layout = QVBoxLayout(stats_box)
        self._stats_label = QLabel("—")
        self._stats_label.setWordWrap(True)
        stats_layout.addWidget(self._stats_label)
        panel_layout.addWidget(stats_box)

        panel_layout.addSpacing(5)

        # Saved settings
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

        # OK / Cancel
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
        # Backward compat: derive min/max/count from a legacy scales list
        # if present and the new fields are missing.
        if "scales" in p and p.get("scales") and "scaleMin" not in p:
            sl = list(p["scales"])
            p["scaleMin"] = float(min(sl))
            p["scaleMax"] = float(max(sl))
            p["scaleCount"] = int(len(sl))
        self._scale_min_spin.setValue(float(p.get("scaleMin", 0.5)))
        self._scale_max_spin.setValue(float(p.get("scaleMax", 2.0)))
        self._scale_count_spin.setValue(int(p.get("scaleCount", 3)))
        self._thr_spin.setValue(float(p["thresholdK"]))
        self._minsep_spin.setValue(float(p["minSeparationFrac"]))
        idx = self._method_combo.findData(p.get("scoringMethod", "max"))
        if idx >= 0:
            self._method_combo.setCurrentIndex(idx)
        self._auto_norm_chk.setChecked(bool(p.get("autoNormalize", True)))
        self._split_chk.setChecked(bool(p.get("splitCoincidents", True)))
        self._split_return_spin.setValue(float(p.get("splitReturnSigma", 3.0)))
        self._split_merge_spin.setValue(float(p.get("splitMergeGapFrac", 0.1)))
        if p.get("runMode") == "live":
            self._mode_live.setChecked(True)
        else:
            self._mode_batch.setChecked(True)
        speeds = [s for s, _ in _SIM_SPEEDS]
        if p.get("simSpeed") in speeds:
            self._speed_combo.setCurrentIndex(speeds.index(p["simSpeed"]))
        self._window_spin.setValue(float(p.get("liveWindowSec", 3.0)))

    def _read_params_from_widgets(self):
        smin = float(self._scale_min_spin.value())
        smax = float(self._scale_max_spin.value())
        if smin > smax:
            QMessageBox.warning(self, "Invalid scale range",
                                "Scale min must be ≤ Scale max.")
            return None
        return {
            "scaleMin":          smin,
            "scaleMax":          smax,
            "scaleCount":        int(self._scale_count_spin.value()),
            "thresholdK":        float(self._thr_spin.value()),
            "minSeparationFrac": float(self._minsep_spin.value()),
            "scoringMethod":     self._method_combo.currentData(),
            "autoNormalize":     bool(self._auto_norm_chk.isChecked()),
            "splitCoincidents":  bool(self._split_chk.isChecked()),
            "splitReturnSigma":  float(self._split_return_spin.value()),
            "splitMergeGapFrac": float(self._split_merge_spin.value()),
            "runMode":           "live" if self._mode_live.isChecked()
                                 else "batch",
            "simSpeed":          self._speed_combo.currentText(),
            "liveWindowSec":     float(self._window_spin.value()),
        }

    # --- Compute + draw --------------------------------------------------

    def _run_batch_and_redraw(self):
        try:
            self._result = detect_cwt(self._signal, self._wavelet_in,
                                      self._params)
        except Exception as e:
            QMessageBox.critical(self, "Detection failed", str(e))
            self._result = None
            return
        self._sim_revealed = len(self._signal)
        self._redraw()

    def _redraw(self):
        # Save the current x/y-limits so the user's pan/zoom survives a redraw.
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
        hint = self._result["coincidentHint"].astype(bool)

        # In live mode, slice to just the sliding window so the y-axis
        # autoscale tracks the visible region (oscilloscope behavior).
        if self._live_view and revealed < n:
            window_sec = float(self._params.get("liveWindowSec", 3.0))
            window_samples = max(2, int(round(window_sec * sr)))
            left_idx = max(0, revealed - window_samples)
            right_idx = revealed
            self._ax_sig.plot(t[left_idx:right_idx],
                              self._signal[left_idx:right_idx],
                              color="#3b6fb6", lw=0.8)
            self._ax_score.plot(t[left_idx:right_idx],
                                score[left_idx:right_idx],
                                color="#e08533", lw=0.8)
            # Shade only regions whose visible portion overlaps the window.
            for i, (a, b) in enumerate(regions):
                a_i, b_i = int(a), int(b)
                if b_i < left_idx or a_i >= right_idx:
                    continue
                a_show = max(a_i, left_idx)
                b_show = min(b_i, right_idx - 1)
                if b_show <= a_show:
                    continue
                color = "#d23f3f" if hint[i] else "#3aa55a"
                self._ax_sig.axvspan(a_show / sr, b_show / sr,
                                     color=color, alpha=0.25)
        else:
            self._ax_sig.plot(t, self._signal, color="#3b6fb6", lw=0.8)
            self._ax_score.plot(t, score, color="#e08533", lw=0.8)
            for i, (a, b) in enumerate(regions):
                color = "#d23f3f" if hint[i] else "#3aa55a"
                self._ax_sig.axvspan(int(a) / sr, int(b) / sr,
                                     color=color, alpha=0.25)

        self._ax_sig.set_ylabel("signal")
        self._ax_sig.set_title("Detection preview")
        thr = self._result["threshold"]
        self._ax_score.axhline(thr, color="#444", ls="--", lw=0.7,
                               label=f"thr={thr:.3g}")
        self._ax_score.set_ylabel("CWT score")
        self._ax_score.set_xlabel("time (s)")
        self._ax_score.legend(loc="upper right", fontsize=8)

        # X-axis policy:
        # - Live sim active: sliding window of `liveWindowSec` anchored to
        #   the right edge at revealed_time. User zoom is NOT preserved
        #   here — the sliding view is the whole point of live mode.
        # - Otherwise: full record, with user pan/zoom preserved if they
        #   zoomed (prev_xlim differs from the full range).
        full_xlim = (0.0, max(t[-1], 1.0 / sr))
        live_active = self._live_view and revealed < n
        if live_active:
            window_sec = float(self._params.get("liveWindowSec", 3.0))
            window_sec = max(window_sec, 1.0 / sr)
            right = revealed / sr
            left = max(0.0, right - window_sec)
            # Until the recording has played long enough to fill the
            # window, anchor the view to the left edge so the empty
            # right side is visible — like an oscilloscope warming up.
            if right < window_sec:
                left, right = 0.0, window_sec
            self._ax_sig.set_xlim(left, right)
            left_idx = max(0, int(round(left * sr)))
            right_idx = max(left_idx + 1, int(round(right * sr)))
            visible = self._signal[left_idx:min(right_idx, len(self._signal))]
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

        # Stats
        n_reg = len(regions)
        n_hint = int(hint.sum())
        sigma = self._result["sigma"]
        med = float(np.median(score)) if score.size else 0.0
        ac_floor = int(self._result.get("autocorrFloor", 0))
        ac_floor_s = ac_floor / sr if sr > 0 else 0.0
        wavelet_len = len(self._result.get("wavelet_used", []))
        user_min_sep = int(round(self._params.get("minSeparationFrac", 1.0)
                                 * max(wavelet_len, 1)))
        eff_min_sep = max(ac_floor, user_min_sep)
        self._stats_label.setText(
            f"Detected: {n_reg}\n"
            f"Coincident hints: {n_hint}\n"
            f"Score median: {med:.3g}\n"
            f"σ̂ (1.4826·MAD): {sigma:.3g}\n"
            f"Threshold: {thr:.3g}\n"
            f"Autocorr floor: {ac_floor} samp ({ac_floor_s:.3f} s)\n"
            f"Effective min sep: {eff_min_sep} samp"
        )

        self._canvas.draw_idle()

    # --- Button handlers -------------------------------------------------

    def _on_apply(self):
        p = self._read_params_from_widgets()
        if p is None:
            return
        self._params = p
        self._stop_sim()
        self._run_batch_and_redraw()

    def _on_run(self):
        p = self._read_params_from_widgets()
        if p is None:
            return
        self._params = p
        # Always recompute batch result first, then either reveal-all
        # (batch mode) or animate (live mode).
        try:
            self._result = detect_cwt(self._signal, self._wavelet_in,
                                      self._params)
        except Exception as e:
            QMessageBox.critical(self, "Detection failed", str(e))
            return
        if p["runMode"] == "batch":
            self._sim_revealed = len(self._signal)
            self._redraw()
            return
        self._start_sim()

    def _on_stop(self):
        # Stop sim and snap to full batch result.
        self._stop_sim()
        if self._result is not None:
            self._sim_revealed = len(self._signal)
            self._live_view = False
            self._progress.setValue(100)
            self._redraw()

    def _on_pause_toggle(self):
        if self._sim_timer is None:
            return
        if self._sim_paused:
            # Resume — re-anchor wall clock so elapsed time only counts
            # from now, not the wall time spent paused.
            self._sim_paused = False
            self._sim_revealed_at_resume = self._sim_revealed
            self._sim_t0 = perf_counter()
            self._sim_timer.start()
            self._pause_btn.setText("❚❚ Pause")
        else:
            # Pause
            self._sim_paused = True
            self._sim_timer.stop()
            self._pause_btn.setText("▶ Resume")

    def _on_accept(self):
        # Ensure the emitted result reflects current widget values.
        p = self._read_params_from_widgets()
        if p is None:
            return
        self._params = p
        self._stop_sim()
        try:
            self._result = detect_cwt(self._signal, self._wavelet_in,
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
            self._redraw()  # final redraw without live-clipping

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
        return saved_templates_dir(os.path.join("Detection", "RealtimeCWT"))

    def _on_save(self):
        if self._current_path:
            target = self._current_path
        else:
            self._on_save_as()
            return
        p = self._read_params_from_widgets()
        if p is None:
            return
        try:
            save_state(p, target)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_save_as(self):
        p = self._read_params_from_widgets()
        if p is None:
            return
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
                f"Realtime Detection (CWT) — {os.path.basename(path)}")
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
                f"Realtime Detection (CWT) — {os.path.basename(path)}")
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._sync_widgets_from_params()
        self._stop_sim()
        self._run_batch_and_redraw()

    # --- Result accessor ------------------------------------------------

    def result_state(self):
        """Return (params, result_dict) reflecting the last accepted run."""
        return dict(self._params), self._result
