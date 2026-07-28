"""RT Detection (Combined): pure helpers and the interactive QDialog.

Causal pulse detection on a *matched-step* score: the difference between
the mean of the most recent `w` samples and the mean of the previous `w`
samples. With `w = 1` this reduces to RT Detection (Threshold)'s
single-sample first difference; with `w > 1` it integrates over the
expected pulse-edge duration, giving roughly √w SNR gain on step-like
edges — useful when the pulse edge is smeared across many samples and
the per-sample diff is buried in noise.

Both the matched-step width (Edge width) and the post-IIR Smoothing
remain user-tunable, so this block subsumes the Threshold detector
(set Edge width = 0 ms to fall back to the original behavior).

Companion to RT Detection (CWT) / RT Detection (Threshold): same dialog
layout, same sliding-window live-feed simulation.
"""

import json
import os
from time import perf_counter

import numpy as np

from utils.paths import saved_templates_dir


_DEFAULT_PARAMS = {
    "diffEnabled":       True,        # if False, skip matched-step threshold/peak detection (CWT alone)
    "thresholdKPos":     3.0,         # rising threshold = +kPos · σ
    "thresholdKNeg":     3.0,         # falling threshold = -kNeg · σ
    "edgeWidthMs":       1.0,         # matched-step half-width in ms (w = round(edgeWidthMs·sr/1000)). 0 ⇒ 1-sample diff.
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
    # Optional Ricker (Mexican-hat) CWT side-channel for low-SNR pulses.
    # Runs in parallel with the matched-step and OR-unions the regions.
    "cwtEnabled":        False,
    "cwtWidthMs":        20.0,        # CWT wavelet half-width in ms
    "cwtThresholdK":     5.0,         # |cwt| > k·σ_cwt counts as a peak
    "cwtWaveletType":    "ricker",    # ricker | haar | dog1 | morlet | template (uses the wavelet input)
    "cwtTemplateEnvelope": False,     # when wavelet=template, multiply by a Gaussian envelope to taper the edges
    "downsampleFactor":  1,           # integer decimation factor reserved for future use; currently a no-op
    "runMode":           "batch",     # "batch" or "live"
    "simSpeed":          "1x",
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
                              pad_before_factor, pad_after_factor,
                              cwt_peaks=None, cwt_half_width=0):
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

    If `cwt_peaks` (an iterable of CWT-origin peak indices) and
    `cwt_half_width` (>0) are supplied, any cluster containing a CWT
    peak gets its padding floor-bounded to `cwt_half_width` samples on
    each side. This keeps isolated CWT crossings (single-peak clusters,
    core_width=1) from being squashed to a 1-sample region — the
    matched-filter response naturally has the kernel's scale, so the
    detected region should reflect that.

    Returns (padded_regions, core_regions). Padded regions are NOT
    merged here — that's a separate post-step.
    """
    if len(peaks_sorted) == 0:
        return [], []
    cwt_set = (set(int(p) for p in cwt_peaks)
               if cwt_peaks is not None and len(cwt_peaks) else None)
    padded = []
    cores = []
    i = 0
    while i < len(peaks_sorted):
        first = int(peaks_sorted[i])
        last = first
        cluster_has_cwt = (cwt_set is not None and first in cwt_set)
        j = i + 1
        while j < len(peaks_sorted):
            if int(peaks_sorted[j]) - last <= max_gap:
                last = int(peaks_sorted[j])
                if cwt_set is not None and last in cwt_set:
                    cluster_has_cwt = True
                j += 1
            else:
                break
        core_width = max(1, last - first + 1)
        pad_before = max(0, int(round(pad_before_factor * core_width)))
        pad_after = max(0, int(round(pad_after_factor * core_width)))
        if cluster_has_cwt and cwt_half_width > 0:
            pad_before = max(pad_before, int(cwt_half_width))
            pad_after = max(pad_after, int(cwt_half_width))
        rs = max(0, first - pad_before)
        re = min(n_signal - 1, last + pad_after)
        padded.append((rs, re))
        cores.append((first, last))
        i = j
    return padded, cores


def _matched_step_score(sig, w):
    """Causal matched-step detector.

    For each sample i, score[i] = mean(sig[i-w+1:i+1]) - mean(sig[i-2w+1:i-w+1])
    — the recent w-sample mean minus the prior w-sample mean. This is a
    causal Haar-style step filter; for a true step embedded in white
    noise it improves SNR by √w over a 1-sample diff.

    For i < 2w-1 the second window doesn't exist yet, so score[i] = 0.
    With w = 1 this exactly reproduces sig[i] - sig[i-1] (np.diff).
    """
    n = sig.size
    if n == 0 or w <= 0:
        return np.zeros(max(0, n - 1), dtype=float)
    if w == 1:
        return np.diff(sig)
    # cumsum trick: csum[k] = sum(sig[:k]), so sum(sig[a:b]) = csum[b] - csum[a].
    csum = np.concatenate(([0.0], np.cumsum(sig, dtype=np.float64)))
    # Score for samples i = 2w-1 .. n-1 (length n - 2w + 1)
    score_len = n - 2 * w + 1
    score = np.zeros(n - 1, dtype=float)
    if score_len > 0:
        i = np.arange(2 * w - 1, n)
        recent = (csum[i + 1] - csum[i + 1 - w]) / float(w)
        prior = (csum[i + 1 - w] - csum[i + 1 - 2 * w]) / float(w)
        # Place at i-1 to match np.diff's index convention (score[k] sits
        # between sig[k] and sig[k+1] when used as np.diff(sig)[k]).
        score[i - 1] = recent - prior
    return score


def _ricker_kernel(width_samples):
    """Mexican-hat / Ricker wavelet at the given width (in samples).

    Built on the canonical (1 − x²) · exp(−x²/2) form with x = t/a, then
    zero-mean and unit-L2 normalized so the convolution score's noise σ
    is independent of `width` for white-noise input.

    Length = 8·width + 1 (covers ±4σ of the underlying Gaussian, where
    most of the wavelet's energy lives).
    """
    a = max(1.0, float(width_samples))
    half = max(1, int(round(4.0 * a)))
    t = np.arange(-half, half + 1, dtype=float)
    x = (t / a) ** 2
    k = (1.0 - x) * np.exp(-0.5 * x)
    k = k - k.mean()
    norm = float(np.sqrt(np.sum(k * k)))
    if norm > 0:
        k = k / norm
    return k


def _haar_kernel(width_samples):
    """Step-edge detector: −1 then +1, each block ≈ width_samples long.
    Picks up sharp baseline shifts (good for square pulse edges)."""
    w = max(1, int(round(float(width_samples))))
    k = np.concatenate([-np.ones(w, dtype=float), np.ones(w, dtype=float)])
    norm = float(np.sqrt(np.sum(k * k)))
    if norm > 0:
        k = k / norm
    return k


def _dog1_kernel(width_samples):
    """First derivative of Gaussian — smooth edge detector. Asymmetric
    response: positive on rising edges, negative on falling edges."""
    a = max(1.0, float(width_samples))
    half = max(1, int(round(4.0 * a)))
    t = np.arange(-half, half + 1, dtype=float)
    k = -(t / (a * a)) * np.exp(-0.5 * (t / a) ** 2)
    k = k - k.mean()
    norm = float(np.sqrt(np.sum(k * k)))
    if norm > 0:
        k = k / norm
    return k


def _morlet_kernel(width_samples):
    """Real Morlet (cosine-modulated Gaussian) at the standard ω₀ = 5.
    Sensitive to oscillatory features near the wavelet's center frequency."""
    a = max(1.0, float(width_samples))
    half = max(1, int(round(4.0 * a)))
    t = np.arange(-half, half + 1, dtype=float)
    omega0 = 5.0
    k = np.exp(-0.5 * (t / a) ** 2) * np.cos(omega0 * t / a)
    k = k - k.mean()
    norm = float(np.sqrt(np.sum(k * k)))
    if norm > 0:
        k = k / norm
    return k


def _build_cwt_kernel(wavelet_type, width_samples, template,
                      template_envelope=False):
    """Dispatch to the requested kernel family. Returns None if the
    'template' option is selected but no template wavelet is supplied.

    If `template_envelope` is True and the wavelet is the input
    template, the raw template is multiplied by a Gaussian envelope
    (centered, σ ≈ length/6) before zero-mean / unit-L2 normalization.
    Tapering the edges suppresses ring-down artifacts for templates
    that don't return cleanly to baseline at their endpoints.
    """
    wt = str(wavelet_type or "ricker").lower()
    if wt == "template":
        if template is None:
            return None
        k = np.asarray(template, dtype=float).ravel().copy()
        if k.size == 0:
            return None
        if template_envelope and k.size > 1:
            n = k.size
            t = np.arange(n, dtype=float) - (n - 1) / 2.0
            sigma_env = max(1.0, n / 6.0)
            env = np.exp(-0.5 * (t / sigma_env) ** 2)
            k = k * env
        k = k - k.mean()
        norm = float(np.sqrt(np.sum(k * k)))
        if norm < 1e-12:
            return None
        return k / norm
    if wt == "haar":
        return _haar_kernel(width_samples)
    if wt == "dog1":
        return _dog1_kernel(width_samples)
    if wt == "morlet":
        return _morlet_kernel(width_samples)
    return _ricker_kernel(width_samples)


def _cwt_score(sig, width_samples):
    """Single-scale Ricker CWT score, same length as `sig`.

    Detrending strategy: subtract a wide rolling-mean baseline so the
    convolution input is locally zero-mean. Without this, slow drift +
    boundary reflection makes the Ricker fire a huge phantom "pulse"
    at each end of the record (the reflected ramp meets the original
    ramp in a V shape that looks just like a downward pulse).

    The detrend window is 4× the kernel length — wide enough that a
    real pulse only occupies a small fraction of it (so the baseline
    isn't pulled into the pulse and the score amplitude is preserved),
    but narrow enough to still track slow drift.

    Positive output values flag upward bumps; negative values flag
    downward dips of the matching width.
    """
    from scipy.ndimage import uniform_filter1d
    n = sig.size
    if width_samples < 1 or n < 2:
        return np.zeros(n, dtype=float)
    k = _ricker_kernel(width_samples)
    baseline_w = min(max(len(k), 4 * len(k)), n)
    baseline = uniform_filter1d(sig, size=baseline_w, mode="nearest")
    sig0 = sig - baseline
    half = (len(k) - 1) // 2
    if half >= n:
        return np.convolve(sig0, k, mode="same")
    sig_padded = np.pad(sig0, half, mode="reflect")
    from scipy.signal import fftconvolve
    score = fftconvolve(sig_padded, k, mode="valid")
    # The kernel can't fully overlap the real signal within `half`
    # samples of either end — even with detrending, residual drift
    # leaks through reflection. Blank those regions so they can't
    # trigger false detections.
    if half > 0:
        score[:half] = 0.0
        score[-half:] = 0.0
    return score


def _cwt_score_with_kernel(sig, kernel):
    """Same detrend + reflect-pad CWT score as `_cwt_score`, but using a
    user-supplied kernel (e.g. a template wavelet) instead of the Ricker.
    The kernel is zero-mean / unit-L2 normalized so the resulting score's
    σ behaves like the Ricker variant.
    """
    from scipy.ndimage import uniform_filter1d
    n = sig.size
    k = np.asarray(kernel, dtype=float).ravel()
    if k.size == 0 or n < 2:
        return np.zeros(n, dtype=float)
    # Reflect-pad + mode="valid" only produces a length-n output when the
    # kernel has odd length. Even-length kernels (common from the Wavelet
    # Editor) yield n-1 samples and the dialog then refuses to plot the
    # CWT trace because len(score) != n. Trim one sample to force odd.
    if k.size % 2 == 0:
        k = k[:-1]
    k = k - k.mean()
    norm = float(np.sqrt(np.sum(k * k)))
    if norm < 1e-12:
        return np.zeros(n, dtype=float)
    k = k / norm
    baseline_w = min(max(len(k), 4 * len(k)), n)
    baseline = uniform_filter1d(sig, size=baseline_w, mode="nearest")
    sig0 = sig - baseline
    half = (len(k) - 1) // 2
    if half >= n:
        return np.convolve(sig0, k, mode="same")
    sig_padded = np.pad(sig0, half, mode="reflect")
    # FFT-based convolution beats direct convolution by orders of
    # magnitude for the long Ricker kernel × long-signal pairs the
    # streaming pipeline produces — turns "Update Scores" on a
    # multi-minute record from tens of seconds into a couple seconds.
    from scipy.signal import fftconvolve
    score = fftconvolve(sig_padded, k, mode="valid")
    if half > 0:
        score[:half] = 0.0
        score[-half:] = 0.0
    return score


def cluster_peaks(all_peaks, n_signal, sample_rate, params,
                  cwt_peaks=None, cwt_half_width=0):
    """Cluster a sorted array of peak sample indices into final padded
    regions and inner cores, using merge-gap, padding factors, and a
    min-event filter — exactly the post-thresholding stage of
    `detect_combined`. Pulled out so 'Update Detected' can re-run only
    this step on cached peaks.

    `cwt_peaks` (subset of `all_peaks` that came from the CWT side
    channel) and `cwt_half_width` together let CWT-containing clusters
    bypass the `minEventMs` noise filter and get a kernel-scale padding
    floor. Without that, isolated CWT crossings (one peak, core_width=1)
    are silently dropped before they ever reach the top plot.
    """
    p = params or {}
    sr = float(sample_rate) if sample_rate else 10000.0
    merge_gap = max(1, int(round(float(p.get("mergeGapMs", 100.0))
                                 * 0.001 * sr)))
    pad_before_factor = max(0.0, float(p.get("padBeforeFactor", 0.2)))
    pad_after_factor = max(0.0, float(p.get("padAfterFactor", 0.2)))
    min_event = max(1, int(round(float(p.get("minEventMs", 5.0))
                                 * 0.001 * sr)))
    if all_peaks is None or len(all_peaks) == 0:
        return (np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int))
    peaks = np.sort(np.asarray(all_peaks, dtype=int).ravel())
    padded_grouped, cores_grouped = _group_peaks_into_regions(
        peaks, n_signal=int(n_signal), max_gap=merge_gap,
        pad_before_factor=pad_before_factor,
        pad_after_factor=pad_after_factor,
        cwt_peaks=cwt_peaks, cwt_half_width=int(cwt_half_width),
    )
    cwt_arr = (np.sort(np.asarray(cwt_peaks, dtype=int).ravel())
               if (cwt_peaks is not None and len(cwt_peaks)) else None)
    keep = []
    for i in range(len(cores_grouped)):
        a, b = cores_grouped[i]
        if (b - a + 1) >= min_event:
            keep.append(i)
            continue
        if cwt_arr is not None:
            lo = int(np.searchsorted(cwt_arr, a, side="left"))
            hi = int(np.searchsorted(cwt_arr, b, side="right"))
            if hi > lo:
                # Cluster contains a CWT peak — skip the min_event noise
                # filter, since CWT crossings are shape-validated rather
                # than 1-sample noise spikes.
                keep.append(i)
    padded_kept = [padded_grouped[i] for i in keep]
    cores_kept = [cores_grouped[i] for i in keep]
    # Each surviving cluster yields exactly one padded region + one core,
    # so regions:cores stay 1:1. (The previous post-padding merge fused
    # adjacent padded regions whose pad windows overlapped, leaving one
    # wide region with multiple inner cores — which read visually like
    # the same pulse marked more than once.)
    regions_arr = (np.asarray(padded_kept, dtype=int)
                   if padded_kept else np.zeros((0, 2), dtype=int))
    cores_arr = (np.asarray(cores_kept, dtype=int)
                 if cores_kept else np.zeros((0, 2), dtype=int))
    return regions_arr, cores_arr


def detect_combined(signal, sample_rate, params=None, wavelet=None,
                    cwt_sigma_override=None,
                    cwt_sigma_pos_override=None,
                    cwt_sigma_neg_override=None):
    """End-to-end: returns dict with detectedPulses, score (signed
    filtered matched-step), coincidentHint, thresholdPos / thresholdNeg,
    sigma.

    score is the filtered matched-step aligned to signal samples (length
    matches signal; first sample is 0).
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

    # Causal matched-step (mean of recent w samples − mean of prior w
    # samples), then one-pole IIR smoothing. w = 1 falls back to the
    # plain first-difference of RT Detection (Threshold).
    edge_ms = max(0.0, float(p.get("edgeWidthMs", 0.0)))
    w = max(1, int(round(edge_ms * 0.001 * sr)))
    diff = _matched_step_score(sig, w)
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
    # When the diff stage is disabled, the score is still computed (for
    # display) but no peaks contribute to detection — useful when only
    # the CWT side-channel is wanted.
    from scipy.signal import find_peaks
    diff_enabled = bool(p.get("diffEnabled", True))
    if diff_enabled:
        pos_idx, _ = find_peaks(diff_filt, height=t_pos)
        neg_idx, _ = find_peaks(-diff_filt, height=-t_neg)
        ms_peaks = np.concatenate([pos_idx + 1, neg_idx + 1]).astype(int)
    else:
        ms_peaks = np.array([], dtype=int)

    # Optional CWT side-channel: a Ricker (Mexican-hat) wavelet at a
    # user-set width finds slow / smeared bumps that the matched-step
    # misses. Peaks from both detectors are unioned before clustering,
    # so a region can be triggered by either kind of evidence.
    cwt_enabled = bool(p.get("cwtEnabled", False))
    cwt_score = np.zeros(n, dtype=float)
    cwt_sigma = 0.0
    cwt_sigma_pos = 0.0
    cwt_sigma_neg = 0.0
    cwt_thr = 0.0
    cwt_thr_pos = 0.0
    cwt_thr_neg = 0.0
    cwt_peaks = np.array([], dtype=int)
    cwt_kernel_half = 0
    # Back-compat: old saved settings used a `useTemplateWavelet` bool.
    wavelet_type = p.get("cwtWaveletType")
    if wavelet_type is None:
        wavelet_type = "template" if p.get("useTemplateWavelet") else "ricker"
    tpl_w = (np.asarray(wavelet, dtype=float).ravel()
             if wavelet is not None else None)
    if cwt_enabled:
        cwt_w = max(1, int(round(float(p.get("cwtWidthMs", 20.0))
                                 * 0.001 * sr)))
        kernel = _build_cwt_kernel(
            wavelet_type, cwt_w, tpl_w,
            template_envelope=bool(p.get("cwtTemplateEnvelope", False)),
        )
        if kernel is None:
            # 'template' selected without a wavelet — fall back to Ricker.
            kernel = _ricker_kernel(cwt_w)
        cwt_kernel_half = max(1, len(kernel) // 2)
        cwt_score = _cwt_score_with_kernel(sig, kernel)
        # Two-sided σ: σ⁺ for the +k·σ threshold (positive peaks),
        # σ⁻ for the −k·σ threshold (negative dips). Caller can pin
        # either explicitly (e.g. cumulative committed σ from the
        # streaming buffer); otherwise we estimate per-side MAD on the
        # chunk's CWT score, falling back to combined MAD when one side
        # is empty.
        if (cwt_sigma_pos_override is not None
                and cwt_sigma_pos_override > 0):
            cwt_sigma_pos = float(cwt_sigma_pos_override)
        elif cwt_sigma_override is not None and cwt_sigma_override > 0:
            cwt_sigma_pos = float(cwt_sigma_override)
        else:
            med_c = float(np.median(cwt_score))
            pos_dev = (cwt_score - med_c)[cwt_score > med_c]
            if pos_dev.size > 8:
                cwt_sigma_pos = 1.4826 * float(np.median(pos_dev))
            else:
                cwt_sigma_pos = 1.4826 * float(
                    np.median(np.abs(cwt_score - med_c)))
        if (cwt_sigma_neg_override is not None
                and cwt_sigma_neg_override > 0):
            cwt_sigma_neg = float(cwt_sigma_neg_override)
        elif cwt_sigma_override is not None and cwt_sigma_override > 0:
            cwt_sigma_neg = float(cwt_sigma_override)
        else:
            med_c = float(np.median(cwt_score))
            neg_dev = (med_c - cwt_score)[cwt_score < med_c]
            if neg_dev.size > 8:
                cwt_sigma_neg = 1.4826 * float(np.median(neg_dev))
            else:
                cwt_sigma_neg = 1.4826 * float(
                    np.median(np.abs(cwt_score - med_c)))
        # Single-σ output for back-compat with existing display paths.
        cwt_sigma = max(cwt_sigma_pos, cwt_sigma_neg)
        if cwt_sigma_pos > 0 or cwt_sigma_neg > 0:
            k_cwt = float(p.get("cwtThresholdK", 5.0))
            cwt_thr_pos = k_cwt * cwt_sigma_pos
            cwt_thr_neg = k_cwt * cwt_sigma_neg
            cwt_thr = max(cwt_thr_pos, cwt_thr_neg)
            pos_c, _ = (find_peaks(cwt_score, height=cwt_thr_pos)
                        if cwt_thr_pos > 0 else (np.array([], int), None))
            neg_c, _ = (find_peaks(-cwt_score, height=cwt_thr_neg)
                        if cwt_thr_neg > 0 else (np.array([], int), None))
            cwt_peaks = np.concatenate([pos_c, neg_c]).astype(int)
        else:
            cwt_thr_pos = 0.0
            cwt_thr_neg = 0.0

    all_peaks = np.sort(np.concatenate([ms_peaks, cwt_peaks])).astype(int)

    regions_arr, cores_arr = cluster_peaks(
        all_peaks, n, sr, p,
        cwt_peaks=cwt_peaks,
        cwt_half_width=cwt_kernel_half,
    )

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
        "cwtScore":       cwt_score,
        "cwtSigma":       float(cwt_sigma),
        "cwtSigmaPos":    float(cwt_sigma_pos),
        "cwtSigmaNeg":    float(cwt_sigma_neg),
        "cwtThreshold":   float(cwt_thr),
        "cwtThresholdPos": float(cwt_thr_pos),
        "cwtThresholdNeg": float(cwt_thr_neg),
        "cwtKernelHalf":  int(cwt_kernel_half),
        "cwtEnabled":     bool(cwt_enabled),
        "diffEnabled":    bool(diff_enabled),
        # Cached pre-clustering peaks so re-clustering with different
        # merge/pad parameters doesn't have to redo find_peaks.
        "msPeaks":        ms_peaks.astype(int),
        "cwtPeaks":       cwt_peaks.astype(int),
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
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QRadioButton,
    QButtonGroup, QGroupBox, QCheckBox, QFileDialog, QMessageBox, QWidget,
    QProgressBar, QProgressDialog,
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure


_SIM_SPEEDS = [
    ("0.5x", 0.5),
    ("1x",   1.0),
    ("10x",  10.0),
    ("100x", 100.0),
    ("asap", 0.0),
]


class RealtimeDetectionCombinedDialog(QDialog):
    """Interactive dialog for tuning + previewing threshold-based pulse
    detection. Layout and live-feed simulation match the CWT block.
    """

    def __init__(self, signal, sample_rate, params,
                 wavelet=None, current_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RT Detection (Combined)")
        self.resize(1500, 800)

        try:
            from utils.dialog_style import apply_dialog_style
            apply_dialog_style(self)
        except ImportError:
            pass

        self._signal = np.asarray(signal, dtype=float).ravel()
        self._sample_rate = float(sample_rate)
        self._wavelet = (np.asarray(wavelet, dtype=float).ravel()
                         if wavelet is not None else None)
        self._params = dict(_DEFAULT_PARAMS)
        self._params.update(params or {})
        # Back-compat: old saved settings used a `useTemplateWavelet` bool.
        if "cwtWaveletType" not in (params or {}) \
                and bool((params or {}).get("useTemplateWavelet")):
            self._params["cwtWaveletType"] = "template"
        if self._wavelet is None and \
                self._params.get("cwtWaveletType") == "template":
            # Saved state asked for the template wavelet but none is wired
            # up — fall back to the default rather than silently misfiring.
            self._params["cwtWaveletType"] = "ricker"
        self._current_path = current_path
        if current_path:
            self.setWindowTitle(
                f"RT Detection (Combined) — {os.path.basename(current_path)}")

        self._result = None
        self._sim_timer = None
        self._sim_revealed = 0
        self._sim_paused = False
        self._sim_t0 = 0.0
        self._sim_revealed_at_resume = 0
        self._live_view = False
        # Threshold drag state: 'pos' / 'neg' / 'cwt_pos' / 'cwt_neg' / None
        self._thr_drag = None
        # Artist refs so motion handler can move the lines smoothly.
        self._thr_line_pos = None
        self._thr_line_neg = None
        self._cwt_line_pos = None
        self._cwt_line_neg = None
        # Cached score-axis background for blit-based dragging.
        self._drag_bg = None
        # Pre-streaming batch result, kept so we can restore the static
        # full-record view when the live sim stops.
        self._batch_result = None
        # Streaming "freeze" state: once a detected pulse's end-padded
        # boundary has been a full 1 s in the past, its detection has
        # converged and we don't re-process those samples on subsequent
        # ticks. The cumulative buffers below hold the frozen scores
        # and signalFiltered values; only samples after _frozen_until
        # are touched by the per-tick detect_combined call.
        self._stream_score = None
        self._stream_cwt = None
        self._stream_sig_filt = None
        self._frozen_regions = []
        self._frozen_cores = []
        self._frozen_until = 0
        # Rolling history of detect_combined() wall times in milliseconds.
        # Each entry corresponds to one full pipeline run (raw signal in
        # → detected pulses out). Capped to keep memory bounded.
        self._proc_times_ms = []
        self._proc_times_max = 200
        # Y-range high-water marks (per axis). The visible-window y_diff
        # is monotonically tracked here so the y-range only grows during
        # a sim — preventing constant rescaling. Reset on start_sim and
        # step_back.
        self._top_y_diff_max = 0.0
        self._bot_y_abs_max = 0.0

        # Optional QProgressDialog driven from inside _run_streaming_full
        # while the user waits for Update Scores to finish.
        self._progress_dialog = None

        self._build_ui()
        self._sync_widgets_from_params()
        self._redraw_wavelet_preview()
        # Detection no longer runs on open — the streaming pipeline is
        # expensive enough that we wait for the user to click Update
        # Scores. Show the raw signal in the meantime so the panel isn't
        # blank.
        self._draw_idle_placeholder()

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
        # Two-column grid of groups. Wider than the previous single-column
        # 300 px to fit two columns of QGroupBox content.
        panel.setFixedWidth(600)
        panel_outer = QVBoxLayout(panel)
        panel_outer.setContentsMargins(5, 5, 5, 5)
        main.addWidget(panel)

        # Two side-by-side columns. Each group sits at its own natural
        # height — heights don't need to line up across columns.
        # Column 0 (left, top→bottom):  Mode, Diff thresholding, CWT,
        #     Wavelet preview, Update Scores.
        # Column 1 (right, top→bottom): Pre-filter, Downsample, Detection
        #     merging, Padding, Update Detected, Info.
        cols = QHBoxLayout()
        cols.setSpacing(8)
        col0 = QVBoxLayout()
        col0.setSpacing(5)
        col1 = QVBoxLayout()
        col1.setSpacing(5)
        cols.addLayout(col0, 1)
        cols.addLayout(col1, 1)
        panel_outer.addLayout(cols)

        # 1) Pre-filter group: causal lowpass + notch applied to the
        # signal before any score is computed. Both can be toggled
        # independently. Placed first because everything downstream
        # operates on the (optionally) filtered signal.
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

        # 2) Diff thresholding group: matched-step (Δsignal) score and
        # its detection thresholds. Owns the toggle that gates whether
        # this score contributes peaks to the union.
        diff_box = QGroupBox("Diff thresholding")
        diff_grid = QGridLayout(diff_box)
        drow = 0

        self._diff_chk = QCheckBox("Enable diff thresholding")
        self._diff_chk.setToolTip(
            "When off, the matched-step (Δsignal) score is still computed "
            "but its threshold lines and peaks don't contribute to "
            "detection — and the trace is hidden from the score plot. "
            "Useful when only the CWT side-channel should fire events.")
        diff_grid.addWidget(self._diff_chk, drow, 0, 1, 2)
        drow += 1

        diff_grid.addWidget(QLabel("Threshold k+:"), drow, 0)
        self._thr_pos_spin = QDoubleSpinBox()
        self._thr_pos_spin.setRange(0.1, 50.0)
        self._thr_pos_spin.setSingleStep(0.5)
        self._thr_pos_spin.setDecimals(2)
        self._thr_pos_spin.setToolTip(
            "Rising-edge threshold multiplier: rising edge fires when "
            "filtered Δsignal > +k·σ. Drag the upper green line in the "
            "bottom plot to adjust visually.")
        diff_grid.addWidget(self._thr_pos_spin, drow, 1)
        drow += 1

        diff_grid.addWidget(QLabel("Threshold k−:"), drow, 0)
        self._thr_neg_spin = QDoubleSpinBox()
        self._thr_neg_spin.setRange(0.1, 50.0)
        self._thr_neg_spin.setSingleStep(0.5)
        self._thr_neg_spin.setDecimals(2)
        self._thr_neg_spin.setToolTip(
            "Falling-edge threshold multiplier: falling edge fires when "
            "filtered Δsignal < −k·σ. Drag the lower red line in the "
            "bottom plot to adjust visually.")
        diff_grid.addWidget(self._thr_neg_spin, drow, 1)
        drow += 1

        diff_grid.addWidget(QLabel("Edge width (ms):"), drow, 0)
        self._edge_spin = QDoubleSpinBox()
        self._edge_spin.setRange(0.0, 1000.0)
        self._edge_spin.setSingleStep(0.5)
        self._edge_spin.setDecimals(2)
        self._edge_spin.setToolTip(
            "Matched-step kernel half-width: score = mean(recent w samples) "
            "− mean(prior w samples), with w = round(edge_width_ms × "
            "sample_rate / 1000). Wider w boosts SNR on slow/smeared "
            "edges by ≈ √w, but blurs short events. Set to 0 ms to fall "
            "back to the 1-sample first-difference (Threshold detector).")
        diff_grid.addWidget(self._edge_spin, drow, 1)
        drow += 1

        diff_grid.addWidget(QLabel("Smoothing:"), drow, 0)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.01, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setToolTip(
            "One-pole causal IIR on the Δ signal: y[n] = α·x[n] + "
            "(1−α)·y[n−1]. Lower = heavier smoothing of sample-to-"
            "sample noise on the diff; α=1 disables smoothing.")
        diff_grid.addWidget(self._alpha_spin, drow, 1)

        # Optional Ricker-CWT side-channel for low-SNR pulses. Runs in
        # parallel with the matched-step; peak unions trigger detection.
        cwt_box = QGroupBox("CWT (low-SNR helper)")
        cwt_grid = QGridLayout(cwt_box)
        crow = 0
        self._cwt_chk = QCheckBox("Enable CWT")
        self._cwt_chk.setToolTip(
            "Run a CWT in parallel with the matched-step. Peaks in "
            "|CWT score| above the CWT threshold are added to the "
            "matched-step peaks before clustering, so either detector "
            "can fire a region.")
        cwt_grid.addWidget(self._cwt_chk, crow, 0, 1, 2)
        crow += 1

        cwt_grid.addWidget(QLabel("Wavelet:"), crow, 0)
        self._cwt_wavelet_combo = QComboBox()
        # (display label, internal key)
        self._cwt_wavelet_options = [
            ("Ricker (Mexican hat)", "ricker"),
            ("Haar (step edge)",     "haar"),
            ("Gaussian deriv. (DoG-1)", "dog1"),
            ("Morlet (real)",        "morlet"),
            ("Template (input)",     "template"),
        ]
        for label, _ in self._cwt_wavelet_options:
            self._cwt_wavelet_combo.addItem(label)
        self._cwt_wavelet_combo.setToolTip(
            "Wavelet family used for the CWT kernel. 'Template (input)' "
            "uses the wavelet wired to the 'wavelet' input port at its "
            "native length and ignores Width (ms); the other options are "
            "synthesized at the chosen Width.")
        # Disable the 'template' entry if no wavelet was provided.
        if self._wavelet is None or self._wavelet.size == 0:
            tpl_idx = next(i for i, (_, k) in enumerate(
                self._cwt_wavelet_options) if k == "template")
            model = self._cwt_wavelet_combo.model()
            item = model.item(tpl_idx)
            if item is not None:
                item.setEnabled(False)
                item.setToolTip("No wavelet input is connected.")
        cwt_grid.addWidget(self._cwt_wavelet_combo, crow, 1)
        self._cwt_wavelet_combo.currentIndexChanged.connect(
            self._on_cwt_wavelet_changed)
        crow += 1

        # Optional Gaussian-envelope taper for the template wavelet —
        # suppresses ring-down on templates whose endpoints don't return
        # cleanly to baseline. Only meaningful when wavelet=template.
        self._cwt_envelope_chk = QCheckBox("Apply envelope to template")
        self._cwt_envelope_chk.setToolTip(
            "Multiply the input template wavelet by a centered Gaussian "
            "envelope (σ ≈ length/6) before normalization. Reduces edge "
            "artifacts when the template doesn't taper to zero on its "
            "own. Only applies when Wavelet = 'Template (input)'.")
        cwt_grid.addWidget(self._cwt_envelope_chk, crow, 0, 1, 2)
        self._cwt_envelope_chk.toggled.connect(
            lambda *_: self._redraw_wavelet_preview())
        crow += 1

        cwt_grid.addWidget(QLabel("Width (ms):"), crow, 0)
        self._cwt_width_spin = QDoubleSpinBox()
        self._cwt_width_spin.setRange(0.5, 1000.0)
        self._cwt_width_spin.setSingleStep(1.0)
        self._cwt_width_spin.setDecimals(2)
        self._cwt_width_spin.setToolTip(
            "Ricker wavelet half-width in ms. Best detection happens when "
            "this matches the expected pulse half-width — the matched-"
            "filter SNR peaks at that scale.")
        cwt_grid.addWidget(self._cwt_width_spin, crow, 1)
        self._cwt_width_spin.valueChanged.connect(
            lambda *_: self._redraw_wavelet_preview())
        crow += 1
        cwt_grid.addWidget(QLabel("Threshold k:"), crow, 0)
        self._cwt_thr_spin = QDoubleSpinBox()
        self._cwt_thr_spin.setRange(0.1, 50.0)
        self._cwt_thr_spin.setSingleStep(0.5)
        self._cwt_thr_spin.setDecimals(2)
        self._cwt_thr_spin.setToolTip(
            "|CWT score| > k · σ_CWT counts as a peak. Drag either "
            "magenta dashed line in the score plot, or set here.")
        cwt_grid.addWidget(self._cwt_thr_spin, crow, 1)

        # 4) Detection merging group: how peaks (from diff and/or CWT)
        # cluster into events, and which clusters survive as detections.
        merge_box = QGroupBox("Detection merging")
        merge_grid = QGridLayout(merge_box)
        mrow = 0
        merge_grid.addWidget(QLabel("Merge gap (ms):"), mrow, 0)
        self._merge_spin = QDoubleSpinBox()
        self._merge_spin.setRange(0.0, 1000.0)
        self._merge_spin.setSingleStep(5.0)
        self._merge_spin.setDecimals(1)
        self._merge_spin.setToolTip(
            "Maximum peak-to-peak spacing within one region. Threshold "
            "crossings closer than this are grouped into the same "
            "detected pulse — set this to be wider than the longest "
            "internal gap of your pulse template.")
        merge_grid.addWidget(self._merge_spin, mrow, 1)
        mrow += 1

        merge_grid.addWidget(QLabel("Min event (ms):"), mrow, 0)
        self._minev_spin = QDoubleSpinBox()
        self._minev_spin.setRange(0.0, 5000.0)
        self._minev_spin.setSingleStep(1.0)
        self._minev_spin.setDecimals(1)
        self._minev_spin.setToolTip(
            "Reject events shorter than this duration as noise.")
        merge_grid.addWidget(self._minev_spin, mrow, 1)

        # 5) Padding group: extend each cluster's span symmetrically by
        # a factor of its own core width.
        pad_box = QGroupBox("Padding")
        pad_grid = QGridLayout(pad_box)
        prow2 = 0
        pad_grid.addWidget(QLabel("Pad before (×):"), prow2, 0)
        self._pad_before_spin = QDoubleSpinBox()
        self._pad_before_spin.setRange(0.0, 5.0)
        self._pad_before_spin.setSingleStep(0.1)
        self._pad_before_spin.setDecimals(2)
        self._pad_before_spin.setToolTip(
            "Pre-event padding as a factor of the cluster core width "
            "(peak-to-peak span). Wider events get proportionally wider "
            "padding, so this scales naturally with pulse size.")
        pad_grid.addWidget(self._pad_before_spin, prow2, 1)
        prow2 += 1

        pad_grid.addWidget(QLabel("Pad after (×):"), prow2, 0)
        self._pad_after_spin = QDoubleSpinBox()
        self._pad_after_spin.setRange(0.0, 5.0)
        self._pad_after_spin.setSingleStep(0.1)
        self._pad_after_spin.setDecimals(2)
        self._pad_after_spin.setToolTip(
            "Post-event padding as a factor of the cluster core width.")
        pad_grid.addWidget(self._pad_after_spin, prow2, 1)

        # Downsample group: integer decimation factor. Currently a stored
        # preference; not yet applied to detection (placeholder for future
        # signal pre-decimation support).
        down_box = QGroupBox("Downsample")
        down_grid = QGridLayout(down_box)
        down_grid.addWidget(QLabel("Factor:"), 0, 0)
        self._down_factor_spin = QSpinBox()
        self._down_factor_spin.setRange(1, 100)
        self._down_factor_spin.setSingleStep(1)
        self._down_factor_spin.setToolTip(
            "Integer decimation factor (1 = no downsampling). Stored with "
            "the detector settings; not yet applied to detection.")
        down_grid.addWidget(self._down_factor_spin, 0, 1)

        run_box = QGroupBox("Mode")
        run_layout = QVBoxLayout(run_box)
        self._mode_group = QButtonGroup(self)
        self._mode_batch = QRadioButton("Process All")
        self._mode_live = QRadioButton("Live simulation")
        self._mode_group.addButton(self._mode_batch)
        self._mode_group.addButton(self._mode_live)
        run_layout.addWidget(self._mode_batch)
        run_layout.addWidget(self._mode_live)

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

        step_btns = QHBoxLayout()
        self._step_back_btn = QPushButton("❙◁ Step back")
        self._step_back_btn.clicked.connect(self._on_step_back)
        self._step_back_btn.setToolTip(
            "Roll the live sim back by one batch (~33 ms of data) and "
            "redraw. Re-runs streaming detection up to the new position "
            "so the displayed result matches a fresh sim at that point.")
        self._step_btn = QPushButton("▷❙ Step")
        self._step_btn.clicked.connect(self._on_step)
        self._step_btn.setToolTip(
            "Advance the live sim by one batch (~33 ms of data) and "
            "redraw. Works whether running or paused; if no sim is "
            "active it starts a paused sim at the beginning.")
        step_btns.addWidget(self._step_back_btn)
        step_btns.addWidget(self._step_btn)
        run_layout.addLayout(step_btns)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        run_layout.addWidget(self._progress)

        # Info group (sits in the bottom-right grid cell).
        stats_box = QGroupBox("Info")
        stats_layout = QVBoxLayout(stats_box)
        self._sr_label = QLabel(f"Sample rate: {int(self._sample_rate)} Hz")
        self._sr_label.setWordWrap(True)
        stats_layout.addWidget(self._sr_label)
        self._stats_label = QLabel("—")
        self._stats_label.setWordWrap(True)
        stats_layout.addWidget(self._stats_label)

        # Wavelet preview group (sits in the (3,0) cell). Refreshes when
        # the combo / width / wavelet input changes.
        wave_box = QGroupBox("Wavelet preview")
        wave_layout = QVBoxLayout(wave_box)
        wave_layout.setContentsMargins(4, 4, 4, 4)
        self._wave_fig = Figure(figsize=(3.0, 1.4), tight_layout=True)
        style_mpl_figure(self._wave_fig)
        self._wave_ax = self._wave_fig.add_subplot(1, 1, 1)
        self._wave_canvas = FigureCanvas(self._wave_fig)
        self._wave_canvas.setFixedHeight(140)
        wave_layout.addWidget(self._wave_canvas)

        # Update buttons: 'Update Scores' re-runs the full pipeline (the
        # old 'Update Detection'); 'Update Detected' only re-clusters the
        # cached peaks with the current merge/padding parameters.
        self._update_scores_btn = QPushButton("Update Scores")
        self._update_scores_btn.setToolTip(
            "Recompute scores and detection with the current pre-filter, "
            "diff, and CWT settings (full pipeline run).")
        self._update_scores_btn.clicked.connect(self._on_apply)

        self._update_detected_btn = QPushButton("Update Detected")
        self._update_detected_btn.setToolTip(
            "Re-cluster the cached peaks using the current Merging and "
            "Padding parameters only — does not recompute scores.")
        self._update_detected_btn.clicked.connect(self._on_recluster)

        # Place groups + buttons + Info in their two columns.
        col0.addWidget(run_box)
        col0.addWidget(diff_box)
        col0.addWidget(cwt_box)
        col0.addWidget(wave_box)
        col0.addWidget(self._update_scores_btn)
        col0.addStretch(1)

        col1.addWidget(pre_box)
        col1.addWidget(down_box)
        col1.addWidget(merge_box)
        col1.addWidget(pad_box)
        col1.addWidget(self._update_detected_btn)
        # Push Info to the bottom of the right column so it sits directly
        # above the OK/Cancel row.
        col1.addStretch(1)
        col1.addWidget(stats_box)

        # Bottom row spans both columns: Save / Save as… / Load… on the
        # left, OK / Cancel on the right.
        bottom = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn = QPushButton("Save as…")
        self._save_as_btn.clicked.connect(self._on_save_as)
        self._load_btn = QPushButton("Load…")
        self._load_btn.clicked.connect(self._on_load)
        bottom.addWidget(self._save_btn)
        bottom.addWidget(self._save_as_btn)
        bottom.addWidget(self._load_btn)
        bottom.addStretch(1)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(ok_btn)
        bottom.addWidget(cancel_btn)
        panel_outer.addLayout(bottom)

    # --- Param ⇄ widget sync ---------------------------------------------

    def _sync_widgets_from_params(self):
        p = self._params
        # Backward compat: old saved settings used a single thresholdK.
        legacy_k = float(p.get("thresholdK", 3.0))
        self._diff_chk.setChecked(bool(p.get("diffEnabled", True)))
        self._thr_pos_spin.setValue(float(p.get("thresholdKPos", legacy_k)))
        self._thr_neg_spin.setValue(float(p.get("thresholdKNeg", legacy_k)))
        self._edge_spin.setValue(float(p.get("edgeWidthMs", 0.0)))
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
        self._cwt_chk.setChecked(bool(p.get("cwtEnabled", False)))
        self._cwt_width_spin.setValue(float(p.get("cwtWidthMs", 20.0)))
        self._cwt_thr_spin.setValue(float(p.get("cwtThresholdK", 5.0)))
        self._down_factor_spin.setValue(int(p.get("downsampleFactor", 1)))
        wt = str(p.get("cwtWaveletType", "ricker")).lower()
        keys = [k for _, k in self._cwt_wavelet_options]
        idx = keys.index(wt) if wt in keys else 0
        # If 'template' was requested but the entry is disabled (no
        # wavelet), fall back to the first enabled entry.
        model = self._cwt_wavelet_combo.model()
        item = model.item(idx)
        if item is not None and not item.isEnabled():
            idx = 0
        self._cwt_wavelet_combo.setCurrentIndex(idx)
        is_template = self._cwt_wavelet_options[idx][1] == "template"
        self._cwt_width_spin.setEnabled(not is_template)
        self._cwt_envelope_chk.setChecked(
            bool(p.get("cwtTemplateEnvelope", False)))
        self._cwt_envelope_chk.setEnabled(is_template)
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
            "diffEnabled":    bool(self._diff_chk.isChecked()),
            "thresholdKPos":  float(self._thr_pos_spin.value()),
            "thresholdKNeg":  float(self._thr_neg_spin.value()),
            "edgeWidthMs":    float(self._edge_spin.value()),
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
            "cwtEnabled":     bool(self._cwt_chk.isChecked()),
            "cwtWidthMs":     float(self._cwt_width_spin.value()),
            "cwtThresholdK":  float(self._cwt_thr_spin.value()),
            "cwtWaveletType": self._cwt_wavelet_options[
                self._cwt_wavelet_combo.currentIndex()][1],
            "cwtTemplateEnvelope": bool(
                self._cwt_envelope_chk.isChecked()),
            "downsampleFactor": int(self._down_factor_spin.value()),
            "runMode":        "live" if self._mode_live.isChecked()
                              else "batch",
            "simSpeed":       self._speed_combo.currentText(),
            "liveWindowSec":  float(self._window_spin.value()),
        }

    # --- Compute + draw --------------------------------------------------

    def _live_batch_samples(self):
        """Live-feed batch size in samples — ~33 ms of data at the
        current sample rate, matching the sim-tick interval at 1×
        speed. This is also a reasonable proxy for the per-chunk size
        a streaming / FPGA pipeline would process at this sample rate.
        """
        return max(64, int(round(self._sample_rate * 0.033)))

    def _detect_timed(self):
        """Produce the overall detection result by replaying the live-sim
        streaming pipeline over the full signal at full speed (no UI
        delay). The detected-pulse list returned here is exactly what a
        real-time feed of this same record would have committed — every
        region survives the freeze buffer or is flushed at end-of-data,
        so it's safe to hand directly to a downstream classifier.
        Per-chunk wall times accumulated during the run feed the Info
        panel's process-time stats.
        """
        self._proc_times_ms = []
        return self._run_streaming_full()

    def _run_streaming_full(self):
        """Drive _update_streaming_result repeatedly with no timer, then
        commit any still-active regions at end-of-record.

        The live-sim runs at ~33 ms per tick to match real-time playback;
        for batch processing the tick size is bumped to ~1 s so the
        per-chunk pipeline gets called orders of magnitude fewer times.
        The freeze rule is sample-distance based, not tick-count based,
        so the eventual committed regions match what live sim would
        produce.
        """
        n = len(self._signal)
        if n < 2:
            return self._empty_stream_result(n)
        self._reset_stream_state()
        self._stream_active_regions = []
        self._stream_active_cores = []
        live_batch = self._live_batch_samples()
        batch = max(live_batch, int(round(self._sample_rate * 1.0)))
        revealed = 0
        # Pump the event loop and update the progress dialog (if any)
        # roughly twice a second so the UI stays responsive on long
        # records. The streaming pipeline itself remains synchronous.
        ui_pump_every = max(1, int(round(0.5 * self._sample_rate / batch)))
        steps = 0
        cancelled = False
        while revealed < n:
            revealed = min(n, revealed + batch)
            self._sim_revealed = revealed
            self._update_streaming_result()
            steps += 1
            if steps % ui_pump_every == 0:
                if self._progress_dialog is not None:
                    pct = int(round(100.0 * revealed / max(n, 1)))
                    self._progress_dialog.setValue(min(pct, 99))
                    elapsed = perf_counter() - getattr(
                        self, "_progress_t0", perf_counter())
                    self._progress_dialog.setLabelText(
                        f"Running streaming detection…\n"
                        f"Elapsed: {elapsed:.1f} s")
                    if self._progress_dialog.wasCanceled():
                        cancelled = True
                QApplication.processEvents()
                if cancelled:
                    break
        # End-of-data flush. If cancelled mid-run, skip — partial
        # result keeps only the regions that were already frozen.
        if not cancelled:
            self._finalize_stream_at_end_of_data()
        if self._progress_dialog is not None and not cancelled:
            self._progress_dialog.setValue(100)
            QApplication.processEvents()
        # Rebuild the result dict so detectedPulses reflects only
        # committed (frozen + flushed) regions.
        self._build_stream_result_dict()
        if len(self._proc_times_ms) > self._proc_times_max:
            self._proc_times_ms = self._proc_times_ms[-self._proc_times_max:]
        return self._result

    def _empty_stream_result(self, n):
        return {
            "detectedPulses": np.zeros((0, 2), dtype=int),
            "coreRegions":    np.zeros((0, 2), dtype=int),
            "score":          np.zeros(n, dtype=float),
            "cwtScore":       np.zeros(n, dtype=float),
            "signalFiltered": self._signal.astype(float).copy(),
            "coincidentHint": np.zeros(0, dtype=np.int8),
            "sigma":          0.0,
            "thresholdPos":   0.0,
            "thresholdNeg":   0.0,
            "threshold":      0.0,
            "cwtSigma":       0.0,
            "cwtThreshold":   0.0,
            "cwtEnabled":     bool(self._params.get("cwtEnabled", False)),
            "diffEnabled":    bool(self._params.get("diffEnabled", True)),
            "msPeaks":        np.array([], dtype=int),
            "cwtPeaks":       np.array([], dtype=int),
        }

    def _run_batch_and_redraw(self):
        try:
            self._result = self._detect_timed()
        except Exception as e:
            QMessageBox.critical(self, "Detection failed", str(e))
            self._result = None
            return
        self._batch_result = self._result
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

        diff_enabled_disp = bool(self._result.get("diffEnabled", True))

        if self._live_view and revealed < n:
            window_sec = float(self._params.get("liveWindowSec", 3.0))
            window_samples = max(2, int(round(window_sec * sr)))
            left_idx = max(0, revealed - window_samples)
            right_idx = revealed
            self._ax_sig.plot(t[left_idx:right_idx],
                              sig_to_plot[left_idx:right_idx],
                              color="#3b6fb6", lw=0.8)
            if diff_enabled_disp:
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
            if diff_enabled_disp:
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
        # Diff thresholds are draggable, so the artist refs need to
        # exist when enabled. When disabled, drop them so the hover/
        # drag handlers see them as absent and the plot isn't cluttered
        # with non-functional lines.
        if diff_enabled_disp:
            self._thr_line_pos = self._ax_score.axhline(
                t_pos, color="#3aa55a", ls="--", lw=1.4,
                label=f"+k·σ={t_pos:.3g}")
            self._thr_line_neg = self._ax_score.axhline(
                t_neg, color="#d23f3f", ls="--", lw=1.4,
                label=f"−k·σ={t_neg:.3g}")
        else:
            self._thr_line_pos = None
            self._thr_line_neg = None
        self._ax_score.axhline(0, color="#888", ls=":", lw=0.5)

        # Optional CWT side-channel overlay.
        #  - Batch / summary view: rescale CWT to the matched-step σ so
        #    both noise bands share a single y-axis.
        #  - Live sim view: plot CWT at *native* amplitude with the
        #    threshold derived from the same σ the detector used (the
        #    cumulative committed σ, carried back as result["cwtSigma"]).
        #    This keeps detected regions and the displayed threshold
        #    consistent — every detection has a CWT crossing visible
        #    above the dashed line.
        cwt_enabled = bool(self._result.get("cwtEnabled", False))
        cwt_score = self._result.get("cwtScore")
        cwt_sigma_raw = float(self._result.get("cwtSigma", 0.0) or 0.0)
        cwt_thr_raw = float(self._result.get("cwtThreshold", 0.0) or 0.0)
        diff_sigma = float(self._result.get("sigma", 0.0) or 0.0)
        live_cwt = (self._live_view and revealed < n
                    and cwt_enabled and cwt_score is not None
                    and len(cwt_score) == n)
        # Reset; populated below when applicable.
        self._cwt_sigma_dynamic = 0.0
        self._cwt_native_scale = False
        if live_cwt:
            window_sec = float(self._params.get("liveWindowSec", 3.0))
            window_samples = max(2, int(round(window_sec * sr)))
            left_idx = max(0, revealed - window_samples)
            right_idx = revealed
            visible_cwt = cwt_score[left_idx:right_idx]
            self._ax_score.plot(
                t[left_idx:right_idx], visible_cwt,
                color="#9b59b6", lw=0.7, alpha=0.85,
                label="CWT (native)")
            # Threshold history: per-sample +k·σ⁺ / −k·σ⁻ traces written
            # at each sample's commit time. Plotted as solid lines so
            # the user can see exactly where the threshold was when each
            # bit of CWT score was decided. The dashed-line "current
            # threshold" representation only made sense when σ was
            # global; with locked-on-commit detection it would be a lie.
            stream_thr_pos = getattr(self, "_stream_thr_pos", None)
            stream_thr_neg = getattr(self, "_stream_thr_neg", None)
            if (stream_thr_pos is not None and stream_thr_neg is not None
                    and len(stream_thr_pos) == n
                    and len(stream_thr_neg) == n):
                vp = stream_thr_pos[left_idx:right_idx]
                vn = stream_thr_neg[left_idx:right_idx]
                tt = t[left_idx:right_idx]
                if np.isfinite(vp).any():
                    self._ax_score.plot(
                        tt, vp, color="#9b59b6", ls="--", lw=1.0,
                        label="+k·σ⁺ at commit")
                if np.isfinite(vn).any():
                    self._ax_score.plot(
                        tt, vn, color="#9b59b6", ls="--", lw=1.0)
            self._cwt_line_pos = None
            self._cwt_line_neg = None
            self._cwt_sigma_dynamic = float(cwt_sigma_raw)
            self._cwt_native_scale = True
        elif (cwt_enabled and cwt_score is not None
                and len(cwt_score) == n):
            # Summary / post-sim: plot the full-record CWT score at
            # native amplitude, overlaid with the per-sample threshold
            # trace that was actually used during the streaming pass.
            # Falls back to a single horizontal ±k·σ axhline if the
            # trace buffers haven't been populated yet (e.g. result
            # restored from a saved workflow without streaming state).
            self._ax_score.plot(
                t, cwt_score,
                color="#9b59b6", lw=0.7, alpha=0.85,
                label="CWT (native)")
            stream_thr_pos = getattr(self, "_stream_thr_pos", None)
            stream_thr_neg = getattr(self, "_stream_thr_neg", None)
            thr_trace_drawn = False
            if (stream_thr_pos is not None and stream_thr_neg is not None
                    and len(stream_thr_pos) == n
                    and len(stream_thr_neg) == n
                    and (np.isfinite(stream_thr_pos).any()
                         or np.isfinite(stream_thr_neg).any())):
                if np.isfinite(stream_thr_pos).any():
                    self._ax_score.plot(
                        t, stream_thr_pos, color="#9b59b6", ls="--", lw=1.0,
                        label="+k·σ⁺ at commit")
                if np.isfinite(stream_thr_neg).any():
                    self._ax_score.plot(
                        t, stream_thr_neg, color="#9b59b6", ls="--", lw=1.0)
                thr_trace_drawn = True
            cwt_thr_raw_disp = float(self._result.get("cwtThreshold", 0.0) or 0.0)
            if not thr_trace_drawn and cwt_thr_raw_disp > 0:
                self._cwt_line_pos = self._ax_score.axhline(
                    cwt_thr_raw_disp, color="#9b59b6", ls="--", lw=1.2,
                    label=f"CWT ±k·σ={cwt_thr_raw_disp:.3g}")
                self._cwt_line_neg = self._ax_score.axhline(
                    -cwt_thr_raw_disp, color="#9b59b6", ls="--", lw=1.2)
            else:
                self._cwt_line_pos = None
                self._cwt_line_neg = None
            # Stash the values needed to set y-limits at the very end of
            # _redraw — set_ylim there is the last write, so nothing
            # downstream can clobber it. Limit covers both the score and
            # the threshold trace at native amplitude.
            arr_c = np.asarray(cwt_score, dtype=float)
            disp_finite = arr_c[np.isfinite(arr_c)]
            if disp_finite.size:
                pending_lo = float(np.min(disp_finite))
                pending_hi = float(np.max(disp_finite))
            else:
                pending_lo, pending_hi = 0.0, 0.0
            if thr_trace_drawn:
                if np.isfinite(stream_thr_pos).any():
                    pending_hi = max(pending_hi,
                                     float(np.nanmax(stream_thr_pos)))
                if np.isfinite(stream_thr_neg).any():
                    pending_lo = min(pending_lo,
                                     float(np.nanmin(stream_thr_neg)))
            elif cwt_thr_raw_disp > 0:
                pending_lo = min(pending_lo, -cwt_thr_raw_disp)
                pending_hi = max(pending_hi, +cwt_thr_raw_disp)
            self._pending_score_ylim = (pending_lo, pending_hi)
            self._cwt_summary_y_set = True
            self._cwt_native_scale = True
        else:
            self._cwt_line_pos = None
            self._cwt_line_neg = None
            self._cwt_summary_y_set = False
            self._pending_score_ylim = None

        self._ax_score.set_ylabel("filtered Δsignal")
        self._ax_score.set_xlabel("time (s)")
        self._ax_score.legend(loc="upper right", fontsize=8)

        # Bottom (score) y-limits: y=0 centered.
        #  - Live sim mode: rolling visible-window max |y|, tracked as
        #    a high-water mark (range only grows).
        #  - Summary mode (default / post-sim / batch): full-record
        #    max |y| across diff and (scaled) CWT.
        window_sec_yl = float(self._params.get("liveWindowSec", 3.0))
        window_samples_yl = max(2, int(round(window_sec_yl * sr)))
        live_for_yl = self._live_view and revealed < n
        score_max = 0.0
        if live_for_yl:
            s_left = max(0, revealed - window_samples_yl)
            s_right = max(s_left + 1, revealed) if revealed > 0 else 0
            if diff_enabled_disp and s_right > s_left:
                seg = score[s_left:s_right]
                if seg.size:
                    score_max = max(score_max, float(np.max(np.abs(seg))))
            if (cwt_enabled and cwt_score is not None
                    and len(cwt_score) == n and s_right > s_left):
                # Live mode: CWT is plotted at native scale, so the
                # y-limit calc must compare apples-to-apples.
                if self._cwt_native_scale:
                    seg_c = cwt_score[s_left:s_right]
                elif cwt_sigma_raw > 0 and diff_sigma > 0:
                    seg_c = cwt_score[s_left:s_right] * (diff_sigma / cwt_sigma_raw)
                else:
                    seg_c = None
                if seg_c is not None and seg_c.size:
                    score_max = max(score_max, float(np.max(np.abs(seg_c))))
            if score_max > self._bot_y_abs_max:
                self._bot_y_abs_max = score_max
            score_max = self._bot_y_abs_max
        else:
            full_score = (self._batch_result.get("score")
                          if self._batch_result else None)
            if full_score is None:
                full_score = score
            full_cwt = (self._batch_result.get("cwtScore")
                        if self._batch_result else None)
            if full_cwt is None:
                full_cwt = cwt_score
            if diff_enabled_disp and full_score is not None and len(full_score):
                score_max = max(score_max, float(np.max(np.abs(full_score))))
            # Fit to whatever was actually plotted on the score axis —
            # iterate the matplotlib lines and take the max |y|. This
            # robustly handles both the rescaled CWT and any future
            # additions without the cached-attribute bookkeeping.
            for line in self._ax_score.get_lines():
                ydata = line.get_ydata()
                if ydata is None or len(ydata) == 0:
                    continue
                y_arr = np.asarray(ydata, dtype=float)
                if y_arr.size:
                    finite = y_arr[np.isfinite(y_arr)]
                    if finite.size:
                        score_max = max(
                            score_max, float(np.max(np.abs(finite))))
        if score_max > 0 and not getattr(self, "_cwt_summary_y_set", False):
            lim = score_max * 1.15
            self._ax_score.set_ylim(-lim, lim)
        # Reset the flag so the next redraw computes fresh.
        self._cwt_summary_y_set = False

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
        elif prev_xlim is not None and not np.allclose(prev_xlim, full_xlim,
                                                       atol=1e-6):
            self._ax_sig.set_xlim(prev_xlim)
        else:
            self._ax_sig.set_xlim(full_xlim)

        # Top-plot y-limits.
        #  - Live sim mode: rolling visible-window fit with y_diff
        #    high-water mark (span only grows), centered on the
        #    current window.
        #  - Summary mode: full-record y_max − y_min with 15% padding,
        #    so the user sees the entire trace framed in one view.
        if live_active:
            left_idx = max(0, revealed - window_samples_yl)
            right_idx = max(left_idx + 1, revealed) if revealed > 0 else 0
            visible = sig_to_plot[left_idx:min(right_idx, len(sig_to_plot))]
            if visible is not None and len(visible):
                y_lo = float(np.min(visible))
                y_hi = float(np.max(visible))
                cur_diff = y_hi - y_lo
                if cur_diff <= 0:
                    cur_diff = max(abs(y_hi), 1.0) * 0.01
                if cur_diff > self._top_y_diff_max:
                    self._top_y_diff_max = cur_diff
                span = self._top_y_diff_max
                center = 0.5 * (y_lo + y_hi)
                pad = 0.15 * span
                half = 0.5 * span + pad
                self._ax_sig.set_ylim(center - half, center + half)
        else:
            full_sig = (self._result.get("signalFiltered")
                        if filters_on else self._signal)
            if full_sig is None or len(full_sig) != n:
                full_sig = self._signal
            if full_sig is not None and len(full_sig):
                y_lo = float(np.min(full_sig))
                y_hi = float(np.max(full_sig))
                rng = y_hi - y_lo
                if rng <= 0:
                    rng = max(abs(y_hi), 1.0) * 0.01
                pad = 0.15 * rng
                self._ax_sig.set_ylim(y_lo - pad, y_hi + pad)
            elif prev_ylim is not None:
                self._ax_sig.set_ylim(prev_ylim)

        n_reg = len(regions)
        batch = self._live_batch_samples()
        batch_ms = batch * 1000.0 / max(sr, 1.0)
        stats = [f"Detected: {n_reg}",
                 f"Batch size: {batch:,} samples ({batch_ms:.1f} ms)"]
        if self._proc_times_ms:
            arr = np.asarray(self._proc_times_ms, dtype=float)
            stats.append("Process time:")
            stats.append(f"  N    = {arr.size}")
            stats.append(f"  Min  = {arr.min():.3f} ms")
            stats.append(f"  Mean = {arr.mean():.3f} ms")
            stats.append(f"  Max  = {arr.max():.3f} ms")
        self._stats_label.setText("\n".join(stats))

        # Final word on the score-axis y-limits in summary CWT mode —
        # applied here, after every other drawing/limit operation, so
        # nothing can overwrite it before draw_idle().
        pending = getattr(self, "_pending_score_ylim", None)
        if pending is not None:
            lo, hi = pending
            if hi <= lo:
                hi = max(abs(hi), 1.0)
                lo = -hi
            span = max(hi - lo, 1e-9)
            pad = 0.15 * span
            self._ax_score.set_ylim(lo - pad, hi + pad, auto=False)
            self._ax_score.set_autoscaley_on(False)
            self._pending_score_ylim = None

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
        # Switch ALL visible threshold lines to animated artists and
        # cache the rest of the score axis as a static background. Blit
        # restores that background each motion event and redraws the
        # animated lines, avoiding a full re-rasterization of the
        # (potentially million-sample) score trace.
        try:
            for line in (self._thr_line_pos, self._thr_line_neg,
                         self._cwt_line_pos, self._cwt_line_neg):
                if line is not None:
                    line.set_animated(True)
            self._canvas.draw()
            self._drag_bg = self._canvas.copy_from_bbox(
                self._ax_score.bbox)
        except Exception:
            self._drag_bg = None

    def _threshold_at(self, event):
        """Return 'pos' / 'neg' / 'cwt_pos' / 'cwt_neg' / None depending
        on which (if any) draggable threshold line the event is within
        the hit-tolerance of. Picks the closest candidate.
        """
        if event.inaxes is not self._ax_score or self._result is None:
            return None
        if event.y is None:
            return None
        candidates = []
        ax = self._ax_score
        if bool(self._result.get("diffEnabled", True)):
            t_pos = self._result.get("thresholdPos", 0.0)
            t_neg = self._result.get("thresholdNeg", -t_pos)
            try:
                y_pos = ax.transData.transform((0, t_pos))[1]
                y_neg = ax.transData.transform((0, t_neg))[1]
                candidates.append(("pos", abs(event.y - y_pos)))
                candidates.append(("neg", abs(event.y - y_neg)))
            except Exception:
                pass
        cwt_enabled = bool(self._result.get("cwtEnabled", False))
        cwt_sigma_raw = float(self._result.get("cwtSigma", 0.0) or 0.0)
        diff_sigma = float(self._result.get("sigma", 0.0) or 0.0)
        # In live sim with native-scale CWT, the σ-rescale math used
        # below doesn't apply, so CWT lines aren't draggable there —
        # the user adjusts via the spinbox.
        if (cwt_enabled and cwt_sigma_raw > 0 and diff_sigma > 0
                and not getattr(self, "_cwt_native_scale", False)):
            cwt_thr_raw = float(self._result.get("cwtThreshold", 0.0) or 0.0)
            cwt_disp = cwt_thr_raw * (diff_sigma / cwt_sigma_raw)
            try:
                y_cp = ax.transData.transform((0, cwt_disp))[1]
                y_cn = ax.transData.transform((0, -cwt_disp))[1]
                candidates.append(("cwt_pos", abs(event.y - y_cp)))
                candidates.append(("cwt_neg", abs(event.y - y_cn)))
            except Exception:
                pass
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[1])
        name, dist = candidates[0]
        return name if dist <= 6.0 else None

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
        # Drag y-coordinate is in the score axis's display units. Diff
        # threshold lines sit at ±k·σ_diff. CWT lines are rescaled to
        # match σ_diff visually, so the same y → k conversion applies
        # — that's the whole point of the σ-rescaling.
        new_k = abs(float(event.ydata) / sigma)
        new_k = max(0.1, min(50.0, new_k))
        if self._thr_drag == "pos":
            self._thr_pos_spin.blockSignals(True)
            self._thr_pos_spin.setValue(new_k)
            self._thr_pos_spin.blockSignals(False)
            if self._thr_line_pos is not None:
                self._thr_line_pos.set_ydata([new_k * sigma, new_k * sigma])
        elif self._thr_drag == "neg":
            self._thr_neg_spin.blockSignals(True)
            self._thr_neg_spin.setValue(new_k)
            self._thr_neg_spin.blockSignals(False)
            if self._thr_line_neg is not None:
                self._thr_line_neg.set_ydata([-new_k * sigma, -new_k * sigma])
        elif self._thr_drag in ("cwt_pos", "cwt_neg"):
            # |cwt| > k·σ_cwt — symmetric, so dragging either CWT line
            # changes the single shared cwtThresholdK and both lines
            # move together.
            self._cwt_thr_spin.blockSignals(True)
            self._cwt_thr_spin.setValue(new_k)
            self._cwt_thr_spin.blockSignals(False)
            disp = new_k * sigma
            if self._cwt_line_pos is not None:
                self._cwt_line_pos.set_ydata([disp, disp])
            if self._cwt_line_neg is not None:
                self._cwt_line_neg.set_ydata([-disp, -disp])
        if self._drag_bg is not None:
            self._canvas.restore_region(self._drag_bg)
            for line in (self._thr_line_pos, self._thr_line_neg,
                         self._cwt_line_pos, self._cwt_line_neg):
                if line is not None:
                    self._ax_score.draw_artist(line)
            self._canvas.blit(self._ax_score.bbox)
        else:
            self._canvas.draw_idle()

    def _on_thr_release(self, event):
        if self._thr_drag is None:
            return
        self._thr_drag = None
        for line in (self._thr_line_pos, self._thr_line_neg,
                     self._cwt_line_pos, self._cwt_line_neg):
            if line is not None:
                line.set_animated(False)
        self._drag_bg = None
        # Recompute detection with the new threshold (cheap; same σ).
        self._params = self._read_params_from_widgets()
        try:
            self._result = self._detect_timed()
        except Exception:
            return
        self._sim_revealed = len(self._signal)
        self._live_view = False
        self._redraw()

    # --- Button handlers -------------------------------------------------

    def _on_cwt_wavelet_changed(self, idx):
        # Width spin only matters for the synthetic wavelets; for the
        # template option we use the wavelet at its native length. The
        # envelope option is the inverse — it only applies to the
        # template wavelet.
        if 0 <= idx < len(self._cwt_wavelet_options):
            key = self._cwt_wavelet_options[idx][1]
            self._cwt_width_spin.setEnabled(key != "template")
            if hasattr(self, "_cwt_envelope_chk"):
                self._cwt_envelope_chk.setEnabled(key == "template")
        self._redraw_wavelet_preview()

    def _redraw_wavelet_preview(self):
        ax = self._wave_ax
        ax.clear()
        ax.set_facecolor("#fafafa")
        ax.tick_params(axis="both", labelsize=7)
        idx = self._cwt_wavelet_combo.currentIndex()
        if not (0 <= idx < len(self._cwt_wavelet_options)):
            self._wave_canvas.draw_idle()
            return
        wt = self._cwt_wavelet_options[idx][1]
        width_ms = float(self._cwt_width_spin.value())
        width_samples = max(1, int(round(width_ms * 0.001
                                         * self._sample_rate)))
        env_on = (hasattr(self, "_cwt_envelope_chk")
                  and self._cwt_envelope_chk.isChecked())
        kernel = _build_cwt_kernel(wt, width_samples, self._wavelet,
                                   template_envelope=env_on)
        if kernel is None or len(kernel) == 0:
            ax.text(0.5, 0.5, "(no wavelet)",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#888")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            t = np.arange(len(kernel)) - (len(kernel) - 1) / 2.0
            ax.plot(t, kernel, color="#9b59b6", linewidth=1.2)
            ax.axhline(0, color="#bbb", linewidth=0.5)
            ax.set_xlabel("samples", fontsize=7)
            ax.margins(x=0)
        ax.grid(True, alpha=0.3)
        self._wave_canvas.draw_idle()

    def _snapshot_view(self):
        """Capture each axis's x and y limits IF that axis already has
        plotted data, so the next redraw can restore the user's zoom/pan
        without overriding the limits of an axis that was empty at
        snapshot time (e.g. the score axis behind the idle placeholder
        starts at matplotlib's default 0..1)."""
        sig_has = self._ax_sig.has_data()
        score_has = self._ax_score.has_data()
        if not sig_has and not score_has:
            return None
        return {
            "sig":   (self._ax_sig.get_xlim(),
                      self._ax_sig.get_ylim()) if sig_has else None,
            "score": (self._ax_score.get_xlim(),
                      self._ax_score.get_ylim()) if score_has else None,
        }

    def _restore_view(self, snap):
        if not snap:
            return
        sig = snap.get("sig")
        if sig is not None:
            xs, ys = sig
            self._ax_sig.set_xlim(xs)
            self._ax_sig.set_ylim(ys)
        score = snap.get("score")
        if score is not None:
            xc, yc = score
            self._ax_score.set_xlim(xc)
            self._ax_score.set_ylim(yc)
        self._canvas.draw_idle()

    def _draw_idle_placeholder(self):
        """Show the raw signal trace + an empty score axis with a hint
        that detection hasn't been computed yet. Called at dialog open
        and any time we want to clear results without running the
        streaming pipeline."""
        n = len(self._signal)
        self._ax_sig.clear()
        self._ax_score.clear()
        if n > 0:
            sr = max(self._sample_rate, 1.0)
            t = np.arange(n) / sr
            self._ax_sig.plot(t, self._signal, color="#3b6fb6", lw=0.8)
            y_lo = float(np.min(self._signal))
            y_hi = float(np.max(self._signal))
            rng = y_hi - y_lo if y_hi > y_lo else max(abs(y_hi), 1.0) * 0.01
            pad = 0.15 * rng
            self._ax_sig.set_ylim(y_lo - pad, y_hi + pad)
            self._ax_sig.set_xlim(0.0, max(t[-1], 1.0 / sr))
        self._ax_sig.set_ylabel("signal")
        self._ax_sig.set_title("Detection preview")
        self._ax_score.set_xlabel("time (s)")
        self._ax_score.set_ylabel("filtered Δsignal")
        self._ax_score.text(
            0.5, 0.5,
            "Click 'Update Scores' to run the streaming detection.",
            ha="center", va="center", fontsize=10, color="#888",
            transform=self._ax_score.transAxes,
        )
        self._stats_label.setText("—")
        self._canvas.draw_idle()

    def _on_apply(self):
        n = len(self._signal)
        if n < 2:
            return
        snap = self._snapshot_view()
        self._params = self._read_params_from_widgets()
        self._stop_sim()
        progress = self._make_progress_dialog("Update Scores")
        self._progress_dialog = progress
        self._progress_t0 = perf_counter()
        try:
            self._run_batch_and_redraw()
        finally:
            self._progress_dialog = None
            progress.close()
        self._restore_view(snap)

    def _make_progress_dialog(self, title):
        """Build the modal QProgressDialog used by Update Scores / OK.
        The label is updated in-place from the streaming loop to show
        both progress percent and elapsed wall time."""
        progress = QProgressDialog(
            "Running streaming detection…\nElapsed: 0.0 s",
            "Cancel", 0, 100, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        # Two-line label needs a bit more vertical room than the default.
        progress.setMinimumWidth(320)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        return progress

    def _on_recluster(self):
        """Re-cluster the cached pre-clustering peaks using the current
        Merging and Padding parameters. Scores and signal traces are
        unchanged — only the detected/core regions in the top plot
        update."""
        if self._result is None:
            return
        snap = self._snapshot_view()
        self._params = self._read_params_from_widgets()
        ms_peaks = np.asarray(
            self._result.get("msPeaks", np.array([], dtype=int)), dtype=int)
        cwt_peaks = np.asarray(
            self._result.get("cwtPeaks", np.array([], dtype=int)), dtype=int)
        # Honour the current diff/CWT enables, in case the user toggled
        # them after the last full Update Scores.
        if not bool(self._params.get("diffEnabled", True)):
            ms_peaks = np.array([], dtype=int)
        if not bool(self._params.get("cwtEnabled", False)):
            cwt_peaks = np.array([], dtype=int)
        all_peaks = np.sort(np.concatenate([ms_peaks, cwt_peaks])).astype(int)
        n = self._signal.size
        # Recompute the kernel half-width from current params so CWT-
        # containing clusters keep their kernel-scale padding floor.
        cwt_half_width = 0
        if bool(self._params.get("cwtEnabled", False)):
            sr = self._sample_rate
            width_ms = float(self._params.get("cwtWidthMs", 20.0))
            cwt_w = max(1, int(round(width_ms * 0.001 * sr)))
            wavelet_type = self._params.get("cwtWaveletType", "ricker")
            kernel = _build_cwt_kernel(
                wavelet_type, cwt_w, self._wavelet,
                template_envelope=bool(
                    self._params.get("cwtTemplateEnvelope", False)),
            )
            if kernel is None:
                kernel = _ricker_kernel(cwt_w)
            cwt_half_width = max(1, len(kernel) // 2)
        regions, cores = cluster_peaks(
            all_peaks, n, self._sample_rate, self._params,
            cwt_peaks=cwt_peaks,
            cwt_half_width=cwt_half_width,
        )
        self._result["detectedPulses"] = regions
        self._result["coreRegions"] = cores
        self._result["coincidentHint"] = np.zeros(len(regions), dtype=np.int8)
        self._batch_result = self._result
        self._redraw()
        self._restore_view(snap)

    def _on_run(self):
        self._params = self._read_params_from_widgets()
        if self._params["runMode"] == "batch":
            # Batch mode shows the whole record statically — pre-compute
            # the streaming result up front, with a progress dialog so
            # the user can track / cancel.
            progress = self._make_progress_dialog("Process All")
            self._progress_dialog = progress
            self._progress_t0 = perf_counter()
            try:
                self._result = self._detect_timed()
            except Exception as e:
                QMessageBox.critical(self, "Detection failed", str(e))
                return
            finally:
                self._progress_dialog = None
                progress.close()
            self._batch_result = self._result
            self._sim_revealed = len(self._signal)
            self._redraw()
            return
        # Live simulation: skip the up-front full-record pass — the
        # streaming sim itself produces the result tick by tick. Going
        # straight to _start_sim makes Run feel instant.
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

    def _on_step(self):
        """Advance the live sim by one batch (~33 ms of data) and
        redraw. Works while running (auto-pauses first), while paused,
        and when no sim is active (initializes a paused sim at t=0 so
        the user can step through from the start).
        """
        # Pick up any widget edits since the sim started.
        self._params = self._read_params_from_widgets()
        # No active sim → initialize one (timer created + immediately
        # paused so Resume works) and reset to t=0.
        if self._sim_timer is None:
            self._start_sim()
            self._sim_paused = True
            self._sim_timer.stop()
            self._pause_btn.setText("▶ Resume")
        # Active but running → pause first so stepping is deterministic.
        elif not self._sim_paused:
            self._sim_paused = True
            self._sim_timer.stop()
            self._pause_btn.setText("▶ Resume")
        # Advance one batch.
        n = len(self._signal)
        batch = self._live_batch_samples()
        self._sim_revealed = min(n, self._sim_revealed + batch)
        self._sim_revealed_at_resume = self._sim_revealed
        self._sim_t0 = perf_counter()
        pct = int(round(100.0 * self._sim_revealed / max(1, n)))
        self._progress.setValue(pct)
        self._update_streaming_result()
        self._redraw()
        if self._sim_revealed >= n:
            # Reached end via stepping — flush active regions like
            # _sim_tick does so the post-sim static view is complete.
            self._stop_sim()
            self._live_view = False
            self._finalize_stream_at_end_of_data()
            self._batch_result = self._result
            self._redraw()

    def _on_step_back(self):
        """Roll the live sim back by one batch and redraw. Recomputes
        the streaming buffers from scratch up to the new revealed point
        so the view matches a fresh sim paused there.
        """
        self._params = self._read_params_from_widgets()
        # No active sim → nothing to step back from. (Step forward
        # initializes a sim at t=0; stepping back from there is a no-op.)
        if self._sim_timer is None:
            return
        # Pause first so the rollback is deterministic.
        if not self._sim_paused:
            self._sim_paused = True
            self._sim_timer.stop()
            self._pause_btn.setText("▶ Resume")
        n = len(self._signal)
        batch = self._live_batch_samples()
        new_revealed = max(0, int(self._sim_revealed) - batch)
        if new_revealed == int(self._sim_revealed):
            return
        # Reset stream state and replay forward to new_revealed in a
        # single chunk. This unfreezes everything and recomputes scores
        # / regions consistently with a fresh sim paused at that point.
        self._reset_stream_state()
        self._sim_revealed = new_revealed
        self._sim_revealed_at_resume = new_revealed
        self._sim_t0 = perf_counter()
        self._live_view = True
        # Y-range high-water marks reset on rollback so the view fits
        # the rolled-back data tightly again.
        self._top_y_diff_max = 0.0
        self._bot_y_abs_max = 0.0
        if new_revealed >= 2:
            self._update_streaming_result()
        else:
            self._build_stream_result_dict()
        pct = int(round(100.0 * new_revealed / max(1, n)))
        self._progress.setValue(pct)
        self._redraw()

    def _on_stop(self):
        self._stop_sim()
        if self._batch_result is not None:
            # Show the full-record batch result, not the last partial
            # streaming snapshot.
            self._result = self._batch_result
            self._sim_revealed = len(self._signal)
            self._live_view = False
            self._progress.setValue(100)
            self._redraw()

    def _on_accept(self):
        self._params = self._read_params_from_widgets()
        self._stop_sim()
        # If the user already ran Update Scores, accept that result
        # directly — no need to spin the streaming pipeline again. Only
        # re-run when there's nothing committed yet.
        if self._result is None and len(self._signal) >= 2:
            progress = self._make_progress_dialog("OK")
            self._progress_dialog = progress
            self._progress_t0 = perf_counter()
            try:
                self._result = self._detect_timed()
            except Exception as e:
                QMessageBox.critical(self, "Detection failed", str(e))
                return
            finally:
                self._progress_dialog = None
                progress.close()
        self.accept()

    # --- Live simulation -------------------------------------------------

    def _speed_factor(self):
        label = self._speed_combo.currentText()
        for lab, factor in _SIM_SPEEDS:
            if lab == label:
                return factor
        return 10.0

    def _reset_stream_state(self):
        """Wipe the freeze-buffer state. Called whenever a fresh sim
        starts so old per-sample scores / frozen detections aren't
        carried over from a previous run.
        """
        n = len(self._signal)
        self._stream_score = np.zeros(n, dtype=float)
        self._stream_cwt = np.zeros(n, dtype=float)
        self._stream_sig_filt = self._signal.astype(float).copy()
        self._frozen_regions = []
        self._frozen_cores = []
        self._frozen_until = 0
        # Position past which streaming-buffer values are committed and
        # will not be overwritten on subsequent ticks. Advances with
        # `revealed` minus a kernel-scale latency margin so each emitted
        # score is written exactly once.
        self._stream_commit_until = 0
        # Locked peak sets: peaks discovered in the just-committed range
        # at each tick are added here and never re-evaluated. These
        # drive monotonic region growth — once a peak's been emitted to
        # downstream (e.g. a classifier), it stays in the set even if
        # later σ updates would have rejected it.
        self._stream_ms_peaks_locked = set()
        self._stream_cwt_peaks_locked = set()
        # Active peaks: re-found each tick from the not-yet-committed
        # tail of the chunk. Can change tick-to-tick until they cross
        # commit_until and migrate into the locked sets.
        self._stream_ms_peaks_active = []
        self._stream_cwt_peaks_active = []
        # Per-sample threshold history (filled at commit time with the
        # σ that was actually used for that sample's peak detection).
        # NaN where not yet emitted. Drives the live-mode threshold
        # trace plot.
        self._stream_thr_pos = np.full(n, np.nan, dtype=float)
        self._stream_thr_neg = np.full(n, np.nan, dtype=float)
        # Aliases retained because 'Update Detected' / serialization
        # paths still reference these names — they're now just unions
        # of locked + active, populated by _build_stream_result_dict.
        self._stream_ms_peaks_set = set()
        self._stream_cwt_peaks_set = set()

    def _start_sim(self):
        self._stop_sim()
        self._sim_revealed = 0
        self._sim_revealed_at_resume = 0
        self._sim_paused = False
        self._live_view = True
        self._sim_t0 = perf_counter()
        self._top_y_diff_max = 0.0
        self._bot_y_abs_max = 0.0
        self._reset_stream_state()

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
        # Streaming detection: pre-process the just-revealed window
        # through the full pipeline (filter → diff/CWT → threshold)
        # and update the displayed result with what came out.
        self._update_streaming_result()
        self._redraw()
        if self._sim_revealed >= len(self._signal):
            self._stop_sim()
            self._live_view = False
            self._finalize_stream_at_end_of_data()
            self._batch_result = self._result
            self._redraw()

    def _update_streaming_result(self):
        """Run detect_combined on the live window, splice partial
        outputs into the cumulative buffers, and freeze regions whose
        end-padded boundary has been a full 1 s in the past — those
        pulses' scores are stable so we won't re-process them.

        Each tick processes signal[_frozen_until : revealed]. The
        scores in [_frozen_until : revealed] get spliced into
        self._stream_score (etc.) — older samples (already frozen) are
        never touched again. Detected regions whose end + 1 s lies
        before the current revealed are moved into self._frozen_regions
        and _frozen_until advances past their end, so the next tick's
        chunk gets shorter.
        """
        if self._stream_score is None:
            self._reset_stream_state()
        sr = max(self._sample_rate, 1.0)
        n = len(self._signal)
        revealed = max(0, min(n, int(self._sim_revealed)))
        process_start = max(0, int(self._frozen_until))
        if revealed - process_start < 2:
            self._build_stream_result_dict()
            return

        chunk = self._signal[process_start:revealed]
        # Per-direction sliding-window σ override for CWT peak detection.
        # Iteratively-clipped MAD over the most recent committed CWT
        # history, computed separately on positive and negative deviations
        # from the median. Each iteration estimates σ from MAD then drops
        # samples beyond k_clip·σ — those are pulse outliers, not noise —
        # and re-estimates σ on the survivors. The result is a robust
        # noise-floor σ that doesn't inflate during pulse-rich stretches:
        # the threshold stays steady where pulses cluster, and only drops
        # if the real background noise drops. MAD alone breaks down once
        # outliers exceed ~50% of one side of the distribution; the clip
        # loop pushes that limit much higher.
        cwt_sigma_pos_override = None
        cwt_sigma_neg_override = None
        if bool(self._params.get("cwtEnabled", False)):
            committed_end = int(getattr(self, "_stream_commit_until", 0))
            if committed_end > 100:
                window_samples = max(int(round(max(sr, 1.0) * 5.0)), 2000)
                hist_lo = max(0, committed_end - window_samples)
                hist = self._stream_cwt[hist_lo:committed_end]
                hist_nz = hist[hist != 0.0]
                if hist_nz.size > 100:
                    med_g = float(np.median(hist_nz))
                    pos_dev = (hist_nz - med_g)[hist_nz > med_g]
                    neg_dev = (med_g - hist_nz)[hist_nz < med_g]

                    def _clipped_half_sigma(half_dev, n_iter=5, k_clip=3.0):
                        # half_dev is a 1D array of non-negative deviations
                        # from the local median (one tail of the score
                        # distribution). For Gaussian noise the 50th
                        # percentile of |z| is 0.6745σ, so 1.4826·median
                        # is an unbiased σ estimator on the noise samples.
                        # Iteratively drop samples beyond k_clip·σ — those
                        # are pulse outliers — and recompute.
                        if half_dev.size < 50:
                            return 0.0
                        work = half_dev
                        sigma = 0.0
                        for _ in range(n_iter):
                            med = float(np.median(work))
                            sigma = 1.4826 * med
                            if sigma <= 0:
                                return sigma
                            keep_mask = work < k_clip * sigma
                            n_keep = int(keep_mask.sum())
                            # Stop when the trim no longer removes anything
                            # or would leave too few survivors for a stable
                            # estimate.
                            if n_keep < 50 or n_keep == work.size:
                                break
                            work = work[keep_mask]
                        med = float(np.median(work))
                        return 1.4826 * med if med > 0 else 0.0

                    sp = _clipped_half_sigma(pos_dev)
                    if sp > 0:
                        cwt_sigma_pos_override = sp
                    sn = _clipped_half_sigma(neg_dev)
                    if sn > 0:
                        cwt_sigma_neg_override = sn
        t0 = perf_counter()
        chunk_result = detect_combined(
            chunk, self._sample_rate, self._params,
            wavelet=self._wavelet,
            cwt_sigma_pos_override=cwt_sigma_pos_override,
            cwt_sigma_neg_override=cwt_sigma_neg_override,
        )
        dt_ms = (perf_counter() - t0) * 1000.0
        self._proc_times_ms.append(dt_ms)
        if len(self._proc_times_ms) > self._proc_times_max:
            self._proc_times_ms = self._proc_times_ms[-self._proc_times_max:]

        # Splice the chunk's per-sample outputs into the streaming
        # buffers. Two write boundaries:
        #   - LEFT:   process_start (= _frozen_until)  — frozen samples
        #             are NEVER overwritten.
        #   - RIGHT:  revealed                         — newest sample.
        # Within that range, samples that are FAR ENOUGH past revealed
        # are also locked: once the convolution kernel + baseline
        # window for sample X fits entirely inside the chunk, X's
        # score doesn't shift any further as more data arrives — so
        # we commit it here and stop rewriting it on later ticks. That
        # turns the displayed CWT trace into a strictly emit-once
        # signal, matching how a real-time output stage would behave.
        chunk_score = chunk_result["score"]
        new_len = revealed - process_start
        # Latency margin = full kernel length when CWT is on, else
        # 2 × edge-width samples (matched-step uses backward window
        # only, so its scores stabilize quickly).
        sr = float(self._sample_rate) if self._sample_rate else 10000.0
        if bool(self._params.get("cwtEnabled", False)):
            cwt_w_samples = max(1, int(round(
                float(self._params.get("cwtWidthMs", 20.0)) * 0.001 * sr)))
            commit_margin = max(1, 4 * cwt_w_samples + 1)  # full Ricker
        else:
            edge_w_samples = max(1, int(round(
                float(self._params.get("edgeWidthMs", 1.0)) * 0.001 * sr)))
            commit_margin = max(1, 2 * edge_w_samples)
        committed_until = int(getattr(self, "_stream_commit_until", 0))
        write_lo = max(process_start, committed_until)
        write_hi = revealed
        if write_hi > write_lo:
            offset = write_lo - process_start
            n_w = write_hi - write_lo
            if len(chunk_score) >= offset + n_w:
                self._stream_score[write_lo:write_hi] = \
                    chunk_score[offset:offset + n_w]
            chunk_cwt = chunk_result.get("cwtScore")
            if chunk_cwt is not None and len(chunk_cwt) >= offset + n_w:
                self._stream_cwt[write_lo:write_hi] = \
                    chunk_cwt[offset:offset + n_w]
            chunk_sf = chunk_result.get("signalFiltered")
            if chunk_sf is not None and len(chunk_sf) >= offset + n_w:
                self._stream_sig_filt[write_lo:write_hi] = \
                    chunk_sf[offset:offset + n_w]
        # Save the σ⁺/σ⁻ that the detector actually used, so the
        # threshold-trace splice (next) records the same numbers.
        used_sigma_pos = float(chunk_result.get("cwtSigmaPos", 0.0) or 0.0)
        used_sigma_neg = float(chunk_result.get("cwtSigmaNeg", 0.0) or 0.0)
        k_cwt = float(self._params.get("cwtThresholdK", 5.0))
        used_thr_pos = k_cwt * used_sigma_pos if used_sigma_pos > 0 else 0.0
        used_thr_neg = k_cwt * used_sigma_neg if used_sigma_neg > 0 else 0.0

        # Splice the per-sample threshold history for newly committed
        # samples. Older entries stay as they were when first emitted —
        # the trace is a record of "what threshold was actually used
        # for THIS sample's peak decision".
        if write_hi > write_lo:
            if used_thr_pos > 0:
                self._stream_thr_pos[write_lo:write_hi] = used_thr_pos
            if used_thr_neg > 0:
                self._stream_thr_neg[write_lo:write_hi] = -used_thr_neg

        # Warmup: hold the commit boundary at process_start until enough
        # data has been processed for the cumulative σ override to be
        # stable. Before this, chunk-internal MAD (or override from a
        # too-short history) can severely underestimate σ — peaks found
        # with that small σ would otherwise lock permanently and create
        # false detections that visibly sit below the later, higher
        # threshold trace. With the boundary held, every peak stays in
        # the "active" pool and gets re-evaluated on each tick using the
        # current (and improving) σ.
        sr_warm = float(self._sample_rate) if self._sample_rate else 10000.0
        warmup_samples = max(int(round(sr_warm * 1.0)), 5 * commit_margin)
        if revealed < warmup_samples:
            new_committed = process_start
        else:
            new_committed = max(committed_until, revealed - commit_margin)
            new_committed = max(new_committed, process_start)
        self._stream_commit_until = new_committed

        # End-of-data flush: when the last sample has been revealed we
        # have no more future context coming, so commit everything that's
        # still in flight on this final tick. The lock loop below will
        # pick up any peaks in [prev_committed, revealed) and the freeze
        # rule promotes every remaining region to frozen.
        end_of_data = revealed >= len(self._signal)
        if end_of_data:
            new_committed = revealed
            self._stream_commit_until = new_committed

        # PEAK EMIT-ONCE.
        # Each peak's commit decision is made exactly once, on the tick
        # its position crosses into [prev_committed, new_committed) — at
        # which point it has full kernel context, σ has been updated
        # from a committed history that includes that sample's vicinity,
        # and find_peaks on the chunk gives the same answer it would
        # give in a true streaming pipeline. Peaks past new_committed
        # are intentionally NOT cached — re-running find_peaks for them
        # on a later tick (with different σ) would amount to revisiting
        # a decision, which is the opposite of emit-once. The cluster
        # below sees only locked peaks, so visible regions on the canvas
        # reflect what a real-time consumer would have committed by now.
        ms_p = chunk_result.get("msPeaks")
        cwt_p = chunk_result.get("cwtPeaks")
        prev_committed = int(committed_until)
        if ms_p is not None and len(ms_p):
            for p in ms_p:
                ap = int(p) + process_start
                if prev_committed <= ap < new_committed:
                    self._stream_ms_peaks_locked.add(ap)
        if cwt_p is not None and len(cwt_p):
            for p in cwt_p:
                ap = int(p) + process_start
                if prev_committed <= ap < new_committed:
                    self._stream_cwt_peaks_locked.add(ap)
        # Legacy attributes kept (always empty) for any external reader
        # that still inspects them.
        self._stream_ms_peaks_active = []
        self._stream_cwt_peaks_active = []

        # Cluster on locked peaks only — the visible regions are a pure
        # function of emit-once decisions made on earlier ticks.
        ms_union = set(self._stream_ms_peaks_locked)
        cwt_union = set(self._stream_cwt_peaks_locked)
        all_union = ms_union | cwt_union
        if all_union:
            all_peaks_arr = np.fromiter(all_union, dtype=int,
                                        count=len(all_union))
        else:
            all_peaks_arr = np.array([], dtype=int)
        if cwt_union:
            cwt_peaks_arr = np.fromiter(cwt_union, dtype=int,
                                        count=len(cwt_union))
        else:
            cwt_peaks_arr = np.array([], dtype=int)
        cwt_kernel_half = int(chunk_result.get("cwtKernelHalf", 0) or 0)
        regions, cores = cluster_peaks(
            all_peaks_arr, n_signal=len(self._signal),
            sample_rate=self._sample_rate, params=self._params,
            cwt_peaks=cwt_peaks_arr, cwt_half_width=cwt_kernel_half,
        )

        # Freeze regions whose right edge can no longer be extended:
        # no future locked peak will land within merge_gap of `b`. At
        # end-of-data every region is frozen unconditionally.
        merge_gap_samples = max(1, int(round(
            float(self._params.get("mergeGapMs", 100.0)) * 0.001 * sr)))
        freeze_thr = (revealed + 1) if end_of_data else (
            new_committed - merge_gap_samples)
        regions_iter = (regions.tolist() if hasattr(regions, "tolist")
                        else list(regions))
        cores_iter = (cores.tolist() if hasattr(cores, "tolist")
                      else list(cores))
        frozen_regions, active_regions = [], []
        for a, b in regions_iter:
            (frozen_regions if int(b) < freeze_thr
             else active_regions).append((int(a), int(b)))
        frozen_cores, active_cores = [], []
        for a, b in cores_iter:
            (frozen_cores if int(b) < freeze_thr
             else active_cores).append((int(a), int(b)))
        self._frozen_regions = frozen_regions
        self._frozen_cores = frozen_cores
        self._stream_active_regions = active_regions
        self._stream_active_cores = active_cores
        # Refresh the legacy peak-set aliases (Update Detected uses these).
        self._stream_ms_peaks_set = ms_union
        self._stream_cwt_peaks_set = cwt_union

        # Advance _frozen_until so the next tick's chunk doesn't re-scan
        # samples whose scores can no longer change. Without this advance
        # the chunk grows linearly with `revealed`, making Process All
        # O(N²); with it, the chunk stays bounded at roughly
        # batch + commit_margin + context_pad regardless of record length.
        #
        # We must keep enough left context that the CWT detrend baseline
        # (uniform_filter1d, width = 4·kernel_length, mode="nearest") and
        # the matched-step's 2w lookback both see real signal — otherwise
        # scores at the chunk's left edge are biased and feed garbage into
        # find_peaks. cwt_kernel_half here = (len(kernel)-1)//2, so the
        # detrend baseline at the writeable edge needs ~4·kernel_half of
        # left context to be fully populated from real data.
        cwt_kernel_half_now = int(chunk_result.get("cwtKernelHalf", 0) or 0)
        edge_w_samples = max(1, int(round(
            float(self._params.get("edgeWidthMs", 1.0)) * 0.001 * sr)))
        context_pad = max(
            4 * cwt_kernel_half_now + 4 if cwt_kernel_half_now > 0 else 0,
            2 * edge_w_samples,
            64,
        )
        advance_to = max(0, new_committed - context_pad)
        if advance_to > self._frozen_until:
            self._frozen_until = int(advance_to)

        self._build_stream_result_dict(latest_meta=chunk_result)

    def _finalize_stream_at_end_of_data(self):
        """End-of-data fixup. With emit-once peak handling the per-tick
        update already commits and freezes everything on the final tick
        (revealed == n) — there are no lingering active peaks or active
        regions to promote. This method just rebuilds the result dict so
        any callers that hit it before the last _update_streaming_result
        ran (cancelled Process All, etc.) still get a consistent view.
        """
        self._stream_ms_peaks_active = []
        self._stream_cwt_peaks_active = []
        # If the caller bypassed the natural end-of-data tick, fold any
        # straggler active regions into frozen so the dialog's region
        # list matches what was actually committed.
        active_regions = list(getattr(self, "_stream_active_regions", []))
        active_cores = list(getattr(self, "_stream_active_cores", []))
        if active_regions:
            self._frozen_regions = list(self._frozen_regions) + active_regions
            self._stream_active_regions = []
        if active_cores:
            self._frozen_cores = list(self._frozen_cores) + active_cores
            self._stream_active_cores = []
        self._stream_ms_peaks_set = set(self._stream_ms_peaks_locked)
        self._stream_cwt_peaks_set = set(self._stream_cwt_peaks_locked)
        self._build_stream_result_dict()

    def _build_stream_result_dict(self, latest_meta=None):
        """Assemble self._result from the cumulative streaming buffers
        plus the latest per-tick metadata (σ̂, thresholds). Carries
        forward old metadata if `latest_meta` is None (e.g. when a
        tick had no new data to process).
        """
        if self._stream_score is None:
            return
        active_regions = getattr(self, "_stream_active_regions", [])
        active_cores = getattr(self, "_stream_active_cores", [])
        all_regions = list(self._frozen_regions) + list(active_regions)
        all_cores = list(self._frozen_cores) + list(active_cores)
        regions_arr = (np.asarray(all_regions, dtype=int)
                       if all_regions else np.zeros((0, 2), dtype=int))
        cores_arr = (np.asarray(all_cores, dtype=int)
                     if all_cores else np.zeros((0, 2), dtype=int))
        meta = dict(latest_meta) if latest_meta else {}
        meta.update({
            "score":          self._stream_score,
            "cwtScore":       self._stream_cwt,
            "signalFiltered": self._stream_sig_filt,
            "detectedPulses": regions_arr,
            "coreRegions":    cores_arr,
            "coincidentHint": np.zeros(len(regions_arr), dtype=np.int8),
            # Union of all peaks the streaming pipeline produced across
            # chunks (absolute sample indices), so 'Update Detected' can
            # re-cluster without rerunning scores.
            "msPeaks":        np.asarray(
                sorted(self._stream_ms_peaks_set), dtype=int)
                if self._stream_ms_peaks_set else np.array([], dtype=int),
            "cwtPeaks":       np.asarray(
                sorted(self._stream_cwt_peaks_set), dtype=int)
                if self._stream_cwt_peaks_set else np.array([], dtype=int),
        })
        # Carry forward σ̂ / threshold scalars from a previous tick if
        # this call didn't get fresh ones.
        for k in ("sigma", "thresholdPos", "thresholdNeg", "threshold",
                  "cwtSigma", "cwtThreshold", "cwtEnabled", "diffEnabled"):
            if k not in meta and self._result is not None:
                meta[k] = self._result.get(k)
        self._result = meta

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
                                                "RealtimeCombined"))

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
                f"RT Detection (Combined) — {os.path.basename(path)}")
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
                f"RT Detection (Combined) — {os.path.basename(path)}")
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._sync_widgets_from_params()
        self._redraw_wavelet_preview()
        self._stop_sim()
        # Don't auto-run the streaming pipeline on load — it's expensive.
        # Clear any stale result and show the placeholder so the user
        # can click Update Scores when ready.
        self._result = None
        self._batch_result = None
        self._draw_idle_placeholder()

    def result_state(self):
        return dict(self._params), self._result
