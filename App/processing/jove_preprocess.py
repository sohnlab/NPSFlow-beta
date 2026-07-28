"""JOVE mNPS preprocessing chain — Python port of the MATLAB pipeline.

Mirrors mNPS_readJOVE.m: rectangular smoothing (fastsmooth) -> decimate ->
zero-phase low-pass with slope-continuous padding -> ASLS baseline detrend.
Each stage is independently toggleable via the params dict.

All functions are pure/Qt-free so they can be unit-tested directly. The
interactive dialog (downsample_ui.py) and the runner both drive `process`.
"""

import numpy as np
from scipy import signal as sig
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# Default stage parameters. The ASLS defaults follow the JOVE *pipeline*
# (mNPS_procJOVE.m), not ASLS.m's standalone defaults.
DEFAULT_PARAMS = {
    "smooth_enable": True, "smooth_width": 200, "smooth_type": 1,
    "ds_enable": True, "ds_factor": 20,
    "lp_enable": True, "lp_cutoff": 50.0, "lp_pad": True,
    "asls_enable": True,
    "asls_lambda": 1e9, "asls_p": 3e-3, "asls_margin": 1e-4, "asls_iter": 20,
}

# Selectable ASLS presets (fill the four ASLS fields).
ASLS_PRESETS = {
    "JOVE pipeline": {"asls_lambda": 1e9, "asls_p": 3e-3,
                      "asls_margin": 1e-4, "asls_iter": 20},
    "ASLS standalone": {"asls_lambda": 1e5, "asls_p": 0.01,
                        "asls_margin": 0.0, "asls_iter": 5},
}

# fastsmooth type index -> label (type 1 is MATLAB's rectangular boxcar).
SMOOTH_TYPES = {1: "Rectangular", 2: "Triangular", 3: "Gaussian (3-pass)",
                4: "Gaussian (4-pass)", 5: "Multi-width"}


# ─────────────────── fastsmooth (T.C. O'Haver) ───────────────────

def _sa(y, w, ends):
    """One rectangular sliding-average pass — port of fastsmooth's `sa`."""
    w = int(round(w))
    L = y.size
    if w < 2 or L == 0:
        return y.copy()
    if w >= L:
        return np.full(L, float(np.mean(y)))
    halfw = int(round(w / 2.0))
    csum = np.concatenate(([0.0], np.cumsum(y)))
    wins = csum[w:] - csum[:-w]              # wins[j] = sum(y[j:j+w]), j=0..L-w
    s = np.zeros(L)
    pos = np.arange(wins.size) + (halfw - 1)  # centre each window sum
    mask = (pos >= 0) & (pos < L)
    s[pos[mask]] = wins[mask]
    smooth = s / w
    if ends == 1:
        # Taper the ends with progressively smaller symmetric windows.
        startpoint = (w + 1) // 2  # MATLAB (w+1)/2, loop truncates fractional
        smooth[0] = (y[0] + y[1]) / 2.0
        for k in range(2, startpoint + 1):
            if k - 1 < L:
                smooth[k - 1] = np.mean(y[0:2 * k - 1])
            lo = L - 2 * k + 1
            if 0 <= L - k and lo >= 0:
                smooth[L - k] = np.mean(y[lo:L])
        smooth[L - 1] = (y[L - 1] + y[L - 2]) / 2.0
    return smooth


def fastsmooth(y, w, smooth_type=1, ends=1):
    """Smooth vector *y* with a width-*w* sliding average.

    Port of fastsmooth.m. type 1=rectangular, 2=triangular, 3/4=pseudo-
    Gaussian, 5=multi-width. ends=1 tapers the signal ends.
    """
    y = np.asarray(y, dtype=float).ravel()
    w = int(round(w))
    if w < 2 or y.size == 0:
        return y.copy()
    t = int(smooth_type)
    if t == 2:
        return _sa(_sa(y, w, ends), w, ends)
    if t == 3:
        return _sa(_sa(_sa(y, w, ends), w, ends), w, ends)
    if t == 4:
        return _sa(_sa(_sa(_sa(y, w, ends), w, ends), w, ends), w, ends)
    if t == 5:
        return _sa(_sa(_sa(_sa(y, round(1.6 * w), ends),
                             round(1.4 * w), ends),
                         round(1.2 * w), ends), w, ends)
    return _sa(y, w, ends)


# ─────────────────── padding (edge-effect reduction) ───────────────────

def pad_data(data, samplerate):
    """Slope-continuous reflect padding — port of pad_data.m. Returns
    (padded, NR) where NR is the samples added on each side."""
    data = np.asarray(data, dtype=float).ravel()
    n = data.size
    NR = int(round(samplerate))
    NR = min(NR, n - 1)
    if NR < 1:
        return data.copy(), 0
    before = 2 * data[0] - data[1:NR + 1][::-1]
    after = 2 * data[-1] - data[-NR - 1:-1][::-1]
    return np.concatenate([before, data, after]), NR


def unpad_data(data_pad, NR):
    """Undo pad_data (strip NR samples per side)."""
    if NR < 1:
        return np.asarray(data_pad, dtype=float)
    return np.asarray(data_pad, dtype=float)[NR:len(data_pad) - NR]


def _matlab_lowpass_fir(x, fs, fpass, steepness=0.85, atten=60.0):
    """Kaiser-window FIR low-pass matching MATLAB's `lowpass` defaults:
    passband edge `fpass`, `steepness`-wide transition, `atten` dB stopband,
    applied as a single zero-phase (linear-phase, delay-compensated) pass.

    A single FIR pass with a wide transition band keeps the sharp pulse edges
    that a Butterworth `filtfilt` (double application) would smooth away.
    """
    x = np.asarray(x, dtype=float).ravel()
    nyq = fs / 2.0
    fpass = min(fpass, nyq * 0.98)
    # MATLAB: transition width = (1 - steepness) * (Nyquist - fpass).
    wt = max((1.0 - steepness) * (nyq - fpass), 1e-9)
    fstop = min(fpass + wt, nyq * 0.999)
    numtaps, beta = sig.kaiserord(atten, wt / nyq)
    numtaps = int(numtaps) | 1  # force odd -> integer group delay
    if x.size <= numtaps or numtaps < 3:
        return x.copy()
    cutoff = (fpass + fstop) / 2.0  # -6 dB point sits mid-transition
    taps = sig.firwin(numtaps, cutoff / nyq, window=("kaiser", beta))
    d = (numtaps - 1) // 2
    # Single forward pass; drop the linear-phase group delay to zero-phase it.
    yf = sig.lfilter(taps, 1.0, x)
    return np.concatenate([yf[d:], np.full(d, yf[-1])])


def lp_filter(y, fs, cutoff, use_padding=True):
    """Zero-phase low-pass. DC removed before filtering and restored after
    (MATLAB `ym-ym(1) ... +ym(1)`), with optional pad_data padding. Uses a
    single-pass Kaiser-FIR to match MATLAB's `lowpass` (see above)."""
    y = np.asarray(y, dtype=float).ravel()
    if not fs or fs <= 0:
        return y.copy()
    nyq = fs / 2.0
    if cutoff <= 0 or cutoff >= nyq:
        return y.copy()
    dc = y[0]
    x = y - dc
    try:
        if use_padding:
            xp, NR = pad_data(x, fs)
            yf = unpad_data(_matlab_lowpass_fir(xp, fs, cutoff), NR)
        else:
            yf = _matlab_lowpass_fir(x, fs, cutoff)
    except Exception:
        return y.copy()
    return yf + dc


# ─────────────────── ASLS baseline (Eilers & Boelens 2005) ───────────────────

def asls(data, lam=1e5, p=0.01, max_iter=5, noise_margin=0.0, progress=None):
    """Asymmetric Least Squares baseline — port of ASLS.m.

    Fits a smooth baseline that hugs points below (baseline+noise_margin),
    down-weighting points above it by *p* each iteration. *progress*, if given,
    is called with a 0..1 fraction after each iteration (the slow part).
    """
    y = np.asarray(data, dtype=float).ravel()
    N = y.size
    if N < 3:
        return y.copy()
    # 2nd-order difference operator (N-2 x N), rows [1, -2, 1].
    D = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(N - 2, N))
    DtD = (D.transpose() @ D).tocsc()
    w = np.ones(N)
    baseline = y.copy()
    iters = max(1, int(max_iter))
    for i in range(iters):
        Z = (sp.diags(w, 0) + lam * DtD).tocsc()
        baseline = spsolve(Z, w * y)
        above = y > (baseline + noise_margin)
        w = p * above + (1.0 - p) * (~above & (y < (baseline + noise_margin)))
        if progress:
            progress((i + 1) / iters)
    return baseline


def asls_baseline(y, lam=1e9, p=3e-3, max_iter=20, noise_margin=1e-4,
                  progress=None):
    """Baseline for downward pulses: MATLAB `-1*ASLS(-1*ym)` — negate so the
    asymmetric fit rides the upper (baseline) envelope."""
    return -asls(-np.asarray(y, dtype=float).ravel(), lam, p, max_iter,
                 noise_margin, progress=progress)


# ─────────────────── pipeline ───────────────────

def _get(params, key):
    return params.get(key, DEFAULT_PARAMS[key])


def process_1d(y, fs, params, progress=None):
    """Run the enabled stages on a 1-D signal. Returns a dict with the
    intermediate + final signals (used by the preview and by `process`).

    *progress*, if given, is called as ``progress(frac, message)`` at each
    stage boundary and once per ASLS iteration (the slow part). It may raise
    to abort the computation (used for cancellation).
    """
    def _p(frac, msg):
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    _p(0.02, "Preparing…")
    y = np.asarray(y, dtype=float).ravel()
    # MATLAB mNPS_readJOVE flips the raw signal when most samples are negative,
    # so the working signal is a positive baseline with downward pulses — the
    # orientation ASLS's -ASLS(-ym) expects. Without it the baseline rides the
    # wrong envelope (bulges up at the pulses).
    inverted = bool(y.size and np.count_nonzero(y < 0) / y.size > 0.5)
    if inverted:
        y = -y

    ds_en = _get(params, "ds_enable")
    N = int(_get(params, "ds_factor") or 1) if ds_en else 1
    if N < 1:
        N = 1

    smoothed = y
    if _get(params, "smooth_enable"):
        _p(0.05, "Smoothing…")
        smoothed = fastsmooth(y, int(_get(params, "smooth_width") or 1),
                              int(_get(params, "smooth_type") or 1), 1)

    _p(0.20, "Downsampling…")
    down = smoothed[::N] if ds_en else smoothed
    fs_out = (float(fs) / N) if fs else fs

    lp = down
    if _get(params, "lp_enable"):
        _p(0.28, "Low-pass filtering…")
        lp = lp_filter(down, fs_out, float(_get(params, "lp_cutoff") or 0),
                       bool(_get(params, "lp_pad")))

    if _get(params, "asls_enable"):
        _p(0.40, "ASLS detrend…")
        trend = asls_baseline(
            lp, float(_get(params, "asls_lambda")),
            float(_get(params, "asls_p")),
            int(_get(params, "asls_iter") or 1),
            float(_get(params, "asls_margin")),
            progress=lambda f: _p(0.40 + 0.58 * f, "ASLS detrend…"))
        data_out = lp - trend
    else:
        trend = np.zeros_like(lp)
        data_out = lp

    _p(1.0, "Done")
    return {"input": y, "inverted": inverted, "smoothed": smoothed,
            "down": down, "lp": lp, "trendline": trend, "data": data_out,
            "N": N, "fs_out": fs_out}


def process(data, fs, params, progress=None):
    """Apply the pipeline per column. Returns a dict of every intermediate +
    final signal, column-stacked per zone:
      smoothed (full length), down, lp, detrended, trendline (all reduced),
      plus N (decimation factor) and fs_out (reduced rate).

    1-D input stays 1-D; 2-D (samples x zones) is processed column-wise and
    restacked so all zones survive. *progress* (optional) is called as
    ``progress(frac, message)`` spanning all columns.
    """
    arr = np.asarray(data, dtype=float)
    single = arr.ndim == 1
    cols = [arr] if single else [arr[:, c] for c in range(arr.shape[1])]
    n = len(cols)
    outs = []
    for ci, c in enumerate(cols):
        pcb = None
        if progress:
            def pcb(f, m, ci=ci):
                progress((ci + f) / n, m if n == 1 else f"Zone {ci + 1}/{n}: {m}")
        outs.append(process_1d(c, fs, params, progress=pcb))

    def stack(key):
        if single:
            return outs[0][key]
        return np.column_stack([o[key] for o in outs])

    return {
        "smoothed": stack("smoothed"),
        "down": stack("down"),
        "lp": stack("lp"),
        "detrended": stack("data"),
        "trendline": stack("trendline"),
        "N": outs[0]["N"],
        "fs_out": outs[0]["fs_out"],
    }
