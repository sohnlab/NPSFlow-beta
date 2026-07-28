"""Decompose coincident (overlapping) pulse regions into individual
single-particle spans, using start/end indicator-zone timing when available
and a template-tiling fallback otherwise. Pure NumPy/SciPy; no Qt.

Reuses the single-block front-end helpers (padding + filter stack) so edge-peak
detection matches Pulse Processing exactly.
"""
import numpy as np
from scipy.signal import find_peaks

# Fallback (no-indicator) significance floor: when the template's fixed
# thresholds gate out every edge-peak in a region, re-gate at this fraction of
# the region's peak |derivative| so the region can still be split rather than
# collapsing to a single particle.
_ADAPTIVE_THRESH_FRAC = 0.3


def detect_indicator_spikes(indicator, lo, hi, prominence=None):
    """Return sorted sample indices of spikes in ``indicator[lo:hi]``.

    ``prominence`` defaults to 4x the robust noise level (MAD) of the window,
    so it adapts per-signal. Indices are absolute (offset by ``lo``).
    """
    lo = max(0, int(lo)); hi = min(len(indicator), int(hi))
    if hi - lo < 3:
        return np.array([], dtype=int)
    seg = np.asarray(indicator[lo:hi], dtype=float)
    # spikes may be positive or negative; work on |detrended|
    detr = seg - np.median(seg)
    mag = np.abs(detr)
    if prominence is None:
        # Folded |noise| peaks reach ~3-4 sigma prominence (~6-8x its MAD),
        # so a 4x floor floods long windows with false spikes — one particle
        # each. 10x (~4.8 sigma) keeps pure-noise windows empty while real
        # indicator spikes stand far above it.
        mad = np.median(np.abs(mag - np.median(mag)))
        prominence = max(10.0 * mad, np.ptp(mag) * 0.2, 1e-12)
    locs, _ = find_peaks(mag, prominence=prominence)
    return np.sort(locs.astype(int)) + lo


def fifo_windows(starts, ends, s, e):
    """Pair ordered starts with ordered ends (no overtaking) into windows.

    N = max(#starts, #ends, 1). Missing starts pad from region start ``s``;
    missing ends pad from region end ``e``. Returns list of (start, end) with
    each end >= its start.
    """
    starts = sorted(int(x) for x in starts)
    ends = sorted(int(x) for x in ends)
    n = max(len(starts), len(ends), 1)
    out = []
    for k in range(n):
        sk = starts[k] if k < len(starts) else int(s)
        ek = ends[k] if k < len(ends) else int(e)
        if ek < sk:
            ek = int(e)
        out.append((sk, ek))
    return out


def measurement_edge_peaks(segment, fs, template_params):
    """Derivative edge-peaks of a measurement segment, matching Pulse
    Processing's front-end (pad -> filter -> diff -> filter -> find_peaks).

    Returns (pos_locs, neg_locs, dseg_filtered) with locs sorted ascending.
    """
    from processing.extract_single_pulse_features import (
        _pad_signal, _apply_filter_stack)
    seg = np.asarray(segment, dtype=float)
    dur = len(seg)
    if dur < 3:
        return np.array([], int), np.array([], int), np.array([])
    pad_cfg = template_params.get("filter_padding") or {}
    pad_len = int(pad_cfg.get("pad_length", 0))
    pad_method = pad_cfg.get("method", "replicate")
    fcfg = template_params.get("filter_config")
    actual_pad = min(pad_len, dur // 4)
    seg_pad = _pad_signal(seg, actual_pad, pad_method)
    seg_filt = _apply_filter_stack(seg_pad, float(fs), fcfg)
    dpad = np.diff(seg_filt)
    dfilt = _apply_filter_stack(dpad, float(fs), fcfg)
    if actual_pad > 0:
        dfilt = dfilt[actual_pad: actual_pad + dur - 1]
    pos, _ = find_peaks(dfilt, height=None)
    neg, _ = find_peaks(-dfilt, height=None)
    return np.sort(pos.astype(int)), np.sort(neg.astype(int)), dfilt


def split_fallback(peaks, n, region_len, seq_len):
    """Split a coincident region by peak layout when no indicators exist.

    peaks: sorted 1-D array of significant edge-peak local positions.
    seq_len: number of edge-peaks in ONE template instance.
    Returns (spans, mode) where spans is a list of (start, end) local indices
    and mode is 'sequential' or 'interleaved'. N is estimated as
    round(len(peaks)/seq_len), clamped >= 1.
    """
    peaks = np.sort(np.asarray(peaks, dtype=int))
    if len(peaks) < 2 or seq_len < 1:
        return [(0, int(region_len))], "sequential"
    N = max(1, int(round(len(peaks) / float(seq_len))))
    if N == 1:
        return [(int(peaks[0]), int(peaks[-1]))], "sequential"
    gaps = np.diff(peaks)
    med = float(np.median(gaps)) if len(gaps) else 0.0
    # candidate boundaries = the N-1 largest gaps, if they stand out
    order = np.argsort(gaps)[::-1]
    big = [i for i in order[:N - 1] if gaps[i] > 1.8 * max(med, 1e-9)]
    if len(big) >= N - 1:
        cut_after = sorted(big)  # indices into `peaks` after which to cut
        spans = []
        start_i = 0
        for c in cut_after:
            spans.append((int(peaks[start_i]), int(peaks[c])))
            start_i = c + 1
        spans.append((int(peaks[start_i]), int(peaks[-1])))
        return spans, "sequential"
    # interleaved: tile the sequence N times, assign peak j to particle j//seq_len
    spans = []
    per = int(np.ceil(len(peaks) / N))
    for k in range(N):
        grp = peaks[k * per:(k + 1) * per]
        if len(grp) == 0:
            continue
        spans.append((int(grp[0]), int(grp[-1])))
    return spans, "interleaved"


def _resolve_seq_len(template_params):
    from processing.extract_single_pulse_features import _resolve_sequence
    seq = _resolve_sequence(template_params.get("peak_sequence"))
    return int(len(seq)) if seq is not None and len(seq) else 1


# ---- additive overlap decomposition (common shape, per-pulse amp/width) ----

def template_landmarks(template):
    """Principal-edge landmarks of the complete-pulse template.

    Anatomy (in time): entry dip (slight fall) -> sharp rise into the
    subpulse series -> sharp fall to idle -> exit rise (particle leaves).
    Edges are diff peaks; selection is ordering-based (first negative =
    entry, first positive = series, last negative = idle, last positive =
    exit), so internal subpulse edges cannot confuse it.

    Returns {"entry", "series", "idle", "exit"} sample indices plus
    "u_series"/"u_idle" fractions of the entry->exit unit timeline; None if
    the structure can't be resolved.
    """
    tpl = np.asarray(template, dtype=float).ravel() if template is not None \
        else np.array([])
    if tpl.size < 8 or not np.all(np.isfinite(tpl)):
        return None
    d = np.diff(tpl)
    floor = 0.2 * float(np.max(np.abs(d)))
    if floor <= 0:
        return None
    pos, _ = find_peaks(d, height=floor)
    neg, _ = find_peaks(-d, height=floor)
    if not len(pos) or not len(neg):
        return None
    i_entry, i_exit = int(neg[0]), int(pos[-1])
    pos_after = pos[pos > i_entry]
    neg_before = neg[neg < i_exit]
    span = i_exit - i_entry
    if span <= 0 or not len(pos_after) or not len(neg_before):
        return None
    i_series, i_idle = int(pos_after[0]), int(neg_before[-1])
    return {"entry": i_entry, "series": i_series, "idle": i_idle,
            "exit": i_exit, "u_series": (i_series - i_entry) / span,
            "u_idle": (i_idle - i_entry) / span}


def unit_shape_from_template(template):
    """Unit pulse shape S from the template waveform: baseline-referenced
    (median of the outer 10% at each end), cropped to the entry->exit
    landmark timeline when resolvable, normalized to max |deviation| = 1.
    Duration normalizes implicitly (index 0..1 on resample). None if
    unusable."""
    tpl = np.asarray(template, dtype=float).ravel() if template is not None \
        else np.array([])
    if tpl.size < 3 or not np.all(np.isfinite(tpl)):
        return None
    k = max(1, tpl.size // 10)
    base = float(np.median(np.concatenate([tpl[:k], tpl[-k:]])))
    dev = tpl - base
    lm = template_landmarks(tpl)
    if lm is not None and lm["exit"] - lm["entry"] >= 3:
        dev = dev[lm["entry"]:lm["exit"] + 1]
    a = float(np.max(np.abs(dev)))
    return dev / a if a > 0 else None


def indicator_edge_peaks(indicator, lo, hi, sign, fs=None, filter_config=None,
                         top=None):
    """Sorted absolute indices of the indicator's rising (sign=+1) or
    falling (sign=-1) diff edges in [lo, hi), gated at a robust 10x-MAD
    prominence floor. Filters with the template stack when configured —
    real edges ramp over many samples, so raw per-sample diffs sink below
    any noise floor. `top` keeps only the N most prominent edges."""
    lo = max(0, int(lo)); hi = min(len(indicator), int(hi))
    if hi - lo < 4:
        return np.array([], dtype=int)
    seg = np.asarray(indicator[lo:hi], dtype=float)
    if filter_config and fs:
        from processing.extract_single_pulse_features import _apply_filter_stack
        seg = _apply_filter_stack(seg, float(fs), filter_config)
    d = np.diff(seg)
    d = d if sign > 0 else -d
    mad = np.median(np.abs(d - np.median(d)))
    prom = max(10.0 * mad, np.ptp(d) * 0.2, 1e-12)
    locs, props = find_peaks(d, prominence=prom)
    if top is not None and len(locs) > int(top):
        locs = locs[np.sort(np.argsort(props["prominences"])[::-1][:int(top)])]
    return np.sort(locs.astype(int)) + lo


def estimate_two_pulse_spans_from_edges(start_rises, end_falls,
                                        meas_falls, meas_rises):
    """Full-extent 2-pulse spans directly from indicator + measurement edges.

    Anatomy per pulse: the measurement FALLING edge right before that pulse's
    start-indicator RISING edge is its beginning (entry dip); the measurement
    RISING edge right after that pulse's end-indicator FALLING edge is its end
    (exit rise). FIFO pairing (first start ↔ first end), so pulse 1 begins and
    ends first and [begin2, end1] is the overlap.

    start_rises : 2 start-indicator rising-edge times
    end_falls   : 2 end-indicator falling-edge times
    meas_falls / meas_rises : measurement falling / rising edge times (abs).
    Returns [(b1,e1),(b2,e2)] sorted by begin, or None.
    """
    sr = sorted(float(x) for x in start_rises)
    ef = sorted(float(x) for x in end_falls)
    if len(sr) != 2 or len(ef) != 2:
        return None
    mf = np.sort(np.asarray(meas_falls, dtype=float))
    mr = np.sort(np.asarray(meas_rises, dtype=float))
    spans = []
    for i in range(2):
        before = mf[mf < sr[i]]       # entry dip before the start spike onset
        after = mr[mr > ef[i]]        # exit rise after the end spike offset
        if not len(before) or not len(after):
            return None
        spans.append((float(before[-1]), float(after[0])))
    spans.sort(key=lambda s: s[0])
    if any(e <= b for b, e in spans):
        return None
    # Expect an overlap: pulse-2 begins before pulse-1 ends.
    if not (spans[0][0] < spans[1][0] < spans[0][1] < spans[1][1]):
        return None
    return spans


def estimate_two_pulse_spans(t_series, t_endind, t_entry1, t_exit2, u_series):
    """Landmark-based full-extent spans for a 2-pulse overlap (uniform
    stretch: fixed device distances = fixed fractions of the pulse timeline).

    t_series : the two start-indicator falling-edge times (series starts)
    t_endind : the two end-indicator rising-edge times
    t_entry1 : pulse 1's clean entry-dip edge (first unowned measurement
               peak), or None when the small entry dip wasn't detected
    t_exit2  : pulse 2's clean exit-rise edge (last unowned measurement peak)
    u_series : series-rise fraction of the entry->exit unit timeline

    Pulse 2 solves exactly from its series edge + observed exit (both far
    apart -> well conditioned); its end-indicator edge then calibrates the
    fixed end-indicator fraction u_ei; pulse 1 solves by least squares over
    its entry (when available), series, and end-indicator anchors.
    Returns ([(t0_1, e_1), (t0_2, e_2)], u_ei) or None.
    """
    ts1, ts2 = sorted(float(x) for x in t_series)
    te1, te2 = sorted(float(x) for x in t_endind)
    if not (0.0 < u_series < 1.0):
        return None
    w2 = (float(t_exit2) - ts2) / (1.0 - u_series)
    if w2 <= 0:
        return None
    t02 = float(t_exit2) - w2
    u_ei = (te2 - t02) / w2
    if not (u_series < u_ei < 1.2):
        return None
    if t_entry1 is None:
        w1 = (te1 - ts1) / (u_ei - u_series)
        t01 = ts1 - u_series * w1
    else:
        A = np.array([[1.0, 0.0], [1.0, u_series], [1.0, u_ei]])
        b = np.array([float(t_entry1), ts1, te1])
        (t01, w1), *_ = np.linalg.lstsq(A, b, rcond=None)
    if w1 <= 0:
        return None
    return [(float(t01), float(t01 + w1)),
            (float(t02), float(t02 + w2))], float(u_ei)


def scaled_unit(shape, n):
    """`shape` resampled to n samples (time-scaled copy, unit amplitude)."""
    if n < 1:
        return np.array([])
    if n == 1:
        return np.array([float(shape[0])])
    x = np.linspace(0.0, 1.0, len(shape))
    return np.interp(np.linspace(0.0, 1.0, n), x, shape)


def decompose_two_pulses(meas, spans, shape, refine=2):
    """Fit-subtract-refine decomposition of a 2-pulse additive overlap.

    Model: meas = B + A1*S((t-t1)/w1) + A2*S((t-t2)/w2) + noise, common unit
    shape S, shared baseline B. A1 seeds on pulse 1's pure (pre-overlap)
    prefix, A2 fits on the p1-subtracted residual, then both amplitudes
    alternate `refine` times. Amplitudes are closed-form least squares.

    meas : 1-D array (full measurement); spans : [(t1,e1),(t2,e2)] sample
    spans; shape : unit shape (baseline 0, max |v| = 1).
    Returns dict: baseline, spans (FIFO-sorted), amps [A1,A2],
    models [m1,m2] (each over its span), outputs [o1,o2] (residual-based:
    raw minus the OTHER pulse's model, over its span).
    """
    meas = np.asarray(meas, dtype=float)
    (t1, e1), (t2, e2) = sorted((tuple(int(v) for v in s) for s in spans))
    t1 = max(0, t1); t2 = max(0, t2)
    e1 = min(len(meas) - 1, e1); e2 = min(len(meas) - 1, e2)
    if e1 <= t1 or e2 <= t2 or shape is None:
        return None

    # Shared baseline from clean context around the event.
    mrg = max(8, (e2 - t1) // 4)
    ctx = np.concatenate([meas[max(0, t1 - mrg):t1], meas[e2 + 1:e2 + 1 + mrg]])
    baseline = float(np.median(ctx if len(ctx) else meas[t1:e2 + 1]))
    y = meas - baseline

    u1 = scaled_unit(shape, e1 - t1 + 1)
    u2 = scaled_unit(shape, e2 - t2 + 1)

    def lsq(seg, u):
        den = float(np.dot(u, u))
        return float(np.dot(seg, u)) / den if den > 0 else 0.0

    # Seed A1 on the pure prefix [t1, min(t2, e1+1)) — empty when the onsets
    # coincide, in which case the refine loop does the separation.
    pre = min(t2, e1 + 1)
    a1 = lsq(y[t1:pre], u1[:pre - t1]) if pre > t1 else 0.0
    p1 = np.zeros_like(y)
    p1[t1:e1 + 1] = a1 * u1
    a2 = lsq((y - p1)[t2:e2 + 1], u2)
    p2 = np.zeros_like(y)
    p2[t2:e2 + 1] = a2 * u2
    for _ in range(max(0, int(refine))):
        a1 = lsq((y - p2)[t1:e1 + 1], u1)
        p1[:] = 0.0
        p1[t1:e1 + 1] = a1 * u1
        a2 = lsq((y - p1)[t2:e2 + 1], u2)
        p2[:] = 0.0
        p2[t2:e2 + 1] = a2 * u2

    return {
        "baseline": baseline,
        "spans": [(t1, e1), (t2, e2)],
        "amps": [a1, a2],
        "models": [baseline + p1[t1:e1 + 1], baseline + p2[t2:e2 + 1]],
        "outputs": [meas[t1:e1 + 1] - p2[t1:e1 + 1],
                    meas[t2:e2 + 1] - p1[t2:e2 + 1]],
    }


def decompose_coincident_regions(
    measurement, fs, coincident_indices, template_params,
    start_indicator=None, end_indicator=None, prominence=None, margin=None,
):
    """Decompose each coincident region into recovered-particle spans.

    Returns dict:
      recovered_indices : (M,2) int  spans into `measurement`
      source_event      : (M,) int   source region index per recovered particle
      event_n           : (M,) int   N particles in that source region
      event_mode        : list[str]  per recovered particle: 'timing'|'sequential'|'interleaved'
    """
    meas = np.asarray(measurement, dtype=float)
    regions = np.atleast_2d(np.asarray(coincident_indices, dtype=int))
    if regions.size == 0:
        return {"recovered_indices": np.zeros((0, 2), int),
                "source_event": [], "event_n": [], "event_mode": []}
    seq_len = _resolve_seq_len(template_params)
    thr = template_params.get("threshold") or {}
    pos_thr = float(thr.get("upper", 0.0))
    neg_thr = float(thr.get("lower", 0.0))
    use_timing = start_indicator is not None or end_indicator is not None

    rec, src, evn, mode = [], [], [], []
    for ri in range(len(regions)):
        s, e = int(regions[ri, 0]), int(regions[ri, 1])
        mrg = int(margin) if margin is not None else (e - s) // 4

        if use_timing:
            # Landmark tier (2-pulse overlap): with exactly two start-falls,
            # two end-rises, a landmark-resolvable template, and clean
            # entry/exit measurement edges, solve the FULL pulse extents
            # (entry dip -> exit rise) under uniform stretch.
            spans = None
            if start_indicator is not None and end_indicator is not None:
                # Full extents reach well past the detected region: search a
                # region-width margin each side.
                mlm = max(mrg, e - s)
                fcfg = template_params.get("filter_config")
                off = max(0, s - mlm)
                # Edge-based full extents: pulse begin = measurement fall just
                # before the start-spike RISING edge; pulse end = measurement
                # rise just after the end-spike FALLING edge.
                sr = indicator_edge_peaks(start_indicator, s - mlm, e + mlm,
                                          +1, fs=fs, filter_config=fcfg, top=2)
                ef = indicator_edge_peaks(end_indicator, s - mlm, e + mlm,
                                          -1, fs=fs, filter_config=fcfg, top=2)
                if len(sr) == 2 and len(ef) == 2:
                    pos_m, neg_m, dflt = measurement_edge_peaks(
                        meas[off:e + mlm + 1], fs, template_params)
                    if len(dflt):
                        # Significance gate: measurement_edge_peaks returns
                        # every local maximum, noise included.
                        mad = np.median(np.abs(dflt - np.median(dflt)))
                        gate = 12.0 * mad
                        pos_m = pos_m[np.abs(dflt[pos_m]) >= gate]
                        neg_m = neg_m[np.abs(dflt[neg_m]) >= gate]
                    est = estimate_two_pulse_spans_from_edges(
                        sr, ef, neg_m + off, pos_m + off)
                    if est is not None:
                        spans = [(max(0, int(round(a))),
                                  min(len(meas) - 1, int(round(b))))
                                 for a, b in est]
                        m = "landmark"
            if spans is None:
                starts = (detect_indicator_spikes(start_indicator, s - mrg, e, prominence)
                          if start_indicator is not None else [])
                ends = (detect_indicator_spikes(end_indicator, s, e + mrg, prominence)
                        if end_indicator is not None else [])
                wins = fifo_windows(starts, ends, s, e)
                spans = [(max(s, a), min(e, b)) for (a, b) in wins]
                m = "timing"
        else:
            seg = meas[s:e + 1]
            pos, neg, dfilt = measurement_edge_peaks(seg, fs, template_params)
            sig = [int(p) for p in pos if p < len(dfilt) and dfilt[p] >= pos_thr]
            sig += [int(p) for p in neg if p < len(dfilt) and dfilt[p] <= neg_thr]
            sig = np.sort(np.unique(sig))
            # adaptive fallback: if fixed thresholds yield nothing, re-gate at a
            # fraction of the region's peak |dfilt| (see _ADAPTIVE_THRESH_FRAC).
            if len(sig) == 0 and len(dfilt) > 0:
                adapt = _ADAPTIVE_THRESH_FRAC * float(np.max(np.abs(dfilt)))
                sig2 = [int(p) for p in pos if p < len(dfilt) and dfilt[p] >= adapt]
                sig2 += [int(p) for p in neg if p < len(dfilt) and dfilt[p] <= -adapt]
                sig = np.sort(np.unique(sig2))
            local_spans, m = split_fallback(sig, n=len(seg),
                                            region_len=len(seg), seq_len=seq_len)
            spans = [(s + a, s + b) for (a, b) in local_spans]

        # clamp + drop degenerate
        spans = [(int(a), int(b)) for (a, b) in spans if b > a]
        if not spans:
            spans = [(s, e)]
        n_part = len(spans)
        for (a, b) in spans:
            rec.append([a, b]); src.append(ri); evn.append(n_part); mode.append(m)

    return {
        "recovered_indices": np.asarray(rec, dtype=int).reshape(-1, 2),
        "source_event": src, "event_n": evn, "event_mode": mode,
    }
