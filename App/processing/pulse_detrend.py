"""Pure iterative baseline-fit detrend for one pulse segment.

Extracted verbatim from the single-block dialog's _detrend_pulse_segment so
both the single and coincident review dialogs share one implementation.
mode: "starting" (anchor to pre-pulse reference) or "mean" (anchor to midpoint).
"""
import numpy as np


def detrend_pulse_segment(data, seg_filt, global_peaks, s, e, mode, ref_samples):
    """Returns (seg_raw_dt, seg_filt_dt, rect_median, rect_mean) or None.

    `data` is the raw segment source sliced [s:e+1]; `seg_filt` the filtered
    counterpart (or None); `global_peaks` absolute peak indices.
    """
    seg_raw = np.asarray(data[s:min(e + 1, len(data))], dtype=float).copy()
    n = len(seg_raw)
    if n < 3:
        return None
    seg_filt = (np.asarray(seg_filt, dtype=float).copy()
                if seg_filt is not None and len(seg_filt) >= n else None)
    local_peaks = (global_peaks - s if global_peaks is not None
                   and len(global_peaks) > 0 else np.array([], dtype=int))
    valid_lp = local_peaks[(local_peaks >= 0) & (local_peaks < n)]
    bounds = np.sort(np.unique(np.concatenate([[0], valid_lp, [n]])))
    idx_arr = np.arange(n, dtype=float)
    bl_mask = np.ones(n, dtype=bool)
    if len(valid_lp) >= 2:
        bl_mask[int(np.min(valid_lp)):int(np.max(valid_lp)) + 1] = False

    is_start_mode = (mode == "starting")
    if is_start_mode:
        first_pk = int(np.min(valid_lp)) if len(valid_lp) > 0 else 0
        anchor_idx = max(first_pk - 1, 0)
        ref_n = max(min(int(ref_samples), first_pk), 1)
        ref_start = first_pk - ref_n
        ref_end = first_pk
    else:
        anchor_idx = ((int(np.min(valid_lp)) + int(np.max(valid_lp))) // 2
                      if len(valid_lp) >= 2 else n // 2)
        ref_start = ref_end = 0

    def _rect_from(sig_in, agg=np.median):
        r = np.zeros(n, dtype=float)
        if len(bounds) < 2:
            r[:] = agg(sig_in)
        else:
            for _k in range(len(bounds) - 1):
                _bi, _be = int(bounds[_k]), int(bounds[_k + 1])
                if _be <= _bi:
                    _be = _bi + 1
                r[_bi:_be] = agg(sig_in[_bi:_be])
        return r

    _max_iter = 15
    _tol = max(1e-12, float(np.ptp(seg_raw)) * 5e-3)
    first_seg_mid = ((int(bounds[0]) + int(bounds[1])) / 2.0
                     if len(bounds) > 1 else 0.0)
    last_seg_mid = ((int(bounds[-2]) + int(bounds[-1])) / 2.0
                    if len(bounds) > 1 else float(n))
    seg_rect = None
    first_val = last_val = 0.0
    for _it in range(_max_iter):
        if _it == 0:
            bl_idx = idx_arr[bl_mask]
            if len(bl_idx) >= 2:
                sig = seg_filt if seg_filt is not None else seg_raw
                coeffs = np.polyfit(bl_idx, sig[:n][bl_mask], 1)
                drift = (np.polyval(coeffs, idx_arr)
                         - np.polyval(coeffs, [anchor_idx])[0])
            else:
                drift = (np.linspace(seg_raw[0], seg_raw[-1], n) - seg_raw[0])
        else:
            residual = last_val - first_val
            span = last_seg_mid - first_seg_mid
            if span <= 0:
                break
            drift = (residual / span) * (idx_arr - float(anchor_idx))
        seg_raw = seg_raw[:n] - drift
        if seg_filt is not None:
            seg_filt = seg_filt[:n] - drift
        if seg_filt is not None:
            seg_rect = _rect_from(seg_filt)
            first_val, last_val = seg_rect[0], seg_rect[-1]
            if abs(first_val - last_val) <= _tol:
                break
        else:
            seg_rect = _rect_from(seg_raw)
            break
    if seg_rect is None:
        return None
    if is_start_mode and ref_end > ref_start:
        sig_ref = seg_filt if seg_filt is not None else seg_raw
        anchor_ref = float(np.mean(sig_ref[ref_start:ref_end]))
        shift = seg_rect[0] - anchor_ref
        seg_raw = seg_raw - shift
        if seg_filt is not None:
            seg_filt = seg_filt - shift
        seg_rect = seg_rect - shift
    sig_for_mean = seg_filt if seg_filt is not None else seg_raw
    seg_rect_mean = _rect_from(sig_for_mean, agg=np.mean)
    return seg_raw, seg_filt, seg_rect, seg_rect_mean
