"""Runner for RecoveryAnalysis block.

Determines recovery time, category, and recovery rate for each pulse.

Inputs:
  refAmplitude:    (M,) reference amplitude per pulse
  recAmplitude:    (N_segments x M_pulses) recovery R' matrix from Recovery Info
  segmentTimes:    (N_segments x M_pulses) relative segment times (ms) from Recovery Info
  squeezeSegment:  (optional) squeeze segment dict — if provided, recovery time is
                   measured from squeeze end; otherwise from first recovery segment

Recovery category (per readJOVE):
    0         = instant (recovered by 1st recovery segment)
    1..N-1    = transient (recovered by segment i+1)
    N         = prolonged (never recovered within measured segments)

Recovery time:
    Time of the first recovered segment (ms).  If squeezeSegment is connected,
    this is relative to squeeze end.  Otherwise relative to first recovery
    segment start.  Inf if never recovered.

Recovery rate:
    Slope of a linear fit of recovery amplitude vs segment time.

Reference: Kim et al. (2018), Lai et al. (2022) JoVE protocol.
"""

import numpy as np


def run(inputs, params, block):
    dI_ref = np.asarray(inputs.get("refAmplitude"), dtype=float).ravel()
    dI_rec = np.asarray(inputs.get("recAmplitude"), dtype=float)
    seg_times = np.asarray(inputs.get("segmentTimes"), dtype=float)
    tol = float(params.get("tolerance", 0.08))

    # recAmplitude and segmentTimes come in as (N_segments x M_pulses)
    # Transpose to (M_pulses x N_segments) for per-pulse iteration
    if dI_rec.ndim == 1:
        dI_rec = dI_rec.reshape(1, -1)
    if seg_times.ndim == 1:
        seg_times = seg_times.reshape(1, -1)

    dI_rec = dI_rec.T        # (M_pulses, N_segments)
    seg_times = seg_times.T  # (M_pulses, N_segments)

    n_pulses, n_segs = dI_rec.shape

    rec_time = np.full(n_pulses, np.inf)
    rec_cat = np.full(n_pulses, n_segs, dtype=float)
    rec_rate = np.zeros(n_pulses)
    r_squared = np.zeros(n_pulses)

    for i in range(n_pulses):
        ref = dI_ref[i]
        if ref == 0:
            continue

        # Determine recovery category and time
        for j in range(n_segs):
            if (ref - dI_rec[i, j]) / ref < tol:
                rec_cat[i] = j
                rec_time[i] = seg_times[i, j]
                break

        # Linear fit: dI_rec vs time for recovery rate
        t = seg_times[i, :]
        y = dI_rec[i, :]

        valid = ~(np.isnan(t) | np.isnan(y))
        t_v = t[valid]
        y_v = y[valid]

        if len(t_v) >= 2:
            p = np.polyfit(t_v, y_v, 1)
            rec_rate[i] = p[0]

            y_pred = np.polyval(p, t_v)
            ss_res = np.sum((y_v - y_pred) ** 2)
            ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
            r_squared[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "recoveryTime": rec_time.tolist(),
        "recoveryCategory": rec_cat.tolist(),
        "recoveryRate": rec_rate.tolist(),
        "rSquared": r_squared.tolist(),
    }
