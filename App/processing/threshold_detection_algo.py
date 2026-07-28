"""Pure-numpy causal threshold detection + metrics for the Threshold Detection block.

Kept free of Qt / matplotlib so it can be unit-tested and (later) ported to an
FPGA-side reference implementation.
"""

import numpy as np


def _apply_iir(diff, alpha):
    """One-pole causal IIR: y[t] = a*x[t] + (1-a)*y[t-1]."""
    if alpha is None or alpha >= 1.0 or alpha <= 0.0:
        return diff
    out = np.empty_like(diff, dtype=np.float64)
    y = float(diff[0])
    a = float(alpha)
    for i in range(len(diff)):
        y = a * diff[i] + (1.0 - a) * y
        out[i] = y
    return out


def detect_causal(data, t_pos, t_neg, filter_alpha=None):
    """Causal pulse detection on running diff.

    Rising edge: diff > t_pos -> pulse begins.
    Falling edge: diff < t_neg -> pulse ends.
    Returns int32 per-sample array shape-matched to `data`, values in {0, 1}.
    """
    n = len(data)
    detected = np.zeros(n, dtype=np.int32)
    if n < 2:
        return detected

    diff = np.diff(np.asarray(data, dtype=np.float64))
    diff = _apply_iir(diff, filter_alpha)

    in_pulse = False
    start = 0
    for t in range(len(diff)):
        d = diff[t]
        if (not in_pulse) and d > t_pos:
            in_pulse = True
            start = t + 1  # pulse starts at the sample after the rising edge
        elif in_pulse and d < t_neg:
            detected[start:t + 2] = 1  # inclusive of the falling edge
            in_pulse = False
    if in_pulse:
        detected[start:] = 1
    return detected


def find_regions(binary):
    """Contiguous True regions as list of (start, end_exclusive)."""
    b = np.asarray(binary, dtype=bool)
    if b.size == 0:
        return []
    edges = np.diff(b.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def compute_metrics(detected, labels, iou_threshold=0.5):
    """Sample-level and event-level metrics.

    `detected` is binary (int or bool). `labels` is int; any nonzero label is
    treated as a true pulse for evaluation.
    """
    det = np.asarray(detected) > 0
    gt = np.asarray(labels) > 0

    tp = int(np.sum(det & gt))
    fp = int(np.sum(det & ~gt))
    fn = int(np.sum(~det & gt))
    tn = int(np.sum(~det & ~gt))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    det_events = find_regions(det)
    gt_events = find_regions(gt)

    matched_gt = set()
    matched_det = set()
    for i, g in enumerate(gt_events):
        best_j = -1
        best_iou = 0.0
        for j, d in enumerate(det_events):
            if j in matched_det:
                continue
            iou = _iou(g, d)
            if iou >= iou_threshold and iou > best_iou:
                best_j = j
                best_iou = iou
        if best_j >= 0:
            matched_gt.add(i)
            matched_det.add(best_j)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "det_events": len(det_events),
        "gt_events": len(gt_events),
        "matched": len(matched_gt),
        "missed": len(gt_events) - len(matched_gt),
        "false_alarms": len(det_events) - len(matched_det),
    }
