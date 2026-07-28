"""Runner for WeightedMean block.

Inputs: multiple segment structs from SegmentProcessing, each containing
  - width:      array of length numPulses
  - startValue: array of length numPulses
  - endValue:   array of length numPulses

For each pulse i, across all connected segments j:
  x_ij = 0.5 * (startValue_ij + endValue_ij)   (segment midpoint)
  w_ij = width_ij / W_i                         (weight)
  W_i  = sum_j(width_ij)                        (total width for pulse i)

Output: weightedMean array of length numPulses
  weightedMean_i = sum_j(w_ij * x_ij)
"""

import numpy as np


def run(inputs, params, block):
    segments = []
    for key, val in inputs.items():
        if key == "addInput" or val is None:
            continue
        if not isinstance(val, dict):
            continue
        width = val.get("width")
        start_val = val.get("startValue")
        end_val = val.get("endValue")
        if width is None or start_val is None or end_val is None:
            continue
        segments.append({
            "width": np.asarray(width, dtype=float),
            "x": 0.5 * (np.asarray(start_val, dtype=float) + np.asarray(end_val, dtype=float)),
        })

    if not segments:
        return {"weightedMean": np.array([])}

    # Stack: shape (num_segments, num_pulses)
    widths = np.stack([s["width"] for s in segments], axis=0)
    xs = np.stack([s["x"] for s in segments], axis=0)

    # Total width per pulse: shape (num_pulses,)
    W = widths.sum(axis=0)

    # Avoid division by zero
    W_safe = np.where(W > 0, W, 1.0)

    # Weighted mean per pulse
    weighted_mean = (widths * xs).sum(axis=0) / W_safe

    # Zero out pulses with no width
    weighted_mean = np.where(W > 0, weighted_mean, 0.0)

    return {"weightedMean": weighted_mean.tolist()}
