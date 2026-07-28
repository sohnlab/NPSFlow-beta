"""Runner for SegmentWidthToTime block.

Converts a segment width matrix (in samples) to transit time using the
global sample rate: time = width / sampleRate, scaled to the chosen units.
Shape is preserved (N_pulses x N_segments); a single-column input yields a
per-pulse time vector.
"""

import numpy as np

# Multiplier applied to the time-in-seconds result.
_UNIT_FACTORS = {"s": 1.0, "ms": 1e3, "µs": 1e6, "us": 1e6}


def run(inputs, params, block):
    width = np.asarray(inputs.get("widthMatrix"), dtype=float)
    if width.size == 0:
        return {"timeMatrix": np.array([]).tolist()}

    sample_rate = float(inputs.get("sampleRate", 200000))
    factor = _UNIT_FACTORS.get(params.get("units", "ms"), 1e3)

    time = width / sample_rate * factor
    return {"timeMatrix": time.tolist()}
