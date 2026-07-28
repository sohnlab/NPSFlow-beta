"""Runner for TotalSegmentWidth block.

Takes two segment inputs and computes the total width (in samples) between
the start of segment 1 and the end of segment 2 for each pulse.

Width = endGlobal[in2] - startGlobal[in1]  (per pulse)
"""

import numpy as np


def run(inputs, params, block):
    seg1 = inputs.get("in1")
    seg2 = inputs.get("in2")

    if seg1 is None or seg2 is None:
        raise ValueError("Both Start Segment (in1) and End Segment (in2) must be connected.")

    if not isinstance(seg1, dict) or not isinstance(seg2, dict):
        raise ValueError("Inputs must be segment structs (dicts with startGlobal/endGlobal).")

    start_global = seg1.get("startGlobal")
    end_global = seg2.get("endGlobal")

    if start_global is None:
        raise ValueError("Start Segment (in1) is missing 'startGlobal' field.")
    if end_global is None:
        raise ValueError("End Segment (in2) is missing 'endGlobal' field.")

    start_arr = np.asarray(start_global, dtype=float).ravel()
    end_arr = np.asarray(end_global, dtype=float).ravel()

    if len(start_arr) != len(end_arr):
        raise ValueError(
            f"Segment arrays have different lengths: "
            f"in1 startGlobal has {len(start_arr)}, "
            f"in2 endGlobal has {len(end_arr)}.")

    width = end_arr - start_arr

    return {"width": width}
