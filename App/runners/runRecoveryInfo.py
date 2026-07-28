"""Runner for RecoveryInfo block.

Takes N dynamic segment-info dict inputs (each with startValue/endValue
and startGlobal/endGlobal arrays of length M pulses).

Outputs:
  Rprime:       N x M matrix — midpoint 0.5*(startValue + endValue) per segment per pulse
  segmentTimes: N x M matrix — time of each segment's startGlobal relative to
                the first input's startGlobal, converted to ms using sampleRate.
                Row 0 is always zeros (the reference).
"""

import numpy as np


def run(inputs, params, block):
    # Get sample rate for time conversion (injected as global var)
    sample_rate = params.get("sampleRate", inputs.get("sampleRate", None))
    if sample_rate is not None:
        sample_rate = float(sample_rate)

    # Collect segment inputs in port order (skip addInput placeholder)
    port_names = [p.name for p in block.input_ports if p.name != "addInput"]

    rprime_rows = []
    start_global_rows = []

    for name in port_names:
        val = inputs.get(name)
        if val is None:
            continue
        if not isinstance(val, dict):
            raise ValueError(
                f"Input '{name}' is not a segment dict (got {type(val).__name__}). "
                "Connect segment info outputs from Data Bus or Segment Processing.")

        # R' from midpoint of start/end values
        sv = val.get("startValue")
        ev = val.get("endValue")
        if sv is None or ev is None:
            raise ValueError(
                f"Segment dict '{name}' missing 'startValue' or 'endValue'.")
        midpoint = 0.5 * (np.asarray(sv, dtype=float) + np.asarray(ev, dtype=float))
        rprime_rows.append(midpoint.ravel())

        # Collect startGlobal for relative time calculation
        sg = val.get("startGlobal")
        if sg is not None:
            start_global_rows.append(np.asarray(sg, dtype=float).ravel())

    if not rprime_rows:
        raise ValueError("No segment inputs connected.")

    rprime = np.vstack(rprime_rows)
    result = {"Rprime": rprime.tolist()}

    # Compute relative segment times: each segment's startGlobal relative
    # to the first input's startGlobal, converted to ms
    if start_global_rows:
        ref_start = start_global_rows[0]  # first input as reference
        rel_rows = []
        for sg in start_global_rows:
            rel_samples = sg - ref_start
            if sample_rate is not None:
                rel_rows.append(rel_samples / sample_rate * 1000.0)
            else:
                rel_rows.append(rel_samples)
        seg_times = np.vstack(rel_rows)
        result["segmentTimes"] = seg_times.tolist()

    return result
