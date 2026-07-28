"""Runner for SegmentProcessing block.

For each connected segment input, computes per pulse:
  - width:      sample count per segment (endLocal - startLocal + 1)
  - startValue: signal value at the start of the segment
  - endValue:   signal value at the end of the segment

Only these three arrays are output (the rest of the segment struct is dropped).

Zone-aware: a single-zone Pulse Slicing wires one segment dict per input port,
which is processed directly. Multi-zone Pulse Slicing wires a per-zone struct
(``{segLabel: segdict, ...}``) per input port; each segment inside is processed
and the output mirrors the same struct so downstream stays zone-grouped.
"""

import numpy as np


def _is_segment(d):
    """True if *d* looks like a single segment struct (has segment fields)."""
    return isinstance(d, dict) and (
        "values" in d or "startLocal" in d or "endLocal" in d)


def _process_segment(segment):
    """Compute width / startValue / endValue (+ pass-through globals) for one
    segment struct. Returns the per-pulse result dict."""
    result = {}

    # Compute width: endLocal - startLocal + 1
    start_local = segment.get("startLocal")
    end_local = segment.get("endLocal")
    if start_local is not None and end_local is not None:
        start_arr = np.asarray(start_local)
        end_arr = np.asarray(end_local)
        result["width"] = (end_arr - start_arr + 1).tolist()
    elif "values" in segment:
        result["width"] = [len(v) for v in segment["values"]]

    # Pass through global indices
    start_global = segment.get("startGlobal")
    end_global = segment.get("endGlobal")
    if start_global is not None:
        result["startGlobal"] = list(start_global)
    if end_global is not None:
        result["endGlobal"] = list(end_global)

    # Compute startValue and endValue from the segment samples
    values = segment.get("values")
    if values is not None:
        start_vals = []
        end_vals = []
        for v in values:
            arr = np.asarray(v, dtype=float).ravel()
            if len(arr) == 0:
                start_vals.append(0.0)
                end_vals.append(0.0)
            else:
                start_vals.append(float(arr[0]))
                end_vals.append(float(arr[-2]) if len(arr) > 1 else float(arr[0]))
        result["startValue"] = start_vals
        result["endValue"] = end_vals

    return result


def run(inputs, params, block):
    """Process each connected input. Returns a flat {output_port_name: result}.

    * A single segment (single-zone Pulse Slicing) mirrors ``inK -> outK``.
    * A per-zone struct (``{segLabel: segdict, ...}`` from multi-zone Pulse
      Slicing) is UNPACKED: each segment becomes its own output port named by
      the segment key. With more than one zone input the keys are prefixed by
      the input port to keep them unique.
    """
    plain = {}            # outK  -> result   (single segments)
    zone_inputs = []      # [(port_name, {segKey: segment}), ...]
    for port_name, val in inputs.items():
        if port_name == "addInput" or val is None:
            continue
        if _is_segment(val):
            plain[port_name.replace("in", "out", 1)] = _process_segment(val)
        elif isinstance(val, dict):
            segs = {k: v for k, v in val.items() if _is_segment(v)}
            if segs:
                zone_inputs.append((port_name, segs))

    output = dict(plain)
    used = set(output.keys())
    multi_zone_inputs = len(zone_inputs) > 1
    for port_name, segs in zone_inputs:
        for seg_key, seg in segs.items():
            # Drop the leading "_" that _make_valid_name prepends to a label
            # starting with a digit ("1" -> "_1") so the output reads "1".
            base = (seg_key[1:] if seg_key.startswith("_") and len(seg_key) > 1
                    else seg_key)
            name = f"{port_name}_{base}" if multi_zone_inputs else base
            if name in used:   # de-prefix collided — keep the raw key
                name = f"{port_name}_{seg_key}" if multi_zone_inputs else seg_key
            used.add(name)
            output[name] = _process_segment(seg)
    return output
