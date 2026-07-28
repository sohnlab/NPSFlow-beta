"""Runner for Reciprocal (1/x) block — takes element-wise reciprocal of each input."""

import numpy as np


def run(inputs, params, block):
    result = {}

    # Identify data port names from the block
    data_port_names = {p.name for p in block.input_ports if p.name != "addInput"}
    out_port_names = {p.name for p in block.output_ports}

    for key, val in inputs.items():
        # Skip non-data keys (global vars, addInput)
        if key not in data_port_names:
            continue
        if val is None:
            continue
        out_name = key.replace("in", "out", 1)
        if out_name not in out_port_names:
            # Output port may not exist yet — still compute and store
            pass
        try:
            arr = np.asarray(val, dtype=float)
        except (ValueError, TypeError):
            result[out_name] = val
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            result[out_name] = np.where(arr != 0, 1.0 / arr, np.nan)

    return result
