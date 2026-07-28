"""Runner for Gain block — multiplies each input vector by a gain factor."""

import numpy as np


def run(inputs, params, block):
    gain = float(params.get("gain", 1.0))

    # Display text
    if gain == int(gain):
        block.parameters["displayText"] = f"\u00d7{int(gain)}"
    else:
        block.parameters["displayText"] = f"\u00d7{gain:g}"

    result = {}

    # Identify data port names from the block
    data_port_names = {p.name for p in block.input_ports if p.name != "addInput"}

    for key, val in inputs.items():
        if key not in data_port_names:
            continue
        if val is None:
            continue
        out_name = key.replace("in", "out", 1)
        try:
            arr = np.asarray(val, dtype=float)
        except (ValueError, TypeError):
            result[out_name] = val
            continue
        result[out_name] = arr * gain

    return result
