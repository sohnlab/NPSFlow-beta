"""Runner for ExtractSegmentWidth block.

Takes multiple segment processing outputs (each a dict with 'width' list),
extracts the width arrays, and stacks them column-wise into a matrix.
Each row is a pulse, each column is a segment (in input order).
"""

import numpy as np


def run(inputs, params, block):
    width_columns = []

    # Iterate input ports in order (skip addInput placeholder)
    for port in block.input_ports:
        if port.name == "addInput":
            continue
        val = inputs.get(port.name)
        if val is None:
            continue

        # Extract width from segment dict
        if isinstance(val, dict):
            w = val.get("width")
            if w is not None:
                width_columns.append(np.asarray(w, dtype=float).ravel())
        elif isinstance(val, (list, np.ndarray)):
            width_columns.append(np.asarray(val, dtype=float).ravel())

    if not width_columns:
        return {"widthMatrix": np.array([])}

    # Stack as columns: each column is a segment, each row is a pulse
    # All columns must have the same length (number of pulses)
    matrix = np.column_stack(width_columns)  # shape (n_pulses, n_segments)

    return {"widthMatrix": matrix.tolist()}
