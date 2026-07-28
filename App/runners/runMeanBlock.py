"""Runner for MeanBlock — general-purpose mean calculation.

Single 1-D input (incl. a row or column vector):  returns the scalar mean.
Single 2-D matrix:  applies the selected mode and returns a column vector [N x 1]
  - column-wise (axis=0): mean of each column → one value per column
  - row-wise   (axis=1): mean of each row    → one value per row
Multiple inputs:  each input becomes one row of a stacked matrix; the selected
                  mode reduces it to a 1-D vector.
"""

import numpy as np


def run(inputs, params, block):
    # Collect data inputs — only consider actual input port names on this block,
    # skipping the addInput placeholder and any injected global variables.
    port_names = {p.name for p in block.input_ports} if block else set(inputs.keys())
    data_ports = sorted(
        [k for k in inputs if k in port_names and k != "addInput"],
        key=lambda k: k,
    )

    if not data_ports:
        raise ValueError("No data provided.")

    def _to_array(val):
        """Convert a value to a 1-D float array.

        Accepts plain numeric arrays or segment dicts with
        startValue/endValue (uses midpoint).
        """
        if isinstance(val, dict):
            sv = val.get("startValue")
            ev = val.get("endValue")
            if sv is not None and ev is not None:
                return 0.5 * (np.asarray(sv, dtype=float).ravel() +
                              np.asarray(ev, dtype=float).ravel())
            # Fall back to 'value' key or first numeric array in the dict
            if "value" in val:
                return np.asarray(val["value"], dtype=float).ravel()
            raise ValueError(
                f"Segment dict missing 'startValue'/'endValue' keys. "
                f"Got keys: {list(val.keys())}")
        return np.asarray(val, dtype=float).ravel()

    def _to_matrix(val):
        """Convert to a float array, preserving 2-D shape for matrix inputs.

        Segment dicts collapse to a 1-D midpoint vector (no matrix meaning).
        Ragged/nested inputs fall back to a flattened 1-D vector.
        """
        if isinstance(val, dict):
            return _to_array(val)
        try:
            a = np.asarray(val, dtype=float)
            if a.dtype != object:
                return a
        except (ValueError, TypeError):
            pass
        parts = [np.asarray(x, dtype=float).ravel() for x in np.atleast_1d(val)]
        return np.concatenate(parts) if parts else np.asarray([], dtype=float)

    mode = params.get("mode", "column")

    # --- Single input ---
    if len(data_ports) == 1:
        arr = np.squeeze(_to_matrix(inputs[data_ports[0]]))
        if arr.ndim <= 1:
            # Scalar, or a 1-D / row / column vector → single scalar mean.
            return {"mean": float(np.nanmean(arr))}
        # True 2-D matrix → per-row (axis 1) or per-column (axis 0) means,
        # returned as a column vector [N x 1].
        axis = 1 if mode == "row" else 0
        return {"mean": np.nanmean(arr, axis=axis).reshape(-1, 1)}

    # --- Multiple inputs: matrix mean ---
    # Convert each input to a 1-D array
    arrays = []
    for k in data_ports:
        arrays.append(_to_array(inputs[k]))

    # Validate: all must be the same length
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Input vectors must be the same length, got {lengths}. "
            "Ensure all connected inputs have equal size.")

    # Stack: each input becomes a row → shape (N_inputs, N_elements)
    matrix = np.vstack(arrays)

    if mode == "column":
        # Mean down columns (across inputs/segments within each pulse)
        # → one value per element (pulse)
        result = np.nanmean(matrix, axis=0)
    else:
        # Mean across columns (across pulses for each input/segment)
        # → one value per input
        result = np.nanmean(matrix, axis=1)

    return {"mean": result}
