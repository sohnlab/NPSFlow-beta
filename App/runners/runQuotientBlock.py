"""Runner for QuotientBlock — element-wise division with scalar broadcasting.

Accepts scalars and 1D vectors (column/row vectors are squeezed to 1D);
genuine 2D arrays are rejected. A scalar operand broadcasts across a vector
(s/v, v/s); two vectors divide element-wise and must match in length.
Division by zero yields inf.
"""

import numpy as np


def _as_operand(value, label):
    if value is None:
        raise ValueError(f"Quotient '{label}' input is not connected.")
    arr = np.squeeze(np.asarray(value, dtype=float))  # (N,1)/(1,N) -> (N,)
    if arr.ndim >= 2:
        raise ValueError(
            f"Quotient '{label}' is a 2D array (shape {np.shape(value)}); "
            "only scalars and 1D vectors are supported.")
    return arr


def run(inputs, params, block):
    a = _as_operand(inputs.get("input1"), "dividend")
    b = _as_operand(inputs.get("input2"), "divisor")

    # Two vectors must line up; a scalar broadcasts across a vector.
    if a.ndim == 1 and b.ndim == 1 and a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Quotient length mismatch: dividend has {a.shape[0]} elements, "
            f"divisor has {b.shape[0]}.")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    # Per spec: divisor == 0 -> inf (also overrides numpy's nan for 0/0).
    result = np.where(b == 0, np.inf, result)

    if np.ndim(result) == 0:
        return {"quotient": float(result)}
    return {"quotient": np.asarray(result).tolist()}
