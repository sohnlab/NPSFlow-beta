"""Runner for Statistics block — computes summary statistics of an input vector."""

import numpy as np


def run(inputs, params, block):
    data = inputs.get("data")
    if data is None:
        raise ValueError("No data provided to Statistics.")

    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    n = len(arr)
    if n == 0:
        block.parameters["displayText"] = "n=0"
        return {"mean": np.nan, "std": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan, "count": 0}

    m = float(np.mean(arr))
    s = float(np.std(arr))
    block.parameters["displayText"] = (
        f"n={n}\n\u03bc={m:.4g}\n\u03c3={s:.4g}")

    return {
        "mean": m,
        "std": s,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": n,
    }
