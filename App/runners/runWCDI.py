"""Runner for WCDI block.

Computes the whole-cell deformability index (wCDI).
Default formula: Vc / Vnp * diameter / H

From Kim et al. (2018) / Lai et al. (2022):
    wCDI = (Lc / (Uflow * H)) * (d / dTc)
         = V_squeeze / V_np * d / H
"""

import numpy as np

DEFAULT_FORMULA = "Vc / Vnp * diameter / H"


def _scalar(x):
    a = np.asarray(x, dtype=float).ravel()
    return float(a[0]) if a.size else float("nan")


def _per_pulse(x):
    """Normalise a per-pulse input to a 1-D vector (col/row vector → 1-D)."""
    return np.squeeze(np.asarray(x, dtype=float))


def run(inputs, params, block):
    # Vc / diameter are per-pulse vectors ([n] or [n×1]); Vnp / H are scalars.
    Vc = _per_pulse(inputs.get("Vc"))
    diameter = _per_pulse(inputs.get("diameter"))
    Vnp = _scalar(inputs.get("Vnp"))
    H = _scalar(inputs.get("H"))

    formula = params.get("formula", DEFAULT_FORMULA)

    safe_ns = {"np": np, "Vc": Vc, "Vnp": Vnp, "diameter": diameter, "H": H}
    try:
        result = eval(formula, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    result = np.asarray(result, dtype=float)
    if result.ndim >= 1 and result.size > 1:
        # One value per pulse → column vector [n×1].
        return {"wCDI": result.reshape(-1, 1)}
    return {"wCDI": float(result)}
