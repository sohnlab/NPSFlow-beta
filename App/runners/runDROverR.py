"""Runner for DROverR block — computes ΔR / R per pulse."""

import numpy as np

DEFAULT_FORMULA = "deltaR / R"


def run(inputs, params, block):
    R = np.asarray(inputs.get("R"), dtype=float)
    deltaR = np.asarray(inputs.get("deltaR"), dtype=float)

    formula = params.get("formula", DEFAULT_FORMULA)

    R_safe = np.where(R != 0, R, np.nan)
    safe_ns = {"np": np, "R": R_safe, "deltaR": deltaR}
    try:
        result = eval(formula, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    return {"dROverR": result.tolist() if hasattr(result, 'tolist') else result}
