"""Runner for Strain1D block — computes 1D strain from free cell diameter and contraction width."""

import numpy as np

DEFAULT_FORMULA = "(diameter - Wc) / diameter"


def run(inputs, params, block):
    diameter = np.asarray(inputs.get("diameter"), dtype=float)
    Wc = np.asarray(inputs.get("Wc"), dtype=float).ravel()
    Wc = Wc.item(0) if Wc.size >= 1 else Wc

    formula = params.get("formula", DEFAULT_FORMULA)

    safe_ns = {"np": np, "diameter": diameter, "Wc": Wc}
    try:
        result = eval(formula, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    return {"strain": result if np.isscalar(result) else result.tolist()}
