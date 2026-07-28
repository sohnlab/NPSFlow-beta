"""Runner for DeformedDiameter block.

Computes the deformed (elongated) diameter of a cell in the contraction channel.
Default formula: 0.01 * (pi/4 * Wc) * d_c^2
"""

import numpy as np

DEFAULT_FORMULA = "0.01 * (np.pi / 4 * Wc) * d_c ** 2"


def run(inputs, params, block):
    d_c = np.asarray(inputs.get("d_c"), dtype=float)
    Wc = np.asarray(inputs.get("Wc"), dtype=float).ravel()
    Wc = Wc.item(0) if Wc.size >= 1 else Wc

    formula = params.get("formula", DEFAULT_FORMULA)

    safe_ns = {"np": np, "d_c": d_c, "Wc": Wc}
    try:
        result = eval(formula, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    return {"deformedDiameter": result if np.isscalar(result) else result.tolist()}
