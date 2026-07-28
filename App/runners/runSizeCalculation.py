"""Runner for SizeCalculation block — computes particle diameter from dR/R, De, Ls."""

import numpy as np

DEFAULT_FORMULA = "np.cbrt((dR_R * D**3 * L) / (D + 0.8 * dR_R * L))"


def run(inputs, params, block):
    D = np.asarray(inputs.get("De"), dtype=float).ravel()
    L = np.asarray(inputs.get("L"), dtype=float).ravel()
    dR_R = np.asarray(inputs.get("dROverR"), dtype=float)

    # D and L come from device geometry — De is typically a scalar,
    # Ls is an array of sizing pore lengths (e.g. [300, 300, 300]).
    # Use the first element so they broadcast against the dR_R array.
    D = D.item(0) if D.size >= 1 else D
    L = L.item(0) if L.size >= 1 else L

    formula = params.get("formula", DEFAULT_FORMULA)

    safe_ns = {"np": np, "D": D, "L": L, "dR_R": dR_R}
    try:
        diameter = eval(formula, {"__builtins__": __builtins__}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    return {"diameter": diameter if np.isscalar(diameter) else diameter.tolist()}
