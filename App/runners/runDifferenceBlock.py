"""Runner for DifferenceBlock — computes input1 - input2 with broadcasting."""

import numpy as np

DEFAULT_FORMULA = "input1 - input2"


def run(inputs, params, block):
    input1 = np.asarray(inputs.get("input1"), dtype=float)
    input2 = np.asarray(inputs.get("input2"), dtype=float)

    formula = params.get("formula", DEFAULT_FORMULA)

    safe_ns = {"np": np, "input1": input1, "input2": input2}
    try:
        result = eval(formula, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    return {"difference": float(result) if np.ndim(result) == 0 else result.tolist()}
