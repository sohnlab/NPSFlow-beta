"""Runner for Constant block - returns a numeric constant value."""

import numpy as np


def run(inputs, params, block):
    value_str = str(params.get("value", "0"))

    try:
        value = eval(value_str, {"__builtins__": {}, "np": np, "numpy": np,
                                  "pi": np.pi, "inf": np.inf, "nan": np.nan})
    except Exception:
        try:
            value = float(value_str)
        except ValueError:
            value = 0.0

    # Display the value on the block
    block.display_name = str(value_str)

    return {"value": value}
