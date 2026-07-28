"""Runner for InvertData block — negates the sign of the input data."""

import numpy as np


def run(inputs, params, block):
    data = inputs.get("data")
    if data is None:
        raise ValueError("InvertData: no input data provided.")
    inverted = -np.asarray(data)
    return {"data": inverted}
