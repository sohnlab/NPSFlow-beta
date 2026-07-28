"""Runner for DisplayBlock - formats input value as displayText on the block."""

import numpy as np


def run(inputs, params, block):
    val = inputs.get("dataIn")
    if val is None:
        text = "(no data)"
    elif isinstance(val, str):
        text = val
    elif isinstance(val, (int, float)):
        text = f"{val:g}" if isinstance(val, float) else str(val)
    elif isinstance(val, np.ndarray):
        text = f"[{val.dtype} {list(val.shape)}]"
    elif isinstance(val, dict):
        text = f"struct ({len(val)} fields)"
    elif isinstance(val, (list, tuple)):
        text = f"list ({len(val)} items)"
    else:
        text = str(val)[:60]

    block.parameters["displayText"] = text
    # Resize block to fit text
    min_w = max(block.size[0], len(text) * 7 + 20)
    block.size = (min_w, block.size[1])
    return {}
