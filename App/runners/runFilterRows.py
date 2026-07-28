"""Runner for FilterRows block — filters rows across all inputs by a criterion
applied to one column.

Each input is expected to be an array-like (list or numpy array) of the same
length.  The block builds a boolean mask from the selected column and operator,
then keeps only the rows where the mask is True.
"""

import numpy as np


_OPS = {
    ">":  np.greater,
    "<":  np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=":  np.equal,
    "!=": np.not_equal,
}


def _to_array(val):
    """Convert input value to a 1-D float array."""
    if isinstance(val, np.ndarray):
        return val.ravel().astype(float)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=float)
    return np.atleast_1d(np.asarray(val, dtype=float))


def run(inputs, params, block):
    column = params.get("column", "").strip()
    op_str = params.get("operator", ">")
    threshold = float(params.get("value", 0))

    op_func = _OPS.get(op_str, np.greater)

    # Collect data ports in order (skip addInput placeholder)
    port_names = [p.name for p in block.input_ports if p.name != "addInput"]
    port_display = {p.name: p.display_name for p in block.input_ports
                    if p.name != "addInput"}

    # Build arrays and display-name lookup
    arrays = {}
    display_to_port = {}
    for pname in port_names:
        val = inputs.get(pname)
        if val is None:
            continue
        arrays[pname] = _to_array(val)
        dname = port_display.get(pname, pname)
        display_to_port[dname] = pname

    if not arrays:
        raise ValueError("No data connected to Filter Rows.")

    # Find the column to filter on (match by display name or port name)
    filter_port = None
    if column in display_to_port:
        filter_port = display_to_port[column]
    elif column in arrays:
        filter_port = column
    else:
        col_lower = column.lower()
        for dname, pname in display_to_port.items():
            if dname.lower() == col_lower:
                filter_port = pname
                break

    if filter_port is None or filter_port not in arrays:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(display_to_port.keys())}")

    # Build mask
    mask = op_func(arrays[filter_port], threshold)

    n_before = len(mask)
    n_after = int(np.sum(mask))
    block.parameters["displayText"] = (
        f"{column} {op_str} {threshold}\n{n_after}/{n_before}")

    # Output: mirror input ports (in1 -> out1), apply mask
    result = {}
    out_port_names = {p.name for p in block.output_ports}
    for pname, arr in arrays.items():
        out_name = pname.replace("in", "out", 1)
        if out_name in out_port_names:
            result[out_name] = arr[mask]

    return result
