"""Runner for PlotData (2D Plot) block – interactive multi-series plotting."""

import numpy as np


def run(inputs, params, block):
    # Collect series from paired x/y ports
    port_names = {p.name for p in block.input_ports} if block else set(inputs.keys())

    # Find max series number
    max_series = 0
    for name in port_names:
        if len(name) > 1 and name[0] in ("x", "y") and name[1:].isdigit():
            max_series = max(max_series, int(name[1:]))

    series = []
    for i in range(1, max_series + 1):
        x_raw = inputs.get(f"x{i}")
        y_raw = inputs.get(f"y{i}")

        if x_raw is None and y_raw is None:
            continue

        # Convert to arrays
        x_data = np.asarray(x_raw, dtype=float).ravel() if x_raw is not None else None
        y_data = np.asarray(y_raw, dtype=float).ravel() if y_raw is not None else None

        x_connected = x_raw is not None
        y_connected = y_raw is not None

        # When one is missing, use indices
        if x_data is None:
            x_data = np.arange(len(y_data), dtype=float)
        if y_data is None:
            y_data = np.arange(len(x_data), dtype=float)

        # Label as "Data 1", "Data 2", etc.
        label = f"Data {i}"

        series.append({
            "x": x_data, "y": y_data, "label": label,
            "x_connected": x_connected, "y_connected": y_connected,
        })

    if not series:
        raise ValueError("No data provided. Connect at least one x or y input.")

    # Open interactive dialog
    from utils.plot_data_ui import plot_data_dialog
    plot_data_dialog(series, block=block)

    return {}
