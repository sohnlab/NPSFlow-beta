"""Runner for SaveFile block — saves connected inputs to NPZ or CSV.

Each connected input becomes a named column/array. The name is taken from
the port's display name (which mirrors the upstream source port name).
NPZ (default) preserves dtypes exactly and is ~5× smaller and ~10-100× faster
to load than CSV. CSV is still available via the file dialog.
"""

import os
from utils.paths import project_root

import numpy as np
import pandas as pd

from PySide6.QtWidgets import QFileDialog, QApplication


def _collect_columns(inputs, block):
    """Return dict of {name: value} from connected input ports, dict inputs flattened."""
    columns = {}
    for port in block.input_ports:
        if port.name in ("addInput", "filename", "trimIndex"):
            continue
        val = inputs.get(port.name)
        if val is None:
            continue

        col_name = port.display_name
        if col_name in ("Add input", port.name) and port.connections:
            wire = port.connections[0]
            if wire.source_port:
                col_name = wire.source_port.display_name or wire.source_port.name

        if isinstance(val, dict):
            for k, v in val.items():
                columns[f"{col_name}.{k}"] = v
        else:
            columns[col_name] = val
    return columns


def _log(block, msg):
    if block and getattr(block, "_engine", None):
        block._engine.log_callback(msg)


def _coerce_trim_index(val):
    """Return [start, end] ints from a trimIndex input, or None if unusable."""
    if val is None:
        return None
    try:
        flat = np.asarray(val).ravel()
        if flat.size != 2:
            return None
        return [int(flat[0]), int(flat[1])]
    except (TypeError, ValueError):
        return None


def _save_npz(path, columns, trim_index=None):
    arrays = {}
    for name, val in columns.items():
        if isinstance(val, np.ndarray):
            arrays[name] = val
        elif isinstance(val, (list, tuple)):
            arrays[name] = np.asarray(val)
        else:
            arrays[name] = np.asarray([val])
    if trim_index is not None:
        arrays["trimIndex"] = np.asarray(trim_index)
    np.savez_compressed(path, **arrays)
    total_items = sum(arr.size for arr in arrays.values())
    return total_items, len(arrays)


def _save_csv(path, columns, trim_index=None):
    flat = {}
    for name, val in columns.items():
        if isinstance(val, np.ndarray):
            flat[name] = val.ravel().tolist()
        elif isinstance(val, (list, tuple)):
            flat[name] = list(val)
        else:
            flat[name] = [val]

    max_len = max(len(v) for v in flat.values())
    for key in flat:
        diff = max_len - len(flat[key])
        if diff > 0:
            flat[key] = flat[key] + [None] * diff

    df = pd.DataFrame(flat)
    with open(path, "w", newline="") as fh:
        if trim_index is not None:
            fh.write(f"# trimIndex: {trim_index[0]}, {trim_index[1]}\n")
        df.to_csv(fh, index=False)
    return len(df), len(df.columns)


def run(inputs, params, block):
    columns = _collect_columns(inputs, block)
    if not columns:
        raise ValueError("No data to save. Connect at least one input.")

    trim_index = _coerce_trim_index(inputs.get("trimIndex"))
    if inputs.get("trimIndex") is not None and trim_index is None:
        _log(block, "  Warning: trimIndex is not a 2-value [start, end]; skipping it.")

    output_dir = os.path.join(project_root(), "Output")
    os.makedirs(output_dir, exist_ok=True)

    filename_input = inputs.get("filename")
    if filename_input and isinstance(filename_input, str):
        suggested = filename_input.strip()
        for ext in (".npz", ".csv"):
            if suggested.lower().endswith(ext):
                suggested = suggested[: -len(ext)]
                break
        default_path = os.path.join(output_dir, suggested + ".npz")
    else:
        default_path = os.path.join(output_dir, "data.npz")

    parent = QApplication.activeWindow()
    path, selected_filter = QFileDialog.getSaveFileName(
        parent,
        "Save Data",
        default_path,
        "NumPy Archive (*.npz);;CSV Files (*.csv);;All Files (*)",
    )
    if not path:
        raise ValueError("Save cancelled by user.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".npz", ".csv"):
        ext = ".npz" if "npz" in (selected_filter or "").lower() else ".csv"
        path += ext

    if ext == ".npz":
        rows, cols = _save_npz(path, columns, trim_index)
        kind = "values"
    else:
        rows, cols = _save_csv(path, columns, trim_index)
        kind = "rows"

    _log(block, f"  Saved {rows} {kind} × {cols} columns to {path}")
    if trim_index is not None:
        _log(block, f"  Recorded trimIndex [{trim_index[0]}, {trim_index[1]}]")

    return {}
