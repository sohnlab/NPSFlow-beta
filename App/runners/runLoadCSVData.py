"""Runner for LoadCSVData block - loads CSV or NPZ files/folders from inputs.

Despite the legacy name, this block loads tabular data from `.csv` files and
array archives from `.npz` files. Multiple files are concatenated by column /
array name. NPZ is the preferred format (faster, smaller, dtype-faithful).
"""

import glob
import os

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog


SUPPORTED_EXTS = (".csv", ".npz")


def _collect_from_path(path):
    """Given a file or folder path, return supported files (csv + npz)."""
    path = path.strip()
    if os.path.isdir(path):
        files = []
        for ext in SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        return sorted(files)
    elif os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTS):
        return [path]
    return []


def _load_one(filepath):
    """Load one file and return a dict of {column_name: ndarray}."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
        return {col: df[col].to_numpy() for col in df.columns}
    if ext == ".npz":
        # allow_pickle=True: object-dtype columns from Save File are pickled.
        with np.load(filepath, allow_pickle=True) as npz:
            return {key: np.asarray(npz[key]) for key in npz.files}
    raise ValueError(f"Unsupported file extension: {ext}")


def run(inputs, params, block):
    files = []

    # Collect from all input ports (dataSource + dynamic inputs)
    skip = {"run", "addInput"}
    for port in block.input_ports:
        if port.name in skip:
            continue
        val = inputs.get(port.name)
        if val and isinstance(val, str) and val.strip():
            files.extend(_collect_from_path(val))

    # If no files from inputs, try saved source from previous run
    if not files:
        saved = block.parameters.get("lastSource", "")
        if saved:
            files = _collect_from_path(saved)

    # If still no files, open folder picker
    if not files:
        folder = QFileDialog.getExistingDirectory(None, "Select Folder with Data Files")
        if not folder:
            raise ValueError("No folder selected")
        files = _collect_from_path(folder)

    if not files:
        raise ValueError("No CSV or NPZ files found")

    # Save first source for next run
    block.parameters["lastSource"] = os.path.dirname(files[0])

    # Load and concatenate each column/array across files
    flat = {}
    loaded = []
    for filepath in files:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        file_data = _load_one(filepath)
        for col, values in file_data.items():
            values = np.asarray(values)
            if col in flat:
                flat[col] = np.concatenate([flat[col], values])
            else:
                flat[col] = values
        loaded.append(os.path.basename(filepath))

    # Strip dot-notation prefixes added by Save File (e.g. "labeledData.data" -> "data")
    combined = {}
    for col, values in flat.items():
        key = col.split(".", 1)[-1] if "." in col else col
        combined[key] = values

    block.parameters["displayText"] = f"{len(loaded)} files"

    from utils.app_logger import logger
    logger.info(f"Loaded {len(loaded)} data files: {loaded}")

    return {"data": combined, "fileNames": loaded}
