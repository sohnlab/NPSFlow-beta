"""Runner for LoadLabeledData block.

Recursively scans a folder for labeled-data files (CSV or NPZ) with two
arrays/columns: data + labels, optionally dot-prefixed as
'labeledData.data'/'labeledData.labels'. Emits a list of per-file dicts
for downstream batch processing.
"""

import glob
import os

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog

from utils.app_logger import logger


DATA_KEYS = ("labeledData.data", "data")
LABEL_KEYS = ("labeledData.labels", "labels")
SUPPORTED_EXTS = (".csv", ".npz")


def _pick(source, candidates):
    """Return the first matching key in source (dict-like), or None."""
    for name in candidates:
        if name in source:
            return name
    return None


def _load_pair(path):
    """Return (data, labels) as numpy arrays, or (None, None) if unreadable/mismatched."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            data_col = _pick(df.columns, DATA_KEYS)
            label_col = _pick(df.columns, LABEL_KEYS)
            if data_col is None or label_col is None:
                return None, None
            return df[data_col].to_numpy(), df[label_col].to_numpy()
        if ext == ".npz":
            # allow_pickle=True: object-dtype columns from Save File are pickled.
            with np.load(path, allow_pickle=True) as npz:
                data_key = _pick(npz.files, DATA_KEYS)
                label_key = _pick(npz.files, LABEL_KEYS)
                if data_key is None or label_key is None:
                    return None, None
                return np.asarray(npz[data_key]), np.asarray(npz[label_key])
    except Exception as exc:
        logger.info(f"[LoadLabeledData] Skipping unreadable file {path}: {exc}")
    return None, None


def _resolve_folder(raw, block):
    if isinstance(raw, str) and raw.strip():
        path = raw.strip().replace("\\", "/")
        if not os.path.isabs(path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            path = os.path.join(project_root, path)
        return path

    saved = block.parameters.get("lastFolder", "")
    if saved and os.path.isdir(saved):
        return saved

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    default_dir = os.path.join(project_root, "Output", "LabeledData")
    start_dir = default_dir if os.path.isdir(default_dir) else project_root

    folder = QFileDialog.getExistingDirectory(
        None, "Select Folder with Labeled Data Files", start_dir)
    if not folder:
        raise ValueError("No folder selected")
    return folder


def run(inputs, params, block):
    folder = _resolve_folder(inputs.get("folder"), block)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
    paths = sorted(paths)

    loaded = []
    skipped = []
    for path in paths:
        data_raw, labels_raw = _load_pair(path)
        if data_raw is None:
            logger.info(f"[LoadLabeledData] Skipping {path}: missing data/labels")
            skipped.append(path)
            continue

        data = np.asarray(data_raw, dtype=np.float64)
        labels_float = np.asarray(labels_raw, dtype=np.float64)
        labels = labels_float.astype(np.int32)
        if not np.array_equal(labels, labels_float):
            logger.info(f"[LoadLabeledData] {path}: labels had non-integer values, truncated to int32")

        loaded.append({
            "data": data,
            "labels": labels,
            "fileName": os.path.relpath(path, folder),
        })

    if not loaded:
        raise ValueError(f"No labeled CSV/NPZ files found in {folder}")

    block.parameters["lastFolder"] = folder
    block.parameters["displayText"] = f"{len(loaded)} files"

    logger.info(
        f"[LoadLabeledData] Loaded {len(loaded)} labeled files from {folder}"
        + (f" (skipped {len(skipped)})" if skipped else "")
    )

    return {"labeledDataList": loaded}
