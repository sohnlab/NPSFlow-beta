"""Unified loader for .mat / .csv / .npz / .json files.

Emits one struct on the ``data`` output port:

    {
        "fileNames": [str],
        "files":     [ {fileName, ...content keys..., labels?} ],
    }

Single-file selection yields ``files=[one entry]``; folder selection scans
recursively and emits one entry per file. Feed the output into an Unpack
block to expose individual fields, or into a Combine Data block to
concatenate all files into single arrays.
"""

import glob
import json
import os

import numpy as np

from utils.paths import project_root
import pandas as pd
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout,
)
from scipy.io import loadmat


MAT_EXTS = (".mat",)
TABULAR_EXTS = (".csv", ".npz", ".json")
ALL_EXTS = MAT_EXTS + TABULAR_EXTS

LABEL_KEYS = ("labeledData.labels", "labels")


# ---------- loaders ----------

def _load_mat(filepath):
    try:
        mat = loadmat(filepath, squeeze_me=True)
        return {k: v for k, v in mat.items() if not k.startswith("_")}
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                if key.startswith("_"):
                    continue
                ds = f[key]
                if isinstance(ds, h5py.Dataset):
                    val = ds[()]
                    if isinstance(val, np.ndarray):
                        val = np.squeeze(val)
                    out[key] = val
                elif isinstance(ds, h5py.Group):
                    group = {}
                    for sub in ds.keys():
                        sds = ds[sub]
                        if isinstance(sds, h5py.Dataset):
                            v = sds[()]
                            if isinstance(v, np.ndarray):
                                v = np.squeeze(v)
                            group[sub] = v
                    out[key] = group
        return out


def _load_csv(filepath):
    df = pd.read_csv(filepath)
    return {col: df[col].to_numpy() for col in df.columns}


def _load_npz(filepath):
    # allow_pickle=True: Save File writes ragged/heterogeneous columns as
    # object-dtype arrays (pickled by savez), so loading must unpickle them.
    with np.load(filepath, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def _load_json(filepath):
    """Best-effort JSON → dict-of-arrays. Tries common shapes."""
    with open(filepath, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        # dict-of-lists → arrays; dict-of-scalars → preserved as-is
        if obj and all(isinstance(v, list) for v in obj.values()):
            return {k: np.asarray(v) for k, v in obj.items()}
        return obj
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj):
            df = pd.DataFrame(obj)
            return {col: df[col].to_numpy() for col in df.columns}
        return {"items": np.asarray(obj, dtype=object)}
    return {"value": obj}


def _load_one(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".mat":
        return _load_mat(filepath)
    if ext == ".csv":
        return _load_csv(filepath)
    if ext == ".npz":
        return _load_npz(filepath)
    if ext == ".json":
        return _load_json(filepath)
    raise ValueError(f"Unsupported extension: {ext}")


# ---------- helpers ----------

def _strip_prefix(d):
    """Strip dot-prefixes added by Save File (e.g. labeledData.data → data)."""
    out = {}
    for col, vals in d.items():
        key = col.split(".", 1)[-1] if "." in col else col
        out[key] = vals
    return out


def _collect_files(path):
    """Expand a file or folder path into a sorted list of supported files."""
    if not path:
        return []
    if os.path.isfile(path):
        return [path] if path.lower().endswith(ALL_EXTS) else []
    if not os.path.isdir(path):
        return []
    found = []
    for ext in ALL_EXTS:
        found.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))
    return sorted(found)


def _start_dir(last_path):
    """Pick a reasonable starting directory for the picker."""
    if last_path:
        if os.path.isdir(last_path):
            return last_path
        parent = os.path.dirname(last_path)
        if os.path.isdir(parent):
            return parent
    root = project_root()
    signal = os.path.join(root, "SignalData")
    return signal if os.path.isdir(signal) else root


# ---------- picker dialog ----------

class _ArrowComboBox(QComboBox):
    """QComboBox that hand-paints its down-triangle. The dialog's QSS border
    styling suppresses the native drop-down arrow on macOS, so draw it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Flatten the drop-down sub-control; otherwise macOS/Fusion paints a
        # shaded native button box behind the hand-painted triangle.
        self.setStyleSheet(
            "QComboBox::drop-down { border: none; background: transparent;"
            " width: 22px; }"
            "QComboBox::down-arrow { image: none; width: 0; height: 0; }")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor("#888")
        painter.setPen(QPen(color))
        painter.setBrush(QBrush(color))
        cx, cy = self.width() - 12, self.height() / 2
        painter.drawPolygon(QPolygonF([
            QPointF(cx - 4, cy - 3), QPointF(cx + 4, cy - 3),
            QPointF(cx, cy + 3)]))
        painter.end()


class _ModePickerDialog(QDialog):
    """Compact dialog: 'Select File' or 'Select Folder', or Cancel."""

    def __init__(self, start_dir, recent_paths=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Data")
        self.setFixedWidth(540)
        self._start_dir = start_dir
        from utils.recent_paths import MAX_RECENT
        self._recent_paths = [p for p in (recent_paths or []) if p][:MAX_RECENT]
        self.selected_path = None
        self.use_last = False

        try:
            from utils.dialog_style import apply_dialog_style
            apply_dialog_style(self)
        except Exception:
            pass

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)
        btn_w = 104  # uniform button width

        # Recent shortcut: reuse a recently-loaded file/folder, treated the
        # same as a wired "/lastpath" (turns on reuse-settings mode downstream).
        # The most recent entry is pre-selected.
        if self._recent_paths:
            layout.addWidget(QLabel("Recent files:"))

            self._recent_combo = _ArrowComboBox()
            for p in self._recent_paths:
                name = os.path.basename(p.rstrip("/\\")) or p
                self._recent_combo.addItem(f"⟲  {name}", p)
                self._recent_combo.setItemData(
                    self._recent_combo.count() - 1, p,
                    Qt.ItemDataRole.ToolTipRole)
            self._recent_combo.setCurrentIndex(0)
            self._recent_combo.currentIndexChanged.connect(
                lambda i: self._recent_combo.setToolTip(
                    self._recent_combo.itemData(i) or ""))
            self._recent_combo.setToolTip(self._recent_combo.currentData() or "")

            # Combo stretches to stay long enough for filenames; the Load
            # button sits inline at its natural width.
            recent_row = QHBoxLayout()
            recent_row.setSpacing(8)
            recent_row.addWidget(self._recent_combo, 1)
            load_btn = QPushButton("Load")
            load_btn.setDefault(True)
            load_btn.setFixedWidth(btn_w)
            load_btn.clicked.connect(self._use_selected)
            recent_row.addWidget(load_btn)
            layout.addLayout(recent_row)

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(sep)

        # Label + equal-width action buttons on one row; Cancel at the right end.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(QLabel("Choose what to load:"))

        file_btn = QPushButton("Single File…")
        file_btn.setFixedWidth(btn_w)
        file_btn.clicked.connect(self._pick_file)
        btn_row.addWidget(file_btn)

        folder_btn = QPushButton("Folder…")
        folder_btn.setFixedWidth(btn_w)
        folder_btn.clicked.connect(self._pick_folder)
        btn_row.addWidget(folder_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(btn_w)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

        hint = QLabel("Supports .mat, .csv, .npz, .json")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", self._start_dir,
            "Data files (*.mat *.csv *.npz *.json);;All files (*)")
        if path:
            self.selected_path = path
            self.accept()

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder", self._start_dir)
        if path:
            self.selected_path = path
            self.accept()

    def _use_selected(self):
        path = self._recent_combo.currentData()
        if not path:
            return
        self.selected_path = path
        self.use_last = True
        self.accept()


def _ask_user(recent_paths):
    """Returns (path, use_last). use_last is True when the user chose a recent
    file — handled the same as a wired "/lastpath"."""
    last_path = recent_paths[0] if recent_paths else ""
    dlg = _ModePickerDialog(_start_dir(last_path), recent_paths=recent_paths)
    if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.selected_path:
        raise ValueError("User cancelled file selection.")
    return dlg.selected_path, dlg.use_last


# ---------- main ----------

def _build_file_entry(filepath, base):
    """Load one file and wrap it into a per-file struct entry."""
    raw = _load_one(filepath)
    if not isinstance(raw, dict):
        raw = {"value": raw}
    raw = _strip_prefix(raw)
    entry = dict(raw)
    entry["fileName"] = os.path.relpath(filepath, base) if base else os.path.basename(filepath)
    # Normalize labels to int32 if present.
    for lk in LABEL_KEYS:
        bare = lk.split(".", 1)[-1]
        if bare in entry:
            try:
                entry[bare] = np.asarray(entry[bare]).astype(np.int32)
            except Exception:
                pass
    return entry


def run(inputs, params, block):
    from utils.app_logger import logger

    # 1) Resolve a path: wired input > picker.  An empty port starts fresh —
    # the picker opens (at the last directory for convenience) rather than
    # silently reloading the last file.  A wired path is persisted to lastPath
    # so the sentinel "/lastpath" can request it explicitly later; "/lastpath"
    # falls back to the picker if nothing is saved.
    path = ""
    reuse = False  # True => reuse previous settings downstream (recent file)
    wired = inputs.get("path")
    if isinstance(wired, str) and wired.strip():
        wired = wired.strip()
        if wired == "/lastpath":
            path = block.parameters.get("lastPath", "") or ""
            reuse = True
        else:
            path = wired
            block.parameters["lastPath"] = path

    # Recent files for this workflow — shared by all Load Data blocks and
    # stored in the workflow file, so it never spans workflows (see
    # utils.recent_paths, kept bound to the active canvas by the app).
    from utils.recent_paths import get_recent, add_recent

    if not path:
        # Fall back to this block's lastPath if the shared list is empty
        # (e.g. workflow opened before the shared list existed).
        recent = get_recent()
        last = block.parameters.get("lastPath", "")
        if last and last not in recent:
            recent = recent + [last]
        path, use_last = _ask_user(recent)
        reuse = bool(use_last)

    # Persist the recent-vs-new choice so reuse-settings mode is deterministic
    # at the start of every run — even when this Load File block is cached and
    # its picker doesn't re-open (re-run from a downstream block, keep-
    # interactive). The engine reads block.parameters["reuseSettings"] in
    # _reuse_settings_mode; recent file => reuse downstream, new file => fresh.
    block.parameters["reuseSettings"] = reuse
    if reuse:
        try:
            from utils.settings_input import set_reuse_when_empty
            set_reuse_when_empty(True)
        except Exception:
            pass

    # 2) Expand to file list.
    files = _collect_files(path)
    if not files:
        raise ValueError(f"No .mat/.csv/.npz/.json files found at: {path}")

    # 3) Determine base for relative fileName.
    base = path if os.path.isdir(path) else os.path.dirname(path)

    # 4) Build per-file entries.
    file_entries = [_build_file_entry(fp, base) for fp in files]

    # 5) Persist last path (per-block, for the "/lastpath" sentinel) and push
    # onto the workflow's shared recent-files list.
    block.parameters["lastPath"] = path
    add_recent(path)

    file_names = [e["fileName"] for e in file_entries]
    logger.info(f"[LoadData] Loaded {len(files)} file(s) from {path}")

    out = {
        "fileNames": file_names,
        "files": file_entries,
    }
    return {"data": out}
