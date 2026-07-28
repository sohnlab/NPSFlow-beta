"""Runner for LoadNPSData block - two-step folder/file selection for NPS .mat data."""

import os

from PySide6.QtWidgets import (
    QFileDialog, QInputDialog, QMessageBox, QDialog, QVBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QDialogButtonBox, QPushButton,
    QHBoxLayout, QScrollArea, QWidget, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QCursor
import numpy as np
from scipy.io import loadmat


def _load_mat_file(filepath):
    """Load a .mat file, with fallback to h5py for MATLAB v7.3 (HDF5) files."""
    try:
        mat_data = loadmat(filepath, squeeze_me=True)
        return {k: v for k, v in mat_data.items() if not k.startswith("_")}
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5 — use h5py
        import h5py
        data = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                if key.startswith("_"):
                    continue
                ds = f[key]
                if isinstance(ds, h5py.Dataset):
                    val = ds[()]
                    if isinstance(val, np.ndarray):
                        val = np.squeeze(val)
                    data[key] = val
                elif isinstance(ds, h5py.Group):
                    # Read group as dict of arrays
                    group_data = {}
                    for sub_key in ds.keys():
                        sub_ds = ds[sub_key]
                        if isinstance(sub_ds, h5py.Dataset):
                            sub_val = sub_ds[()]
                            if isinstance(sub_val, np.ndarray):
                                sub_val = np.squeeze(sub_val)
                            group_data[sub_key] = sub_val
                    data[key] = group_data
        return data


class FileBrowserDialog(QDialog):
    """List-based file browser for selecting .mat files."""

    BREADCRUMB_STYLE = """
        QPushButton {
            border: none;
            padding: 2px 4px;
            color: #0066CC;
            font-size: 12px;
        }
        QPushButton:hover {
            text-decoration: underline;
            color: #0044AA;
        }
    """
    SEPARATOR_STYLE = "color: #888; font-size: 12px; padding: 0 1px;"

    def __init__(self, start_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a MAT File")
        self.setMinimumSize(600, 450)
        self.current_path = start_path
        self.selected_file = None

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        layout = QVBoxLayout(self)

        # Breadcrumb path bar
        self._breadcrumb_widget = QWidget()
        self._breadcrumb_widget.setStyleSheet(
            "QWidget { background: #F5F5F5; border: 1px solid #CCC; "
            "border-radius: 4px; }")
        self._breadcrumb_layout = QHBoxLayout(self._breadcrumb_widget)
        self._breadcrumb_layout.setContentsMargins(6, 2, 6, 2)
        self._breadcrumb_layout.setSpacing(0)
        layout.addWidget(self._breadcrumb_widget)

        self._list = QListWidget()
        import sys as _sys
        _mono = "Menlo" if _sys.platform == "darwin" else "Consolas"
        self._list.setFont(QFont(_mono, 12))
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._list)

        btn_layout = QHBoxLayout()
        self._open_btn = QPushButton("Open")
        self._open_btn.clicked.connect(self._on_open)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self._open_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._populate()

    def _build_breadcrumbs(self):
        """Build clickable breadcrumb buttons: root › ... › last3 › last2 › last1."""
        # Clear existing breadcrumbs
        while self._breadcrumb_layout.count():
            child = self._breadcrumb_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        parts = self.current_path.replace("/", os.sep).split(os.sep)
        # Build (display_name, full_path) for each segment
        segments = []
        accumulated = ""
        for i, part in enumerate(parts):
            if i == 0 and part == "":
                # Unix root
                accumulated = "/"
                segments.append(("/", "/"))
            elif i == 0 and len(part) == 2 and part[1] == ":":
                # Windows drive letter (e.g. "G:")
                accumulated = part + os.sep
                segments.append((part, accumulated))
            else:
                accumulated = os.path.join(accumulated, part)
                segments.append((part, accumulated))

        # Collapse middle: show first + ... + last 3
        TAIL = 3
        if len(segments) > TAIL + 1:
            visible = [segments[0], None] + segments[-TAIL:]
        else:
            visible = segments

        for i, seg in enumerate(visible):
            if i > 0:
                sep = QLabel("\u203A")
                sep.setStyleSheet(self.SEPARATOR_STYLE)
                self._breadcrumb_layout.addWidget(sep)

            if seg is None:
                ellipsis = QLabel("...")
                ellipsis.setStyleSheet(self.SEPARATOR_STYLE)
                self._breadcrumb_layout.addWidget(ellipsis)
                continue

            display, path = seg
            btn = QPushButton(display)
            btn.setStyleSheet(self.BREADCRUMB_STYLE)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.clicked.connect(lambda checked, p=path: self._navigate(p))
            if seg is visible[-1]:
                f = btn.font()
                f.setBold(True)
                btn.setFont(f)
            self._breadcrumb_layout.addWidget(btn)

        self._breadcrumb_layout.addStretch()

    def _navigate(self, path):
        """Navigate to a path from breadcrumb click."""
        if os.path.isdir(path):
            self.current_path = path
            self._populate()

    def _populate(self):
        """Fill the list with folders and .mat files for current_path."""
        self._list.clear()
        self._build_breadcrumbs()

        # Subfolders
        entries = sorted(os.listdir(self.current_path))
        for entry in entries:
            full = os.path.join(self.current_path, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                item = QListWidgetItem(f"\U0001F4C1  {entry}")
                item.setData(Qt.ItemDataRole.UserRole, ("folder", full))
                self._list.addItem(item)

        # .mat files
        for entry in entries:
            if entry.lower().endswith(".mat"):
                full = os.path.join(self.current_path, entry)
                item = QListWidgetItem(f"\U0001F4C4  {entry}")
                item.setData(Qt.ItemDataRole.UserRole, ("file", full))
                self._list.addItem(item)

    def _on_item_double_clicked(self, item):
        kind, path = item.data(Qt.ItemDataRole.UserRole)
        if kind == "file":
            self.selected_file = path
            self.accept()
        else:
            self.current_path = path
            self._populate()

    def _on_open(self):
        item = self._list.currentItem()
        if item is None:
            return
        kind, path = item.data(Qt.ItemDataRole.UserRole)
        if kind == "file":
            self.selected_file = path
            self.accept()
        else:
            self.current_path = path
            self._populate()


def run(inputs, params, block):
    # Check if a file path was provided via input port
    input_path = inputs.get("filePath", None)
    if input_path and isinstance(input_path, str) and input_path.strip():
        ext = os.path.splitext(input_path)[1].lower()
        if ext in (".mat", ".csv") and os.path.isfile(input_path):
            filename = os.path.basename(input_path)
            try:
                if ext == ".mat":
                    data = _load_mat_file(input_path)
                else:
                    import pandas as pd
                    df = pd.read_csv(input_path)
                    data = {col: df[col].values for col in df.columns}
                from utils.app_logger import logger; logger.info(f"Loaded file: {filename}")
                return {"data": data, "filename": filename}
            except Exception as exc:
                from utils.app_logger import logger; logger.warning(f"Failed to load provided path: {exc}")
                # Fall through to manual selection

    # Determine start folder
    # Root is two levels up from runners/ → App/ → root/
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    signal_data_path = os.path.join(project_dir, "SignalData")
    if not os.path.isdir(signal_data_path):
        signal_data_path = project_dir

    # Browse for .mat file using list-based browser
    dlg = FileBrowserDialog(signal_data_path)
    if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.selected_file:
        raise ValueError("User cancelled.")
    selected_mat_file = dlg.selected_file

    # Step 3: Load the selected file
    filename = os.path.basename(selected_mat_file)
    ext = os.path.splitext(selected_mat_file)[1].lower()
    try:
        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(selected_mat_file)
            data = {col: df[col].values for col in df.columns}
        else:
            data = _load_mat_file(selected_mat_file)
        from utils.app_logger import logger; logger.info(f"Loaded file: {filename}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load '{filename}': {exc}") from exc

    return {"data": data, "filename": filename}
