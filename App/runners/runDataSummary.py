"""Runner for DataSummary block - shows summary dialog, passes inputs through."""

import numpy as np
from PySide6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QLabel, QGridLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


def _formatted_number(n):
    """Format number with comma separators."""
    return f"{int(n):,}"


def run(inputs, params, block):
    filename = inputs.get("filename", inputs.get("fileName", "(unknown)"))
    data = inputs.get("data")
    platform = inputs.get("platform", None)
    sample_rate = inputs.get("sampleRate", None)

    # Number of data points
    n_points = 0
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, np.ndarray):
                n_points = max(n_points, v.size)
    elif isinstance(data, np.ndarray):
        n_points = data.size
    elif data is not None:
        try:
            n_points = len(data)
        except TypeError:
            pass

    # Duration
    duration_str = "(unknown)"
    if n_points > 0 and sample_rate and sample_rate > 0:
        duration_s = n_points / sample_rate
        duration_str = f"{duration_s:.3f} s"

    # Build rows: (label, value)
    rows = [
        ("File:", filename if filename else "(unknown)"),
        ("Data points:", _formatted_number(n_points)),
        ("Sample rate:", f"{sample_rate:g} Hz" if sample_rate and sample_rate > 0 else "(unknown)"),
        ("Duration:", duration_str),
        ("Platform:", platform if platform else "(not detected)"),
    ]

    # Custom dialog with 2-column grid layout
    dlg = QDialog()
    dlg.setWindowTitle("Data Summary")
    dlg.setMinimumWidth(420)

    from utils.dialog_style import apply_dialog_style
    apply_dialog_style(dlg)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(24, 20, 24, 16)
    outer.setSpacing(16)

    grid = QGridLayout()
    grid.setHorizontalSpacing(24)
    grid.setVerticalSpacing(8)

    label_font = QFont()
    label_font.setBold(True)
    label_font.setPointSize(13)

    value_font = QFont()
    value_font.setPointSize(13)

    for i, (label, value) in enumerate(rows):
        lbl = QLabel(label)
        lbl.setFont(label_font)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(lbl, i, 0)

        val = QLabel(str(value))
        val.setFont(value_font)
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(val, i, 1)

    outer.addLayout(grid)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    buttons.setCenterButtons(True)
    buttons.accepted.connect(dlg.accept)
    outer.addWidget(buttons)

    dlg.exec()

    return {
        "data": data,
        "info": {
            "filename": filename,
            "platform": platform,
        },
    }
