"""Sample-rate Information group shared by processing dialogs."""
import numpy as np
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout


def fmt_hz(v):
    """Format a frequency with a unit prefix."""
    v = float(np.ravel(v)[0])
    if v >= 1e6:
        return f"{v / 1e6:g} MHz"
    if v >= 1e3:
        return f"{v / 1e3:g} kHz"
    return f"{v:g} Hz"


def make_sr_info_group(fs, ds_factor=None):
    """Build the Information group: base SR, DS factor, effective SR, Nyquist.

    *fs* is the effective rate the dialog operates at; base = fs * ds_factor.
    """
    factor = ds_factor or 1
    fs = float(np.ravel(fs)[0])
    group = QGroupBox("Information")
    lay = QVBoxLayout(group)
    for name, val in (("Base SR:", fmt_hz(fs * factor)),
                      ("Downsample factor:", f"{factor:g}"),
                      ("Effective SR:", fmt_hz(fs)),
                      ("Nyquist f:", fmt_hz(fs / 2))):
        r = QHBoxLayout()
        r.addWidget(QLabel(name))
        r.addStretch()
        r.addWidget(QLabel(val))
        lay.addLayout(r)
    return group
