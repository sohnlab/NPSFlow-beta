"""Multizone support for the Data Labeling block (processing/training_data.py).

Offscreen; no exec. Run: python App/test_training_data_zones.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtWidgets import QApplication

N = 3000


def _synth(n_zones):
    t = np.linspace(0.0, 6.28, N)
    cols = []
    for zi in range(n_zones):
        base = 1.0e6 + 5.0e4 * zi
        cols.append(base + 2.0e4 * np.sin(t * (zi + 1)))
    if n_zones == 1:
        return cols[0]                     # 1-D single zone
    return np.column_stack(cols)           # N x Z multizone


def _regions():
    return {"single": np.array([[100, 300], [800, 1000]]),
            "coincident": None, "noise": None, "uncertain": None}


def _build_dialog(n_zones):
    from processing.training_data import _TrainingDataDialog
    QApplication.instance() or QApplication([])
    return _TrainingDataDialog(_synth(n_zones), _regions(), sample_rate=0)


def test_csv_single_zone():
    from processing.training_data import _build_csv_text
    txt = _build_csv_text([np.array([1.0, 2.0, 3.0])], np.array([0, 1, 0]))
    lines = txt.strip().split("\n")
    assert lines[0] == "sample_index,value,label_id", lines[0]
    assert len(lines) == 4, len(lines)                 # header + 3 rows
    assert lines[1].split(",") == ["0", "1.000000", "0"], lines[1]
    assert len(lines[2].split(",")) == 3, lines[2]


def test_csv_multizone_header_and_width():
    from processing.training_data import _build_csv_text
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    txt = _build_csv_text([a, b], np.array([0, 2, 0]))
    lines = txt.strip().split("\n")
    assert lines[0] == "sample_index,value_zone1,value_zone2,label_id", lines[0]
    assert len(lines) == 4, len(lines)
    # index + 2 zone values + label = 4 fields
    assert len(lines[1].split(",")) == 4, lines[1]
    assert lines[1].split(",")[-1] == "0"
    assert lines[2].split(",")[-1] == "2"
    assert lines[1].split(",")[1] == "1.000000", lines[1]   # value_zone1
    assert lines[1].split(",")[2] == "4.000000", lines[1]   # value_zone2


def test_multizone_intake_shapes():
    dlg = _build_dialog(3)
    assert dlg._is_mz is True
    assert dlg._n == N, dlg._n
    assert len(dlg._zones) == 3, len(dlg._zones)
    assert dlg._labels.shape == (N,), dlg._labels.shape


def test_single_zone_intake_shapes():
    dlg = _build_dialog(1)
    assert dlg._is_mz is False
    assert len(dlg._zones) == 1, len(dlg._zones)
    assert dlg._n == N, dlg._n
    assert dlg._labels.shape == (N,), dlg._labels.shape


def test_multizone_plot_lines_ylabel_and_annotations():
    dlg = _build_dialog(3)
    dlg._refresh_plot()
    assert len(dlg._ax.lines) == 3, len(dlg._ax.lines)
    assert dlg._ax.get_ylabel() == "Resistance [MΩ] (offset)", \
        dlg._ax.get_ylabel()
    # one left-edge "Zone N" annotation per zone
    assert len(dlg._ax.texts) == 3, len(dlg._ax.texts)


def test_single_zone_plot_unchanged():
    dlg = _build_dialog(1)
    dlg._refresh_plot()
    assert len(dlg._ax.lines) == 1, len(dlg._ax.lines)
    assert dlg._ax.get_ylabel() == "Amplitude", dlg._ax.get_ylabel()
    assert len(dlg._ax.texts) == 0, len(dlg._ax.texts)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
