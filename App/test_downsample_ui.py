"""Preprocess dialog: single axis, Confirm gated on Apply, accent migration.
Runs offscreen with a real QApplication but never enters the modal loop."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


def _dlg():
    from processing.downsample_ui import _DownsampleDialog
    x = np.arange(200, dtype=float)
    return _DownsampleDialog(x, 1000, init_params={
        "smooth_enable": False, "ds_enable": True, "ds_factor": 4,
        "lp_enable": False, "asls_enable": False})


def test_single_axis():
    d = _dlg()
    assert hasattr(d, "ax") and not hasattr(d, "ax_bot")
    assert len(d.fig.axes) == 1


def test_zoom_reveals_more_detail_than_full_view():
    from processing.downsample_ui import _DownsampleDialog
    n = 200000
    y = np.sin(np.arange(n, dtype=float) / 50.0)   # fine feature, > 5000 pts
    d = _DownsampleDialog(y, 1000, init_params={
        "smooth_enable": False, "ds_enable": False,
        "lp_enable": False, "asls_enable": False})
    d._recompute()                                  # output plotted, full range
    line = d._plot_series[0][2]
    fx = np.asarray(line.get_xdata())
    full_in_window = int(np.sum((fx >= 100.0) & (fx <= 101.0)))
    d._set_xlim(100.0, 101.0)                        # zoom to a 1 s window
    zx = np.asarray(line.get_xdata())
    zoom_in_window = int(np.sum((zx >= 100.0) & (zx <= 101.0)))
    # zooming re-decimates to the visible window, so it shows far more real
    # points there than the coarse full-range decimation did.
    assert zoom_in_window > 300, zoom_in_window
    assert zoom_in_window > full_in_window * 3, (full_in_window, zoom_in_window)


def test_asls_enabled_overlays_signal_and_trendline():
    from processing.downsample_ui import _DownsampleDialog
    y = -np.abs(np.sin(np.arange(4000, dtype=float) / 20.0)) + 5.0
    # ASLS on -> two series (data + trendline); ASLS off -> one (output).
    d_on = _DownsampleDialog(y, 1000, init_params={
        "smooth_enable": False, "ds_enable": False,
        "lp_enable": False, "asls_enable": True})
    d_on._recompute()
    labels_on = [ln.get_label() for _, _, ln in d_on._plot_series]
    assert labels_on == ["signal", "ASLS baseline"], labels_on

    d_off = _DownsampleDialog(y, 1000, init_params={
        "smooth_enable": False, "ds_enable": False,
        "lp_enable": False, "asls_enable": False})
    d_off._recompute()
    labels_off = [ln.get_label() for _, _, ln in d_off._plot_series]
    assert labels_off == ["signal"], labels_off


def test_confirm_disabled_until_apply():
    d = _dlg()
    assert d.btn_confirm.isEnabled() is False
    assert d.btn_apply.objectName() == "confirmBtn"      # accent starts on Apply
    d._recompute()                                       # simulate Apply
    assert d.btn_confirm.isEnabled() is True
    assert d.btn_confirm.objectName() == "confirmBtn"    # accent moved to Confirm
    assert d.btn_apply.objectName() == ""                # Apply de-accented


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("ok ", fn.__name__)
        except Exception as e:
            failed += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
