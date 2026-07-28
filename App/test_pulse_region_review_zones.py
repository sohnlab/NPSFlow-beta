"""Zone-aware Detection Review dialog: single-zone unchanged, multi-zone aware."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


REGIONS = np.array([[100, 200], [500, 600], [900, 1000]])
FS = 1000
N = 1500


def _signal(scale=1.0, phase=0.0):
    t = np.arange(N) / FS
    return scale * np.sin(2 * np.pi * 5 * t + phase)


def test_single_zone_no_selector():
    _app()
    from processing.pulse_region_review import _PulseReviewDialog
    dlg = _PulseReviewDialog(_signal(), REGIONS, FS)
    assert dlg._n_zones == 1
    assert dlg._zone_btn_group is None
    assert not hasattr(dlg, "_overlay_chk")
    assert dlg._active_zone == 0
    assert dlg._overlay_mode == 'active'
    dlg._render_page()  # must not raise
    dlg.close()


def test_multi_zone_selector_and_switch():
    _app()
    from processing.pulse_region_review import _PulseReviewDialog
    data = np.column_stack([_signal(1.0), _signal(0.5, 0.5), _signal(0.25, 1.0)])
    assert data.shape == (N, 3)
    dlg = _PulseReviewDialog(data, REGIONS, FS)
    assert dlg._n_zones == 3
    assert dlg._zone_btn_group is not None
    assert dlg._active_zone == 0
    assert np.array_equal(dlg._data, data[:, 0])

    # Switch to zone 2 (index 2 -> "Zone 3").
    dlg._on_zone_selected(2)
    assert dlg._active_zone == 2
    assert np.array_equal(dlg._data, data[:, 2])
    assert np.array_equal(dlg._data_lp, dlg._zones_lp[2])
    dlg.close()


def test_overlay_checkbox_toggles_mode():
    _app()
    from processing.pulse_region_review import _PulseReviewDialog
    data = np.column_stack([_signal(1.0), _signal(0.5, 0.5)])
    dlg = _PulseReviewDialog(data, REGIONS, FS)
    assert hasattr(dlg, "_overlay_chk")
    # Multi-zone default: "All Zones" checked → overlay mode 'all'.
    assert dlg._overlay_chk.isChecked()
    assert dlg._overlay_mode == 'all'
    dlg._overlay_chk.setChecked(False)
    assert dlg._overlay_mode == 'active'
    dlg._overlay_chk.setChecked(True)
    assert dlg._overlay_mode == 'all'
    dlg.close()


def test_overlay_ylim_driven_by_active_zone_scale():
    """Zones with huge baseline offsets must not blow up the active y-scale."""
    _app()
    from processing.pulse_region_review import _PulseReviewDialog
    # Each zone shares the same ~2.0 peak-to-peak shape but a huge baseline.
    data = np.column_stack([_signal(1.0) + k * 1e6 for k in range(3)])
    dlg = _PulseReviewDialog(data, REGIONS, FS)
    assert dlg._n_zones == 3

    # Active zone 0; overlay all; render.
    dlg._on_zone_selected(0)
    dlg._overlay_chk.setChecked(True)
    assert dlg._overlay_mode == 'all'
    dlg._render_page()  # must not raise

    # The active zone's in-window range is ~2 (sin amplitude 1, scale 1) and the
    # stacked offsets are multiples of arng*1.5 (~3), NOT the 1e6 baseline
    # spread — so the total ylim span stays small, nowhere near 1e6.
    ax = dlg._axes_grid[0]
    lo, hi = ax.get_ylim()
    span = hi - lo
    assert span < 100.0, f"ylim span {span} suggests raw baseline spread leaked in"
    assert span > 0.0
    dlg.close()


def test_multi_zone_flat_segment_no_crash():
    _app()
    from processing.pulse_region_review import _PulseReviewDialog
    # One zone is flat (zero range) — y-limit guard must hold under overlay.
    flat = np.zeros(N)
    data = np.column_stack([_signal(1.0), flat])
    dlg = _PulseReviewDialog(data, REGIONS, FS)
    assert dlg._n_zones == 2
    dlg._overlay_chk.setChecked(True)
    dlg._render_page()  # must not raise on flat overlay zone
    dlg.close()


def test_delete_pulse_and_undo():
    _app()
    from processing.pulse_region_review import _PulseReviewDialog
    dlg = _PulseReviewDialog(_signal(1.0) + 1000.0, REGIONS, FS)
    assert len(dlg._pulse_regions) == 3
    dlg._delete_pulse(1)  # remove the 2nd region on page 0
    assert len(dlg._pulse_regions) == 2 and len(dlg._labels) == 2
    assert [int(r[0]) for r in dlg._pulse_regions] == [100, 900]
    dlg._undo()  # delete is undoable
    assert [int(r[0]) for r in dlg._pulse_regions] == [100, 500, 900]
    dlg.close()


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
