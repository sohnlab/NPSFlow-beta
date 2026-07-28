"""Pure detrend parity with the single-block dialog method. No exec()."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def _seg_with_drift():
    n = 300
    x = np.arange(n, dtype=float)
    seg = np.zeros(n)
    seg += -20.0 * np.exp(-((x - 120) ** 2) / (2 * 6 ** 2))  # a dip
    seg += 0.02 * x                                          # linear drift
    peaks = np.array([90, 160], dtype=int)                  # global == local (s=0)
    return seg, peaks


def test_detrend_segment_flattens_plateaus_start_mode():
    from processing.pulse_detrend import detrend_pulse_segment
    seg, peaks = _seg_with_drift()
    out = detrend_pulse_segment(seg, seg_filt=seg.copy(), global_peaks=peaks,
                                s=0, e=len(seg) - 1, mode="starting",
                                ref_samples=20)
    assert out is not None
    _, _, rect_med, _ = out
    assert abs(rect_med[0] - rect_med[-1]) <= 0.05 * np.ptp(seg)


def test_pure_matches_dialog_method():
    # Build the real dialog headlessly and compare its _detrend_pulse_segment
    # to the pure function on the same segment.
    from test_extract_zone_compute import _two_zone_computes, _make_dialog
    zones, pidx, fs = _two_zone_computes()
    dlg = _make_dialog(zones, pidx, fs)
    try:
        dlg._rb_starting.setChecked(True)
        dlg._spin_ref_samples.setValue(15)
        seg, peaks = _seg_with_drift()
        ref = dlg._detrend_pulse_segment(seg.copy(), seg.copy(), peaks, 0, len(seg) - 1)
        from processing.pulse_detrend import detrend_pulse_segment
        got = detrend_pulse_segment(seg.copy(), seg.copy(), peaks, 0, len(seg) - 1,
                                    mode="starting", ref_samples=15)
        assert ref is not None and got is not None
        for a, b in zip(ref, got):
            if a is None:
                assert b is None
            else:
                assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
    finally:
        dlg.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("ok ", fn.__name__)
        except Exception as e:
            failed += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
