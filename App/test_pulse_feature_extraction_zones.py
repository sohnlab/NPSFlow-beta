"""Pulse Feature Extraction multi-zone dialog + entry. Mocks the Qt dialog."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np


def _make_tpz(template, peaks, methods):
    return {
        "pulse_template_rec": np.asarray(template, dtype=float),
        "peak_locations": np.asarray(peaks, dtype=int),
        "methods": list(methods),
    }


def _zone_inputs(tp):
    """Mirror how the entry builds per-zone dialog inputs."""
    from utils.zones import n_zones
    from utils.tpsettings_io import get_zone
    out = []
    for k in range(n_zones(tp)):
        tpz = get_zone(tp, k)
        pl = np.atleast_1d(tpz.get("peak_locations", [])).ravel().astype(int)
        out.append({
            "signal_rect": np.atleast_1d(tpz["pulse_template_rec"]).ravel().astype(float),
            "peak_locations": pl,
            "template_original": tpz.get("template_original"),
            "methods": tpz.get("methods"),
        })
    return out


def test_dialog_multizone_selector_and_roundtrip():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_feature_extraction import _SegmentDialog
    from utils.zones import wrap_zones

    tpz0 = _make_tpz(np.linspace(0, 1, 30), [10, 20], ["mean", "min", "max"])
    tpz1 = _make_tpz(np.linspace(0, 1, 50), [15, 30, 40], ["mean", "mean", "max", "min"])
    tp = wrap_zones([tpz0, tpz1])

    dlg = _SegmentDialog(zones=_zone_inputs(tp), active_zone=0)
    assert len(dlg._zones) == 2
    assert dlg._zone_btn_group is not None
    assert dlg._active_zone == 0
    assert dlg._n == 30

    # Edit zone 0.
    dlg._boundaries = [0, 12, 29]
    dlg._labels = ["z0a", "z0b"]
    dlg._methods = ["max", "min"]

    dlg._on_zone_changed(1)
    assert dlg._active_zone == 1
    assert dlg._n == 50
    assert np.array_equal(dlg._signal, dlg._zones[1]["signal"])

    # Edit zone 1 differently.
    dlg._boundaries = [0, 25, 49]
    dlg._labels = ["z1a", "z1b"]

    # Switch back: zone 0 edits must survive.
    dlg._on_zone_changed(0)
    assert dlg._active_zone == 0
    assert dlg._boundaries == [0, 12, 29]
    assert dlg._labels == ["z0a", "z0b"]
    assert dlg._methods == ["max", "min"]

    # And zone 1 retained its own edits.
    assert dlg._zones[1]["boundaries"] == [0, 25, 49]
    assert dlg._zones[1]["labels"] == ["z1a", "z1b"]
    dlg.close()


def test_dialog_single_zone_no_selector():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_feature_extraction import _SegmentDialog

    # Legacy single-template constructor.
    sig = np.linspace(0, 1, 40)
    peaks = np.array([10, 25])
    dlg = _SegmentDialog(sig, peaks, methods=["mean", "min", "max"])
    assert len(dlg._zones) == 1
    assert dlg._zone_btn_group is None
    assert dlg._active_zone == 0
    assert dlg._n == 40
    # Boundaries derive from peaks, exactly as before.
    assert dlg._boundaries == [0, 10, 25, 39]
    assert dlg._methods == ["mean", "min", "max"]
    dlg.close()


class _FakeDialog:
    """Stand-in for _SegmentDialog: confirms with per-zone edits, no exec UI."""
    def __init__(self, signal_rect=None, peak_locations=None,
                 template_original=None, methods=None, parent=None,
                 zones=None, active_zone=0):
        self.confirmed = False
        if zones is None:
            zones = [{"signal_rect": signal_rect, "peak_locations": peak_locations,
                      "methods": methods}]
        self._zones = []
        for z in zones:
            sig = np.atleast_1d(z.get("signal_rect")).ravel().astype(float)
            n = len(sig)
            self._zones.append({
                "signal": sig.copy(),
                "boundaries": [0, n // 2, n - 1],
                "labels": ["a", "b"],
                "methods": ["mean", "max"],
            })

    def exec(self):
        self.confirmed = True
        return 1


def test_entry_zonegrouped_and_bare(monkeypatch=None):
    import processing.pulse_feature_extraction as mod
    from utils.zones import wrap_zones

    saved = mod._SegmentDialog
    mod._SegmentDialog = _FakeDialog
    try:
        # Two zones -> {num_zones, zones:[...]}
        tpz0 = _make_tpz(np.linspace(0, 1, 30), [10, 20], ["mean"])
        tpz1 = _make_tpz(np.linspace(0, 1, 50), [15, 30], ["mean"])
        tp2 = wrap_zones([tpz0, tpz1])
        out2 = mod.pulse_feature_extraction(tp2, 1000)
        assert isinstance(out2, dict) and out2.get("num_zones") == 2
        assert len(out2["zones"]) == 2
        assert out2["zones"][0]["signal"].shape == (30,)
        assert out2["zones"][1]["signal"].shape == (50,)
        for zr in out2["zones"]:
            for key in ("bounds", "labels", "methods", "lines_x", "signal",
                        "peak_locations", "segment_start_peaks",
                        "segment_numbers", "segment_names"):
                assert key in zr

        # One zone -> bare segment dict (same keys as legacy output).
        tp1 = wrap_zones([_make_tpz(np.linspace(0, 1, 30), [10, 20], ["mean"])])
        out1 = mod.pulse_feature_extraction(tp1, 1000)
        assert "num_zones" not in out1
        for key in ("bounds", "labels", "methods", "lines_x", "signal",
                    "peak_locations", "segment_start_peaks",
                    "segment_numbers", "segment_names"):
            assert key in out1
    finally:
        mod._SegmentDialog = saved


def test_ok_finish_gating():
    """Non-last zones show OK; last zone shows Finish, enabled only once every
    other zone is accepted; single zone shows an enabled Finish."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_feature_extraction import _SegmentDialog
    from utils.zones import wrap_zones

    tp = wrap_zones([
        _make_tpz(np.linspace(0, 1, 40), [10, 20], ["mean"] * 3),
        _make_tpz(np.linspace(0, 1, 40), [12, 24], ["mean"] * 3),
        _make_tpz(np.linspace(0, 1, 40), [15, 30], ["mean"] * 3),
    ])
    dlg = _SegmentDialog(zones=_zone_inputs(tp), active_zone=0)
    assert dlg._btn_finish.text() == "OK" and dlg._btn_finish.isEnabled()
    dlg._on_zone_changed(2)  # last zone, none accepted yet
    assert dlg._btn_finish.text() == "Finish" and not dlg._btn_finish.isEnabled()
    dlg._on_zone_changed(0)
    dlg._on_finish()  # OK zone 0 -> advance to 1
    assert dlg._active_zone == 1 and dlg._zones[0]["accepted"]
    dlg._on_finish()  # OK zone 1 -> advance to 2 (last)
    assert dlg._active_zone == 2 and dlg._btn_finish.text() == "Finish"
    assert dlg._btn_finish.isEnabled()  # zones 0,1 accepted
    dlg.close()

    # Single zone: Finish enabled, no selector.
    d1 = _SegmentDialog(zones=_zone_inputs(wrap_zones([
        _make_tpz(np.linspace(0, 1, 40), [20], ["mean", "mean"])])),
        active_zone=0)
    assert d1._zone_btn_group is None
    assert d1._btn_finish.text() == "Finish" and d1._btn_finish.isEnabled()
    d1.close()


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
