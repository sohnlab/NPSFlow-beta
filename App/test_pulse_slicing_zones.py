"""Pulse Slicing multi-zone slicing + per-segment zone grouping.

Headless: monkeypatches the Qt dialog to a no-op so no GUI is created.
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from utils.zones import wrap_zones, zone_list
import runners.runPulseSlicing as mod


# --- segdef + batch-array fixtures -----------------------------------------

def _segdef(length, peaks, labels, methods):
    """A pulseSegments/pulseFeatures struct: two segments split at the peak."""
    bounds = []
    bnds = [0] + list(peaks) + [length - 1]
    for i in range(len(bnds) - 1):
        bounds.append([bnds[i], bnds[i + 1]])
    return {
        "bounds": np.asarray(bounds, dtype=int),
        "labels": list(labels),
        "methods": list(methods),
        "signal": np.linspace(0.0, 1.0, length),
        "peak_locations": np.asarray(peaks, dtype=int),
    }


def _rect_pulses(n_pulses, length):
    """n_pulses simple rectangularized pulse signals (bare arrays)."""
    return [np.linspace(0.0, float(p + 1), length) for p in range(n_pulses)]


_FIELD_KEYS = {
    "label", "method", "segmentId", "pulseIndex",
    "startLocal", "endLocal", "startGlobal", "endGlobal", "values",
}


# --- tests ------------------------------------------------------------------

def test_single_zone_bare_segdicts():
    saved = mod._plot_pulse_segments
    mod._plot_pulse_segments = lambda *a, **kw: None
    try:
        length = 20
        segdef = _segdef(length, [10], ["seg_a", "seg_b"], ["mean", "min"])
        rect = _rect_pulses(3, length)
        block = types.SimpleNamespace(parameters={})
        out = mod.run(
            {"pulseSegments": segdef, "rectangularizedPulses": rect},
            {}, block)

        assert isinstance(out, dict) and "num_zones" not in out
        assert set(out.keys()) == {"seg_a", "seg_b"}
        for field, sd in out.items():
            # Bare segdict, not zone-wrapped.
            assert isinstance(sd, dict)
            assert "zones" not in sd
            assert set(sd.keys()) == _FIELD_KEYS
            assert isinstance(sd["pulseIndex"], list)
            assert len(sd["values"]) == len(sd["pulseIndex"]) == 3
        assert out["seg_a"]["label"] == "seg_a"
        assert out["seg_a"]["method"] == "mean"
        assert out["seg_a"]["segmentId"] == 1
        assert out["seg_b"]["segmentId"] == 2
    finally:
        mod._plot_pulse_segments = saved


def test_single_zone_matches_legacy_via_slice_zone():
    """The bare output is exactly _slice_zone(zone0)'s segment_data."""
    length = 20
    segdef = _segdef(length, [10], ["seg_a", "seg_b"], ["mean", "min"])
    rect = _rect_pulses(3, length)
    sd_z, vis, labels = mod._slice_zone(segdef, rect, None, None, None)

    saved = mod._plot_pulse_segments
    mod._plot_pulse_segments = lambda *a, **kw: None
    try:
        block = types.SimpleNamespace(parameters={})
        out = mod.run(
            {"pulseSegments": segdef, "rectangularizedPulses": rect}, {}, block)
    finally:
        mod._plot_pulse_segments = saved

    assert set(out.keys()) == set(sd_z.keys())
    for f in out:
        assert out[f]["label"] == sd_z[f]["label"]
        assert out[f]["pulseIndex"] == sd_z[f]["pulseIndex"]
        assert out[f]["startLocal"] == sd_z[f]["startLocal"]
        assert all(np.array_equal(a, b)
                   for a, b in zip(out[f]["values"], sd_z[f]["values"]))


def test_two_zone_per_zone_ports():
    """Multi-zone: one output port per zone, each a plain segment struct."""
    saved = mod._plot_pulse_segments
    mod._plot_pulse_segments = lambda *a, **kw: None
    try:
        seg0 = _segdef(20, [10], ["seg_a", "seg_b"], ["mean", "min"])
        seg1 = _segdef(30, [15], ["seg_a", "seg_b"], ["max", "mean"])
        rect0 = _rect_pulses(3, 20)
        rect1 = _rect_pulses(4, 30)

        block = types.SimpleNamespace(parameters={})
        out = mod.run({
            "pulseSegments": wrap_zones([seg0, seg1]),
            "rectangularizedPulses": wrap_zones([rect0, rect1]),
        }, {}, block)

        assert set(out.keys()) == {"Zone 1", "Zone 2"}
        for zkey in ("Zone 1", "Zone 2"):
            zstruct = out[zkey]
            assert isinstance(zstruct, dict) and "zones" not in zstruct
            assert set(zstruct.keys()) == {"seg_a", "seg_b"}
            for sd in zstruct.values():
                # Plain segdict, NOT zone-wrapped — Segment Processing reads it
                # directly after the zone port is unpacked.
                assert set(sd.keys()) == _FIELD_KEYS
        # Zone 0 had 3 pulses, zone 1 had 4.
        assert len(out["Zone 1"]["seg_a"]["values"]) == 3
        assert len(out["Zone 2"]["seg_a"]["values"]) == 4
        # Methods come from each zone's own template, not zone 0's.
        assert out["Zone 1"]["seg_a"]["method"] == "mean"
        assert out["Zone 2"]["seg_a"]["method"] == "max"
    finally:
        mod._plot_pulse_segments = saved


def test_zones_with_different_segment_counts():
    """Each zone keeps its OWN segments — different counts are preserved
    (the reported bug: zone 0's segment set was forced onto every zone)."""
    saved = mod._plot_pulse_segments
    mod._plot_pulse_segments = lambda *a, **kw: None
    try:
        seg0 = _segdef(20, [10], ["a", "b"], ["mean", "min"])           # 2 segs
        seg1 = _segdef(40, [12, 26], ["x", "y", "z"],
                       ["mean", "mean", "max"])                          # 3 segs
        rect0 = _rect_pulses(3, 20)
        rect1 = _rect_pulses(3, 40)
        block = types.SimpleNamespace(parameters={})
        out = mod.run({
            "pulseSegments": wrap_zones([seg0, seg1]),
            "rectangularizedPulses": wrap_zones([rect0, rect1]),
        }, {}, block)
        assert set(out.keys()) == {"Zone 1", "Zone 2"}
        assert set(out["Zone 1"].keys()) == {"a", "b"}
        assert set(out["Zone 2"].keys()) == {"x", "y", "z"}
    finally:
        mod._plot_pulse_segments = saved


def test_broadcast_bare_batch_across_zones():
    """A single bare batch array broadcasts across all zones."""
    saved = mod._plot_pulse_segments
    mod._plot_pulse_segments = lambda *a, **kw: None
    try:
        seg0 = _segdef(20, [10], ["seg_a", "seg_b"], ["mean", "min"])
        seg1 = _segdef(20, [10], ["seg_a", "seg_b"], ["mean", "min"])
        rect = _rect_pulses(3, 20)  # bare, shared by both zones

        block = types.SimpleNamespace(parameters={})
        out = mod.run({
            "pulseSegments": wrap_zones([seg0, seg1]),
            "rectangularizedPulses": rect,
        }, {}, block)
        assert set(out.keys()) == {"Zone 1", "Zone 2"}
        assert len(out["Zone 1"]["seg_a"]["values"]) == 3
        assert len(out["Zone 2"]["seg_a"]["values"]) == 3
    finally:
        mod._plot_pulse_segments = saved


def test_zone_count_mismatch_raises():
    saved = mod._plot_pulse_segments
    mod._plot_pulse_segments = lambda *a, **kw: None
    try:
        seg0 = _segdef(20, [10], ["seg_a", "seg_b"], ["mean", "min"])
        seg1 = _segdef(20, [10], ["seg_a", "seg_b"], ["mean", "min"])
        seg2 = _segdef(20, [10], ["seg_a", "seg_b"], ["mean", "min"])
        rect0 = _rect_pulses(3, 20)
        rect1 = _rect_pulses(3, 20)
        block = types.SimpleNamespace(parameters={})
        raised = False
        try:
            mod.run({
                "pulseSegments": wrap_zones([seg0, seg1, seg2]),  # 3 zones
                "rectangularizedPulses": wrap_zones([rect0, rect1]),  # 2 zones
            }, {}, block)
        except ValueError:
            raised = True
        assert raised, "expected ValueError on zone-count mismatch"
    finally:
        mod._plot_pulse_segments = saved


def test_engine_label_lookup_segment_and_zone_port():
    """Mirror the engine's _seg_label: a segment dict shows its string label;
    a per-zone struct ({segLabel: segdict}) has no top-level string label, so
    the port shows its field name ("Zone N")."""
    def seg_label(value, field):
        zones = zone_list(value)
        first = zones[0] if zones else {}
        if isinstance(first, dict):
            lbl = first.get("label")
            if isinstance(lbl, str) and lbl:
                return lbl
        return field

    bare = {"label": "Bare Seg", "method": "mean"}
    assert seg_label(bare, "fld") == "Bare Seg"

    # Per-zone struct: keys are segment labels, values are segdicts; the
    # top-level dict has no string "label", so fall back to the field name.
    zone_struct = {"seg_a": {"label": "seg_a"}, "seg_b": {"label": "seg_b"}}
    assert seg_label(zone_struct, "Zone 1") == "Zone 1"

    # Falls back to the field name when label absent.
    assert seg_label({}, "fld") == "fld"


def _make_vis(n, length=500):
    out = []
    for _ in range(n):
        seg = np.zeros(length)
        seg[length // 2:] = 1.0
        out.append({"seg": seg,
                    "segments": [{"bounds": (10, 20), "segmentIdx": 1,
                                  "label": "a"}]})
    return out


def test_sliced_pulses_dialog_multizone_tabs():
    """Multi-zone sliced-pulses dialog has one tab per zone, each with its own
    pagination; switching tabs remembers each zone's page."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    dlg = mod._SlicedPulsesDialog(
        [_make_vis(10), _make_vis(5), _make_vis(12)],   # 10/5/12 pulses
        [3, 11, 6],                                      # per-zone seg counts
        [list("abc"), list("abcdefghijk"), list("abcdef")])
    try:
        assert dlg._n_zones == 3 and dlg._zone_bar.count() == 3
        assert dlg._zone_total_pages == [2, 1, 2]
        dlg._on_next()                       # zone 0 -> page 2
        assert dlg._current_page == 2
        dlg._zone_bar.setCurrentIndex(1)     # zone 1: own page, 11 seg colours
        assert dlg._active_zone == 1 and dlg._current_page == 1
        assert len(dlg._seg_colors) == 11
        assert not dlg._btn_next.isEnabled()  # single page
        dlg._zone_bar.setCurrentIndex(0)     # back -> remembered page 2
        assert dlg._current_page == 2
    finally:
        dlg.close()


def test_sliced_pulses_dialog_single_zone_no_tabs():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    dlg = mod._SlicedPulsesDialog([_make_vis(4)], [3], [list("abc")])
    try:
        assert dlg._zone_bar is None        # no tab bar for single zone
        assert dlg._n_zones == 1
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
