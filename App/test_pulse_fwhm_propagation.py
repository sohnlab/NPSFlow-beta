"""FWHM edge method propagates from the template into per-pulse processing.

Covers the pure boundary mapping (_refine_pulse_bounds) and the _compute_zone
path that produces the exported per-pulse boundaries + rect segmentation.
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from processing.extract_single_pulse_features import (
    _refine_pulse_bounds, _compute_zone)


def _box_pulse(baseline=0.0, depth=-1.0, ramp=21, plateau=120, pad=140):
    """One box pulse: flat baseline, step down to a plateau, step up back.

    Returns (signal, down_apex, up_apex) with half-cosine transitions so the
    derivative has one clean min (down) and one clean max (up).
    """
    def ramp_seg(a, b):
        k = np.arange(1, ramp)
        return a + (b - a) * (1 - np.cos(np.pi * k / ramp)) / 2
    parts = [np.full(pad, baseline)]
    down_apex = len(parts[0]) - 1 + (ramp + 1) // 2
    parts.append(ramp_seg(baseline, depth))
    parts.append(np.full(plateau, depth))
    up_start = sum(len(p) for p in parts) - 1
    up_apex = up_start + (ramp + 1) // 2
    parts.append(ramp_seg(depth, baseline))
    parts.append(np.full(pad, baseline))
    return np.concatenate(parts), down_apex, up_apex


def test_refine_pulse_bounds_fwhm_shifts_apex_peaks_stay():
    sig, da, ua = _box_pulse()
    s = 1000                                   # arbitrary global offset
    gp = np.array([da, ua]) + s                # global apex peaks
    peaks = _refine_pulse_bounds(gp, sig, s, "peaks")
    fwhm = _refine_pulse_bounds(gp, sig, s, "fwhm")
    assert list(peaks) == list(gp)             # peaks mode untouched
    # down edge: crossing of half depth (-0.5) on baseline->depth ramp = apex
    # (symmetric), up edge likewise -> both near apex here (symmetric box)
    assert len(fwhm) == 2
    assert abs(int(fwhm[0]) - int(gp[0])) <= 2
    assert abs(int(fwhm[1]) - int(gp[1])) <= 2


def test_refine_pulse_bounds_asymmetric_shifts_off_apex():
    # baseline(0) -> deep(-1.0) -> shoulder(-0.3) -> baseline(0). At the
    # deep->shoulder boundary the LOWER segment is the deep plateau, whose
    # baseline is its higher flank = the 0 baseline (not the -0.3 shoulder),
    # so its half level -0.5 sits at 71% of the -1.0->-0.3 ramp: the FWHM
    # boundary is pulled several samples off the apex toward the shoulder.
    ramp = 21
    def ramp_seg(a, b):
        k = np.arange(1, ramp)
        return a + (b - a) * (1 - np.cos(np.pi * k / ramp)) / 2
    parts = [np.full(120, 0.0), ramp_seg(0.0, -1.0), np.full(120, -1.0),
             ramp_seg(-1.0, -0.3), np.full(60, -0.3),
             ramp_seg(-0.3, 0.0), np.full(120, 0.0)]
    sig = np.concatenate(parts)
    apex, off = [], 0
    for seg in parts:
        if len(seg) == ramp - 1:               # a ramp segment
            apex.append(off + (ramp + 1) // 2 - 1)
        off += len(seg)
    gp = np.array(apex, dtype=int)             # 3 apex boundaries
    fwhm = _refine_pulse_bounds(gp, sig, 0, "fwhm")
    peaks = _refine_pulse_bounds(gp, sig, 0, "peaks")

    assert list(peaks) == list(gp)             # peaks mode identical
    assert np.all(np.diff(fwhm) > 0)
    # deep->shoulder boundary (index 1) shifts toward the shoulder (higher x)
    frac = (-0.5 - -1.0) / (-0.3 - -1.0)       # 0.714
    expected = gp[1] - (ramp + 1) // 2 + 1 + ramp / np.pi * np.arccos(1 - 2 * frac)
    assert int(fwhm[1]) > int(gp[1]) + 2       # meaningfully off the apex
    assert abs(int(fwhm[1]) - expected) <= 2


def _box_zone(edge_method):
    """Run _compute_zone on one box pulse; return (result, apex_peaks_local)."""
    sig, da, ua = _box_pulse()
    n = len(sig)
    data = sig.copy()
    pulse_indices = np.array([[0, n - 1]], dtype=int)
    tp = dict(
        threshold={"upper": 0.002, "lower": -0.002},
        edge_margin={"first_peak_prop": 0.0, "last_peak_prop": 0.0},
        peak_counts={"positive": 1, "negative": 1},
        peak_sequence=[0, 1],
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],
        key_peaks=[], key_peak_properties=[],
        peak_locations=np.array([], dtype=float),
        template_original=np.linspace(-1.0, 0.0, 30),
        exclusion_zones=[], end_zones={},
        key_peak_seq_indices_in=[], exclusion_zone_seq_bounds_in=[],
        edge_method=edge_method,
    )
    return _compute_zone(data, 1000.0, None, pulse_indices, **tp), (da, ua)


def test_compute_zone_returns_edge_method_for_per_zone_export():
    # Each zone's compute dict carries its own edge_method so the review
    # dialog refines every zone's exported boundaries with the right method
    # (channels can have different template edge methods).
    r_pk, _ = _box_zone("peaks")
    r_fw, _ = _box_zone("fwhm")
    assert r_pk["edge_method"] == "peaks"
    assert r_fw["edge_method"] == "fwhm"


def test_compute_zone_apex_locs_unchanged_by_edge_method():
    r_pk, _ = _box_zone("peaks")
    r_fw, _ = _box_zone("fwhm")
    # pulse_peak_locs stay at apex positions in BOTH modes (markers/sequence)
    assert list(r_pk["pulse_peak_locs"][0]) == list(r_fw["pulse_peak_locs"][0])


def test_compute_zone_rect_boundaries_track_edge_method():
    r_pk, _ = _box_zone("peaks")
    r_fw, _ = _box_zone("fwhm")
    # The rect step positions are where the rectangularized signal changes
    # level; compare the two modes. For a symmetric box they can coincide, so
    # assert the rect arrays are valid 3-segment step functions in both and
    # the segment count matches the apex boundary count + 1.
    for r in (r_pk, r_fw):
        rect = np.asarray(r["rect_pulses"][0], dtype=float)
        steps = np.count_nonzero(np.diff(rect) != 0)
        assert steps == len(r["pulse_peak_locs"][0])   # one step per boundary


def test_key_peaks_survive_fwhm_via_apex_positions():
    # Review finding: template key_peaks are apex positions, but FWHM shifts
    # peak_locations away from them; matching key_peaks against the refined
    # peak_locations under a 5-sample tolerance silently drops them. Passing
    # peak_locations_apex (apex-to-apex match) fixes it. Template peak
    # geometry is what drives this resolution, so pass explicit arrays with a
    # boundary shifted well past the tolerance.
    from processing.extract_single_pulse_features import _compute_zone
    sig, da, ua = _box_pulse()
    n = len(sig)
    apex = np.array([da, ua], dtype=int)
    refined = apex + 12                         # simulate FWHM shift > tol
    tp_common = dict(
        threshold={"upper": 0.002, "lower": -0.002},
        edge_margin={"first_peak_prop": 0.0, "last_peak_prop": 0.0},
        peak_counts={"positive": 1, "negative": 1},
        peak_sequence=[0, 1],
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],
        key_peaks=[int(da)],                    # marked at the apex
        key_peak_properties=["min"],
        template_original=np.linspace(-1.0, 0.0, 30),
        exclusion_zones=[], end_zones={},
        key_peak_seq_indices_in=[], exclusion_zone_seq_bounds_in=[],
    )
    pulse_indices = np.array([[0, n - 1]], dtype=int)

    # With peak_locations_apex present, the apex key peak resolves.
    r_apex = _compute_zone(
        sig.copy(), 1000.0, None, pulse_indices,
        peak_locations=refined, peak_locations_apex=apex,
        edge_method="fwhm", **tp_common)
    assert list(r_apex["template_key_seq_indices"]) == [0]
    assert r_apex["template_key_polarities"] == ["min"]

    # Legacy settings (no apex field) fall back to refined peak_locations and
    # would drop the key peak — documents the pre-fix failure mode.
    r_legacy = _compute_zone(
        sig.copy(), 1000.0, None, pulse_indices,
        peak_locations=refined, peak_locations_apex=None,
        edge_method="fwhm", **tp_common)
    assert list(r_legacy["template_key_seq_indices"]) == []


def _multilevel_pulse():
    """baseline -> deep -> shoulder -> baseline; FWHM shifts the deep->shoulder
    boundary off the apex. Returns (sig, tp_params, pulse_indices)."""
    ramp = 41
    def rs(a, b):
        k = np.arange(1, ramp)
        return a + (b - a) * (1 - np.cos(np.pi * k / ramp)) / 2
    sig = np.concatenate([
        np.full(200, 0.0), rs(0, -1.0), np.full(200, -1.0),
        rs(-1.0, -0.3), np.full(80, -0.3), rs(-0.3, 0.0), np.full(200, 0.0)])
    n = len(sig)
    tp = dict(
        threshold={"upper": 0.002, "lower": -0.002},
        edge_margin={"first_peak_prop": 0, "last_peak_prop": 0},
        peak_counts={"positive": 2, "negative": 1},
        peak_sequence=[0, 1, 1],
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[], key_peaks=[], key_peak_properties=[],
        peak_locations=np.array([], float),
        template_original=np.linspace(-1, 0, 30),
        exclusion_zones=[], end_zones={},
        key_peak_seq_indices_in=[], exclusion_zone_seq_bounds_in=[])
    return sig, tp, np.array([[0, n - 1]], dtype=int)


def _dialog_from(sig, tp, pidx, edge_method):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.extract_single_pulse_features import (
        _compute_zone, _PulseFeatureReviewDialog as D)
    c = _compute_zone(sig.copy(), 1000.0, None, pidx,
                      edge_method=edge_method, peak_locations_apex=None, **tp)
    c["data"] = sig.copy(); c["pulse_indices"] = pidx; c["trendline"] = None
    dlg = D(
        c["data"], pidx, c["filtered_pulses"], c["rect_pulses"],
        c["pulse_peak_locs"], c["pulse_diffs"], c["thresholds_used"],
        c["counts_used"], c["sequence_matches"], c["template_seq"], [0], 1000.0,
        pulse_excl_ranges=c["pulse_excl_ranges_list"],
        pulse_key_peaks=c["pulse_key_peaks_list"],
        pulse_key_peak_pols=c["pulse_key_peak_pols_list"],
        pulse_key_peak_seqs=c["pulse_key_peak_seqs_list"],
        template_key_seq_indices=c["template_key_seq_indices"],
        template_key_polarities=c["template_key_polarities"],
        trendline=None, filter_config=None,
        pad_length=c["pad_length"], pad_method=c["pad_method"],
        rect_pulses_mean=c["rect_pulses_mean"],
        zones=[c], active_zone=0, edge_method=edge_method)
    dlg._detrend = False
    dlg.acceptance[:] = 1
    return dlg


def _export(dlg):
    return list(np.asarray(
        dlg.zone_results()[0]["pulse_peak_locations"][0], dtype=int))


def test_dialog_toggle_reslices_and_round_trips():
    sig, tp, pidx = _multilevel_pulse()
    dlg = _dialog_from(sig, tp, pidx, "peaks")
    assert dlg._edge_method == "peaks"
    assert dlg._rb_edge_peaks.isChecked()
    peaks_export = _export(dlg)

    dlg._rb_edge_fwhm.setChecked(True)
    dlg._on_edge_method_changed()
    assert dlg._edge_method == "fwhm"
    fwhm_export = _export(dlg)
    rect = np.asarray(dlg.zone_results()[0]["rectangularized_pulses"][0], float)
    steps = sorted(int(x) for x in np.where(np.diff(rect) != 0)[0] + 1)

    assert fwhm_export != peaks_export           # boundaries moved
    assert sorted(fwhm_export) == steps          # export matches rect steps

    dlg._rb_edge_peaks.setChecked(True)
    dlg._on_edge_method_changed()
    assert _export(dlg) == peaks_export          # idempotent round-trip
    dlg.close()


def test_dialog_toggle_default_reflects_template():
    sig, tp, pidx = _multilevel_pulse()
    dlg = _dialog_from(sig, tp, pidx, "fwhm")
    assert dlg._edge_method == "fwhm"
    assert dlg._rb_edge_fwhm.isChecked()
    dlg.close()


def test_dialog_edge_method_persists_in_state():
    sig, tp, pidx = _multilevel_pulse()
    dlg = _dialog_from(sig, tp, pidx, "peaks")
    dlg._rb_edge_fwhm.setChecked(True)
    dlg._on_edge_method_changed()
    st = dlg._get_state()
    assert st["edge_method"] == "fwhm"

    dlg2 = _dialog_from(sig, tp, pidx, "peaks")
    dlg2._apply_state(st)
    assert dlg2._edge_method == "fwhm"
    assert dlg2._rb_edge_fwhm.isChecked()
    dlg.close(); dlg2.close()


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
