"""FWHM edge slicing: boundary refinement math + dialog integration."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import processing.pulse_template_processing as mod


def _ramp_template(levels, plateau=100, ramp=21):
    """Plateaus at *levels* joined by half-cosine transitions of *ramp*
    intervals (odd, so the derivative has one strict maximum — a linear ramp
    would give a flat derivative whose FP noise breaks find_peaks).

    Returns (signal, apex_idx) where apex_idx are the derivative apexes plus
    the 0 / n-1 sentinels — the same shape _update_all produces.
    """
    parts = [np.full(plateau, float(levels[0]))]
    apexes = []
    for lv_prev, lv in zip(levels, levels[1:]):
        p_last = sum(len(p) for p in parts) - 1   # last sample at lv_prev
        k = np.arange(1, ramp)
        parts.append(lv_prev + (lv - lv_prev)
                     * (1 - np.cos(np.pi * k / ramp)) / 2)
        apexes.append(p_last + (ramp + 1) // 2)
        parts.append(np.full(plateau, float(lv)))
    sig = np.concatenate(parts)
    all_idx = np.array([0] + apexes + [len(sig) - 1], dtype=int)
    return sig, all_idx


def _cos_cross(apex, lv_from, lv_to, level, ramp=21):
    """Sample position where the half-cosine ramp under *apex* crosses
    *level* (transition runs lv_from -> lv_to over *ramp* intervals)."""
    frac = (level - lv_from) / (lv_to - lv_from)
    k = ramp / np.pi * np.arccos(1 - 2 * frac)
    return apex - (ramp + 1) // 2 + k


def test_boundary_count_and_order_preserved():
    sig, idx = _ramp_template([2.0, 0.7, 1.5, -4.2, 1.0])
    out = mod._fwhm_refine_boundaries(sig, idx)
    assert len(out) == len(idx)
    assert np.all(np.diff(out) > 0)
    assert out[0] == 0 and out[-1] == len(sig) - 1


def test_symmetric_transition_keeps_apex():
    # baseline -> pore: the pore's half level is the transition midpoint,
    # crossing = ramp center = apex.
    sig, idx = _ramp_template([2.0, 0.7])
    out = mod._fwhm_refine_boundaries(sig, idx)
    assert abs(int(out[1]) - int(idx[1])) <= 1


def test_boundary_uses_lower_segment_half_level():
    # pore(0.7) -> node bump(1.7) inside a 2.0 baseline: the LOWER segment
    # is the pore, whose half level is (2.0+0.7)/2 = 1.35 — crossed at 81%
    # of the rise, past the apex. The bump's own half level (1.2, the
    # midpoint) would sit AT the apex, so this pins the lower-wins rule.
    sig, idx = _ramp_template([2.0, 0.7, 1.7, 0.7, 2.0])
    out = mod._fwhm_refine_boundaries(sig, idx)
    apex = int(idx[2])                       # pore -> node rise
    expected = _cos_cross(apex, 0.7, 1.7, 1.35)
    assert abs(int(out[2]) - expected) <= 2
    assert int(out[2]) >= apex + 2


def test_pulse_boundaries_at_its_half_depth():
    # bump(1.5) -> deep pulse(-4.2) -> after(1.0): both pulse boundaries use
    # the PULSE's half level (-1.35, baseline = its higher flank 1.5), i.e.
    # its exported width is its FWHM width. The v1 shallower-wins rule put
    # the left boundary at the bump corner (~apex-7); half-depth is ~apex.
    sig, idx = _ramp_template([2.0, 0.7, 1.5, -4.2, 1.0])
    out = mod._fwhm_refine_boundaries(sig, idx)

    left = _cos_cross(int(idx[3]), 1.5, -4.2, -1.35)
    right = _cos_cross(int(idx[4]), -4.2, 1.0, -1.35)
    assert abs(int(out[3]) - left) <= 2
    assert abs(int(out[4]) - right) <= 2
    corner = _cos_cross(int(idx[3]), 1.5, -4.2, 1.1)
    assert int(out[3]) - corner >= 4         # not the v1 bump-corner edge


def test_stair_step_falls_back_to_midpoint():
    # 2.0 -> 0.7 -> -4.2: the deep segment's half level (-1.75) equals the
    # midpoint here; the middle segment's (1.35) is outside the transition.
    # Boundary lands at the midpoint crossing = apex.
    sig, idx = _ramp_template([2.0, 0.7, -4.2])
    out = mod._fwhm_refine_boundaries(sig, idx)
    assert abs(int(out[2]) - int(idx[2])) <= 1


def test_boundary_stays_on_own_transition():
    # Short shallow bump (-0.3) between baseline (0) and a much deeper pulse
    # (-4.0): pass-2 mean contamination once dragged the baseline->bump
    # boundary onto the deep pulse's ramp. The apex-cell search window keeps
    # each boundary on its own transition.
    # short bump plateau stresses ramp contamination
    sig, idx = _ramp_template([0.0, -0.3, -4.0, 0.0], plateau=10, ramp=41)
    out = mod._fwhm_refine_boundaries(sig, idx)
    left_apex = int(idx[1])                  # baseline -> bump
    left_ramp_end = left_apex + (41 + 1) // 2
    # boundary must remain on the baseline->bump transition, not jump onto
    # the bump->deep ramp that starts ~a plateau later
    assert int(out[1]) <= left_ramp_end + 2
    assert np.all(np.diff(out) > 0)


def test_no_crossing_keeps_apex():
    # A constant signal never crosses any level: the boundary stays put.
    sig = np.full(300, 2.0)
    fake = np.array([0, 150, len(sig) - 1], dtype=int)
    out = mod._fwhm_refine_boundaries(sig, fake)
    assert int(out[1]) == 150


def test_short_input_passthrough():
    sig = np.ones(10)
    idx = np.array([0, 9])
    out = mod._fwhm_refine_boundaries(sig, idx)
    assert list(out) == [0, 9]


def _make_dialog(levels=(2.0, 0.7, 1.5, -4.2, 1.0)):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    sig, idx = _ramp_template(list(levels))
    dlg = mod._ThresholdDialog(sig.copy(), sig.copy(), 1000.0)
    # Well below the SMALLEST transition's derivative peak (a half-cosine
    # ramp's diff peaks at |dlv|*sin(pi/42) ~ |dlv|/13.4) so find_peaks
    # detects every transition, at the _ramp_template apex.
    step = float(np.min(np.abs(np.diff(np.asarray(levels, float))))) / 22.0
    dlg._pos_thresh = 0.5 * step
    dlg._neg_thresh = -0.5 * step
    dlg._update_all()
    return dlg, sig, idx


def test_dialog_fwhm_default_refines_boundaries():
    dlg, sig, idx = _make_dialog()
    assert dlg._edge_method == "fwhm"
    refined = mod._fwhm_refine_boundaries(sig, idx)
    assert list(dlg._last_all_idx) == list(refined)
    assert list(dlg._peak_locations) == list(refined[1:-1])
    # apex space preserved alongside, same count
    assert list(dlg._apex_locations) == list(idx[1:-1])
    assert len(dlg._peak_locations) == len(dlg._apex_locations)
    dlg.close()


def test_dialog_peaks_mode_matches_apex():
    dlg, sig, idx = _make_dialog()
    dlg._edge_method = "peaks"
    dlg._update_all()
    assert list(dlg._last_all_idx) == list(idx)
    assert list(dlg._peak_locations) == list(idx[1:-1])
    dlg.close()


def test_sequence_signs_stay_apex_based():
    dlg, sig, idx = _make_dialog()
    td = np.diff(sig)
    expected = [1 if td[a - 1] >= 0 else 0 for a in idx[1:-1]]
    assert list(dlg._peak_sequence) == expected
    dlg._edge_method = "peaks"
    dlg._update_all()
    assert list(dlg._peak_sequence) == expected
    dlg.close()


def test_methods_pair_with_peak_locations():
    # Downstream zips methods against [0] + peak_locations + [n-1].
    dlg, sig, idx = _make_dialog()
    res = dlg._build_result({}, {'pad_length': 0, 'method': 'replicate'})
    assert len(res["methods"]) == len(res["peak_locations"]) + 1
    dlg._segment_methods[int(dlg._last_all_idx[3])] = "median"
    res = dlg._build_result({}, {'pad_length': 0, 'method': 'replicate'})
    assert res["methods"][3] == "median"
    dlg.close()


def test_edge_method_in_result_and_roundtrip():
    dlg, sig, idx = _make_dialog()
    res = dlg._build_result({}, {'pad_length': 0, 'method': 'replicate'})
    assert res["edge_method"] == "fwhm"

    from utils.tpsettings_io import get_zone
    zone = get_zone(mod._wrap_zones([res]), 0)
    dlg2, _, _ = _make_dialog()
    dlg2._edge_method = "peaks"
    dlg2._apply_loaded_settings(zone)
    assert dlg2._edge_method == "fwhm"
    assert dlg2._radio_edge_fwhm.isChecked()
    dlg.close(); dlg2.close()


def test_legacy_settings_without_key_load_as_peaks():
    dlg, sig, idx = _make_dialog()
    res = dlg._build_result({}, {'pad_length': 0, 'method': 'replicate'})
    res.pop("edge_method")
    from utils.tpsettings_io import get_zone
    zone = get_zone(mod._wrap_zones([res]), 0)
    dlg2, _, _ = _make_dialog()
    dlg2._apply_loaded_settings(zone)
    assert dlg2._edge_method == "peaks"
    assert dlg2._radio_edge_peaks.isChecked()
    dlg.close(); dlg2.close()


def test_load_reslices_with_restored_edge_method():
    # A peaks-mode settings loaded into the default-FWHM dialog must re-slice:
    # _apply_loaded_settings' internal _update_all ran under the old method, so
    # without the trailing re-slice the boundaries stay FWHM while the radio
    # says Peaks (review finding: inconsistent Confirm export).
    dlg, sig, idx = _make_dialog()
    dlg._edge_method = "peaks"
    dlg._update_all()
    res_peaks = dlg._build_result({}, {'pad_length': 0, 'method': 'replicate'})
    assert res_peaks["edge_method"] == "peaks"

    from utils.tpsettings_io import get_zone
    zone = get_zone(mod._wrap_zones([res_peaks]), 0)
    dlg2, _, _ = _make_dialog()                # defaults to fwhm
    assert dlg2._edge_method == "fwhm"
    dlg2._apply_loaded_settings(zone)
    assert dlg2._edge_method == "peaks"
    # boundaries now reflect Peaks (apex), not the pre-load FWHM
    assert list(dlg2._last_all_idx) == list(idx)
    assert dlg2._radio_edge_peaks.isChecked()
    dlg.close(); dlg2.close()


def test_result_carries_apex_peak_locations():
    dlg, sig, idx = _make_dialog()
    res = dlg._build_result({}, {'pad_length': 0, 'method': 'replicate'})
    assert list(res["peak_locations_apex"]) == list(idx[1:-1])
    # in fwhm mode the exported boundaries differ from the apexes somewhere
    assert list(res["peak_locations"]) != list(res["peak_locations_apex"]) \
        or dlg._edge_method == "peaks"
    dlg.close()


def test_radio_click_reslices():
    dlg, sig, idx = _make_dialog()
    dlg._radio_edge_peaks.click()
    assert dlg._edge_method == "peaks"
    assert list(dlg._last_all_idx) == list(idx)
    dlg._radio_edge_fwhm.click()
    assert dlg._edge_method == "fwhm"
    assert list(dlg._last_all_idx) == list(
        mod._fwhm_refine_boundaries(sig, idx))
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
