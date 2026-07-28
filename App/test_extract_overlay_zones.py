"""All-zones overlay on the Pulse Processing (Single Pulse Feature Review)
left plots. The right (derivative) plots must stay on the active zone.

Mirrors the Detection Review "All Zones" toggle. Offscreen; no exec.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util

import numpy as np
from PySide6.QtWidgets import QApplication


def _load_synth():
    spec = importlib.util.spec_from_file_location(
        "tz", os.path.join(os.path.dirname(__file__),
                           "test_extract_zone_compute.py"))
    tz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tz)
    return tz


def _build_dialog(n_zones=2):
    from processing.extract_single_pulse_features import (
        _compute_zone, _PulseFeatureReviewDialog as D)
    tz = _load_synth()
    fs, data, pulse_indices, tp = tz._synthetic()
    QApplication.instance() or QApplication([])
    computes = []
    for k in range(n_zones):
        d = data * (1.0 - 0.2 * k) + 5.0 * k   # distinct per-zone baseline
        c = _compute_zone(d, fs, None, pulse_indices, **tp)
        c["data"] = d
        c["pulse_indices"] = pulse_indices
        c["trendline"] = None
        computes.append(c)
    r0 = computes[0]
    si = list(range(len(r0["pulse_peak_locs"])))
    return D(
        r0["data"], pulse_indices, r0["filtered_pulses"], r0["rect_pulses"],
        r0["pulse_peak_locs"], r0["pulse_diffs"], r0["thresholds_used"],
        r0["counts_used"], r0["sequence_matches"], r0["template_seq"], si, fs,
        pulse_excl_ranges=r0["pulse_excl_ranges_list"],
        pulse_key_peaks=r0["pulse_key_peaks_list"],
        pulse_key_peak_pols=r0["pulse_key_peak_pols_list"],
        pulse_key_peak_seqs=r0["pulse_key_peak_seqs_list"],
        template_key_seq_indices=r0["template_key_seq_indices"],
        template_key_polarities=r0["template_key_polarities"],
        trendline=None, filter_config=None,
        pad_length=r0["pad_length"], pad_method=r0["pad_method"],
        rect_pulses_mean=r0["rect_pulses_mean"],
        zones=computes, active_zone=0)


def test_multizone_overlay_default_on_and_checkbox():
    dlg = _build_dialog(2)
    assert dlg._overlay_all_zones is True
    assert dlg._chk_all_zones is not None and dlg._chk_all_zones.isChecked()


def test_overlay_adds_left_traces_right_unchanged():
    dlg = _build_dialog(2)

    dlg._overlay_all_zones = True
    dlg._render_page()
    left_on = len(dlg._axes_left[0].lines)
    right_on = len(dlg._axes_right[0].lines)

    dlg._overlay_all_zones = False
    dlg._render_page()
    left_off = len(dlg._axes_left[0].lines)
    right_off = len(dlg._axes_right[0].lines)

    # Left plots gain the other zone's overlay trace; right plots are identical.
    assert left_on > left_off
    assert right_on == right_off


def test_single_zone_has_no_overlay_control():
    dlg = _build_dialog(1)
    assert dlg._overlay_all_zones is False
    assert dlg._chk_all_zones is None
    dlg._render_page()  # must not raise


def test_zone_switch_still_works_with_overlay():
    dlg = _build_dialog(2)
    dlg._on_zone_changed(1)
    assert dlg._active_zone == 1


def test_overlay_draws_rect_for_other_zones():
    # Each non-active zone contributes a muted rect line ("#555555") to the
    # left plot, with detrending both off and on.
    dlg = _build_dialog(3)
    dlg._overlay_all_zones = True
    for detrend in (False, True):
        dlg._detrend = detrend
        dlg._render_page()
        muted = [ln for ln in dlg._axes_left[0].lines
                 if ln.get_color() == "#555555"]
        assert len(muted) == 2, (detrend, len(muted))  # 3 zones -> 2 overlays


def test_all_zones_detrended_in_output_no_display_shift():
    dlg = _build_dialog(3)
    dlg._detrend = True
    res = dlg.zone_results()
    # A non-active zone's output rect must equal a fresh per-zone detrend
    # (proves it's detrended too) and differ from its undetrended stored rect.
    z2 = dlg._zones[1]
    rect_med, rect_mean = dlg._detrended_rects_for_zone(z2)
    # Compare against whichever method the dialog currently exports.
    fresh = rect_mean if dlg._rect_method == "mean" else rect_med
    stored = (z2["_rect_mean"] if dlg._rect_method == "mean"
              else z2["_rect"])
    out = np.asarray(res[1]["rectangularized_pulses"][0], dtype=float)
    assert np.allclose(out, np.asarray(fresh[0], dtype=float))
    assert not np.allclose(out, np.asarray(stored[0], dtype=float))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
        passed += 1
    print(f"\n{passed}/{len(fns)} passed")
