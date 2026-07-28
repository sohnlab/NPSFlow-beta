"""Coincident review dialog — headless construction + results. No exec()."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from test_coincident_compute import _tpz, _capture


def _make():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import processing.extract_coincident_pulse_features as mod
    from utils.zones import zone_columns
    data, regions = _capture()
    cols = zone_columns(data)
    decomp, compute = mod.recompute(cols, 1, 0, 2, _tpz(), 1000.0, regions)
    dlg = mod._CoincidentReviewDialog(
        compute=compute, decomp=decomp, cols=cols,
        measurement_zone=1, start_zone=0, end_zone=2,
        tp=_tpz(), fs=1000.0, pulse_indices=regions)
    return mod, dlg


def test_dialog_constructs_and_lists_particles():
    mod, dlg = _make()
    try:
        assert dlg._n_particles == 2
        assert dlg._measurement_zone == 1
        # role dropdowns exist and reflect the chosen roles
        assert dlg._cmb_measurement.currentData() == 1
        assert dlg._cmb_start.currentData() == 0
        assert dlg._cmb_end.currentData() == 2
    finally:
        dlg.close()


def test_dialog_results_shape_and_acceptance():
    mod, dlg = _make()
    try:
        dlg.acceptance[:] = 1
        res = dlg.results()
        assert len(res["pulse_peak_locations"]) == 2
        assert np.asarray(res["acceptance_id"]).dtype == bool
        assert np.all(res["acceptance_id"] == True)   # noqa: E712
        assert len(res["coincidence_map"]) == 2
        assert res["coincidence_map"][0]["N"] == 2
    finally:
        dlg.close()


def test_role_change_redecomposes():
    mod, dlg = _make()
    try:
        before = dlg._n_particles
        # Drop the start+end indicators -> fallback mode; particle count may change
        i_none_start = dlg._cmb_start.findData(None)
        i_none_end = dlg._cmb_end.findData(None)
        dlg._cmb_start.setCurrentIndex(i_none_start)
        dlg._cmb_end.setCurrentIndex(i_none_end)
        dlg._on_roles_changed()
        assert dlg._start_zone is None and dlg._end_zone is None
        # recomputed compute/decomp are internally consistent
        assert dlg._n_particles == len(dlg._compute["pulse_peak_locs"])
        assert len(dlg.acceptance) == dlg._n_particles
    finally:
        dlg.close()


def test_manual_N_override_resplits_event():
    mod, dlg = _make()
    try:
        # Force event 0 to N=3 via the programmatic override hook.
        dlg.set_event_n(event=0, n=3)
        # event 0 now contributes 3 recovered particles
        counts = [m for m in dlg._decomp["source_event"] if m == 0]
        assert len(counts) == 3
        assert dlg._n_particles == len(dlg._compute["pulse_peak_locs"])
        assert len(dlg.acceptance) == dlg._n_particles
    finally:
        dlg.close()


def test_window_edit_updates_span():
    mod, dlg = _make()
    try:
        ri = np.asarray(dlg._decomp["recovered_indices"]).copy()
        # move particle 0's start later by 15 samples
        dlg.set_particle_window(particle=0, start=int(ri[0, 0] + 15),
                                end=int(ri[0, 1]))
        got = np.asarray(dlg._decomp["recovered_indices"])[0, 0]
        assert got == ri[0, 0] + 15
        # compute re-ran for the edited span
        assert len(dlg._compute["pulse_peak_locs"]) == dlg._n_particles
    finally:
        dlg.close()


def test_window_edit_preserves_other_particle_acceptance():
    mod, dlg = _make()
    try:
        # reject particle 1, keep particle 0 accepted
        dlg.acceptance[:] = 1
        dlg.acceptance[1] = 0
        ri = np.asarray(dlg._decomp["recovered_indices"]).copy()
        # edit particle 0's window; particle 1's span is unchanged
        dlg.set_particle_window(particle=0, start=int(ri[0, 0] + 10),
                                end=int(ri[0, 1]))
        assert dlg.acceptance[1] == 0, "unchanged particle's rejection was lost"
    finally:
        dlg.close()


def _make2():
    """Two coincident events (regions), each decomposed into 2 particles."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import processing.extract_coincident_pulse_features as mod
    from utils.zones import zone_columns
    n = 4000
    x = np.arange(n)
    rng = np.random.default_rng(5)
    meas = 0.2 * rng.standard_normal(n)
    start_ind = np.zeros(n)
    end_ind = np.zeros(n)
    sig = lambda c: 30.0 / (1 + np.exp(np.clip(-(x - c) / 3.0, -60, 60)))
    for off in (0, 2000):
        for c0, c1 in [(560 + off, 700 + off), (650 + off, 860 + off)]:
            meas += sig(c0) - sig(c1)
        start_ind += 50.0 * np.exp(-((x - 520 - off) ** 2) / (2 * 3 ** 2)) \
            + 50.0 * np.exp(-((x - 610 - off) ** 2) / (2 * 3 ** 2))
        end_ind += 50.0 * np.exp(-((x - 780 - off) ** 2) / (2 * 3 ** 2)) \
            + 50.0 * np.exp(-((x - 900 - off) ** 2) / (2 * 3 ** 2))
    data = np.column_stack([start_ind, meas, end_ind])
    regions = np.array([[500, 950], [2500, 2950]], dtype=int)
    cols = zone_columns(data)
    decomp, compute = mod.recompute(cols, 1, 0, 2, _tpz(), 1000.0, regions)
    dlg = mod._CoincidentReviewDialog(
        compute=compute, decomp=decomp, cols=cols,
        measurement_zone=1, start_zone=0, end_zone=2,
        tp=_tpz(), fs=1000.0, pulse_indices=regions)
    return mod, dlg


def test_event_resplit_preserves_other_events_edits():
    """Re-splitting one event (the Update button) must not touch another
    event's committed template edits, locks, or manual peaks."""
    mod, dlg = _make2()
    try:
        events = dict(dlg._events)
        assert set(events) == {0, 1}
        pid = events[1][-1]  # a particle of event 1
        s, e = (int(v) for v in dlg._decomp["recovered_indices"][pid])
        levels = [1.5, -2.5]
        dlg._tpl_edit[pid] = {"edges": [float(s), (s + e) / 2.0, float(e)],
                              "levels": list(levels)}
        dlg._tpl_locks[pid] = {1}
        manual_pk = (s + e) // 2 + 7
        locs = np.asarray(dlg._compute["pulse_peak_locs"][pid], int)
        dlg._compute["pulse_peak_locs"][pid] = np.sort(
            np.append(locs[locs != manual_pk], manual_pk))

        dlg.set_event_n(event=0, n=3)  # edit the OTHER event

        new_pids = dict(dlg._events)[1]
        carried = [p for p in new_pids
                   if dlg._tpl_edit.get(p, {}).get("levels") == levels]
        assert carried, "event 1's committed template edit was lost"
        assert dlg._tpl_locks.get(carried[0]) == {1}, "template lock was lost"
        assert any(manual_pk in np.asarray(dlg._compute["pulse_peak_locs"][p], int)
                   for p in new_pids), "event 1's manual peak was lost"
    finally:
        dlg.close()


def test_window_edit_preserves_other_events_edits():
    """Dragging one particle's span must not clear another event's edits."""
    mod, dlg = _make2()
    try:
        events = dict(dlg._events)
        pid_b = events[1][0]
        s, e = (int(v) for v in dlg._decomp["recovered_indices"][pid_b])
        dlg._tpl_edit[pid_b] = {"edges": [float(s), (s + e) / 2.0, float(e)],
                                "levels": [3.0, -1.0]}
        p0 = events[0][0]
        ri = np.asarray(dlg._decomp["recovered_indices"]).copy()
        dlg.set_particle_window(particle=p0, start=int(ri[p0, 0] + 10),
                                end=int(ri[p0, 1]))
        assert dlg._tpl_edit.get(pid_b, {}).get("levels") == [3.0, -1.0], \
            "event 1's committed template edit was lost"
    finally:
        dlg.close()


def test_output_rects_page_invariant_with_detrend():
    """Detrended output rects must use each particle's OWN event baseline,
    so the exported rects cannot depend on which page is displayed."""
    mod, dlg = _make2()
    try:
        dlg._chk_detrend.setChecked(True)
        dlg._page = 0
        dlg._render_page()
        r0 = [np.asarray(r, float).copy() for r in dlg._rects_out()]
        dlg._page = 1
        dlg._render_page()
        r1 = [np.asarray(r, float).copy() for r in dlg._rects_out()]
        assert len(r0) == len(r1)
        for a, b in zip(r0, r1):
            np.testing.assert_allclose(a, b)
    finally:
        dlg.close()


def test_save_for_later_saves_and_closes():
    mod, dlg = _make()
    try:
        closed = []
        dlg._autosave = lambda: True  # don't touch the real autosave file
        dlg.reject = lambda: closed.append(True)
        dlg._on_save()
        assert dlg.saved_for_later is True
        assert not dlg.confirmed
        assert closed, "Save for later must close the dialog"
        # a failed write keeps the dialog open and the flag unset
        dlg.saved_for_later = False
        closed.clear()
        dlg._autosave = lambda: False
        dlg._on_save()
        assert not dlg.saved_for_later and not closed
    finally:
        dlg.close()


def test_save_for_later_stops_pipeline_with_clear_message():
    import processing.extract_coincident_pulse_features as mod

    class _SavedFake:
        def __init__(self, **kw):
            self.confirmed = False
            self.saved_for_later = True
        def load_state_from_file(self, path): pass
        def exec(self): return 0

    saved = mod._CoincidentReviewDialog
    mod._CoincidentReviewDialog = _SavedFake
    try:
        data, regions = _capture()
        try:
            mod.extract_coincident_pulse_features(
                data_in=data, tp=_tpz(), sample_rate=1000.0,
                pulse_indices=regions,
                measurement_zone=1, start_zone=0, end_zone=2)
            assert False, "saved-for-later must stop the pipeline"
        except ValueError as e:
            assert "saved" in str(e).lower()
    finally:
        mod._CoincidentReviewDialog = saved


def test_runner_captures_on_save_for_later():
    """Save-for-later must stash the autosave into block.lastSettings before
    the pipeline stops — otherwise the next run's restore() rolls temp.json
    back to the workflow's stale snapshot and the saved progress is lost."""
    import processing.extract_coincident_pulse_features as mod
    import runners.runExtractCoincidentPulseFeatures as runner
    import utils.lastsettings_capture as lsc

    class _SavedFake:
        def __init__(self, **kw):
            self.confirmed = False
            self.saved_for_later = True
        def load_state_from_file(self, path): pass
        def exec(self): return 0

    class _Blk:
        parameters = {}

    data, regions = _capture()
    captured = []
    saved_dlg, saved_cap = mod._CoincidentReviewDialog, lsc.capture
    mod._CoincidentReviewDialog = _SavedFake
    lsc.capture = lambda block, path: captured.append(path)
    try:
        try:
            runner.run({"dataIn": data, "pulseLabels": regions,
                        "TPsettings": _tpz(), "sampleRate": 1000.0}, {}, _Blk())
            assert False, "runner must re-raise SavedForLater"
        except mod.SavedForLater:
            pass
        assert captured, "runner must capture the saved state before raising"
    finally:
        mod._CoincidentReviewDialog = saved_dlg
        lsc.capture = saved_cap


def test_accept_boxes_survive_page_change():
    """The accept-checkbox scroll area must keep a non-zero height after a
    page-change rebuild on a shown dialog (layout sizeHint reads 0 for
    pending-show widgets, which collapsed it)."""
    from PySide6.QtWidgets import QApplication
    mod, dlg = _make2()
    try:
        dlg.show()
        QApplication.processEvents()
        assert dlg._accept_scroll.height() > 0
        dlg._on_next()
        QApplication.processEvents()
        assert len(dlg._accept_boxes) == 2
        assert dlg._accept_scroll.height() > 0, \
            "accept checkboxes collapsed after page change"
        assert all(c.isVisible() for c in dlg._accept_boxes)
    finally:
        dlg.close()


def test_snap_edges_also_snaps_levels():
    """Snap Edges pins interior edges to the raw-rect steps AND each segment
    level to the overlap-dominant raw-rect level; locked levels hold."""
    mod, dlg = _make()
    try:
        dlg._active_tpl = 0
        dlg._enlarge_panel = object()  # skip the full re-render
        dlg._render_enlarge = lambda: None
        dlg._raw_rect = (np.array([0.0, 10.0, 40.0, 50.0]),
                         np.array([0.0, 2.0, 0.5]))
        dlg._tpl_edit[0] = {"edges": [0.0, 12.0, 45.0, 50.0],
                            "levels": [0.3, 1.7, 0.4]}
        dlg._tpl_locks[0] = {0}  # level 0 pinned
        dlg._snap_active_to_raw()
        ed = dlg._tpl_edit[0]
        assert ed["edges"][1] == 10.0
        assert ed["edges"][2] == 40.0
        assert ed["levels"][0] == 0.3   # locked: untouched
        assert ed["levels"][1] == 2.0   # [10,40] -> raw step [10,40]
        assert ed["levels"][2] == 0.5   # [40,50] -> raw step [40,50]

        # a segment spanning several steps takes the overlap-dominant one
        dlg._tpl_locks.pop(0)
        dlg._tpl_edit[0] = {"edges": [0.0, 5.0, 45.0, 50.0],
                            "levels": [0.1, 0.9, 0.4]}
        dlg._snap_active_to_raw()
        assert dlg._tpl_edit[0]["levels"][1] == 2.0  # [5,45]: 30 of 40 on step 2
    finally:
        dlg._enlarge_panel = None
        dlg.close()


def test_regenerate_resets_event_templates():
    mod, dlg = _make()
    try:
        pids = list(dlg._event_pids)
        assert pids, "current event should have particles"
        for p in pids:
            dlg._tpl_edit[p] = {"edges": [0.0, 1.0, 2.0], "levels": [0.0, 1.0]}
            dlg._tpl_locks[p] = {0}
        dlg._regenerate_templates()
        assert not any(p in dlg._tpl_edit for p in pids)
        assert not any(p in dlg._tpl_locks for p in pids)
    finally:
        dlg.close()


def test_ramped_staircase_softens_edges():
    """The residual subtraction uses an edge-ramped staircase: identical to
    the ideal staircase away from edges, linear through each transition."""
    mod, dlg = _make()
    try:
        edges, levels = [0.0, 100.0, 200.0], [0.0, 1.0]
        x = np.arange(200)
        y = dlg._eval_staircase_ramped(edges, levels, x, w=10, outside=0.0)
        ideal = dlg._eval_staircase(edges, levels, x, 0.0)
        np.testing.assert_allclose(y[50:90], ideal[50:90])
        np.testing.assert_allclose(y[110:150], ideal[110:150])
        assert abs(y[100] - 0.5) < 1e-9      # edge centre = halfway
        assert np.all(np.diff(y[94:107]) >= 0)  # monotonic ramp, no spike
    finally:
        dlg.close()


def test_template_warp_anchors_key_peaks():
    """The seed warp must pin the template's key peaks (start/max/min/end)
    onto the pulse's same-role peaks and stretch linearly in between."""
    mod, dlg = _make()
    try:
        tpk = np.array([10, 30, 60, 100])
        dlg._compute["template_seq"] = [1, 1, 0, 0]
        dlg._compute["template_key_seq_indices"] = [1, 2]
        dlg._compute["template_key_polarities"] = ["max", "min"]
        dlg._compute["pulse_peak_locs"][0] = np.array([1000, 1200, 1500, 2000])
        dlg._peak_roles = {1000: "start", 1200: "max",
                           1500: "min", 2000: "end"}
        anchors = dlg._tpl_warp_anchors(0, tpk, 1000, 2000)
        assert anchors == [(10.0, 1000.0), (30.0, 1200.0),
                           (60.0, 1500.0), (100.0, 2000.0)]
        t_a, g_a = zip(*anchors)
        # key peaks land exactly; between-anchor points scale proportionally
        np.testing.assert_allclose(
            dlg._piecewise_x(tpk, t_a, g_a), [1000, 1200, 1500, 2000])
        np.testing.assert_allclose(
            dlg._piecewise_x([45.0], t_a, g_a), [1350.0])
        # flanks extrapolate with the end segments' slopes
        np.testing.assert_allclose(
            dlg._piecewise_x([0.0], t_a, g_a), [900.0])

        # an out-of-order pulse anchor is dropped, keeping the map monotonic
        dlg._peak_roles = {1000: "start", 1600: "max",
                           1400: "min", 2000: "end"}
        dlg._compute["pulse_peak_locs"][0] = np.array([1000, 1400, 1600, 2000])
        anchors = dlg._tpl_warp_anchors(0, tpk, 1000, 2000)
        assert anchors == [(10.0, 1000.0), (30.0, 1600.0), (100.0, 2000.0)]
    finally:
        dlg.close()


def test_state_roundtrip_preserves_roles_and_edits():
    mod, dlg = _make()
    try:
        dlg.set_event_n(event=0, n=3)
        dlg.acceptance[:] = 0
        dlg.acceptance[1] = 1
        st = dlg._get_state()
    finally:
        dlg.close()
    assert st["measurement_zone"] == 1
    assert st["start_zone"] == 0 and st["end_zone"] == 2
    assert len(st["recovered_indices"]) == dlg._n_particles

    mod2, dlg2 = _make()
    try:
        dlg2._apply_state(st)
        assert dlg2._measurement_zone == 1
        assert np.asarray(dlg2._decomp["recovered_indices"]).shape[0] == len(st["recovered_indices"])
        assert list(np.asarray(dlg2.acceptance)) == st["acceptance"]
    finally:
        dlg2.close()


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
