"""_compute_zone (per-pulse compute) shape/type contract. No dialog exec."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


# Keys the dialog constructor consumes from the compute result.
_EXPECTED_KEYS = {
    "data",
    "filtered_pulses", "rect_pulses", "rect_pulses_mean", "pulse_peak_locs",
    "pulse_diffs", "thresholds_used", "counts_used", "sequence_matches",
    "template_seq", "pulse_excl_ranges_list", "pulse_key_peaks_list",
    "pulse_key_peak_pols_list", "pulse_key_peak_seqs_list",
    "template_key_seq_indices", "template_key_polarities",
    "pad_length", "pad_method", "pulse_start_indices",
}


def _synthetic():
    """~3000-sample signal with a few pulse-like dips + 3 pulse regions."""
    fs = 1000.0
    n = 3000
    data = np.zeros(n, dtype=float)
    regions = [(300, 500), (1200, 1450), (2100, 2350)]
    rng = np.random.default_rng(0)
    data += 0.01 * rng.standard_normal(n)
    for (s, e) in regions:
        c = (s + e) // 2
        x = np.arange(s, e)
        # a dip with a small positive shoulder -> +/- derivative peaks
        data[s:e] += -np.exp(-((x - c) ** 2) / (2 * (((e - s) / 6) ** 2)))
    pulse_indices = np.array(regions, dtype=int)
    template_params = dict(
        threshold={"upper": 0.001, "lower": -0.001},
        edge_margin={"first_peak_prop": 0.0, "last_peak_prop": 0.0},
        peak_counts={"positive": 1, "negative": 1},
        peak_sequence=[0, 1],            # one neg then one pos
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],                # no filtering
        key_peaks=[],
        key_peak_properties=[],
        peak_locations=np.array([], dtype=float),
        template_original=np.linspace(-1.0, 0.0, 30),
        exclusion_zones=[],
        end_zones={},
        key_peak_seq_indices_in=[],
        exclusion_zone_seq_bounds_in=[],
    )
    return fs, data, pulse_indices, template_params


def _run():
    from processing.extract_single_pulse_features import _compute_zone
    fs, data, pulse_indices, tp = _synthetic()
    return _compute_zone(data, fs, None, pulse_indices, **tp), len(pulse_indices)


def test_returns_expected_keys():
    r, _ = _run()
    assert isinstance(r, dict)
    missing = _EXPECTED_KEYS - set(r.keys())
    assert not missing, f"missing keys: {missing}"


def test_per_pulse_array_lengths():
    r, n = _run()
    for key in ("filtered_pulses", "rect_pulses", "rect_pulses_mean",
                "pulse_peak_locs", "pulse_diffs", "thresholds_used",
                "counts_used", "pulse_excl_ranges_list", "pulse_key_peaks_list",
                "pulse_key_peak_pols_list", "pulse_key_peak_seqs_list"):
        assert len(r[key]) == n, f"{key}: len {len(r[key])} != n_pulses {n}"
    assert len(r["sequence_matches"]) == n
    assert len(r["pulse_start_indices"]) == n


def test_rect_entries_are_arrays():
    r, n = _run()
    for i in range(n):
        assert isinstance(r["rect_pulses"][i], np.ndarray), f"rect_pulses[{i}]"
        assert isinstance(r["rect_pulses_mean"][i], np.ndarray), f"rect_mean[{i}]"
        assert isinstance(r["filtered_pulses"][i], np.ndarray), f"filtered[{i}]"
        # median/mean rect mirror the filtered length
        assert len(r["rect_pulses"][i]) == len(r["filtered_pulses"][i])
        assert len(r["rect_pulses_mean"][i]) == len(r["filtered_pulses"][i])


def test_sequence_matches_dtype_and_values():
    r, n = _run()
    sm = np.asarray(r["sequence_matches"])
    assert sm.shape == (n,)
    assert set(np.unique(sm)).issubset({0, 1, 2})


def test_scalar_passthrough_params():
    r, _ = _run()
    assert r["pad_length"] == 10
    assert r["pad_method"] == "replicate"
    # template_seq resolved from peak_sequence=[0,1]
    assert np.asarray(r["template_seq"]).tolist() == [0, 1]


def test_start_indices_match_input():
    r, _ = _run()
    _, _, pulse_indices, _ = _synthetic()
    assert np.asarray(r["pulse_start_indices"]).tolist() == \
        pulse_indices[:, 0].tolist()


# ---------------------------------------------------------------------------
# Entry-level zone fan-out (mock the dialog; no GUI)
# ---------------------------------------------------------------------------

class _FakeDialog:
    """Stand-in for _PulseFeatureReviewDialog: confirms, accepts per defaults.

    Mirrors the attributes the entry reads for zone 0: ``confirmed``,
    ``acceptance`` (seeded from the passed sequence_matches), ``rect_method``.
    """
    def __init__(self, data, pulse_indices, *a, **kw):
        # Positional order after data/pulse_indices:
        # a[0]=filtered, a[1]=rect, a[2]=peak_locs, a[3]=diffs, a[4]=thr,
        # a[5]=counts, a[6]=sequence_matches, a[7]=template_seq,
        # a[8]=single_indices, a[9]=fs.
        sequence_matches = a[6]
        single_indices = a[8]
        self.acceptance = np.array(
            [sequence_matches[idx] for idx in single_indices], dtype=int)
        self.confirmed = True
        self.rect_method = "median"
        self._zones_in = kw.get("zones") or [None]

    def load_state_from_file(self, path):
        pass

    def exec(self):
        return 1

    def zone_results(self):
        # Mirror the real dialog: each zone accepts per its compute defaults.
        out = []
        for z in self._zones_in:
            if z is None:  # single-zone construction path (zones=None)
                seq = self.acceptance
                out.append({
                    "pulse_peak_locations": [],
                    "rectangularized_pulses": [],
                    "acceptance_id": np.asarray(seq) >= 1,
                    "pulse_start_indices": np.array([], dtype=int),
                })
                continue
            seq = np.asarray(z["sequence_matches"])
            out.append({
                "pulse_peak_locations": list(z["pulse_peak_locs"]),
                "rectangularized_pulses": list(z["rect_pulses"]),
                "acceptance_id": seq >= 1,
                "pulse_start_indices": np.asarray(
                    z["pulse_indices"])[:, 0].copy(),
            })
        return out


def _tpz(threshold_up=0.001):
    """One minimal flat per-zone tp dict (legacy single-zone shape)."""
    return dict(
        threshold={"upper": threshold_up, "lower": -threshold_up},
        edge_margin={"first_peak_prop": 0.0, "last_peak_prop": 0.0},
        peak_counts={"positive": 1, "negative": 1},
        peak_sequence=[0, 1],
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],
        key_peaks=[],
        key_peak_properties=[],
        peak_locations=np.array([], dtype=float),
        template_original=np.linspace(-1.0, 0.0, 30),
        exclusion_zones=[],
        end_zones={},
        key_peak_seq_indices=[],
        exclusion_zone_seq_bounds=[],
        methods=None,
    )


def _patch_dialog():
    """Monkeypatch the dialog; returns (module, restore-fn)."""
    import processing.extract_single_pulse_features as mod
    saved = mod._PulseFeatureReviewDialog
    mod._PulseFeatureReviewDialog = _FakeDialog
    return mod, (lambda: setattr(mod, "_PulseFeatureReviewDialog", saved))


def test_entry_multizone_outputs_wrapped():
    from utils.zones import wrap_zones
    mod, restore = _patch_dialog()
    try:
        _, data1d, pulse_indices, _ = _synthetic()
        data_2d = np.column_stack([data1d, data1d * 0.5])
        tp = wrap_zones([_tpz(), _tpz()])   # 2-zone grouped struct
        out = mod.extract_single_pulse_features(
            data_in=data_2d, tp=tp, sample_rate=1000.0,
            pulse_indices=pulse_indices,
        )
        for key in ("acceptance_id", "pulse_peak_locations",
                    "rectangularized_pulses", "pulse_start_indices"):
            w = out[key]
            assert isinstance(w, dict) and w.get("num_zones") == 2, \
                f"{key} not zone-wrapped: {type(w)}"
            assert len(w["zones"]) == 2
        # Zone 1 came from compute defaults: acceptance_id == (seq_matches>=1).
        from utils.tpsettings_io import get_zone
        r1 = mod._compute_zone(
            data_2d[:, 1].astype(float), 1000.0, None, pulse_indices,
            **mod._unpack_tp_zone(get_zone(tp, 1)))
        expected = np.asarray(r1["sequence_matches"]) >= 1
        got = out["acceptance_id"]["zones"][1]
        assert np.array_equal(np.asarray(got), expected), \
            "zone 1 acceptance did not match compute default"
    finally:
        restore()


def test_entry_singlezone_outputs_bare():
    mod, restore = _patch_dialog()
    try:
        _, data1d, pulse_indices, _ = _synthetic()
        tp = _tpz()   # bare flat dict (legacy single zone)
        out = mod.extract_single_pulse_features(
            data_in=data1d, tp=tp, sample_rate=1000.0,
            pulse_indices=pulse_indices,
        )
        n = len(pulse_indices)
        # Bare: NOT zone-wrapped structs.
        for key in ("acceptance_id", "pulse_peak_locations",
                    "rectangularized_pulses", "pulse_start_indices"):
            v = out[key]
            assert not (isinstance(v, dict) and "num_zones" in v), \
                f"{key} should be bare for single zone, got {v!r}"
        assert np.asarray(out["acceptance_id"]).dtype == bool
        assert len(out["acceptance_id"]) == n
        assert len(out["pulse_peak_locations"]) == n
        assert len(out["rectangularized_pulses"]) == n
        assert np.asarray(out["pulse_start_indices"]).tolist() == \
            pulse_indices[:, 0].tolist()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Headless dialog: real _PulseFeatureReviewDialog, multi-zone, no exec()
# ---------------------------------------------------------------------------

def _two_zone_computes():
    """Two real _compute_zone dicts on different synthetic columns, augmented
    with the shared pulse_indices + per-zone trendline (as the entry does)."""
    from processing.extract_single_pulse_features import _compute_zone
    fs, data1d, pulse_indices, tp = _synthetic()
    col0 = data1d.astype(float)
    col1 = (data1d * 0.5).astype(float)
    r0 = _compute_zone(col0, fs, None, pulse_indices, **tp)
    r1 = _compute_zone(col1, fs, None, pulse_indices, **tp)
    for r in (r0, r1):
        r["pulse_indices"] = pulse_indices
        r["trendline"] = None
    return [r0, r1], pulse_indices, fs


def _make_dialog(zones, pulse_indices, fs):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.extract_single_pulse_features import _PulseFeatureReviewDialog
    r0 = zones[0]
    single_indices = list(range(len(r0["pulse_peak_locs"])))
    return _PulseFeatureReviewDialog(
        r0["data"], pulse_indices, r0["filtered_pulses"], r0["rect_pulses"],
        r0["pulse_peak_locs"], r0["pulse_diffs"], r0["thresholds_used"],
        r0["counts_used"], r0["sequence_matches"], r0["template_seq"],
        single_indices, fs,
        pulse_excl_ranges=r0["pulse_excl_ranges_list"],
        pulse_key_peaks=r0["pulse_key_peaks_list"],
        pulse_key_peak_pols=r0["pulse_key_peak_pols_list"],
        pulse_key_peak_seqs=r0["pulse_key_peak_seqs_list"],
        template_key_seq_indices=r0["template_key_seq_indices"],
        template_key_polarities=r0["template_key_polarities"],
        trendline=r0.get("trendline"),
        filter_config=None,
        pad_length=r0["pad_length"],
        pad_method=r0["pad_method"],
        rect_pulses_mean=r0["rect_pulses_mean"],
        zones=zones,
        active_zone=0,
    )


def test_dialog_multizone_selector_and_roundtrip():
    zones, pulse_indices, fs = _two_zone_computes()
    dlg = _make_dialog(zones, pulse_indices, fs)
    try:
        assert len(dlg._zones) == 2
        assert dlg._zone_btn_group is not None  # selector exists
        assert dlg._active_zone == 0
        # Active plain attrs point at zone 0's data.
        assert dlg._data is zones[0]["data"]

        # Accept all in zone 0.
        dlg.acceptance[:] = 1
        z0_acc = dlg.acceptance.copy()

        # Switch to zone 1: plain attrs repoint at zone 1.
        dlg._on_zone_changed(1)
        assert dlg._active_zone == 1
        assert dlg._data is zones[1]["data"]
        assert dlg._filtered is dlg._zones[1]["_filtered"]

        # Edit zone-1 acceptance differently (reject all).
        dlg.acceptance[:] = 0
        z1_acc = dlg.acceptance.copy()

        # Back to zone 0: zone-0 edits survived the round-trip.
        dlg._on_zone_changed(0)
        assert dlg._active_zone == 0
        assert np.array_equal(dlg.acceptance, z0_acc), \
            "zone 0 acceptance did not survive the switch"

        results = dlg.zone_results()
        assert len(results) == 2
        # Per-zone edits reflected: zone 0 all accepted, zone 1 all rejected.
        assert np.all(results[0]["acceptance_id"] == True)   # noqa: E712
        assert np.all(results[1]["acceptance_id"] == False)  # noqa: E712
        assert not np.array_equal(results[0]["acceptance_id"],
                                  results[1]["acceptance_id"])
    finally:
        dlg.close()


def test_dialog_singlezone_no_selector():
    zones, pulse_indices, fs = _two_zone_computes()
    dlg = _make_dialog([zones[0]], pulse_indices, fs)  # one zone only
    try:
        assert len(dlg._zones) == 1
        assert dlg._zone_btn_group is None   # NO selector for single zone
        res = dlg.zone_results()
        assert len(res) == 1
        assert "acceptance_id" in res[0]
    finally:
        dlg.close()


def test_state_save_restore_is_per_zone():
    """Each zone's per-pulse peak edits save/restore to its OWN zone. The bug:
    state was active-zone-only, so one zone's peaks landed on another on load."""
    zones, pidx, fs = _two_zone_computes()
    dlg = _make_dialog(zones, pidx, fs)
    try:
        s0 = int(dlg._pulse_indices[0, 0])
        dlg._peak_locs[0] = np.array([s0 + 10, s0 + 20], dtype=int)        # zone 0
        dlg._zones[1]["_peak_locs"][0] = np.array([s0 + 111, s0 + 122], dtype=int)
        st = dlg._get_state()
    finally:
        dlg.close()
    assert len(st.get("zones", [])) == 2
    assert st["zones"][0]["peak_locs"][0] == [s0 + 10, s0 + 20]
    assert st["zones"][1]["peak_locs"][0] == [s0 + 111, s0 + 122]

    dlg2 = _make_dialog(zones, pidx, fs)
    try:
        dlg2._apply_state(st)
        assert np.asarray(dlg2._zones[0]["_peak_locs"][0]).tolist() == [s0 + 10, s0 + 20]
        assert np.asarray(dlg2._zones[1]["_peak_locs"][0]).tolist() == [s0 + 111, s0 + 122]
        # Active zone live reflects zone 0, not zone 1.
        assert np.asarray(dlg2._peak_locs[0]).tolist() == [s0 + 10, s0 + 20]
    finally:
        dlg2.close()


def test_template_indicator_updates_on_zone_switch():
    """The 'Template: ...' sequence indicator refreshes to the active zone's
    template when switching zones (it was built once and never updated)."""
    zones, pidx, fs = _two_zone_computes()
    zones[0]["template_seq"] = np.array([1, 0])
    zones[1]["template_seq"] = np.array([0, 1, 0, 1])
    dlg = _make_dialog(zones, pidx, fs)
    try:
        t0 = dlg._lbl_seq.text()
        dlg._on_zone_changed(1)
        t1 = dlg._lbl_seq.text()
        assert t0 != t1, "template indicator did not change on zone switch"
        dlg._on_zone_changed(0)
        assert dlg._lbl_seq.text() == t0, "did not restore on switch back"
    finally:
        dlg.close()


def test_weak_spike_anchor_not_relocated_far():
    """A weak single spike (below the template threshold) with a 1-max/1-min
    template keeps both anchors AT the spike. Regression: the degeneracy
    refinement used the full pulse duration as the width when no peak cleared
    the threshold, inflating expected_dist and relocating the + anchor to a
    far noise peak."""
    from processing.extract_single_pulse_features import _compute_zone
    fs = 10000.0
    n = 800
    x = np.arange(n)
    rng = np.random.default_rng(1)
    sig = 2.0 * rng.standard_normal(n)
    sig += 20.0 * np.exp(-((x - 200) ** 2) / (2 * 4 ** 2))   # weak spike (<thr)
    sig += 4.0 * np.exp(-((x - 700) ** 2) / (2 * 4 ** 2))    # far noise bump
    pulse_indices = np.array([[0, 799]])
    tp = dict(
        threshold={"upper": 38.0, "lower": -38.0},
        edge_margin={"first_peak_prop": 0.0, "last_peak_prop": 0.0},
        peak_counts={"positive": 1, "negative": 1},
        peak_sequence=[1, 0],
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],
        key_peaks=[368, 444], key_peak_properties=["max", "min"],
        peak_locations=[368, 444], template_original=np.zeros(4436),
        exclusion_zones=[], end_zones={},
        key_peak_seq_indices_in=[0, 1], exclusion_zone_seq_bounds_in=[])
    r = _compute_zone(sig, fs, None, pulse_indices, **tp)
    locs = np.asarray(r["pulse_peak_locs"][0])
    assert locs.max() < 350, f"anchor relocated far: {locs.tolist()}"


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
