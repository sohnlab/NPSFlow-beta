"""Coincident entry: zone selection, decompose->_compute_zone, output shape.
Fakes the dialog (no GUI), mirroring test_extract_zone_compute._FakeDialog."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def _tpz():
    return dict(
        threshold={"upper": 5.0, "lower": -5.0},
        edge_margin={"first_peak_prop": 0.0, "last_peak_prop": 0.0},
        peak_counts={"positive": 1, "negative": 1},
        peak_sequence=[1, 0],
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],
        key_peaks=[], key_peak_properties=[],
        peak_locations=np.array([], dtype=float),
        template_original=np.linspace(0.0, 1.0, 30),
        exclusion_zones=[], end_zones={},
        key_peak_seq_indices=[], exclusion_zone_seq_bounds=[], methods=None,
    )


def _capture():
    """3-column capture: col0=start ind, col1=measurement, col2=end ind.
    One coincident region [500,950] with 2 particles."""
    n = 2000
    x = np.arange(n)
    rng = np.random.default_rng(5)
    meas = 0.2 * rng.standard_normal(n)
    # two overlapping particle edge-pairs inside the measurement
    for c0, c1 in [(560, 700), (650, 860)]:
        meas += 30.0 / (1 + np.exp(-(x - c0) / 3.0)) - 30.0 / (1 + np.exp(-(x - c1) / 3.0))
    start_ind = 50.0 * np.exp(-((x - 520) ** 2) / (2 * 3 ** 2)) \
        + 50.0 * np.exp(-((x - 610) ** 2) / (2 * 3 ** 2))
    end_ind = 50.0 * np.exp(-((x - 780) ** 2) / (2 * 3 ** 2)) \
        + 50.0 * np.exp(-((x - 900) ** 2) / (2 * 3 ** 2))
    data = np.column_stack([start_ind, meas, end_ind])
    regions = np.array([[500, 950]], dtype=int)
    return data, regions


class _FakeDialog:
    def __init__(self, *a, **kw):
        self.confirmed = True
        self._compute = kw.get("compute")
        self._decomp = kw.get("decomp")
    def load_state_from_file(self, path): pass
    def exec(self): return 1
    def results(self):
        c = self._compute
        n = len(c["pulse_peak_locs"])
        return {
            "pulse_peak_locations": list(c["pulse_peak_locs"]),
            "rectangularized_pulses": list(c["rect_pulses"]),
            "acceptance_id": np.asarray(c["sequence_matches"]) >= 1,
            "pulse_start_indices": np.asarray(c["pulse_start_indices"]),
            "coincidence_map": [
                {"source_event": int(se), "N": int(nn)}
                for se, nn in zip(self._decomp["source_event"],
                                  self._decomp["event_n"])],
        }


def _patch():
    import processing.extract_coincident_pulse_features as mod
    saved = mod._CoincidentReviewDialog
    mod._CoincidentReviewDialog = _FakeDialog
    return mod, (lambda: setattr(mod, "_CoincidentReviewDialog", saved))


def test_entry_timing_mode_recovers_two_particles():
    mod, restore = _patch()
    try:
        data, regions = _capture()
        out = mod.extract_coincident_pulse_features(
            data_in=data, tp=_tpz(), sample_rate=1000.0, pulse_indices=regions,
            measurement_zone=1, start_zone=0, end_zone=2)
        assert len(out["pulse_peak_locations"]) == 2
        assert len(out["rectangularized_pulses"]) == 2
        assert np.asarray(out["acceptance_id"]).dtype == bool
        assert len(out["pulse_start_indices"]) == 2
        assert [m["N"] for m in out["coincidence_map"]] == [2, 2]
        # flat/bare output (not zone-wrapped)
        assert not isinstance(out["pulse_peak_locations"], dict)
    finally:
        restore()


def test_entry_auto_guesses_measurement_when_unset():
    mod, restore = _patch()
    try:
        data, regions = _capture()
        out = mod.extract_coincident_pulse_features(
            data_in=data, tp=_tpz(), sample_rate=1000.0, pulse_indices=regions)
        # highest-variance column is the measurement (col 1 has the big edges);
        # with no start/end selected, fallback still yields >=1 particle.
        assert len(out["pulse_start_indices"]) >= 1
    finally:
        restore()


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
