"""Coincident decomposition (pure). No Qt."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def _spike(n, centers, width=3, amp=50.0):
    x = np.arange(n, dtype=float)
    y = np.zeros(n, dtype=float)
    for c in centers:
        y += amp * np.exp(-((x - c) ** 2) / (2 * width ** 2))
    return y


def test_detect_indicator_spikes_orders_by_position():
    from processing.coincident_decomposition import detect_indicator_spikes
    n = 1000
    ind = _spike(n, [200, 620]) + 0.5 * np.random.default_rng(0).standard_normal(n)
    locs = detect_indicator_spikes(ind, lo=0, hi=n)
    assert len(locs) == 2
    assert list(locs) == sorted(locs)
    assert abs(locs[0] - 200) <= 5 and abs(locs[1] - 620) <= 5


def test_detect_indicator_spikes_windowed():
    from processing.coincident_decomposition import detect_indicator_spikes
    n = 1000
    ind = _spike(n, [100, 500, 900])
    locs = detect_indicator_spikes(ind, lo=400, hi=700)
    assert len(locs) == 1 and abs(locs[0] - 500) <= 5


def test_fifo_windows_pairs_starts_and_ends():
    from processing.coincident_decomposition import fifo_windows
    # two particles: starts 210,300 ; ends 450,560 ; region [200,600]
    wins = fifo_windows(starts=[210, 300], ends=[450, 560], s=200, e=600)
    assert wins == [(210, 450), (300, 560)]


def test_fifo_windows_pads_missing_end_with_region_bound():
    from processing.coincident_decomposition import fifo_windows
    # 2 starts, 1 end -> second particle's end clamps to region end e
    wins = fifo_windows(starts=[210, 300], ends=[450], s=200, e=600)
    assert wins[0] == (210, 450)
    assert wins[1] == (300, 600)


def _template_params(seq=(1, 0)):
    return dict(
        threshold={"upper": 5.0, "lower": -5.0},
        peak_counts={"positive": sum(1 for v in seq if v == 1),
                     "negative": sum(1 for v in seq if v == 0)},
        peak_sequence=list(seq),
        filter_padding={"pad_length": 10, "method": "replicate"},
        filter_config=[],
    )


def test_measurement_edge_peaks_found():
    from processing.coincident_decomposition import measurement_edge_peaks
    n = 400
    x = np.arange(n)
    # one up-edge then one down-edge (derivative +peak then -peak)
    sig = 30.0 / (1 + np.exp(-(x - 150) / 3.0)) - 30.0 / (1 + np.exp(-(x - 250) / 3.0))
    pos, neg, dseg = measurement_edge_peaks(sig, fs=1000.0,
                                            template_params=_template_params())
    assert len(pos) >= 1 and len(neg) >= 1
    assert pos[0] < neg[0]  # up-edge before down-edge


def test_split_sequential_cuts_at_large_gap():
    from processing.coincident_decomposition import split_fallback
    # two template instances back-to-back: peaks near 50/70 and 250/270
    peaks = np.array([50, 70, 250, 270])
    spans, mode = split_fallback(peaks, n=200 + 120, region_len=340, seq_len=2)
    assert mode == "sequential"
    assert len(spans) == 2
    # first span covers the first pair, second covers the second pair
    assert spans[0][0] <= 50 and spans[0][1] >= 70 and spans[0][1] < 250
    assert spans[1][0] > 70 and spans[1][0] <= 250


def test_split_interleaved_when_no_clear_gap():
    from processing.coincident_decomposition import split_fallback
    # evenly spaced peaks, 4 of them, 2 instances of len-2 -> interleaved
    peaks = np.array([40, 60, 80, 100])
    spans, mode = split_fallback(peaks, n=140, region_len=140, seq_len=2)
    assert mode == "interleaved"
    assert len(spans) == 2


def test_decompose_timing_mode_two_particles():
    from processing.coincident_decomposition import decompose_coincident_regions
    n = 2000
    rng = np.random.default_rng(3)
    meas = 0.2 * rng.standard_normal(n)
    start_ind = _spike(n, [520, 610])           # two entries
    end_ind = _spike(n, [780, 900])             # two exits
    regions = np.array([[500, 950]], dtype=int)
    out = decompose_coincident_regions(
        meas, fs=1000.0, coincident_indices=regions,
        template_params=_template_params(),
        start_indicator=start_ind, end_indicator=end_ind)
    ri = np.asarray(out["recovered_indices"])
    assert ri.shape == (2, 2)
    assert list(out["event_n"]) == [2, 2]
    assert list(out["source_event"]) == [0, 0]
    assert out["event_mode"][0] == "timing"
    # FIFO windows: particle 0 ~ [520,780], particle 1 ~ [610,900]
    assert abs(ri[0, 0] - 520) <= 6 and abs(ri[0, 1] - 780) <= 6


def test_decompose_fallback_mode_no_indicators():
    from processing.coincident_decomposition import decompose_coincident_regions
    n = 700
    x = np.arange(n)
    # two sequential template instances (up/down edges) in one region
    seg = (30.0 / (1 + np.exp(-(x - 120) / 3.0)) - 30.0 / (1 + np.exp(-(x - 170) / 3.0))
           + 30.0 / (1 + np.exp(-(x - 400) / 3.0)) - 30.0 / (1 + np.exp(-(x - 450) / 3.0)))
    regions = np.array([[50, 650]], dtype=int)
    out = decompose_coincident_regions(
        seg, fs=1000.0, coincident_indices=regions,
        template_params=_template_params(seq=(1, 0)),
        start_indicator=None, end_indicator=None)
    ri = np.asarray(out["recovered_indices"])
    assert ri.shape[0] == 2
    assert out["event_mode"][0] in ("sequential", "interleaved")


def test_decompose_empty_regions():
    from processing.coincident_decomposition import decompose_coincident_regions
    out = decompose_coincident_regions(
        np.zeros(100), fs=1000.0, coincident_indices=np.zeros((0, 2), int),
        template_params=_template_params())
    assert np.asarray(out["recovered_indices"]).shape == (0, 2)
    assert list(out["source_event"]) == []


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
