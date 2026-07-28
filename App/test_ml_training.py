"""Tests for the ML Training core (processing/ml_training_core.py).

Run:  python App/test_ml_training.py
No pytest dependency — plain asserts, prints PASS/FAIL per test.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing import ml_training_core as core  # noqa: E402

FS = 50000.0


# ----------------------------------------------------------------------------
# helpers to synthesize signals
# ----------------------------------------------------------------------------

def _bump(length, center, width, amp):
    x = np.arange(length)
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def _feat(vec, n_zones, name):
    names = core.feature_names(n_zones)
    return vec[names.index(name)]


def _make_recording(rng, n=200000, singles=(), coincidents=(), noise=0.5,
                    amp=30.0, width=20, span=1000, sep=150):
    """Baseline noise + labeled pulse events. Returns (data (n,3), labels)."""
    z = rng.normal(scale=noise, size=n)
    labels = np.zeros(n, dtype=np.int32)
    for c in singles:
        z += _bump(n, c, width, amp)
        labels[c - span // 2:c + span // 2] = 1
    for c in coincidents:
        z += _bump(n, c - sep // 2, width, amp)
        z += _bump(n, c + sep // 2, width, amp)
        labels[c - span // 2:c + span // 2] = 2
    return np.column_stack([z, z * 0.3, z * 0.1]), labels


# ----------------------------------------------------------------------------
# ported pure-function tests
# ----------------------------------------------------------------------------

def test_parse_events():
    labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0, 3, 3, 3])
    ev = core.parse_events(labels)
    assert ev == [(0, 2, 0), (2, 5, 1), (5, 6, 0),
                  (6, 8, 2), (8, 9, 0), (9, 12, 3)], ev


def test_feature_vector_length():
    for z in (1, 3, 5):
        names = core.feature_names(z)
        data = np.random.default_rng(0).normal(size=(1500, z))
        vec = core.extract_features(data, 500, 400, FS)
        assert len(vec) == len(names), (z, len(vec), len(names))


def test_single_vs_double_pulse_subpulses():
    rng = np.random.default_rng(1)
    base_len, win_len = 400, 1000

    def make(centers):
        z = rng.normal(scale=0.5, size=base_len + win_len)
        for c in centers:
            z[base_len:] += _bump(win_len, c, 20, 30.0)
        return np.column_stack([z, z * 0.3, z * 0.1])

    f_single = core.extract_features(make([500]), base_len, win_len, FS)
    f_double = core.extract_features(make([250, 750]), base_len, win_len, FS)
    assert _feat(f_single, 3, "n_subpulses") == 1
    assert _feat(f_double, 3, "n_subpulses") >= 2


def test_causality_ignores_future():
    rng = np.random.default_rng(3)
    a = rng.normal(scale=0.5, size=(2000, 3))
    b = a.copy()
    b[1000:] += 50.0  # huge disturbance strictly after the window
    fa = core.extract_features(a, 400, 400, FS)
    fb = core.extract_features(b, 400, 400, FS)
    assert np.allclose(fa, fb), "future samples leaked into features"


def test_sample_noise_windows_within_background():
    labels = np.zeros(30000, dtype=np.int32)
    labels[10000:11000] = 1
    labels[20000:20500] = 2
    events = core.parse_events(labels)
    rng = np.random.default_rng(4)
    wins = core.sample_noise_windows(events, FS, 20, [1000, 1500], rng)
    assert len(wins) > 0
    for (s, e, v) in wins:
        assert v == 0
        assert np.all(labels[s:e] == 0), (s, e)


def test_build_file_dataset_shapes():
    rng = np.random.default_rng(5)
    data, labels = _make_recording(
        rng, n=60000, singles=[5500, 15500, 35500, 45500],
        coincidents=[25500])
    horizons = [1.0, 2.0, 5.0, 10.0]
    ds = core.build_file_dataset(data, labels, FS, horizons, rng,
                                 noise_ratio=1.0)
    n_ex = len(np.unique(ds["example_id"]))
    assert len(ds["y"]) == n_ex * (len(horizons) + 1)
    assert set(np.unique(ds["y"])) == {0, 1, 2}
    assert "full" in set(ds["horizon"])


def test_oversample_balances_coincident():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 6))
    y = np.array([1] * 20 + [2] * 4, dtype=np.int32)
    horizon = np.array(["5ms"] * 24)
    for kind in ("smote", "random"):
        Xo, yo = core.oversample_train(X, y, horizon, rng, kind,
                                       ratio=1.0, k=3)
        assert int(np.sum(yo == 2)) == 20, (kind, int(np.sum(yo == 2)))
        assert Xo.shape[0] == len(yo)


def test_two_stage_predict_and_proba():
    rng = np.random.default_rng(8)

    def block(center, label, n):
        return (rng.normal(loc=center, scale=0.4, size=(n, 4)),
                np.full(n, label, dtype=np.int32))

    Xn, yn = block(0.0, 0, 60)
    Xs, ys = block(5.0, 1, 60)
    Xc, yc = block(10.0, 2, 8)
    X = np.vstack([Xn, Xs, Xc])
    y = np.concatenate([yn, ys, yc])
    horizon = np.array(["full"] * len(y))
    bundle = core.train_classifier(X, y, horizon, "two_stage", "rf", 0, rng,
                                   oversample="smote", ratio=1.0, k=3)
    pred = core.predict_classifier(bundle, X)
    assert (pred == y).mean() > 0.8
    proba = core.predict_proba_classifier(bundle, X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (np.argmax(proba, axis=1) == pred).mean() > 0.95


# ----------------------------------------------------------------------------
# new: splitting
# ----------------------------------------------------------------------------

def test_auto_split_fractions_and_overrides():
    counts = {f"f{i}": {"single": 50, "coincident": i % 3, "uncertain": 0}
              for i in range(10)}
    split = core.auto_split(counts, {"train": 0.6, "val": 0.2, "test": 0.2},
                            seed=0)
    assert set(split) == set(counts)
    sizes = {k: sum(1 for a in split.values() if a == k)
             for k in ("train", "val", "test")}
    assert sizes["train"] >= sizes["val"] >= 1
    assert sizes["test"] >= 1
    # explicit assignment survives
    split2 = core.auto_split(counts, {"train": 0.6, "val": 0.2, "test": 0.2},
                             seed=0, assignments={"f0": "test"})
    assert split2["f0"] == "test"
    # every split still nonempty with tiny file counts
    small = {n: counts[n] for n in ("f0", "f1", "f2")}
    split3 = core.auto_split(small, {"train": 0.98, "val": 0.01,
                                     "test": 0.01}, seed=1)
    assert set(split3.values()) == {"train", "val", "test"}


# ----------------------------------------------------------------------------
# new: streaming classifier
# ----------------------------------------------------------------------------

def _train_synthetic_bundle(rng):
    data, labels = _make_recording(
        rng, n=400000,
        singles=[20000 + 30000 * i for i in range(8)],
        coincidents=[35000 + 30000 * i for i in range(8)])
    ds = core.build_file_dataset(data, labels, FS, [1, 2, 5, 10], rng)
    bundle = core.train_classifier(
        ds["X"], ds["y"], ds["horizon"], "two_stage", "rf", 0, rng,
        oversample="smote", ratio=1.0, k=3)
    return {"bundle": bundle, "fs": FS, "n_zones": 3,
            "horizons_ms": [1, 2, 5, 10],
            "trigger": dict(core.DEFAULT_TRIGGER)}


def test_stream_classifier_detects_and_classifies():
    rng = np.random.default_rng(10)
    saved = _train_synthetic_bundle(rng)
    data, labels = _make_recording(
        rng, n=200000, singles=[30000, 90000, 150000],
        coincidents=[60000, 120000])
    events = core.simulate_stream(saved, data)
    pos = [e for e in events if e["label"] > 0]
    score = core.score_stream(events, labels, FS)
    assert score["n_events"] == 5
    assert score["detection_recall"] == 1.0, score
    assert score["classification_accuracy"] >= 0.8, score
    assert score["false_positives"] == 0, score
    # early updates were emitted at the configured horizons
    assert all(len(e["updates"]) > 0 for e in pos)
    # onsets land near the true bump starts
    gt_onsets = sorted(s for (s, e, v) in core.parse_events(labels)
                       if v in (1, 2))
    for ev, s in zip(sorted(pos, key=lambda e: e["onset"]), gt_onsets):
        assert abs(ev["onset"] - s) < 600, (ev["onset"], s)


def test_stream_chunk_size_invariance():
    rng = np.random.default_rng(11)
    saved = _train_synthetic_bundle(rng)
    data, _labels = _make_recording(
        rng, n=120000, singles=[30000], coincidents=[70000])
    big = [e for e in core.simulate_stream(saved, data, chunk_size=120000)
           if e["label"] > 0]
    small = [e for e in core.simulate_stream(saved, data, chunk_size=997)
             if e["label"] > 0]
    assert len(big) == len(small) == 2, (len(big), len(small))
    for a, b in zip(big, small):
        assert a["label"] == b["label"]
        assert abs(a["onset"] - b["onset"]) <= 8, (a["onset"], b["onset"])


def test_score_stream_counts():
    labels = np.zeros(60000, dtype=np.int32)
    labels[10000:11000] = 1
    labels[30000:31500] = 2
    events = [
        {"onset": 10050, "end": 10900, "label": 1},   # correct single
        {"onset": 30100, "end": 31300, "label": 1},   # coincident missed as 1
        {"onset": 50000, "end": 50400, "label": 2},   # false positive
        {"onset": 55000, "end": 55100, "label": 0},   # rejected trigger
    ]
    score = core.score_stream(events, labels, FS)
    assert score["n_events"] == 2
    assert score["n_detected"] == 2
    assert score["n_correct"] == 1
    assert score["false_positives"] == 1
    assert score["rejected_triggers"] == 1
    assert score["confusion"] == [[0, 1, 0], [0, 1, 0]]


def test_segment_entries():
    rng = np.random.default_rng(13)
    data, labels = _make_recording(
        rng, n=200000, singles=[30000, 90000, 150000],
        coincidents=[60000, 120000])
    segs = core.segment_entries([{"fileName": "rec.npz", "data": data,
                                  "labels": labels}])
    assert len(segs) == 5
    # segments tile the file exactly, in order
    assert sum(len(s["labels"]) for s in segs) == len(labels)
    joined = np.concatenate([s["labels"] for s in segs])
    assert np.array_equal(joined, labels)
    # exactly one event per segment, fully contained
    for s in segs:
        runs = [r for r in core.parse_events(s["labels"]) if r[2] != 0]
        assert len(runs) == 1, s["fileName"]
        assert runs[0][0] > 0 and runs[0][1] < len(s["labels"])
    # data slices are views, not copies
    assert segs[0]["data"].base is not None
    # no-event entry passes through whole
    plain = {"fileName": "bg.npz", "data": data[:1000],
             "labels": np.zeros(1000, dtype=np.int32)}
    assert core.segment_entries([plain]) == [plain]


def test_modular_entries_roundtrip(tmp_model="/tmp/_test_modular.joblib"):
    rng = np.random.default_rng(12)
    entries = []
    for i in range(4):
        data, labels = _make_recording(
            rng, n=150000,
            singles=[20000 + 30000 * k for k in range(4)],
            coincidents=[35000 + 60000 * k for k in range(2)])
        entries.append({"fileName": f"rec{i}.npz", "data": data,
                        "labels": labels})

    counts = {e["fileName"]: core.event_counts(e["labels"])
              for e in entries}
    split = core.auto_split(counts, {"train": 0.5, "val": 0.25,
                                     "test": 0.25}, seed=0)
    assert set(split.values()) == {"train", "val", "test"}

    train_entries = [e for e in entries if split[e["fileName"]] == "train"]
    test_entries = [e for e in entries if split[e["fileName"]] == "test"]
    info = core.train_from_entries(
        train_entries, {"fs": FS, "horizons_ms": [1, 2, 5, 10],
                        "modelPath": tmp_model, "smote_k": 3})
    assert os.path.isfile(info["modelPath"])
    assert info["rows"] > 0 and info["n_zones"] == 3

    res = core.evaluate_on_entries(info["modelPath"], test_entries)
    assert "full" in res["metrics"]
    assert res["metrics"]["full"]["accuracy"] > 0.6, \
        res["metrics"]["full"]["accuracy"]
    assert res["stream"] is None
    res2 = core.evaluate_on_entries(info["modelPath"], test_entries,
                                    stream_eval=True)
    name = test_entries[0]["fileName"]
    assert res2["stream"][name]["n_events"] == 6
    os.remove(tmp_model)


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
