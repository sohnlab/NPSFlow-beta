"""Causal Filter block: forward-only filtering (no lookahead), parity with
the realtime front-end's causal lowpass, and Load-Data-struct preservation
in the runner (only data replaced; labels and extra keys untouched)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing.causal_filter_ui import apply_causal_chain, apply_causal_filter


FS = 10000.0


def test_causal_no_lookahead():
    # Impulse at k: output must be exactly zero before k for every type.
    k = 500
    x = np.zeros(1000)
    x[k] = 1.0
    for ftype, c1, c2 in (
        ("lowpass", 1000, 0),
        ("highpass", 100, 0),
        ("bandpass", 100, 2000),
        ("notch", 60, 0),
    ):
        y = apply_causal_filter(x, FS, ftype, c1, c2, "freqBW", 2.0, order=2)
        assert np.all(y[:k] == 0), f"{ftype}: nonzero output before impulse"
        assert np.any(y[k:] != 0), f"{ftype}: no response"


def test_matches_realtime_frontend_lowpass():
    from processing.realtime_detection_threshold import _causal_lowpass

    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000) + 3.0
    mine = apply_causal_filter(x, FS, "lowpass", 800, 0, "freqBW", 2.0, order=2)
    ref = _causal_lowpass(x, 800, FS, order=2)
    assert np.allclose(mine, ref, atol=1e-8), np.max(np.abs(mine - ref))


def test_no_startup_transient():
    # Constant signal through a lowpass must stay constant (steady-state zi).
    x = np.full(2000, 5.0)
    y = apply_causal_filter(x, FS, "lowpass", 500, 0, "freqBW", 2.0, order=4)
    assert np.allclose(y, 5.0, atol=1e-9)
    # Highpass of a constant must be ~zero from the first sample.
    y = apply_causal_filter(x, FS, "highpass", 100, 0, "freqBW", 2.0, order=2)
    assert np.allclose(y, 0.0, atol=1e-9)


def test_chain_2d_per_channel():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3000, 3))
    chain = [
        {"type": "lowpass", "cutoff1": 1000.0, "order": 2},
        {"type": "notch", "cutoff1": 60.0, "notch_mode": "freqBW", "notch_bw": 2.0},
    ]
    y = apply_causal_chain(x, FS, chain)
    assert y.shape == x.shape
    # Each column independently equals the 1-D chain result.
    for ch in range(3):
        y1 = apply_causal_chain(x[:, ch], FS, chain)
        assert y1.ndim == 1
        assert np.allclose(y[:, ch], y1)


class _Port:
    def __init__(self, name, connected):
        self.name = name
        self.is_connected = connected


class _Blk:
    def __init__(self, config_connected):
        self.parameters = {}
        self.input_ports = [
            _Port("data", True),
            _Port("filterConfigIn", config_connected),
        ]


def _load_struct():
    rng = np.random.default_rng(2)
    files = []
    for i in range(3):
        files.append(
            {
                "data": rng.standard_normal((1000, 3)),
                "labels": rng.integers(0, 5, 1000).astype(np.int32),
                "fileName": f"f{i}.npz",
            }
        )
    return {"fileNames": [e["fileName"] for e in files], "files": files}


def test_runner_preserves_structure():
    from runners.runCausalFilterUI import run

    struct = _load_struct()
    chain = [{"type": "lowpass", "cutoff1": 1000.0, "order": 2}]
    out = run(
        {"data": struct, "sampleRate": FS, "filterConfigIn": chain},
        {},
        _Blk(config_connected=True),
    )

    filtered = out["filtered"]
    assert filtered["fileNames"] == struct["fileNames"]
    assert len(filtered["files"]) == 3
    for src, dst in zip(struct["files"], filtered["files"]):
        assert dst["fileName"] == src["fileName"]
        assert dst["labels"] is src["labels"]  # untouched, not copied
        assert dst["data"].shape == src["data"].shape
        assert not np.allclose(dst["data"], src["data"])
        assert np.allclose(dst["data"], apply_causal_chain(src["data"], FS, chain))
    # Source struct not mutated.
    assert struct["files"][0]["data"] is not filtered["files"][0]["data"]
    assert out["filterConfig"] == chain


def test_runner_bare_array_and_empty_chain():
    from runners.runCausalFilterUI import run

    x = np.random.default_rng(3).standard_normal(500)
    out = run(
        {
            "data": x,
            "sampleRate": FS,
            "filterConfigIn": [{"type": "lowpass", "cutoff1": 500.0}],
        },
        {},
        _Blk(config_connected=True),
    )
    assert isinstance(out["filtered"], np.ndarray)
    assert out["filtered"].shape == x.shape

    struct = _load_struct()
    out = run(
        {"data": struct, "sampleRate": FS, "filterConfigIn": []},
        {},
        _Blk(config_connected=True),
    )
    assert out["filtered"]["files"][0]["data"] is struct["files"][0]["data"]


def test_filter_ui_runner_struct():
    from processing.filter_ui import apply_filter_chain
    from runners.runFilterUI import run

    struct = _load_struct()
    chain = [
        {"type": "lowpass", "cutoff1": 1000.0},
        {"type": "notch", "cutoff1": 60.0, "notch_mode": "freqBW", "notch_bw": 2.0},
    ]
    out = run(
        {"data": struct, "sampleRate": FS, "filterConfigIn": chain},
        {},
        _Blk(config_connected=True),
    )

    filtered = out["filteredData"]
    assert filtered["fileNames"] == struct["fileNames"]
    assert len(filtered["files"]) == 3
    for src, dst in zip(struct["files"], filtered["files"]):
        assert dst["labels"] is src["labels"]
        assert not np.allclose(dst["data"], src["data"])
        assert np.allclose(dst["data"], apply_filter_chain(src["data"], FS, chain))
    assert out["filterConfig"] == chain


def test_filter_ui_runner_bare_legacy():
    from runners.runFilterUI import run

    x = np.random.default_rng(4).standard_normal(2000)
    out = run(
        {
            "data": x,
            "sampleRate": FS,
            "filterConfigIn": [{"type": "lowpass", "cutoff1": 500.0}],
        },
        {},
        _Blk(config_connected=True),
    )
    y = out["filteredData"]
    assert isinstance(y, np.ndarray) and y.shape == x.shape
    assert not np.allclose(y, x)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
