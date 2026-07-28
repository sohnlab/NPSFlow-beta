"""jove_preprocess.process(): dict of all intermediate signals, 1-D and 2-D."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from processing import jove_preprocess as jp

# Stages off except decimation, so lengths are exact and math is trivial.
P = {"smooth_enable": False, "ds_enable": True, "ds_factor": 4,
     "lp_enable": False, "asls_enable": False}


def test_process_returns_all_keys():
    x = np.arange(40, dtype=float)
    r = jp.process(x, 1000, P)
    assert isinstance(r, dict)
    for k in ("smoothed", "down", "lp", "detrended", "trendline", "N", "fs_out"):
        assert k in r, k
    assert r["N"] == 4
    assert r["fs_out"] == 250.0


def test_process_1d_lengths_and_values():
    x = np.arange(40, dtype=float)
    r = jp.process(x, 1000, P)
    assert len(r["smoothed"]) == 40                 # full length (pre-decimation)
    assert np.array_equal(r["down"], x[::4])        # decimated
    assert len(r["detrended"]) == 10                # reduced
    # ASLS off -> detrended == lp == down, trendline all-zero.
    assert np.array_equal(r["detrended"], r["down"])
    assert np.array_equal(r["trendline"], np.zeros(10))


def test_process_2d_multizone_columns():
    x = np.arange(80, dtype=float).reshape(40, 2)   # 40 samples x 2 zones
    r = jp.process(x, 1000, P)
    assert np.asarray(r["smoothed"]).shape == (40, 2)
    assert np.asarray(r["down"]).shape == (10, 2)
    assert np.asarray(r["detrended"]).shape == (10, 2)
    assert np.array_equal(np.asarray(r["down"]), x[::4, :])


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
