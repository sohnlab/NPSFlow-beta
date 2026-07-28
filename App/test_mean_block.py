"""Tests for runMeanBlock — single/matrix/multi-input mean (no Qt).

Run:  python App/test_mean_block.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from runners.runMeanBlock import run


class _P:
    def __init__(self, name):
        self.name = name


class _B:
    def __init__(self, names):
        self.input_ports = [_P(n) for n in names]


def _blk(*names):
    return _B(["addInput", *names])


# ---- single 2-D matrix → mode applied → COLUMN VECTOR [N x 1] --------------

def test_single_matrix_row_wise_is_per_row_col_vector():
    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]   # 2 rows x 3 cols
    out = np.asarray(run({"in1": M}, {"mode": "row"}, _blk("in1"))["mean"])
    assert out.shape == (2, 1) and np.allclose(out.ravel(), [2.0, 5.0]), out


def test_single_matrix_column_wise_is_per_column_col_vector():
    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out = np.asarray(run({"in1": M}, {"mode": "column"}, _blk("in1"))["mean"])
    assert out.shape == (3, 1) and np.allclose(out.ravel(), [2.5, 3.5, 4.5]), out


def test_single_matrix_default_mode_is_column():
    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out = np.asarray(run({"in1": M}, {}, _blk("in1"))["mean"])
    assert out.shape == (3, 1) and np.allclose(out.ravel(), [2.5, 3.5, 4.5]), out


# ---- any 1-D input (flat / row vector / column vector) → scalar ------------

def test_single_vector_is_scalar_mean():
    out = run({"in1": [1.0, 2.0, 3.0, 4.0]}, {"mode": "row"}, _blk("in1"))["mean"]
    assert isinstance(out, float) and abs(out - 2.5) < 1e-9, out


def test_row_vector_is_scalar():
    out = run({"in1": [[1.0, 2.0, 3.0, 4.0]]}, {"mode": "row"}, _blk("in1"))["mean"]
    assert isinstance(out, float) and abs(out - 2.5) < 1e-9, out


def test_column_vector_is_scalar():
    out = run({"in1": [[1.0], [2.0], [3.0], [4.0]]}, {"mode": "column"},
              _blk("in1"))["mean"]
    assert isinstance(out, float) and abs(out - 2.5) < 1e-9, out


def test_single_vector_with_nan():
    out = run({"in1": [2.0, np.nan, 4.0]}, {"mode": "column"}, _blk("in1"))["mean"]
    assert abs(out - 3.0) < 1e-9, out


# ---- multiple inputs unchanged (regression) --------------------------------

def test_multi_input_column_wise():
    out = run({"a": [1.0, 3.0], "b": [3.0, 5.0]}, {"mode": "column"},
              _blk("a", "b"))["mean"]
    assert np.allclose(np.asarray(out), [2.0, 4.0]), out  # per element across inputs


def test_multi_input_row_wise():
    out = run({"a": [1.0, 3.0], "b": [3.0, 5.0]}, {"mode": "row"},
              _blk("a", "b"))["mean"]
    assert np.allclose(np.asarray(out), [2.0, 4.0]), out  # per input across elements


# ---- segment-dict single input collapses to scalar (no matrix meaning) -----

def test_single_segment_dict_scalar():
    seg = {"startValue": [0.0, 10.0], "endValue": [2.0, 12.0]}  # midpoints 1, 11
    out = run({"in1": seg}, {"mode": "row"}, _blk("in1"))["mean"]
    assert abs(out - 6.0) < 1e-9, out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
