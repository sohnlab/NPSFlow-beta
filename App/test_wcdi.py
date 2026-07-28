"""Tests for runWCDI — per-pulse [n×1] output, robust to vector orientation.

Run:  python App/test_wcdi.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from runners.runWCDI import run


class _B:
    input_ports = []


def _wcdi(Vc, diameter, Vnp=50.0, H=20.0):
    return run({"Vc": Vc, "Vnp": Vnp, "diameter": diameter, "H": H}, {}, _B())["wCDI"]


def _expect(Vc, diameter, Vnp=50.0, H=20.0):
    Vc = np.asarray(Vc, float).ravel()
    diameter = np.asarray(diameter, float).ravel()
    return Vc / Vnp * diameter / H


def test_flat_vectors_give_col_vector():
    Vc = [5.0, 5.5, 5.0, 5.4]
    d = [13.2, 19.1, 19.9, 22.9]
    out = np.asarray(_wcdi(Vc, d))
    assert out.shape == (4, 1), out.shape
    assert np.allclose(out.ravel(), _expect(Vc, d)), out


def test_column_vectors_give_col_vector():
    Vc = [[5.0], [5.5], [5.0], [5.4]]      # [n×1]
    d = [[13.2], [19.1], [19.9], [22.9]]
    out = np.asarray(_wcdi(Vc, d))
    assert out.shape == (4, 1), out.shape
    assert np.allclose(out.ravel(), _expect(Vc, d)), out


def test_mixed_orientation_no_broadcast_explosion():
    # Vc as [n×1] column vector, diameter as flat [n] — must NOT become [n×n].
    Vc = [[5.0], [5.5], [5.0], [5.4]]
    d = [13.2, 19.1, 19.9, 22.9]
    out = np.asarray(_wcdi(Vc, d))
    assert out.shape == (4, 1), out.shape
    assert np.allclose(out.ravel(), _expect(Vc, d)), out


def test_scalar_inputs_give_scalar():
    out = _wcdi(5.0, 13.2)
    assert isinstance(out, float), type(out)
    assert abs(out - (5.0 / 50.0 * 13.2 / 20.0)) < 1e-9, out


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
