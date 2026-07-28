"""Tests for utils.zones — pure functions, numpy only (no Qt/scipy)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.zones import is_multizone, zone_columns, zone_list, n_zones, wrap_zones


def test_is_multizone():
    assert is_multizone(np.zeros((10, 3))) is True
    assert is_multizone(np.zeros((10, 1))) is False
    assert is_multizone(np.zeros(10)) is False
    assert is_multizone(None) is False


def test_zone_columns():
    a = np.arange(6).reshape(3, 2)            # 3 samples, 2 zones
    cols = zone_columns(a)
    assert len(cols) == 2
    assert np.array_equal(cols[0], a[:, 0])
    assert np.array_equal(cols[1], a[:, 1])
    assert len(zone_columns(np.arange(5))) == 1
    assert np.array_equal(zone_columns(np.arange(5))[0], np.arange(5))
    assert len(zone_columns(np.arange(5).reshape(5, 1))) == 1


def test_zone_list_and_n_zones():
    bare = {"a": 1}
    assert zone_list(bare) == [bare]
    assert n_zones(bare) == 1
    wrapped = {"num_zones": 2, "zones": [{"a": 1}, {"a": 2}]}
    assert zone_list(wrapped) == [{"a": 1}, {"a": 2}]
    assert n_zones(wrapped) == 2


def test_wrap_zones_inverse_for_single():
    x = {"a": 1}
    assert wrap_zones([x]) is x                # bare passthrough, identity
    assert zone_list(wrap_zones([x])) == [x]


def test_wrap_zones_multi():
    out = wrap_zones([{"a": 1}, {"a": 2}, {"a": 3}])
    assert out["num_zones"] == 3
    assert out["zones"] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert zone_list(out) == [{"a": 1}, {"a": 2}, {"a": 3}]


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
