"""Regression tests for Unpack Variable output port ordering on run.

Running an Unpack block rebuilds its output ports from the struct fields.
That rebuild must respect a port order the user set via Rearrange Ports
(stored in block.output_ports / outputPortDefs, not in selectedPaths),
instead of snapping back to the selectedPaths/flatten order.

Run:  QT_QPA_PLATFORM=offscreen python App/test_unpack_order.py
"""
import os, sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from block_registry import BlockRegistry
from workflow_engine import WorkflowEngine
from port import Port

_APP = QApplication.instance() or QApplication([])
_REG = BlockRegistry(); _REG.scan_block_defs()


def _engine():
    e = WorkflowEngine(_REG)
    e.wire_remove_callback = lambda w: None
    return e


def _unpack(selected, existing_order):
    blk = _REG.create_block("UnpackStruct", (0, 0))
    blk.parameters["selectedPaths"] = list(selected)
    ports = []
    for n in existing_order:
        p = Port(n, "output", "any", True, f"Field: {n}")
        p.display_name = n; p.parent_block = blk; ports.append(p)
    blk.output_ports = ports
    return blk


def _run(blk, struct):
    _engine()._execute_unpack(blk, {"structIn": struct}, [])
    return [p.name for p in blk.output_ports]


def test_run_preserves_reordered_outputs():
    blk = _unpack(["alpha", "beta", "gamma"], ["gamma", "alpha", "beta"])
    out = _run(blk, {"alpha": 1, "beta": 2, "gamma": 3})
    assert out == ["gamma", "alpha", "beta"], out


def test_new_field_appended_after_custom_order():
    # User has ports [gamma, alpha] reordered; selectedPaths adds beta.
    blk = _unpack(["alpha", "beta", "gamma"], ["gamma", "alpha"])
    out = _run(blk, {"alpha": 1, "beta": 2, "gamma": 3})
    assert out == ["gamma", "alpha", "beta"], out


def test_first_run_uses_natural_order():
    blk = _unpack(["alpha", "beta", "gamma"], [])   # no ports yet
    out = _run(blk, {"alpha": 1, "beta": 2, "gamma": 3})
    assert out == ["alpha", "beta", "gamma"], out


def test_outputportdefs_and_data_follow_order():
    blk = _unpack(["alpha", "beta", "gamma"], ["gamma", "alpha", "beta"])
    _engine()._execute_unpack(blk, {"structIn": {"alpha": 1, "beta": 2, "gamma": 3}}, [])
    defs_order = [d["name"] for d in blk.parameters["outputPortDefs"]]
    assert defs_order == ["gamma", "alpha", "beta"], defs_order


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
