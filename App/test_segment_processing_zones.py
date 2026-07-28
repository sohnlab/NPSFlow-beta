"""SegmentProcessing runner: single segment (single-zone) and per-zone struct
(multi-zone Pulse Slicing output) both processed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import runners.runSegmentProcessing as sp


def _seg(vals):
    """A segment struct like Pulse Slicing emits (one per detected pulse)."""
    n = len(vals)
    return {
        "label": "a", "method": "mean", "segmentId": 1,
        "pulseIndex": list(range(1, n + 1)),
        "startLocal": [0] * n,
        "endLocal": [len(v) - 1 for v in vals],
        "startGlobal": [10 * (i + 1) for i in range(n)],
        "endGlobal": [10 * (i + 1) + len(vals[i]) - 1 for i in range(n)],
        "values": vals,
    }


def test_single_segment_processed():
    vals = [np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0])]
    out = sp.run({"in1": _seg(vals), "addInput": None}, {}, None)
    assert out["out1"]["width"] == [3, 2]
    assert out["out1"]["startValue"] == [5.0, 1.0]
    assert "values" not in out["out1"]      # raw segment dropped


def test_per_zone_struct_unpacked_to_segment_ports():
    """A single zone-struct input is UNPACKED: each segment becomes its own
    top-level output, named by the segment label with the leading "_" dropped
    ("_1" -> "1")."""
    vals = [np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0])]
    zone_struct = {"_1": _seg(vals), "_2": _seg([np.array([9.0, 9.0, 9.0, 9.0])])}
    out = sp.run({"in1": zone_struct}, {}, None)
    assert set(out.keys()) == {"1", "2"}
    assert out["1"]["width"] == [3, 2]
    assert out["2"]["width"] == [4]
    assert "values" not in out["1"]


def test_multiple_zone_inputs_prefixed():
    """Two zone inputs with overlapping segment keys are prefixed by the input
    port so they stay unique (segment part de-prefixed)."""
    vals = [np.array([5.0, 6.0, 7.0])]
    z = {"_1": _seg(vals), "_2": _seg(vals)}
    out = sp.run({"in1": z, "in2": z}, {}, None)
    assert set(out.keys()) == {"in1_1", "in1_2", "in2_1", "in2_2"}
    assert out["in2_1"]["width"] == [3]


def test_none_and_addinput_skipped():
    out = sp.run({"in1": None, "addInput": None}, {}, None)
    assert out == {}


def _engine_block(in_names=("in1",)):
    import types
    from port import Port
    blk = types.SimpleNamespace(input_ports=[], output_ports=[],
                                parameters={}, size=(180, 200),
                                output_data={})
    for nm in in_names:
        ip = Port(name=nm, direction="input")
        ip.display_name = nm
        ip.parent_block = blk
        ip.connections.append(types.SimpleNamespace())   # wired
        blk.input_ports.append(ip)
    add = Port(name="addInput", direction="input")
    add.parent_block = blk
    blk.input_ports.append(add)
    return blk


def test_engine_unpacks_zone_input_to_segment_ports():
    """The engine builds one output PORT per segment of a zone-struct input."""
    from workflow_engine import WorkflowEngine
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.wire_remove_callback = lambda w: None
    blk = _engine_block(("in1",))
    zone_struct = {"_1": _seg([np.array([1.0, 2.0, 3.0])]),
                   "_2": _seg([np.array([4.0, 5.0])]),
                   "_3": _seg([np.array([6.0])])}
    out = eng._execute_segment_processing(blk, {"in1": zone_struct}, wires=[])
    port_names = [p.name for p in blk.output_ports]
    assert port_names == ["1", "2", "3"], port_names
    assert set(out.keys()) == {"1", "2", "3"}


def test_engine_preserves_ports_on_incomplete_inputs():
    """Auto-run with a wired input missing data must NOT tear down ports."""
    from workflow_engine import WorkflowEngine
    eng = WorkflowEngine.__new__(WorkflowEngine)
    removed = []
    eng.wire_remove_callback = lambda w: removed.append(w)
    blk = _engine_block(("in1", "in2"))
    # Pre-existing ports (e.g. from a prior complete run).
    from port import Port
    for nm in ("_1", "_2"):
        op = Port(name=nm, direction="output"); op.parent_block = blk
        blk.output_ports.append(op)
    blk.output_data = {"_1": {"width": [3]}, "_2": {"width": [2]}}
    # in2 is wired but produced no data this run -> keep ports/wires.
    eng._execute_segment_processing(blk, {"in1": {"_1": _seg([np.array([1.0])])}},
                                    wires=[])
    assert {p.name for p in blk.output_ports} == {"_1", "_2"}
    assert removed == []


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
