"""Reproduction for: Segment Processing output connections disappear when the
block is auto-run (as a downstream dependency) with incomplete upstream data.

Run:  python App/test_segment_processing_ports.py
"""
import os, sys, types

sys.path.insert(0, os.path.dirname(__file__))

from port import Port
from workflow_engine import WorkflowEngine


class _Block:
    """Minimal duck-typed SegmentProcessing block."""
    def __init__(self):
        self.input_ports = []
        self.output_ports = []
        self.parameters = {}
        self.size = (180, 200)
        self.is_collapsed = False


def _make_block_with_wires(n=3):
    """SegmentProcessing with in1..inN + addInput, out1..outN each wired to a sink."""
    blk = _Block()
    for i in range(1, n + 1):
        ip = Port(name=f"in{i}", direction="input")
        ip.display_name = str(i)
        ip.parent_block = blk
        # These inputs are wired (the data may just be transiently missing on a
        # given run) — model that so is_connected reflects the real graph.
        ip.connections.append(types.SimpleNamespace(source_port=None,
                                                    dest_port=ip))
        blk.input_ports.append(ip)
    add = Port(name="addInput", direction="input")
    add.parent_block = blk
    blk.input_ports.append(add)

    wires = []
    for i in range(1, n + 1):
        op = Port(name=f"out{i}", direction="output")
        op.display_name = str(i)
        op.parent_block = blk
        op.index = i
        wire = types.SimpleNamespace(source_port=op, dest_port=None)
        op.connections.append(wire)
        wires.append(wire)
        blk.output_ports.append(op)
    return blk, wires


def test_partial_inputs_preserve_output_ports_and_wires():
    eng = WorkflowEngine.__new__(WorkflowEngine)  # skip __init__/Qt
    removed = []
    eng.wire_remove_callback = lambda w: removed.append(w)

    blk, _ = _make_block_with_wires(3)
    # Simulate the cascade: in2's upstream produced no data this run.
    inputs = {"in1": {"values": [[1, 2, 3]]}, "in3": {"values": [[4, 5, 6]]}}

    eng._execute_segment_processing(blk, inputs, wires=[])

    out_names = {p.name for p in blk.output_ports}
    assert out_names == {"out1", "out2", "out3"}, \
        f"expected all 3 mirror outputs, got {sorted(out_names)}"
    assert removed == [], f"no wires should be removed on partial data, removed {len(removed)}"


def test_removed_input_drops_its_output_and_wire():
    """Sanity: when an input port is genuinely gone, its output + wire are removed."""
    eng = WorkflowEngine.__new__(WorkflowEngine)
    removed = []
    eng.wire_remove_callback = lambda w: removed.append(w)

    blk, wires = _make_block_with_wires(3)
    # Genuinely remove in2 (a structural change), keep in1/in3 connected.
    blk.input_ports = [p for p in blk.input_ports if p.name != "in2"]
    inputs = {"in1": {"values": [[1]]}, "in3": {"values": [[4]]}}

    eng._execute_segment_processing(blk, inputs, wires=[])

    out_names = {p.name for p in blk.output_ports}
    assert out_names == {"out1", "out3"}, f"got {sorted(out_names)}"
    assert len(removed) == 1, f"exactly the out2 wire should be removed, got {len(removed)}"


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
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
