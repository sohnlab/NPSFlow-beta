"""Topological sort adds implicit ordering edges so a Data Bus that WRITES a
pool variable runs before a Data Bus that READS it (they share a pool, not a
wire). Regression: the reader's consumer (e.g. Detection Review) ran first,
found an empty pool, errored, and halted the run before the writer ran."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow_engine import WorkflowEngine


class _Port:
    def __init__(self, name, disp=None, desc="", idx=1, parent=None):
        self.name = name
        self.display_name = disp or name
        self.description = desc
        self.index = idx
        self.parent_block = parent
        self.required = False
        self.is_connected = False
        self.has_data = False
        self.connections = []


class _Blk:
    def __init__(self, bid, dn, params=None):
        self.id = bid
        self.definition_name = dn
        self.parameters = params or {}
        self.input_ports = []
        self.output_ports = []


class _Wire:
    def __init__(self, sp, dp):
        self.source_port = sp
        self.dest_port = dp


class _Reg:
    DEFS = {"StartBlock": {"isStart": True}, "FilterUI": {},
            "DataBus": {"isDataBus": True}, "DetectionReview": {}}

    def has(self, n):
        return n in self.DEFS

    def get(self, n):
        return self.DEFS[n]


def _pipeline(writer_label="X", reader_label="X", pool_w="Default", pool_r="Default"):
    start = _Blk("start", "StartBlock")
    start.output_ports = [_Port("trigger", parent=start)]
    filt = _Blk("filt", "FilterUI")
    filt.input_ports = [_Port("data", parent=filt)]
    filt.output_ports = [_Port("filteredData", parent=filt)]
    writer = _Blk("writer", "DataBus", {"pool": pool_w})
    writer.input_ports = [_Port("in1", disp=writer_label, parent=writer)]
    reader = _Blk("reader", "DataBus", {"pool": pool_r})
    reader.output_ports = [_Port("out1", disp=reader_label, desc=reader_label,
                                 parent=reader)]
    review = _Blk("review", "DetectionReview")
    review.input_ports = [_Port("dataIn", parent=review)]
    review.output_ports = [_Port("single", parent=review)]
    wires = [
        _Wire(start.output_ports[0], filt.input_ports[0]),
        _Wire(filt.output_ports[0], writer.input_ports[0]),
        _Wire(reader.output_ports[0], review.input_ports[0]),
    ]
    return [start, filt, writer, reader, review], wires


def test_writer_ordered_before_reader_and_consumer():
    blocks, wires = _pipeline()
    eng = WorkflowEngine(registry=_Reg())
    order = [b.id for b in eng.topological_sort(blocks, wires)]
    assert order.index("writer") < order.index("reader"), order
    assert order.index("writer") < order.index("review"), order
    assert "writer" in eng._find_reachable_from_start(blocks, wires)


def test_mismatched_label_no_edge():
    blocks, wires = _pipeline(writer_label="X", reader_label="Y")
    eng = WorkflowEngine(registry=_Reg())
    adj = {b.id: [] for b in blocks}
    eng._add_databus_pool_edges(adj, blocks, wires)
    assert ("reader") not in [d for _, ns in adj.items() for d, _ in ns]


def test_different_pool_no_edge():
    blocks, wires = _pipeline(pool_w="A", pool_r="B")
    eng = WorkflowEngine(registry=_Reg())
    adj = {b.id: [] for b in blocks}
    eng._add_databus_pool_edges(adj, blocks, wires)
    assert ("writer", "reader") not in [(s, d) for s, ns in adj.items()
                                        for d, _ in ns]


def test_run_to_block_refreshes_stale_databus_reader():
    """A done Data Bus reader in the cone must re-run during run_to_block:
    reusing its cached output serves the PREVIOUS run's value even after a
    re-run upstream block refreshed the pool through the writer bus."""
    class _Reg2(_Reg):
        DEFS = dict(_Reg.DEFS, Seq={"isPriority": True})

    start = _Blk("start", "StartBlock")
    start.output_ports = [_Port("trigger", parent=start)]
    filt = _Blk("filt", "FilterUI")
    filt.input_ports = [_Port("data", parent=filt)]
    filt.output_ports = [_Port("filteredData", parent=filt)]
    writer = _Blk("writer", "DataBus", {"pool": "Default"})
    writer.input_ports = [_Port("in1", disp="X", parent=writer)]
    reader = _Blk("reader", "DataBus", {"pool": "Default"})
    reader.output_ports = [_Port("out1", disp="X", desc="X", parent=reader)]
    seq = _Blk("seq", "Seq")
    seq.input_ports = [_Port("Run", parent=seq)]
    seq.output_ports = [_Port("out1", parent=seq)]
    blocks = [start, filt, writer, reader, seq]
    wires = [
        _Wire(start.output_ports[0], filt.input_ports[0]),
        _Wire(filt.output_ports[0], writer.input_ports[0]),
        _Wire(reader.output_ports[0], seq.input_ports[0]),
    ]

    eng = WorkflowEngine(registry=_Reg2())
    # Prior run left everything done with 'old' in the pool and outputs.
    for b in blocks:
        b.status = "done"
        b.display_name = b.id
        b.error_message = ""
    start.output_data = {"trigger": True}
    filt.output_data = {"filteredData": "old"}
    writer.output_data = {}
    reader.output_data = {"out1": "old"}
    seq.output_data = {"out1": "old"}
    eng.data_bus_store = {"Default": {"X": "old"}}

    # The upstream block re-ran (e.g. interactive dialog) with a new result.
    filt.output_data = {"filteredData": "new"}

    eng.run_to_block(seq, blocks, wires)
    assert reader.output_data.get("out1") == "new", reader.output_data
    assert seq.output_data.get("out1") == "new", seq.output_data


def test_duplicate_pool_writer_warning():
    """Two wired writers of the same pool variable must trigger a warning —
    they silently overwrite each other (regression: Downsample.trendline and
    PulseDetectionBC.trendline both writing 'trendline2')."""
    src_a = _Blk("srcA", "FilterUI")
    src_a.output_ports = [_Port("outA", parent=src_a)]
    src_b = _Blk("srcB", "FilterUI")
    src_b.output_ports = [_Port("outB", parent=src_b)]
    bus_a = _Blk("busA", "DataBus", {"pool": "Data"})
    bus_a.input_ports = [_Port("in1", disp="trendline2", parent=bus_a)]
    bus_a.display_name = "Bus A"
    bus_b = _Blk("busB", "DataBus", {"pool": "Data"})
    bus_b.input_ports = [_Port("in1", disp="trendline2", parent=bus_b)]
    bus_b.display_name = "Bus B"
    blocks = [src_a, src_b, bus_a, bus_b]
    wires = [_Wire(src_a.output_ports[0], bus_a.input_ports[0]),
             _Wire(src_b.output_ports[0], bus_b.input_ports[0])]

    eng = WorkflowEngine(registry=_Reg())
    msgs = []
    eng.log_callback = msgs.append
    eng._warn_duplicate_pool_writers(blocks, wires)
    assert any("trendline2" in m and "2 writers" in m for m in msgs), msgs

    # Distinct labels: no warning.
    bus_b.input_ports[0].display_name = "aslsTrend"
    msgs.clear()
    eng._warn_duplicate_pool_writers(blocks, wires)
    assert not msgs, msgs


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
