"""Tests for pool-wide Data Bus variable rename propagation (no Qt app).

Run:  python App/test_databus_rename.py
"""
import os, sys, types

sys.path.insert(0, os.path.dirname(__file__))

from port import Port
from workflow_app import NPSWorkflowApp as WorkflowApp


def _port(name, direction, display, desc=""):
    p = Port(name=name, direction=direction, description=desc)
    p.display_name = display
    return p


def _bus(pool, inputs, outputs):
    return types.SimpleNamespace(
        definition_name="DataBus",
        parameters={"pool": pool},
        input_ports=inputs,
        output_ports=outputs,
    )


def _app(blocks):
    app = WorkflowApp.__new__(WorkflowApp)
    app.registry = types.SimpleNamespace(
        has=lambda n: True, get=lambda n: {"isDataBus": True})
    app._tab_widget = None  # canvas property falls back to _canvas_fallback
    app._canvas_fallback = types.SimpleNamespace(
        blocks=blocks, update_block_graphics=lambda b: None)
    return app


def test_rename_propagates_to_reader_output():
    writer = _bus("Segments",
                  [_port("Run", "input", "Run"),
                   _port("in1", "input", "sizing1"),
                   _port("addInput", "input", "Add input")],
                  [_port("Run", "output", "Run")])
    reader = _bus("Segments",
                  [_port("Run", "input", "Run")],
                  [_port("Run", "output", "Run"),
                   _port("out1", "output", "sizing1", desc="sizing1")])
    app = _app([writer, reader])
    app._propagate_data_bus_rename("Segments", {"sizing1": "sizing_A"}, exclude=writer)

    op = reader.output_ports[1]
    assert op.description == "sizing_A", op.description
    assert op.display_name == "sizing_A", op.display_name  # no alias -> follows


def test_custom_alias_preserved():
    reader = _bus("Segments",
                  [_port("Run", "input", "Run")],
                  [_port("out1", "output", "S1", desc="sizing1")])  # alias "S1"
    app = _app([reader])
    app._propagate_data_bus_rename("Segments", {"sizing1": "sizing_A"}, exclude=None)

    op = reader.output_ports[0]
    assert op.description == "sizing_A", op.description   # canonical renamed
    assert op.display_name == "S1", op.display_name       # alias kept


def test_other_pool_untouched():
    other = _bus("Analysis",
                 [_port("in1", "input", "sizing1")],
                 [])
    app = _app([other])
    app._propagate_data_bus_rename("Segments", {"sizing1": "sizing_A"}, exclude=None)
    assert other.input_ports[0].display_name == "sizing1"  # different pool


def test_excluded_block_untouched():
    writer = _bus("Segments", [_port("in1", "input", "sizing1")], [])
    app = _app([writer])
    app._propagate_data_bus_rename("Segments", {"sizing1": "sizing_A"}, exclude=writer)
    assert writer.input_ports[0].display_name == "sizing1"  # excluded


def test_empty_description_output_follows():
    reader = _bus("Segments", [],
                  [_port("out1", "output", "sizing1", desc="")])  # canonical = display
    app = _app([reader])
    app._propagate_data_bus_rename("Segments", {"sizing1": "sizing_A"}, exclude=None)
    assert reader.output_ports[0].display_name == "sizing_A"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
