"""Regression tests for undo/redo fidelity on the canvas.

Covers two bugs where undo failed to restore the exact prior state:
  1. A block's custom color was discarded on undo (load_from_struct re-derived
     it from the registry instead of honoring the snapshot).
  2. Dynamic ports (e.g. Data Bus inputs) vanished on undo because the snapshot
     aliased the live `block.parameters` dict — a later in-place rebind of
     `inputPortDefs` corrupted the snapshot already on the undo stack.

Run:  QT_QPA_PLATFORM=offscreen python App/test_undo_fidelity.py
"""
import os, sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication

from block_registry import BlockRegistry
from workflow_canvas import WorkflowCanvas

_APP = QApplication.instance() or QApplication([])


def _canvas():
    reg = BlockRegistry(); reg.scan_block_defs()
    return WorkflowCanvas(reg, None)


def test_undo_preserves_custom_color():
    """Fix 2: undo restores the exact saved color, not the registry color."""
    c = _canvas()
    blk = c.add_block("ConstantBlock", (0, 0))
    blk.color = (0.95, 0.10, 0.40)            # user-picked color
    assert tuple(round(x, 3) for x in blk.color) != \
        tuple(round(x, 3) for x in c.registry.get("ConstantBlock")["color"])
    c.push_undo()
    c.add_block("ConstantBlock", (0, 200))    # any later, undoable action
    c.undo()
    restored = next(b for b in c.blocks if b.definition_name == "ConstantBlock")
    assert tuple(round(x, 3) for x in restored.color) == (0.95, 0.10, 0.40), \
        f"color not preserved: {restored.color}"


def test_serialize_state_is_detached_from_live_params():
    """Fix 1 (root cause): an in-place edit to a live block's parameters must
    NOT mutate a snapshot already pushed onto the undo stack."""
    c = _canvas()
    blk = c.add_block("ConstantBlock", (0, 0))
    blk.parameters["marker"] = "before"
    c.push_undo()
    snap = c._undo_stack[-1]
    blk.parameters["marker"] = "after"        # mutate live params after snapshot
    snap_block = next(b for b in snap["blocks"] if b["id"] == blk.id)
    assert snap_block["parameters"]["marker"] == "before", \
        f"snapshot aliased live params: {snap_block['parameters']['marker']}"


def test_undo_restores_dropped_databus_port():
    """End-to-end (real app): deleting a wire drops a Data Bus dynamic input;
    undo must bring the port and wire back. Builds its own scenario so it does
    not depend on whatever workflow happens to auto-load."""
    from workflow_app import NPSWorkflowApp
    w = NPSWorkflowApp()
    c = w.canvas

    db = c.add_block("DataBus", (900, 900))
    for y in (900, 1050):
        src = c.add_block("ConstantBlock", (600, y))
        add = next(p for p in db.input_ports if p.name == "addInput")
        c.add_wire(src.output_ports[0], add)   # app handler promotes -> in1, in2
    ins = lambda b: [p.name for p in b.input_ports if p.name.startswith("in")]
    assert ins(db) == ["in1", "in2"], ins(db)

    wire = next(p for p in db.input_ports if p.name == "in2").connections[0]
    c.push_undo()                              # mimics delete_selected / _delete_wire
    c.remove_wire(wire)
    db2 = next(b for b in c.blocks if b.id == db.id)
    assert ins(db2) == ["in1"], ins(db2)

    c.undo()
    db3 = next(b for b in c.blocks if b.id == db.id)
    assert ins(db3) == ["in1", "in2"], f"dropped port not restored on undo: {ins(db3)}"


def test_collapsed_block_keeps_port_labels_after_undo():
    """A collapsed block shows only its connected ports. load_from_struct draws
    block graphics before restoring wires, so collapsed blocks must be redrawn
    afterward or their port labels vanish on undo (no _refresh_dynamic_ports)."""
    c = _canvas()
    src = c.add_block("ConstantBlock", (0, 0))
    dst = c.add_block("InvertData", (220, 0))   # static 'data' input
    di = next(p for p in dst.input_ports if p.name == "data")
    c.add_wire(src.output_ports[0], di)
    dst.is_collapsed = True
    c.update_block_graphics(dst)
    labels = lambda b: len(c._block_items[b.id].label_items)
    assert labels(dst) >= 1, "precondition: collapsed connected port should have a label"

    c.push_undo()
    src.position = (src.position[0] + 30, src.position[1])  # any undoable change
    c.undo()
    dst2 = next(b for b in c.blocks if b.id == dst.id)
    assert labels(dst2) >= 1, \
        f"collapsed block lost its port labels after undo: {labels(dst2)}"


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
