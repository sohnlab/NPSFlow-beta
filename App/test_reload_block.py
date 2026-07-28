"""Regression tests for single-block reload (Reload Block).

reload_block rescans block definitions, which resets every defn to its
category *scan* color (assign_dynamic_colors). init_from_definition then
repainted the block with that scan color, so a blue (section-colored) block
turned green on reload. The reload should update logic/ports, not appearance.

Run:  QT_QPA_PLATFORM=offscreen python App/test_reload_block.py
"""
import os, sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from workflow_app import NPSWorkflowApp

_APP = QApplication.instance() or QApplication([])
_WIN = NPSWorkflowApp()


def test_reload_preserves_section_color():
    c = _WIN.canvas
    blk = c.add_block("TrimData", (100, 100))
    before = tuple(round(x, 3) for x in blk.color)   # palette section color
    scan = tuple(round(x, 3) for x in _reg_scan_color("TrimData"))
    assert before != scan, "precondition: section color should differ from scan color"
    _WIN._reload_block(blk)
    after = tuple(round(x, 3) for x in blk.color)
    assert after == before, f"reload changed color {before} -> {after} (scan={scan})"


def test_reload_preserves_custom_color():
    c = _WIN.canvas
    blk = c.add_block("TrimData", (300, 100))
    blk.color = (0.90, 0.10, 0.50)                   # user-picked color
    _WIN._reload_block(blk)
    after = tuple(round(x, 3) for x in blk.color)
    assert after == (0.90, 0.10, 0.50), f"reload lost custom color: {after}"


def _reg_scan_color(name):
    from block_registry import BlockRegistry
    r = BlockRegistry(); r.scan_block_defs()
    return r.get(name)["color"]


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
