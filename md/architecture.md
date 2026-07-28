# Architecture

## Core Components

**NPSWorkflowApp** (`App/workflow_app.py`)
- QMainWindow: toolbar (top), canvas (center), log panel (bottom)
- Toolbar: File ops, editing, undo/redo, layout, Reload, settings, help
- Floating "Add Block" button for block palette access
- Owns BlockRegistry, WorkflowEngine, WorkflowCanvas, DataInspector
- `root_dir` = project root, `app_dir` = App/ folder

**WorkflowCanvas** (`App/workflow_canvas.py`)
- QGraphicsView/QGraphicsScene for blocks and wires
- Mouse: drag blocks, draw wires, pan, zoom (scroll wheel)
- Keyboard: Ctrl+Z/Y undo/redo, Delete, Ctrl+A auto-layout, Ctrl+=/- zoom, Ctrl+0 fit
- Right-click context menus on blocks/canvas/wires
- Undo/redo stack for all operations

**WorkflowEngine** (`App/workflow_engine.py`)
- DFS-based topological sort with Sequence/Priority ordering
- Gathers inputs from upstream wires, validates required inputs
- Calls runner function, stores outputs, updates status
- Stops immediately on first error (all run modes)
- `run_from_block`: resets status of target + downstream blocks before re-running
- ReferenceStore / DataBusStore: global variable stores
- GlobalVars: auto-injected variables (e.g. sampleRate)

**BlockNode** (`App/block_node.py`)
- Data model: id, definitionName, displayName, category, color, position, size
- Status: "pending" | "running" | "done" | "error" | "skipped"
- Parameters dict, OutputData dict
- Flags: isInteractive, isVisualizationOnly, isCompact, isDynamic, etc.

**DataInspector** (`App/data_inspector.py`)
- Opens for any block with output data OR input data (no output required)
- Variable table with detail panel and matplotlib plotting

## Block Definition Format

Each file in `App/blockdefs/` exports `get_definition()`:

```python
def get_definition():
    return {
        "name": "MyBlock",
        "displayName": "My Block",
        "category": "Preprocessing",
        "color": [0.3, 0.6, 0.9],
        "inputs": [{"name": "signal", "type": "any", "required": True}],
        "outputs": [{"name": "result", "type": "any"}],
        "runner": "runMyBlock",
        # Optional flags:
        "isInteractive": False,
        "isDynamic": False,
        "isPlotData": False,       # Paired x/y dynamic ports
        "defaultParameters": {},
    }
```

Categories: FlowControl, DataIO (Input/Output), Variables, Preprocessing, Filters, Template, Detection, FeatureExtraction, Analysis, Utility

## Dynamic Port System

Blocks with `isDynamic: True` use an `addInput` placeholder port:
- Connecting a wire to `addInput` creates a numbered port (in1, in2, ...)
- A fresh `addInput` appears at the bottom
- Fixed ports (defined in blockdef, e.g. "filename", "Run") are preserved
- PlotData blocks create paired x/y ports per series with "Add series"

## Runner Function Format

Each file in `App/runners/` implements `run(inputs, params, block)`:

```python
def run(inputs, params, block):
    signal = inputs.get("signal")
    return {"result": processed_signal}
```

Interactive runners open PySide6 dialogs. Closing without confirming raises ValueError to stop pipeline.

## Execution Flow

1. `NPSflow.py` → single-instance lock → QApplication → NPSWorkflowApp
2. Registry scans `App/blockdefs/`, engine and canvas created
3. User builds pipeline: add blocks, connect wires, configure parameters
4. Run: BFS from Start block → topological sort → execute reachable blocks → stop at first error
5. Per-block: gather inputs → call runner → store outputs → update status
6. Inspect: right-click block → DataInspector shows input/output data

## Settings Save/Load System

All interactive blocks auto-save to `SavedTemplates/<folder>/temp.json` on confirm, and auto-load `temp.json` on dialog open.

| Block | Folder |
|-------|--------|
| Trim Data | SavedTemplates/Trim/ |
| Detection (BC) | SavedTemplates/Detection/ |
| Detection Review | SavedTemplates/Review/ |
| Batch Processing | SavedTemplates/Processing/ |
| Template Settings | SavedTemplates/PulseShape/ |
| Device Geometry | SavedTemplates/Geometry/ |

Loading: connect a Text block to the `settingsFile`/`loadFile` port. Accepts:
- Relative path from project root (e.g. `SavedTemplates/Detection/temp.json`)
- Plain name (e.g. `Sorter Demo`) → resolved to `SavedTemplates/<folder>/<name>.json`
- Absolute path

Serialization: all numpy scalars must be converted to Python ints/floats before `json.dump`. Use `ndarray.tolist()` or explicit `int(x)` casts. File writes use atomic tmp+rename to avoid truncated files on error.

Template Settings saves/loads using sequence-based mapping:
- `key_peak_seq_indices`: position of each key peak within all detected peaks
- `exclusion_zone_seq_bounds`: zone boundaries as **key peak** indices (-1 = before first key peak, N = after last key peak). Key peaks are used as anchors because exclusion zones are defined between consecutive key peaks.
- On load: thresholds are applied first to re-detect peaks, key peaks are restored via sequence indices, then exclusion zones are mapped using the restored key peak positions
