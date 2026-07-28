# NPSflow Quick Start

A five-minute walkthrough of running your first pipeline. It assumes you have
already installed the requirements and can launch the app — see
[Readme.md](Readme.md) if not.

## 1. Launch

```bash
python NPSflow.py
```

The app opens on the **Home** tab with **New Workflow** and **Open
Workflow...** buttons and a list of recent files.

## 2. Open the example pipeline

Click **Open Workflow...** and choose
[workflows/Sorter_v5.json](workflows/Sorter_v5.json). The canvas fills with a
sorter pipeline: load → preprocess → detect → review → extract features →
analyze.

Blocks are connected left-to-right by wires. You can drag blocks to rearrange
them, zoom with the scroll wheel, and pan by dragging the empty canvas.

## 3. Point the Load File block at a recording

Make sure you have a sample `.mat` file under `SignalData/` (see **Sample
data** in [Readme.md](Readme.md)).

**Double-click the "Load File" block.** A file picker opens — select your
recording. The block remembers the path for next time.

> Any block with an interactive UI opens a dialog when you double-click it.
> Detection, review, and template blocks work the same way.

## 4. Run the pipeline

Click the green **▶ Run All** button at the top-left of the canvas. (You can
also double-click the **Start** block, or right-click the canvas → **Run
All**.)

Blocks change color as they execute — **gray** pending, **yellow** running,
**green** done, **red** error. Interactive blocks pause the run and open their
dialog; confirm each dialog to continue. **Closing an interactive dialog
without confirming stops the pipeline** — that is the intended way to abort.

If a block turns red, an **Issues** panel opens with the error message and a
button to re-run from that block after you fix the cause.

## 5. Get your results

- **Save / export blocks** write CSVs to the `Output/` folder in the project
  directory.
- Plot blocks and review dialogs have an **Export Figure** button (quick-save,
  plus a menu to customize or copy the figure).

## Where to go next

- Right-click any block for a short description of what it does.
- Full per-block reference documentation is in [md/](md/).
- To build your own pipeline, start from **New Workflow** and drag blocks from
  the palette on the left; wire an output port to a compatible input port.
