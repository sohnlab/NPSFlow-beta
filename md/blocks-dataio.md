# Data I/O Blocks

## Load Data (`LoadData`) — interactive
Unified loader for `.mat`, `.csv`, `.npz`, and `.json` files. Replaces the
three legacy load blocks (`LoadNPSData`, `LoadCSVData`, `LoadLabeledData`) in
new workflows; the legacy blocks remain registered for backward compatibility.

- Inputs: `run`, `path` (optional file or folder; both unrequired)
- Output: `data` — single struct meant to be fed into an **Unpack** block
  (or a **Combine Data** block to concatenate all files):
  ```
  {
    fileNames: [str],          # basenames / relative paths
    files:     [ {fileName, ...content..., labels?} ],
  }
  ```
- **Runtime picker**: when no `path` is wired and no `lastPath` is saved
  (or saved path is missing), a dialog asks the user to choose **Single
  File…** or **Folder…**. The native picker then opens filtered to
  `.mat .csv .npz .json`.
- Single-file selection → one entry in `files`.
- Folder selection → recursive scan; `files` holds one entry per file.
- Labeled CSV/NPZ files (with `labels` / `labeledData.labels` column) get
  their `labels` array auto-cast to `int32` inside each file entry.
- Remembers `lastPath`; default folder: `SignalData/`.
- (A former `combined` key — an eager column-wise concat of all tabular
  files — was removed: it doubled load time and memory and nothing consumed
  it. Use the Combine Data block instead.)

## Combine Data (`CombineData`)
Concatenates the per-file entries of a Load File struct into single arrays,
in file order. Explicit replacement for Load File's former `combined` key —
combination now happens only when a pipeline asks for it, and the seams
between files are reported instead of hidden.

- Input: `dataIn` — the Load File struct (`{files: [...]}`) or a bare list
  of file entries
- Outputs: `data` (concat along axis 0), `labels` (concat; `None` when any
  file lacks them), `boundaries` (length N+1 start indices; file k spans
  `boundaries[k]:boundaries[k+1]` — use to mask the artificial
  discontinuities at file joins), `fileNames`
- The data key is `data` when the entries share one; otherwise the single
  common numeric key. Mixed shapes (e.g. 3-zone vs 1-zone files) raise a
  clear error.

## Load NPS (`LoadNPSData`) — interactive (legacy)
Loads .mat signal data files. Opens file browser dialog defaulting to `SignalData/` folder.
- Outputs: `data`, `sampleRate`, `platform`

## Save File (`SaveFile`) — interactive, dynamic
Saves connected inputs as NPZ (default) or CSV. Each input becomes a column.
- Fixed input: `Filename` (optional string, suggests output filename)
- Fixed input: `trimIndex` (optional 2-value `[start, end]`, e.g. from Trim
  Data) — recorded as metadata, not as a padded data column. In NPZ it is a
  2-element `trimIndex` array; in CSV it is a `# trimIndex: start, end` header
  comment above the table. Malformed (non-2-value) input is skipped with a warning.
- Dynamic inputs: `Add input` (data columns)
- Default save folder: `Output/`
- Skips `filename`, `trimIndex`, and `addInput` ports when collecting data

## Text (`TextBlock`)
Outputs a constant string. Has separate Display Text and Output Text.
- Outputs: `text`
- Double-click to edit

## Constant (`ConstantBlock`)
Outputs a constant numeric value.
- Outputs: `value`

## Display (`DisplayBlock`)
Shows input data in a popup.
- Inputs: `data`

## Data Summary (`DataSummary`) — interactive
Shows summary statistics of input data.
- Inputs: `data`

## Unpack Variable (`UnpackStruct`) — dynamic, interactive
Unpacks fields of a (possibly nested) struct, list, or array into
individual output ports. Supports arbitrary dotted paths into nested
data, so a single block can replace a chain of cascaded unpackers.

- Inputs: `structIn` (any type — wrapped as `{"value": x}` if not a dict)
- Dynamic outputs: one per selected path

**Edit Node** dialog (right-click → Edit Node) shows a Data
Inspector–style structure tree with three columns (Name / Size / Type).
Each node has a checkbox; checked nodes become output ports. Tree
descends into both nested dicts (key-named children) and lists/tuples
whose items are dicts or arrays (`[i]`-named children), so paths like
`value.[0].data` work directly.

The structure tree is populated from the last successful run's cached
shape (`structShape` parameter). Run the pipeline once before opening
the dialog so the tree has data to display.

**Modes** (in order of priority on each run):
1. `selectedPaths` — explicit list of dotted paths from the dialog
2. `unpackDepth` + `visibleOutputs` — legacy uniform depth flattening

## Pack Variable (`PackVariable`) — dynamic
The inverse of **Unpack Variable**: collects every connected input into a
single `struct` (dict) output. Wire variables into the `Add input` port and
new numbered ports grow automatically.

- Dynamic inputs: `Add input` (auto-grows on connect)
- Output: `structOut` (`struct`) — `{label: value}` for each connected input

Each input's key is the port label, which defaults to the upstream
variable's name (same convention as Data Bus). Colliding keys get a `_2`,
`_3` suffix; injected globals (e.g. sample rate) are skipped. Right-click →
**Rename Inputs** (or Cmd/Ctrl-click) to customize the dict keys. The
`struct` output feeds straight into an Unpack Variable for a clean
round-trip.

## Pack Output (`CompileTable`)
Compiles multiple inputs into a table/struct.
- Dynamic inputs: `Add input`
- Outputs: `table`

## Export Pulses (`ExportAcceptedPulses`)
Exports accepted pulse data to file.
- Inputs: `data`, `acceptanceID`, `pulseIndices`

## Device Geometry (`DeviceGeometry`) — interactive
Defines NPS device geometry (nodes, pores, dimensions).
- Inputs: `geometry` (optional, load saved)
- Outputs: `geometry` (struct)
- Auto-saves to `SavedTemplates/Geometry/temp.json`
- Parameters: components list, channel height, De_np
