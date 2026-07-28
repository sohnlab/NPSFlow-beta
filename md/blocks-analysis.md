# Analysis Blocks

## Mean (`MeanBlock`) — dynamic
Computes mean of connected inputs. Single input → scalar mean. Multiple → column/row mean.
- Dynamic inputs: `Add input`
- Outputs: `mean`
- Parameter: `mode` (column/row)

## W. Mean (`WeightedMean`)
Weighted mean calculation from segment data.
- Dynamic inputs: segment structs
- Outputs: weighted mean values

## dR/R (`DROverR`)
Computes relative resistance change (dR/R).
- Inputs: segment data
- Outputs: `dROverR`

## Difference (`DifferenceBlock`)
Computes difference between two inputs.
- Inputs: `in1`, `in2`
- Outputs: `difference`

## Quotient (`QuotientBlock`)
Element-wise division `dividend / divisor`. Accepts scalars and 1D vectors
(column/row vectors are squeezed to 1D; genuine 2D arrays are rejected). A
scalar operand broadcasts across a vector (`s/v`, `v/s`); two vectors divide
element-wise and must match in length. Division by zero returns `inf`.
- Inputs: `dividend`, `divisor`
- Outputs: `quotient`

## Calculate Diameter (`SizeCalculation`)
Calculates particle diameter from NPS measurements using device geometry.
- Inputs: `dROverR`, `geometry`
- Outputs: `diameter`

## Deformed Diameter (`DeformedDiameter`)
Calculates deformed diameter from strain measurements.
- Inputs: measurement data, geometry
- Outputs: `deformedDiameter`

## 1D Strain (`Strain1D`)
Computes 1D strain from diameter measurements.
- Inputs: diameter data
- Outputs: `strain`

## Recovery Analysis (`RecoveryAnalysis`)
Analyzes recovery behavior of pulses.
- Inputs: segment data
- Outputs: recovery metrics

## Recovery Info (`RecoveryInfo`) — dynamic
Extracts recovery information from segments.
- Dynamic inputs
- Outputs: recovery data

## Velocity (`NodePoreVelocity`)
Calculates transit velocity through node-pore segments from time and length.
- Inputs: `Time` (N_pulses x N_segments), `ChannelLength`
- Outputs: `velocity` in the selected output unit (same shape; single column → per-pulse vector)
- Units (configurable via Edit Node): `timeUnit` (ms/s), `lengthUnit` (µm/mm/m), `outputUnit` (m/mm/µm per s/ms, default mm/s)
- Inputs are normalised to (mm, s); the editable formula (`ChannelLength / Time`) yields mm/s, then the result is scaled to `outputUnit`
- Pair with **Width → Time** to convert a sample-count width matrix to time.

## Width → Time (`SegmentWidthToTime`)
Converts a segment width matrix (samples) to transit time using the global sample rate.
- Inputs: `widthMatrix` (N_pulses x N_segments, samples)
- Outputs: `timeMatrix` (same shape; single column → per-pulse time vector)
- Parameter: `units` — `ms` / `s` (default `ms`); edit via Cmd/Ctrl+click or right-click → Edit Node
- Calculation: `width / sampleRate`, scaled to units

## wCDI (`WCDI`)
Weighted Cell Deformability Index calculation.
- Inputs: deformability measurements
- Outputs: `wCDI`

## Analyze (`AnalysisMethodSelector`) — interactive
Interactive method selection for analysis operations.

## Segment Processing (`SegmentProcessing`) — dynamic
Processes segment data with configurable operations.
- Dynamic inputs/outputs

## Detection Agreement

Compares two detector outputs on the same trace symmetrically (neither is treated as ground truth). Use it to validate the realtime detector against the offline detector, or to compare parameter sweeps.

**Category:** Analysis
**Interactive:** yes (modal dialog with preview)

**Inputs:**
- `dataIn` (required) — underlying 1-D signal. Multi-channel input uses column 0.
- `pulsesA` (required) — Nx2 `[start, end]` indices from detector A (e.g. Pulse Detection BC).
- `pulsesB` (required) — Mx2 `[start, end]` indices from detector B (e.g. RT Detection Combined).
- `sampleRate` — injected from the global Sample Rate / Define Parameters block. Used for the ms-axis in the preview only; match math is in samples. The block errors out if no global sample rate is available.

**Outputs:**
- `score` — scalar in `[0, 1]`. The formula (F1, Jaccard, or Mean IoU) is selected in the dialog.
- `matchTable` — Kx5 array: `[side, idx, partner_idx, iou, midpoint_offset_samples]`. `side` is 0 for A, 1 for B. `partner_idx` is the row index in the *other* set, or `-1` if unmatched. For unmatched rows, `iou` and `midpoint_offset_samples` describe the best non-gated candidate (useful for diagnosing why a match was rejected).

**Match rule:** Two pulses match iff **both** enabled gates pass:
1. *IoU gate* — `IoU(a, b) ≥ iou_thr` (default 0.5).
2. *Midpoint gate* — `|midpoint(a) − midpoint(b)| ≤ midpoint_tol_samples` (default 50).

Each gate can be toggled off independently. Pairing is greedy by IoU desc, tiebroken by smaller midpoint offset.

**Scores (all three computed every run):**
- `F1 = 2·TP / (2·TP + FP + FN)` — TP = matched, FP = B-only, FN = A-only.
- `Jaccard = TP / (TP + FP + FN)`.
- `MeanIoU` = mean IoU over matched pairs (0 if none).

**Settings** are auto-saved on Confirm to `SavedTemplates/Analysis/DetectionAgreement/temp.json` and re-loaded on next run. The dialog also has Save / Save As / Load buttons for named presets in the same folder.
