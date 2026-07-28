# Feature Extraction Blocks

## Pulse Processing (`ExtractSinglePulseFeatures`) — interactive
Per-pulse peak detection, sequence matching, and accept/reject review. Paginated 3-per-page layout.

- Inputs: `dataIn`, `trendline` (optional), `pulseLabels` (optional if `dataIn` is a pulseList), `TPsettings`, `pulseFeatures` (optional), `settingsFile` (optional)
- Outputs: `pulsePeakLocations`, `rectangularizedPulses`, `acceptanceID`, `pulseStartIndices`, `pulseFeatures`
- `dataIn` also accepts Extract Labeled Pulses' `pulseList` (per-file `{fileName, data, single, ...}` entries): files are concatenated in order, each file's `single` regions become the pulse set (offset into the combined signal), and the file names drive the per-file LP Filter group
- Auto-saves to `SavedTemplates/Processing/temp.json`; auto-loads on dialog open
- `settingsFile`: accepts absolute path, relative path from project root, or plain name (e.g. "Sorter Demo" → `SavedTemplates/Processing/Sorter Demo.json`)

**Detection algorithm** (purely sequence-based):

1. Filter and compute derivative per pulse using TPsettings filter config
2. Find ▲ max key peak (strongest positive derivative) and ▼ min key peak (strongest negative) globally
3. Find ★ regular key peaks by counting threshold-exceeding peaks from nearest anchor in sequence order
4. Build exclusion zones from sequence-mapped boundaries
5. Detect remaining ● peaks using region-aware adaptive thresholding

**Interactive features:**
- Left plot: Signal & Rectangularized with peak markers
- Right plot: Derivative with thresholds, peaks, exclusion zones
- Accept/Reject classification table
- Summary: Total, Accepted (green), Rejected (red)
- Green/red outline on left plots based on acceptance
- Peak adjust mode: add/remove peaks on derivative plot (cursor circle, auto-cancel on leave)
- Expand/Trim tools for pulse boundaries
- Live sequence checking during peak adjustment
- Detrend toggle with Starting Value / Mean Value modes
- Median / Mean rectangularization toggle
- **Edge: Peaks / FWHM toggle** — global (all zones/channels); overrides the template's `edge_method`. FWHM slices each pulse's segment boundaries at the crossing of the lower adjacent plateau's half level (its own FWHM width) instead of the derivative-peak apexes. Re-slices live; the exported `pulsePeakLocations` and rect steps stay aligned. Defaults to the template's edge method.
- **LP Filter group** — per-file independent low-pass on top of the TPsettings filter stack: pick a file in the combo, enable, set cutoff (Hz). Only that file's pulses are re-filtered/re-detected (all zones); other files keep their own settings. Single-array input shows one "input" entry.
- Save/Load settings buttons, Export Figure
- Saved state includes: acceptance, rect method, edge method, detrend toggle & mode, show trendline, page, pulse indices, peak locations, key peaks & polarities, exclusion ranges, expand %, cursor radius, per-file LP filters

**Peak markers:** ▲ max (triangle up), ▼ min (triangle down), ★ regular (star), ● non-key (circle). All color-coded green/red by derivative sign.

## Coincident Processing (`ExtractCoincidentPulseFeatures`) — interactive
Decomposes coincident (overlapping) pulse regions into individual
single-particle pulses; emits the same schema as Pulse Processing as a
standalone branch.

- Inputs: `dataIn` (multizone capture), `trendline` (opt), `pulseLabels`
  (Nx2 coincident regions), `TPsettings`, `pulseFeatures` (opt),
  `settingsFile` (opt)
- Outputs: `pulsePeakLocations`, `rectangularizedPulses`, `acceptanceID`,
  `pulseStartIndices`, `pulseFeatures`, `coincidenceMap`
- Auto-saves to `SavedTemplates/Coincident/temp.json`

**Zones as roles:** the multizone `dataIn` columns are *roles* — one
**measurement** (required) plus optional **start**/**end** indicator columns.
Roles are picked in the review dialog (dropdowns) and persisted.

**Decomposition:**
1. *Timing mode* (start/end indicators present): detect entry spikes (start
   zone) and exit spikes (end zone); FIFO-pair them into per-particle
   occupancy windows.
2. *Fallback* (no indicators): detect measurement edge-peaks, estimate N =
   round(peaks ÷ template length), split at large gaps (sequential) or tile
   the template sequence (interleaved). If the template's fixed thresholds
   gate out every edge-peak in a region, an adaptive floor (a fraction of the
   region's peak derivative) re-gates so the region can still be split.
3. Each recovered span runs the single-pulse engine (`_compute_zone`) →
   peaks, rectangularized span, accept/warn/reject.

**Interactive:** paginated event cards (2/page) with color-coded recovered
particles, start/end markers, occupancy shading; per-event N override, window
edit (click-to-snap the nearest window edge), per-particle accept/reject;
detrend toggle (Starting/Mean-Value), median/mean rectangularization,
Save/Load, Export Figure.

**Detrending:** per recovered particle, the iterative baseline-fit
(`processing.pulse_detrend`) flattens the raised overlap plateau; also available
as an output toggle.

## Template Slicing (`PulseFeatureExtraction`) — interactive
Interactive segment definition UI for feature extraction.
- Inputs: `TPsettings`
- Outputs: `pulseFeatures` (struct with bounds, labels, methods)
- Feature Definition table: #, Label (editable), Method (read-only from TPsettings), Start, End

## Pulse Slicing (`PulseSlicing`) — interactive, dynamic
Slices pulses into segments based on feature definitions.
- Inputs: `pulseSegments`, `pulsePeakLocations`, `rectangularizedPulses`, ...
- Dynamic outputs: segment data structs per feature

## Segment Processing (`SegmentProcessing`) — dynamic
Processes segment data (width, startValue, endValue).
- Dynamic inputs/outputs: in1→out1, in2→out2, etc.

## Segment Width (`ExtractSegmentWidth`)
Extracts width from each connected segment, stacks as matrix.
- Dynamic inputs: `Add input`
- Outputs: `widthMatrix` (N_pulses x N_segments)

## Total Segment Width (`TotalSegmentWidth`)
Computes span from start of segment 1 to end of segment 2.
- Inputs: `Start Segment` (in1), `End Segment` (in2)
- Outputs: `width` (samples per pulse)
- Calculation: endGlobal[in2] - startGlobal[in1]

## Segment Info (`ExtractSegmentInfo`)
Extracts segment info (start/end values, width) from segment structs.

## Select Regions (`RegionSelectionUI`) — interactive
Interactive region selection for analysis.
