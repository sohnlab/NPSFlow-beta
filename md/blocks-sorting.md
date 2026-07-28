# Sorting Blocks

Real-time-style detection blocks designed for the sorting pipeline (low
latency, simple parameters, sliding-window live simulation for tuning).
Both blocks share the same dialog layout and live-feed simulation behavior.

## RT Detection (CWT) (`RealtimeDetectionCWT`) — interactive
Matched-filter detection. Convolves the input signal with the mother
wavelet at several width scales, combines coefficients across scales into
a 1-D detection score, peak-picks above a robust MAD threshold, and uses
signal-domain segmentation within each region to split coincident events.

- Inputs: `dataIn`, `wavelet` (from Wavelet Editor), `loadFile` (optional)
- Outputs: `detectedPulses` (Nx2), `score` (full-length), `coincidentHint`
  (length-N int8; 1 = split-coincident sub-region)
- Auto-saves to `SavedTemplates/Detection/RealtimeCWT/temp.json`
- Sample rate read from the global `Global Sample Rate` block (defaults to
  10 kHz if absent).

**Best for**: low-SNR data, smooth single-bump templates. Multi-plateau
templates produce autocorrelation side-lobes — the block compensates with
a template-aware autocorrelation floor, but for high-SNR plateau pulses,
the simpler `RT Detection (Threshold)` block usually performs comparably
with fewer knobs.

## RT Detection (Threshold) (`ThresholdDetection`) — interactive

Causal pulse detection on the running first-difference of the signal.
Same approach as `Pulse Detection BC` but causal (one-pole IIR instead
of `filtfilt`) so the same algorithm runs in batch and real-time.

**Why diff-then-threshold**: a slowly varying baseline produces a near-
zero diff, so drift is rejected naturally. Sharp pulse edges produce
strong rising/falling diff peaks that cluster into events.

- Inputs: `dataIn`, `loadFile` (optional)
- Outputs: `detectedPulses` (Nx2 padded extents), `coreRegions` (Mx2
  peak-cluster spans, M ≥ N when merging happened), `score` (filtered Δ
  signal), `coincidentHint` (always 0 for this block)
- Auto-saves to `SavedTemplates/Sorting/RealtimeThreshold/temp.json`
- Sample rate read from the global block (defaults to 10 kHz).

### Pipeline

```
signal → [Pre-filter: notch + LPF, causal] → [diff] → [IIR α smoothing]
       → ±k·σ threshold (find_peaks both signs)
       → peak grouping (merge_gap)
       → padding (factor of cluster core width)
       → post-padding region merging (gap ≤ merge_gap)
       → drop core spans < min_event
       → output detectedPulses + coreRegions
```

### Parameters

**Detection**:
- `Threshold k+` / `Threshold k−` (default 3.0 each): rising/falling
  thresholds in units of σ. Independently draggable on the bottom plot.
- `Smoothing` (default 0.3): one-pole IIR α applied to the diff. Lower
  α = heavier smoothing, smaller noise σ but smaller pulse-edge peaks.
  α = 1 disables smoothing.
- `Merge gap (ms)` (default 100): max peak-to-peak spacing within one
  region. Two purposes: (1) groups multi-segment-pulse internal edges
  into a single cluster, (2) merges adjacent padded regions if the gap
  between them is smaller than this.

**Event width** (asymmetric padding around each peak cluster):
- `Pad before (×)` / `Pad after (×)` (default 0.2 each): padding as a
  *factor* of the cluster's core width (peak-to-peak span). Wider events
  get proportionally wider padding — narrow noise-blip detections stay
  tight, wide multi-segment pulses get wide context.

**Other**:
- `Min event (ms)` (default 5): drop any cluster whose CORE span (peak-
  to-peak, padding excluded) is shorter than this.

**Pre-filter (signal)** — applied before the diff:
- `Lowpass` checkbox + cutoff Hz spinbox: causal Butterworth lowpass,
  `lfilter_zi` initial conditions to avoid startup transients.
- `Notch` checkbox + frequency Hz + Q: causal IIR notch (mains
  interference rejection). Default 60 Hz at Q = 30 (~2 Hz wide).

### Visualization

- **Light green** band = the full padded detection region (one per
  output row in `detectedPulses`).
- **Dark green** band overlay = the **core span** (first to last peak)
  of each underlying cluster. After merging, multiple dark cores can
  appear inside one light region.

### Best for

High-SNR multi-plateau pulses (typical NPS data). Much cheaper than CWT
in compute and FPGA implementation. The diff + IIR + threshold pipeline
maps directly to a few DSP slices.

## RT Detection (Combined) (`CombinedDetection`) — interactive

Two detectors running in parallel — a causal matched-step (same idea
as Threshold but with a tunable kernel width) plus an optional Ricker
(Mexican-hat) CWT side-channel — with peak unions feeding the same
clustering / padding / merge pipeline as the other RT blocks.

```
peaks = peaks_from(matched_step, if diff_enabled)
      ∪ peaks_from(ricker_cwt,   if cwt_enabled)
       → cluster (mergeGap) → drop core < minEvent → pad → post-merge
```

- Same I/O as Threshold; auto-saves under
  `SavedTemplates/Sorting/RealtimeCombined/temp.json`.
- Either detector can be toggled on/off independently. With **only**
  the diff enabled (`Enable Ricker CWT` off, `Enable diff thresholding`
  on), the block behaves like `RT Detection (Threshold)` with a
  configurable matched-step width.

### Diff thresholding (matched-step)

Causal matched-step score:

```
score[i] = mean(sig[i-w+1 : i+1]) − mean(sig[i-2w+1 : i-w+1])
```

i.e. recent w-sample mean minus prior w-sample mean. With `w = 1` this
reduces to `np.diff` exactly. With `w > 1` it integrates over the
expected pulse-edge duration, improving SNR by ≈√w on step-like
edges — useful when the pulse edge is smeared across many samples and
the per-sample diff is buried in noise.

- `Enable diff thresholding` (default on): when off, the score is still
  computed and shown for diagnostics but its threshold lines / peaks
  don't contribute to detection.
- `Threshold k+ / k−` (default 3.0): rising/falling thresholds in σ̂
  units. Independently draggable green/red dashed lines.
- `Edge width (ms)` (default 1.0): converted to `w = round(edge_width_ms
  × sample_rate / 1000)`. Set to 0 ms to fall back to the 1-sample diff.
- `Smoothing` (default 0.3): one-pole IIR α applied to the matched-step
  output. Lower = heavier smoothing.
- Tradeoff: a wide `w` blurs short events and merges close pulses.
  Size it to your slowest expected edge; for sharp high-SNR pulses
  use a small `w` (1–10 samples) or the Threshold block.

### CWT side-channel (low-SNR helper)

Optional Ricker (Mexican-hat) CWT runs in parallel with the matched-
step. Either detector firing is enough to open a region.

- `Enable Ricker CWT` (default off)
- `Width (ms)` (default 20.0): wavelet half-width. Best detection when
  this matches the expected pulse half-width — at that scale the Ricker
  is roughly the matched filter for a Gaussian-shaped bump.
- `Threshold k` (default 5.0): `|cwt| > k · σ_cwt` counts as a peak.

The CWT trace appears in magenta on the score plot, rescaled so its
σ matches the matched-step σ̂ — both noise floors and thresholds share
a single y-axis. CWT threshold lines (±) are draggable just like the
diff thresholds; both lines move together since the rule is symmetric.

### Detection merging / Padding (separate panel groups)

The cluster→pad→merge pipeline parameters live in their own UI groups
in the right panel:

- **Detection merging** — `Merge gap (ms)` (default 100), `Min event (ms)`
  (default 5).
- **Padding** — `Pad before (×)` and `Pad after (×)`, both factors of
  the cluster's core width (default 0.2 each).

### Live simulation — freeze-and-skip streaming

When run in `Live simulation` mode, the dialog re-runs `detect_combined`
on `signal[_frozen_until : revealed]` each tick (~33 ms wall-time at
1× speed), splices the chunk's score / cwt / signalFiltered into
cumulative full-length buffers, and decides per-region whether to
freeze it.

Freeze rule: a detected region whose end + 1 s lies before the current
revealed sample has converged — the chunk has moved well past it, so
its boundaries won't change. It moves into a frozen list and
`_frozen_until` advances past the region's end. Subsequent ticks
exclude those samples from processing entirely (the chunk shrinks,
re-processing cost drops). Re-detections that overlap an already-
frozen entry are dropped, so a multi-plateau pulse whose core
freezes before its padded region won't accumulate duplicate cores
during the freeze-window lag.

## Shared UI (RT detection blocks)

- **Mode group**:
  - `Process All` — one-shot detection on the full record.
  - `Live simulation` — chunked streaming with freeze-and-skip (see
    above). Sim speed (`1× / 10× / 100× / asap`), `Window (s)`, and
    `Run / Pause / Step / Stop` controls.
- **Step button** advances the live sim by one ~33 ms batch and
  redraws. Works while running (auto-pauses), while paused, or from
  cold (initializes a paused sim at t = 0). Re-clickable to step
  through batch-by-batch.
- Matplotlib navigation toolbar (zoom/pan/home/save), with synced
  x-axes between the signal plot (top) and the score plot (bottom).
- The top plot shows the **filtered signal** when LPF/Notch are on;
  raw signal otherwise. In live mode the entire processed history
  `[0, revealed]` is plotted (xlim auto-scrolls to the live window;
  Home / zoom-out reveals everything processed so far).
- **Threshold drag**: drag any threshold line in the bottom plot to
  retune (toolbar must not be in pan/zoom mode). Hovering near a line
  shows a vertical-resize cursor. Drag uses blit-based redraw so it
  stays smooth on long records.
- **Info panel** shows `Sample rate`, cumulative `Detected` count,
  live-feed `Batch size` (= `sample_rate × 33 ms` ≈ 330 samples at
  10 kHz), and `Process time` distribution (`N / Min / Mean / Max ms`)
  measured per chunk during streaming + 10 sample-time chunks per
  Update Detection click. Useful as a real-time / FPGA feasibility
  check.

## ML Training (`MLTraining`) — interactive

Trains a machine-learning event classifier — **baseline (0) / single (1) /
coincident (2)** — from a folder of labeled recordings (Data Labeling
output, `.npz` or `.csv`, scanned recursively). Noise is treated as
baseline; uncertain labels are dropped. Designed for real-time use: the
saved model classifies a live stream through a causal trigger +
grow-the-window pipeline (`processing/ml_training_core.py`,
`StreamEventClassifier`).

- Inputs: `run` (optional), `folder` (optional — else Browse in dialog),
  `sampleRate` (optional, overrides dialog), `settingsFile` (optional)
- Outputs: `modelFile` (path to `.joblib` bundle), `metrics` (struct)
- Auto-saves settings to `SavedTemplates/ML/temp.json`

**Pipeline**: files are split **train / validation / test at the file
level** (no leakage; auto split spreads coincident-rich files, or pin a
file via its Split combo). Each labeled event is trigger-aligned
(`refine_event_extent`) so training windows match what the live trigger
sees, then featurized causally at several latency horizons (default 1–40 ms
+ full) with a pre-onset baseline. Negatives = random background windows +
**hard negatives** (background stretches where the trigger actually fires).
Models: Random Forest (default) or Gradient Boosting; two-stage strategy
(event-vs-baseline, then single-vs-coincident with SMOTE oversampling) or
flat 3-way. The shipped model is refit on train(+val) after evaluation.

**Streaming engine**: per-sample max-zone |z| against a rolling baseline,
smoothed by a short causal moving average; the baseline only adapts while
deeply quiet, freezes during events, and re-seeds after each event (level
shifts self-heal). Sustained activity above Trigger σ opens an event, the
model re-classifies at each horizon as samples arrive (early decisions),
sustained quiet below Release σ (or Max event) closes it. Enable *Simulate
real-time stream on test* to score this end-to-end on the held-out test
files (detection recall, classification accuracy, FP/min).

**Results tabs**: per-horizon accuracy/macro-F1/recalls (val + test),
score-vs-latency and confusion plots, stream-test table, feature
importances.

**Reading the stream-test FP number**: predicted events are scored against
the per-sample labels, so any *unlabeled* real pulse the detector finds
counts as a false positive. On sparsely-labeled recordings (e.g.
`dev2_t2`: ~16 % of samples labeled, background full of pulse-like
activity) the FP/min figure is an upper bound, not a model error rate —
cross-check the window-level test metrics, which use only labeled spans.

## ML Classify (`MLClassify`) — interactive

Streams a signal through a trained classifier bundle from ML Training and
reviews the predictions. The dialog runs the causal StreamEventClassifier
in a background thread (progress bar), then plots **predicted events as a
colored strip above the signal vs true labels as shaded regions** — false
positives get a red outline — with an event-level score line (detected /
correct / FP per min) when labels are present.

- Inputs: `modelFile` (from ML Training), `dataIn` (raw array, Load File
  struct/entry, or Data Labeling output — labels found inside become the
  ground truth), `trueLabels` (optional override)
- Outputs: `predLabels` (per-sample 0/1/2), `events` (dicts with onset,
  end, label, proba, per-horizon updates), `summary` (score struct, None
  without labels)
- A multi-file Load File struct classifies the first file (use Combine
  Data upstream to stream them all as one).

The `Sorter_v5_ML_training.json` workflow chains this after ML Training:
Load File picks a held-out test recording, ML Classify replays it through
the freshly trained model, and the review plot answers "what did the model
call each event, and what was it really?".

## Modular ML chain (`MLSplit` / `MLTrain` / `MLEvaluate`)

Non-interactive building blocks that decompose the ML Training wizard into
a plain dataflow pipeline (the wizard remains as the all-in-one
alternative). `workflows/Sorter_v5_ML_training.json` is the reference
layout:

```
Text (folder) ─► Load File ─► ML Split ─┬► ML Train ─┬► Evaluate (val)  ─► Display
Constant 50 kHz ─────────► ML Train     │ (trainSet) ├► Evaluate (test) ─► Display
                                        │            └► ML Classify     ─► Display
                                        └ valSet / testSet feed the evaluators
```

- **ML Split** — train/val/test split (params: split mode, fractions,
  seed). *By file* (default): whole recordings per split — the leak-free,
  coincident-aware assignment the wizard uses; scores estimate performance
  on unseen recordings. *By event*: every file is cut into per-event
  segments (cuts at background-gap midpoints, one event per segment with
  its margins) and the pooled segments are mixed across splits — all files
  contribute to train AND test, but splits then share each recording's
  noise/drift character, so scores read optimistic (Sorter V5: test
  macro-F1 0.98 by event vs 0.86 by file — the gap is the leakage).
  Outputs three `{fileNames, files}` structs + `splitInfo`.
- **ML Train** — trains on `trainSet` only and saves the bundle
  (params: model rf/gb, strategy, oversampling, baseline/hard-negative
  ratios, horizons, seed, sample rate, model path; `sampleRate` port
  overrides the parameter). Outputs `modelFile` + `trainInfo`.
- **ML Evaluate** — scores any labeled set against a bundle; windows are
  rebuilt with the config stored in the bundle so numbers are comparable.
  Set *Stream eval = yes* for the (slower) per-file real-time simulation.
- **ML Classify** closes the loop: wire `testSet` into `dataIn` and the
  review dialog plots predictions vs true labels for one test file
  (*File index* parameter selects which).

## Extract Labeled Pulses (`ExtractLabeledPulses`)

Converts a per-sample labeled signal into Detection-Review-style pulse
regions so labeled recordings can drive the feature pipeline directly —
no detection or interactive review pass needed.

- Input `labeledData` accepts a Load File struct (`{files: [...]}`), a
  single `{data, labels}` entry (dot-prefixed `labeledData.*` keys fine),
  a list of entries, or a bare data array with the `labels` port wired
  separately.
- Regions are contiguous runs of the same label id, using the standard
  class convention: 1 = single, 2 = coincident, 3 = noise, 4 = uncertain
  (0 = baseline, ignored). End indices are inclusive, matching Pulse
  Processing's `data[s:e+1]` slicing.
- Output `pulseList`: one dict per file — `{fileName, data, single,
  coincident, noise, uncertain}` — where each class field is an Nx2
  `[start, end]` index array local to that file's data (multizone
  columns preserved). Files are never concatenated, so regions can't
  fuse across file seams. Wire it straight into Pulse Processing's
  `dataIn` (no separate `pulseLabels` needed); its per-file LP Filter
  group then lists these files.
- Interactive: after extraction a paginated 3×3 grid viewer opens (same
  style as Debug - Labeling) with every zone overlaid and the labeled
  span shaded. **File** and **Class** selectors filter the grid; context
  margins are clamped to each pulse's own file so cells never plot across
  file seams. Close continues the pipeline.
