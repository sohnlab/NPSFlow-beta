# Detection Blocks

## Pulse Detection (BC) (`PulseDetectionBC`) — interactive
Pulse detection with baseline correction. Two-tab UI:

1. **Detection** — Derivative filtering, threshold adjustment, peak detection
2. **Baseline** — Baseline interpolation and smoothing

- Inputs: `dataIn`, `pulseTemplate` (optional), `filterConfig` (optional), `loadFile` (optional)
- Outputs: `trendline`, `detectedPulses`
- Auto-saves to `SavedTemplates/Detection/temp.json`
- Save/Load settings buttons; settings include all thresholds, margins, smooth window

**Multi-zone (mzNPS):** when `dataIn` has more than one column, a zone tab bar
appears above the Detection/Baseline stages; each zone is detected and
baseline-corrected independently. Outputs: `trendline` is N×Z (one column per
zone) and `detectedPulses` is zone-grouped (`{num_zones, zones:[Nx2,…]}`).
Single-zone data is unchanged (1-D trendline, bare Nx2).

## Detection Review (`PulseRegionReview`) — interactive
Paginated 3×3 grid review of detected pulse regions.

- Inputs: `detectedPulses`, `dataIn`, `settingsFile` (optional)
- Outputs: `single`, `coincident`, `noise`, `uncertain` (Nx2 index arrays per class)
- Auto-saves to `SavedTemplates/Review/temp.json`

**Features:**
- Quick classify buttons: All Single, All Coincident, All Noise, All Uncertain
- Per-pulse tools: Split, Expand, Trim (hover to reveal)
- Display mode: Both (Raw + Filtered), Raw Only, Filtered Only
- Color-coded outlines: green=single, orange=coincident, red=noise, grey=uncertain
- Summary stats (per-class counts, color-coded, multi-line)
- Save/Load settings preserves adjusted regions (splits/trims/expands) and labels

## CWT Evaluator (`CwtEvaluator`) — interactive
Compares detection performance across multiple wavelet/scoring
configurations on the same signal. Each "config" is one tab in the
right panel; user can add, clone, rename, and remove tabs (minimum 1).
Every config picks one mother-wavelet shape — the user-supplied
template (optional) or standard Ricker / Morlet / DoG / Haar-step — and
its own threshold k, scoring method (max / rss), and scale range. This
lets the user compare different wavelet shapes, *and* different scoring
settings for the same wavelet (e.g. Template-max vs Template-rss, or
Template across two scale ranges).

Each config runs through the same CWT pipeline as the live RT Detection
(CWT) block (`build_kernels` → `compute_score` → `mad_threshold` →
`detect_regions`) so the comparison is apples-to-apples. Detected
regions are then post-processed with shared common settings (merge gap
/ min event / pad before / pad after) — same model as the Combined
detection block.

- Inputs: `dataIn`, `templateWavelet` (optional), `groundTruth`
  (optional Nx2), `loadFile` (optional)
- Outputs: `detectedPulses` (winner's regions), `score` (winner's
  score), `metrics` (per-config dict), `winnerName`
- Auto-saves to `SavedTemplates/Detection/CwtEvaluator/temp.json`

**Per-tab settings:** name · enabled · wavelet shape (dropdown) ·
threshold k · scoring method · scale min/max/count · base length (ms)
for non-template wavelets · invert polarity (Template only) · Morlet ω
(Morlet only).

**Common settings:** merge gap (ms), min event (ms), pad before (×),
pad after (×), IoU threshold for ground-truth matching.

**Metrics columns:** Name · Shape · Method · Count · mean peak SNR ·
threshold. With `groundTruth`: TP / FP / FN · precision · recall · F1 ·
mean IoU of true positives (greedy IoU matching, adjustable cutoff,
default 0.30).

**Plot:** signal on top, one stacked score axis per enabled config
(color-coded by wavelet shape, with detected regions shaded).

**Winner picker:** selects which config's regions / score flow to the
downstream outputs; "auto" prefers highest-F1 if ground truth is
present, else the first Template config, else the config with the most
detections.
