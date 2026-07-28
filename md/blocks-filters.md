# Filter Blocks

## Filter (`FilterUI`) — interactive
Interactive zero-phase filter design UI with preview (`filtfilt`; no group
delay, not causal — use Causal Filter for real-time training data).
- Inputs: `data` (bare array, or Load Data struct `{fileNames, files}`),
  `filterConfigIn` (optional; apply programmatically without UI)
- Outputs: `filteredData`, `filterConfig`
- Bare array: legacy behavior — one dialog per zone, per-zone chains.
- Struct: opens the multi-file preview dialog (same as Causal Filter, but
  zero-phase); one chain applied to all files and zones, output preserves
  the struct (only each entry's `data` replaced; `labels` etc. pass
  through). Sample rate from the global/wired `sampleRate` (10 kHz
  fallback).

## Causal Filter (`CausalFilterUI`) — interactive
Filter UI variant for real-time / training-data pipelines: every stage is
applied forward-only (`sosfilt`/`lfilter` with steady-state initial
conditions, never `filtfilt`), so the output matches what a streaming
front-end produces sample by sample.
- Inputs: `data` (Load Data struct `{fileNames, files}` or bare array),
  `sample rate` (optional; falls back to 10 kHz), `filterConfigIn`
  (optional; apply programmatically without UI)
- Outputs: `filtered` (same structure as input — only each entry's `data`
  is replaced; `labels` and other keys pass through), `filterConfig`
- UI: same tabbed chain design as Filter UI plus a Butterworth order field
  and a preview file/zone selector; switching the preview replays the
  current chain. The confirmed chain is applied to all files and zones.

## Filter Config (`FilterConfig`) — dynamic
Define a filter chain configuration (no data processing).
- Dynamic inputs: `Add input` (cascade multiple filters)
- Outputs: `filterConfig` (struct passed to other blocks)

## Lowpass (`FilterLowpass`)
Lowpass Butterworth filter.
- Inputs: `data`
- Outputs: `filtered`
- Parameters: cutoff frequency, order

## Highpass (`FilterHighpass`)
Highpass Butterworth filter.
- Inputs: `data`
- Outputs: `filtered`
- Parameters: cutoff frequency, order

## Bandpass (`FilterBandpass`)
Bandpass Butterworth filter.
- Inputs: `data`
- Outputs: `filtered`
- Parameters: low cutoff, high cutoff, order

## Notch (`FilterNotch`)
Notch (band-stop) filter for removing specific frequencies.
- Inputs: `data`
- Outputs: `filtered`
- Parameters: center frequency, bandwidth/Q
