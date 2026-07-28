# Preprocessing Blocks

## Trim Data (`TrimData`) — interactive
Interactive data trimming with range selection.
- Inputs: `data`, `platform` (optional), `zones` (optional), `settingsFile` (optional)
- Outputs: `data`
- Features: visual trim, zone selection (mzNPS), reset, export figure
- Save/Load buttons; auto-saves to `SavedTemplates/Trim/temp.json`
- Tracks cumulative trim indices for save/load

## Platform Detection (`PlatformDetection`)
Auto-detects mzNPS vs mechanoNPS platform.
- Inputs: `data`
- Outputs: `platform`

## Zone Selection (`ZoneSelection`)
Selects specific zones from multi-zone mzNPS data.
- Inputs: `data`, `zones`
- Outputs: `data`

## Preprocess (`Downsample`)
The JOVE mNPS preprocessing chain (interactive). Four toggleable stages —
rectangular smoothing → decimate by N → zero-phase low-pass → ASLS baseline
detrend — tuned in a dialog and applied per zone. Signal maths live in
`processing/jove_preprocess.py`; the dialog in `processing/downsample_ui.py`.
- Inputs: `data` (1-D or 2-D samples×zones), `sample rate` (base rate in Hz,
  optional — wired value wins, else the injected global), `settings` (saved
  JSON path or `/lastSettings`)
- Outputs: `dataSmoothed` (full length), `dataDownsampled`, `dataFiltered`,
  `dataDetrended`, `trendline` (all at the reduced rate), `DSfactor` (the
  decimation N, scalar)
- Parameters: per-stage enables / smoothing width & type / low-pass cutoff /
  ASLS settings (see `jove_preprocess.DEFAULT_PARAMS`); default `ds_factor` = 20.
- **Does not** publish the global sample rate. Wire `DSfactor` into a Global
  Sample Rate block (and the base rate into both this block and that one); the
  Global Sample Rate block computes the effective rate `base/N` and publishes
  it for downstream blocks.
- Pulse indices downstream are in **downsampled space**; recover original
  samples via `index × DSfactor`.
- Place early (right after Load, after any anti-alias filter) — but **after
  Platform Detection**, which classifies mzNPS vs mechanoNPS by absolute
  sample rate (`<= 15000 Hz`); downsampling first would push a mechanoNPS rate
  below that threshold and misclassify it.
