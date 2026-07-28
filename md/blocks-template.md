# Template Blocks

## Slicing (`PulseTemplateExtractor`) — interactive
Extract a single pulse template from signal data by selecting a region.
- Inputs: `data`, `loadFile` (optional)
- `data` accepts a bare signal array **or a Load File struct**
  (`{fileNames, files}` — also a single entry or list of entries). With
  multiple files the dialog shows a **File** selector; each file keeps its
  own view/selection while switching, and the chosen file persists via
  saved settings (matched by name, index fallback).
- Outputs:
  - `dataSegment` (struct) — `template`, `start_idx`, `end_idx`, `file_index`/`file_name` (which loaded file), `accepted_pulses`, `accepted_indices`
  - `indexVector` (numeric) — absolute sample indices of the extracted template (`start_idx` … `end_idx`)
  - `timeVector` (numeric) — time in seconds = `indexVector / sampleRate` (uses the global sample rate)
- **X-axis unit** selector (`Index` / `Time (s)`) switches the plot axis and the Start/End fields between sample index and seconds; the choice is display-only (selection stays sample-based) and persists via saved settings.

## Linear Detrending (`PulseTemplateDetrend`) — interactive
Detrend pulse template (remove linear baseline drift).
- Inputs: `template`
- Outputs: `detrended`

## Template Settings (`PulseTemplateProcessing`) — interactive
4-step interactive UI for pulse template processing:

1. **Thresholds** — Set upper/lower derivative thresholds
2. **Key Peaks** — Mark peaks as key peaks (Regular ★, Max ▲, Min ▼)
3. **Exclusion Zones** — Define zones where non-key peaks should not be detected
4. **Methods** — Set per-segment rectangularization method

Step 4 also selects **Edge Indices**: `FWHM` (default) keeps one boundary
per detected transition but moves it from the derivative-peak apex to the
signal's crossing of the **lower** adjacent plateau's half level (midpoint
of its plateau mean and its higher flanking plateau mean). Both boundaries
of a deep pulse use the pulse's own half-depth level, so its width is its
FWHM width. `Peaks` uses the derivative peak apexes (legacy behavior —
settings saved before this option restore as `Peaks`).

The choice propagates to **Batch Pulses Processing**: each detected pulse's
own segment boundaries (exported `pulsePeakLocations` and the
rectangularized segments) are refined to the FWHM crossings of that pulse's
filtered signal, so Pulse Slicing cuts real pulses at the same edge
definition the template uses. The derivative peak markers in the review
dialog stay at the apexes. (The Coincident block's interactive re-slicing
is not yet wired; its initial per-pulse rect follows the setting.)

- Inputs: `templateIn`, `filterConfigIn` (optional), `loadFile` (optional)
- Outputs: `TPsettings` (struct)
- Auto-saves to `SavedTemplates/PulseShape/temp.json`

**Key peak types:**
- **Max** (▲) — strongest positive derivative peak; found globally by amplitude
- **Min** (▼) — strongest negative derivative peak; found globally by amplitude
- **Regular** (★) — found by sequence counting relative to min/max anchors

**Save/Load:** Settings are mapped by sequence index, not absolute position. Key peaks are mapped via `key_peak_seq_indices` (indices into all detected peaks). Exclusion zones are mapped via `exclusion_zone_seq_bounds` using **key peak** indices as anchors (not all-peak indices), since zones are defined between consecutive key peaks. On load, the template re-detects peaks with saved thresholds, restores key peaks first, then maps zone boundaries to the current key peak positions.

**TPsettings struct fields:** See [data-structs.md](data-structs.md)

## Wavelet Editor (`WaveletEditor`) — interactive
Interactive editor for the rectangularized template, producing a
1-D vector usable as a mother wavelet (matched filter, CWT, etc.).

- Inputs: `TPsettings` (struct, required), `loadFile` (string, optional)
- Outputs:
  - `wavelet` — 1-D numpy array, post-edit, post-normalization
  - `segments` — struct with `widths`, `amplitudes`, `boundaries`,
    `normalization`, `n_segments`
- Auto-saves to `SavedTemplates/Wavelet/temp.json` on Confirm and
  auto-loads it on next dialog open (when no explicit `loadFile` is
  wired).
- **Saved settings** group inside the dialog: **Save** / **Save as…** /
  **Load…** to persist the current state to a named JSON file under
  `SavedTemplates/Wavelet/` independent of the auto-save. Use this to
  keep a few canonical wavelets around. The wavelet vector itself can
  be exported via a downstream `Save File` block on the `wavelet`
  output.

**Interactions in the dialog:**
- Vertical drag on a segment's top edge → change its amplitude.
- Horizontal drag on an internal boundary → shift the boundary; the
  two adjacent segment widths trade samples (total length preserved).
- Click rows in the sidebar table to highlight the matching segments
  on the plot. Press Delete (or the **Delete selected** button) to
  remove them from the output.
- Edit `Width` / `Amplitude` cells in the sidebar table for precise
  values.

**Normalization modes:** `None`, `Unit L2 norm`, `Zero-mean + unit L2 (wavelet)` (default), `Peak magnitude = 1`, `Dominant peak = +1`.

**Cancel** halts the pipeline (project convention for interactive
blocks). **Reset** restores widths/amplitudes from the upstream
`TPsettings`. Restoring deleted segments requires using **Reset** in
combination with closing the dialog without saving — re-running the
block will rebuild from the current `TPsettings`.

See `docs/superpowers/specs/2026-04-24-wavelet-editor-design.md` for
full design details.

