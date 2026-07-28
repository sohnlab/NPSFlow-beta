# NPSflow

A visual, node-based workflow editor for **Node Pore Sensing (NPS)** signal
processing. Build a pipeline by wiring blocks on a canvas — load a recording,
preprocess it, detect and review pulses, extract per-pulse features, and
compute diameters or transit metrics — then run it and export the results.

Built with PySide6. Developed with the assistance of GitHub Copilot and
Claude Code.

## Requirements

- **Python 3.10 or newer** (developed and tested on 3.12)
- The Python packages in [App/requirements.txt](App/requirements.txt): PySide6,
  numpy, scipy, matplotlib, pandas, h5py, scikit-learn, joblib

## Install

```bash
git clone <your-repo-url> NPSflow
cd NPSflow
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r App/requirements.txt
```

On first launch NPSflow also detects any missing packages and offers to
install them for you from the terminal.

## Run

```bash
python NPSflow.py
```

## Sample data

Sample `.mat` recordings are **not** bundled with the repository (each is tens
of megabytes). Download them from the shared data folder and place them under
`SignalData/`:

<!-- TODO: replace this line with your shared data-folder link (e.g. Google Drive) -->
> **Data download:** _add your shared folder link here._

An `.mat` recording is expected to contain a `data` array (the raw signal) and
a `sampleRate` scalar; `ampsPerVolt` and related fields are used when present.
CSV and `.npz` inputs are also supported.

## First steps

See [QUICKSTART.md](QUICKSTART.md) for a walkthrough: open the example
workflow, point it at a recording, run it, and export results. The example
pipeline is [workflows/Sorter_v5.json](workflows/Sorter_v5.json).

## Documentation

Per-block reference documentation lives in [md/](md/):

- [md/architecture.md](md/architecture.md) — core components and block format
- [md/blocks-dataio.md](md/blocks-dataio.md) — Load, Save, Display, …
- [md/blocks-preprocessing.md](md/blocks-preprocessing.md) — Trim, Baseline, Filter, …
- [md/blocks-detection.md](md/blocks-detection.md) — pulse detection and review
- [md/blocks-features.md](md/blocks-features.md) — feature extraction
- [md/blocks-analysis.md](md/blocks-analysis.md) — means, diameter, recovery
- …and the other `md/blocks-*.md` files for sorting, filters, variables, flow control, and utilities.

## A note on trust

Workflow `.json` files are programs, not just data: a **Custom Code** block
stores Python that runs when you execute the workflow. Only open and run
workflows from people you trust.

## License

Released under the [MIT License](LICENSE).
