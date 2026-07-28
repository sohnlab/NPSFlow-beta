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

### Method 1:
```bash
git clone https://github.com/sohnlab/NPSFlow-beta.git NPSflow
cd NPSflow
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r App/requirements.txt
```

### Method 2:
Download the package.
On first launch, NPSflow detects any missing packages and offers to
install them for you from the terminal.

## Run

```bash
python NPSflow.py
```

## Sample data

One sample `.mat` signal recording is provided with the repository. Place them under
`SignalData/`:

An `.mat` recording is expected to contain a `data` array (the raw signal) and
a `sampleRate` scalar; `ampsPerVolt` and related fields are used when present.
CSV and `.npz` inputs are also supported.

## Demo workflow

See [QUICKSTART.md](QUICKSTART.md) for a walkthrough: open the example
workflow, point it at a recording, run it, and export results. The example
pipeline is [workflows/mNPS_COH_pipeline.json](workflows/mNPS_COH_pipeline.json).


## License

Released under the [MIT License](LICENSE).
