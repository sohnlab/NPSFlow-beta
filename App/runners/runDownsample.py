"""Runner for the Downsample block — the JOVE preprocessing chain.

Opens an interactive dialog to tune four toggleable stages (rectangular
smoothing -> decimate -> zero-phase LPF with padding -> ASLS detrend), then
applies them per zone. Emits the five intermediate/final signals
(dataSmoothed, dataDownsampled, dataFiltered, dataDetrended, trendline) and
the decimation factor (DSfactor). The effective sample rate is published by
the Global Sample Rate block, not here. Signal maths live in
processing/jove_preprocess.py; the UI in processing/downsample_ui.py.
"""

import os


def run(inputs, params, block):
    from processing import jove_preprocess as jp

    data = inputs.get("data")
    if data is None:
        raise ValueError("Downsample: no input data provided.")

    sr = inputs.get("sampleRate")
    fs = float(sr) if sr not in (None, 0) else 0.0

    # Base params from the block (defaults + persisted). Map the legacy
    # scalar `factor` param (old decimate-only block) onto ds_factor.
    init = {k: params.get(k, jp.DEFAULT_PARAMS[k]) for k in jp.DEFAULT_PARAMS}
    if "ds_factor" not in params and "factor" in params:
        try:
            init["ds_factor"] = int(params["factor"])
        except (TypeError, ValueError):
            pass

    # Settings file: `/lastSettings` -> block snapshot/autosave; a path ->
    # that file; empty -> fresh (block defaults), or last settings in reuse mode.
    from utils.settings_input import load_settings_json
    from utils.lastsettings_capture import capture
    from utils.paths import saved_templates_dir
    autosave_abs = os.path.join(saved_templates_dir("Downsample"), "temp.json")
    saved = load_settings_json(inputs.get("settingsFile"), "Downsample",
                               block=block)
    if isinstance(saved, dict):
        init.update({k: v for k, v in saved.items()
                     if k in jp.DEFAULT_PARAMS})

    # Interactive tuning dialog (auto-saves confirmed params to temp.json).
    from processing.downsample_ui import downsample_ui
    confirmed = downsample_ui(data, fs, init_params=init)

    # Persist confirmed params so they travel with the workflow JSON.
    for k, v in confirmed.items():
        block.parameters[k] = v
    capture(block, autosave_abs)

    res = jp.process(data, fs, confirmed)
    return {
        "dataSmoothed": res["smoothed"],
        "dataDownsampled": res["down"],
        "dataFiltered": res["lp"],
        "dataDetrended": res["detrended"],
        "trendline": res["trendline"],
        "DSfactor": res["N"],
    }
