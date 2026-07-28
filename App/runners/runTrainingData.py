"""Runner for TrainingData block - generates labeled training data for ML pulse classification."""

import os


def run(inputs, params, block):
    from processing.training_data import training_data_ui

    data = inputs.get("dataIn")
    if data is None:
        raise ValueError("No data provided to TrainingData.")

    single = inputs.get("single")
    coincident = inputs.get("coincident")
    noise = inputs.get("noise")
    uncertain = inputs.get("uncertain")
    sample_rate = inputs.get("sampleRate", 1)

    # Seed the UI from (priority order):
    #   1. block.parameters — state saved with the workflow
    #   2. settingsFile input — external JSON
    # Both are optional; UI falls back to defaults if neither is present.
    init_settings = None
    if isinstance(params, dict):
        if params.get("classes") or params.get("ignored"):
            init_settings = {
                "classes": params.get("classes", []),
                "ignored": params.get("ignored", []),
            }

    from utils.settings_input import load_settings_json
    from utils.lastsettings_capture import capture
    from utils.paths import saved_templates_dir
    autosave_abs = os.path.join(saved_templates_dir("Training"), "temp.json")
    if init_settings is None:
        init_settings = load_settings_json(
            inputs.get("settingsFile"), "Training", block=block)

    result = training_data_ui(
        data, single, coincident, noise, uncertain,
        sample_rate=sample_rate,
        init_settings=init_settings,
    )

    # Persist class layout into the block so it round-trips through the
    # workflow JSON.
    block.parameters["classes"] = result.get("classes", [])
    block.parameters["ignored"] = result.get("ignored", [])
    block.parameters.pop("displayText", None)

    # Mirror the autosave temp.json into block params so settings travel
    # with the workflow JSON.
    capture(block, autosave_abs)

    return {
        "labeledData": {
            "data": data,
            "labels": result["labels"],
        },
    }
