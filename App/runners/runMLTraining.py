"""Runner for MLTraining block — trains the baseline/single/coincident
event classifier from a folder of labeled files."""

import json
import os
from utils.paths import project_root


def run(inputs, params, block):
    from processing.ml_training import ml_training_ui

    folder = inputs.get("folder") or ""
    if not isinstance(folder, str):
        folder = str(folder)
    sample_rate = inputs.get("sampleRate")

    # Seed the UI from (priority order): block.parameters saved with the
    # workflow, then the settingsFile input / autosave temp.json.
    init_settings = None
    if isinstance(params, dict) and params.get("mlSettings"):
        init_settings = params["mlSettings"]

    from utils.settings_input import resolve_settings_input
    from utils.lastsettings_capture import capture
    root = project_root()
    autosave_abs = os.path.join(root, "SavedTemplates", "ML", "temp.json")
    if init_settings is None:
        settings_file = inputs.get("settingsFile")
        path, _ = resolve_settings_input(
            settings_file, autosave_abs, block=block)
        if path and os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    init_settings = json.load(f)
            except Exception as exc:
                from utils.app_logger import logger
                logger.warning(
                    f"[MLTraining] Failed to load settings file: {exc}")

    result = ml_training_ui(
        folder=folder,
        sample_rate=sample_rate,
        init_settings=init_settings,
    )

    # Persist the dialog settings so they round-trip through the workflow
    # JSON, and mirror the autosave temp.json onto the block.
    block.parameters["mlSettings"] = result.get("settings", {})
    block.parameters.pop("displayText", None)
    capture(block, autosave_abs)

    return {
        "modelFile": result["modelFile"],
        "metrics": result["metrics"],
    }
