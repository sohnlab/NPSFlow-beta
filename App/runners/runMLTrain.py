"""Runner for MLTrain block — trains the event classifier on the wired
training set (split upstream by ML Split)."""

import numpy as np


def run(inputs, params, block):
    from processing.ml_training_core import (
        extract_entries, parse_floats, train_from_entries)

    entries = extract_entries(inputs.get("trainSet"))
    if not entries:
        raise ValueError("ML Train: no file entries on trainSet.")

    fs = inputs.get("sampleRate")
    if fs is None:
        fs = params.get("sampleRate", 50000)
    horizons = parse_floats(params.get("horizonsMs", ""))

    cfg = {
        "fs": float(np.ravel(fs)[0]),
        "model": params.get("model", "rf"),
        "strategy": params.get("strategy", "two_stage"),
        "oversample": params.get("oversample", "smote"),
        "oversample_ratio": float(params.get("oversampleRatio", 1.0)),
        "noise_ratio": float(params.get("noiseRatio", 1.0)),
        "hard_negative_ratio": float(params.get("hardNegativeRatio", 1.0)),
        "horizons_ms": horizons or None,
        "seed": int(params.get("seed", 0)),
        "modelPath": params.get("modelPath") or None,
    }

    from utils.app_logger import logger
    info = train_from_entries(entries, cfg, progress=logger.info)

    block.parameters["displayText"] = (
        f"{len(info['files'])} files, {info['rows']:,} rows")
    return {"modelFile": info["modelPath"], "trainInfo": info}
