"""Runner for the CWT Evaluator block."""

import os

import numpy as np

from utils.paths import project_root


_AUTOSAVE_RELPATH = os.path.join(
    "SavedTemplates", "Detection", "CwtEvaluator", "temp.json")


def _resolve_load_path(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if os.path.isabs(s):
        return s
    if os.sep not in s and "/" not in s and "\\" not in s and "." not in s:
        return os.path.join(project_root(),
                            "SavedTemplates", "Detection", "CwtEvaluator",
                            s + ".json")
    return os.path.join(project_root(), s)


def run(inputs, params, block):
    from processing.cwt_evaluator import (
        load_state, save_state, _DEFAULT_PARAMS, CwtEvaluatorDialog,
    )
    from utils.app_logger import logger

    data = inputs.get("dataIn")
    if data is None:
        raise ValueError("CwtEvaluator: dataIn input is required")

    signal = np.asarray(data, dtype=float)
    if signal.ndim > 1:
        logger.info("CwtEvaluator: multi-channel input; using column 0")
        signal = signal[:, 0]
    signal = signal.ravel()

    template_in = inputs.get("templateWavelet")
    template = (np.asarray(template_in, dtype=float).ravel()
                if template_in is not None else None)

    gt_in = inputs.get("groundTruth")
    ground_truth = None
    if gt_in is not None:
        gt = np.asarray(gt_in, dtype=int)
        if gt.size:
            if gt.ndim == 1:
                gt = gt.reshape(-1, 2)
            if gt.shape[-1] == 2:
                ground_truth = gt.reshape(-1, 2)
            else:
                logger.warning(
                    "CwtEvaluator: groundTruth must be Nx2; ignoring")

    sample_rate = inputs.get("sampleRate")
    sr_value = float(sample_rate) if sample_rate is not None else 10000.0

    from utils.settings_input import (
        resolve_settings_input, LAST_SETTINGS_SENTINEL, reuse_when_empty,
    )
    raw_load = inputs.get("loadFile")
    # Empty port + global reuse-mode → reuse this block's last settings.
    if (not isinstance(raw_load, str) or not raw_load.strip()) and reuse_when_empty():
        raw_load = LAST_SETTINGS_SENTINEL
    autosave_abs = os.path.join(project_root(), _AUTOSAVE_RELPATH)
    if isinstance(raw_load, str) and raw_load.strip() == LAST_SETTINGS_SENTINEL:
        load_path, auto_load = resolve_settings_input(
            raw_load, autosave_abs, block=block)
    else:
        load_path = _resolve_load_path(raw_load)
        auto_load = False

    detector_params = dict(_DEFAULT_PARAMS)
    if load_path is not None and os.path.isfile(load_path):
        try:
            detector_params = load_state(load_path)
            logger.info(f"CwtEvaluator: loaded settings from {load_path}")
        except (OSError, ValueError) as e:
            logger.warning(f"CwtEvaluator: failed to load {load_path}: {e}")

    current_path = load_path if (load_path and not auto_load) else None

    dlg = CwtEvaluatorDialog(
        signal, sr_value, template, detector_params,
        ground_truth=ground_truth, current_path=current_path,
    )
    if dlg.exec() != dlg.DialogCode.Accepted:
        raise ValueError("CWT Evaluator cancelled")

    accepted_params, metrics, winner_name, winner = dlg.result_state()
    if winner is None:
        raise ValueError("CWT Evaluator: no valid wavelet results")

    try:
        autosave_path = os.path.join(project_root(), _AUTOSAVE_RELPATH)
        os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
        save_state(accepted_params, autosave_path)
        from utils.lastsettings_capture import capture
        capture(block, autosave_path)
    except OSError as e:
        logger.warning(f"CwtEvaluator: auto-save failed: {e}")

    # Strip large arrays out of the metrics struct that flows downstream
    # (regions, score, wavelet are in the dedicated outputs / would bloat
    # logged dicts). Keep the scalar metrics so consumers can compare.
    summary = {}
    for name, m in metrics.items():
        if "error" in m:
            summary[name] = {"error": m["error"]}
            continue
        summary[name] = {
            "count":       m["count"],
            "threshold":   m["threshold"],
            "sigma":       m["sigma"],
            "peakSnrMean": m["peakSnrMean"],
            "peakSnrMax":  m["peakSnrMax"],
            "tp":          m["tp"],
            "fp":          m["fp"],
            "fn":          m["fn"],
            "precision":   m["precision"],
            "recall":      m["recall"],
            "f1":          m["f1"],
            "meanIoU":     m["meanIoU"],
        }

    return {
        "detectedPulses": np.asarray(winner["regions"], dtype=int),
        "score":          np.asarray(winner["score"], dtype=float),
        "metrics":        summary,
        "winnerName":     str(winner_name),
    }
