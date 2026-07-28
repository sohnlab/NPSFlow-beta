"""Runner for the Detection Agreement block."""

import os

import numpy as np

from utils.paths import project_root


_AUTOSAVE_RELPATH = os.path.join(
    "SavedTemplates", "Analysis", "DetectionAgreement", "temp.json")


def run(inputs, params, block):
    from processing.detection_agreement import (
        compare, load_state, save_state, _DEFAULT_PARAMS,
        DetectionAgreementDialog,
    )
    from utils.app_logger import logger

    data = inputs.get("dataIn")
    if data is None:
        raise ValueError("Detection Agreement: dataIn input is required")
    signal = np.asarray(data, dtype=float)
    if signal.ndim > 1:
        logger.info(
            "Detection Agreement: multi-channel input; using column 0")
        signal = signal[:, 0]
    signal = signal.ravel()

    pulses_a_raw = inputs.get("pulsesA")
    pulses_b_raw = inputs.get("pulsesB")
    if pulses_a_raw is None:
        raise ValueError("Detection Agreement: pulsesA input is required")
    if pulses_b_raw is None:
        raise ValueError("Detection Agreement: pulsesB input is required")
    pulses_a = np.asarray(pulses_a_raw, dtype=int).reshape(-1, 2) \
        if np.size(pulses_a_raw) else np.zeros((0, 2), dtype=int)
    pulses_b = np.asarray(pulses_b_raw, dtype=int).reshape(-1, 2) \
        if np.size(pulses_b_raw) else np.zeros((0, 2), dtype=int)

    sample_rate = inputs.get("sampleRate")
    if sample_rate is None:
        raise ValueError(
            "Detection Agreement: no global sampleRate available — "
            "add a Global Sample Rate or Define Parameters block "
            "upstream")
    sr_value = float(sample_rate)

    # Load auto-saved params if present, else defaults
    autosave = os.path.join(project_root(), _AUTOSAVE_RELPATH)
    detector_params = dict(_DEFAULT_PARAMS)
    if os.path.isfile(autosave):
        try:
            detector_params = load_state(autosave)
            logger.info(
                f"Detection Agreement: loaded autosave from {autosave}")
        except (OSError, ValueError) as e:
            logger.warning(
                f"Detection Agreement: failed to load autosave: {e}")

    dlg = DetectionAgreementDialog(
        signal, pulses_a, pulses_b, sr_value, detector_params)
    from PySide6.QtWidgets import QDialog
    if dlg.exec() != QDialog.DialogCode.Accepted:
        raise ValueError("Detection Agreement: cancelled by user")

    final_params = dlg.get_params()
    try:
        save_state(autosave, final_params)
    except OSError as e:
        logger.warning(f"Detection Agreement: autosave failed: {e}")

    match_table, counts, scores = compare(
        pulses_a, pulses_b,
        final_params["iou_thr"],
        final_params["midpoint_tol_samples"],
        final_params["iou_enabled"],
        final_params["midpoint_enabled"])

    logger.info(
        f"Detection Agreement: |A|={counts['A']} |B|={counts['B']} "
        f"TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} "
        f"F1={scores['F1']:.3f} Jaccard={scores['Jaccard']:.3f} "
        f"meanIoU={scores['MeanIoU']:.3f}")

    return {
        "score": float(scores[final_params["primary_score"]]),
        "matchTable": match_table,
    }
