"""Runner for the RT Detection (Combined) block."""

import os

import numpy as np

from utils.paths import project_root


_AUTOSAVE_RELPATH = os.path.join(
    "SavedTemplates", "Sorting", "RealtimeCombined", "temp.json")


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
                            "SavedTemplates", "Sorting", "RealtimeCombined",
                            s + ".json")
    return os.path.join(project_root(), s)


def run(inputs, params, block):
    from processing.realtime_detection_combined import (
        load_state, save_state, _DEFAULT_PARAMS,
        RealtimeDetectionCombinedDialog,
    )
    from utils.app_logger import logger

    data = inputs.get("dataIn")
    if data is None:
        raise ValueError("RT Detection (Combined): dataIn input is required")

    signal = np.asarray(data, dtype=float)
    if signal.ndim > 1:
        logger.info("RT Detection (Combined): multi-channel input; "
                    "using column 0")
        signal = signal[:, 0]
    signal = signal.ravel()

    wavelet_raw = inputs.get("wavelet")
    wavelet = (np.asarray(wavelet_raw, dtype=float).ravel()
               if wavelet_raw is not None else None)

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
            logger.info(
                f"RT Detection (Combined): loaded settings from {load_path}")
        except (OSError, ValueError) as e:
            logger.warning(
                f"RT Detection (Combined): failed to load {load_path}: {e}")

    current_path = load_path if (load_path and not auto_load) else None

    dlg = RealtimeDetectionCombinedDialog(
        signal, sr_value, detector_params,
        wavelet=wavelet,
        current_path=current_path,
    )
    if dlg.exec() != dlg.DialogCode.Accepted:
        raise ValueError("RT Detection (Combined) cancelled")

    accepted_params, result = dlg.result_state()
    if result is None:
        raise ValueError("RT Detection (Combined) produced no result")

    try:
        autosave_path = os.path.join(project_root(), _AUTOSAVE_RELPATH)
        os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
        save_state(accepted_params, autosave_path)
        from utils.lastsettings_capture import capture
        capture(block, autosave_path)
    except OSError as e:
        logger.warning(f"RT Detection (Combined): auto-save failed: {e}")

    return {
        "detectedPulses": np.asarray(result["detectedPulses"], dtype=int),
        "score":          np.asarray(result["score"], dtype=float),
        "coincidentHint": np.asarray(result["coincidentHint"], dtype=np.int8),
    }
