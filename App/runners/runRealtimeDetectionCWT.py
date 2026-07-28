"""Runner for the Realtime Detection (CWT) block."""

import os

import numpy as np

from utils.paths import project_root


_AUTOSAVE_RELPATH = os.path.join(
    "SavedTemplates", "Detection", "RealtimeCWT", "temp.json")


def _resolve_load_path(raw):
    """Resolve a non-sentinel loadFile input string to an absolute path or None.

    The `/lastSettings` sentinel is handled separately via
    ``resolve_settings_input``. Bare names (no separator, no extension) are
    interpreted as named templates in the block's SavedTemplates folder.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if os.path.isabs(s):
        return s
    if os.sep not in s and "/" not in s and "\\" not in s and "." not in s:
        return os.path.join(project_root(),
                            "SavedTemplates", "Detection", "RealtimeCWT",
                            s + ".json")
    return os.path.join(project_root(), s)


def run(inputs, params, block):
    from processing.realtime_detection_cwt import (
        load_state, save_state, _DEFAULT_PARAMS,
        RealtimeDetectionCWTDialog,
    )
    from utils.app_logger import logger

    data = inputs.get("dataIn")
    wavelet = inputs.get("wavelet")
    if data is None:
        raise ValueError("RealtimeDetectionCWT: dataIn input is required")
    if wavelet is None:
        raise ValueError("RealtimeDetectionCWT: wavelet input is required")

    signal = np.asarray(data, dtype=float)
    if signal.ndim > 1:
        # Use first column; matches BC detection convention.
        logger.info("RealtimeDetectionCWT: multi-channel input; using column 0")
        signal = signal[:, 0]
    signal = signal.ravel()
    wavelet = np.asarray(wavelet, dtype=float).ravel()

    # Sample rate from injected globals; default 10 kHz.
    sample_rate = inputs.get("sampleRate")
    if sample_rate is None:
        sr_value = 10000.0
        sr_source = "default"
    else:
        sr_value = float(sample_rate)
        sr_source = "global"

    # Resolution: /lastSettings → autosave temp.json; path → that path;
    # empty → start fresh (no implicit autosave restore).
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
                f"RealtimeDetectionCWT: loaded settings from {load_path}")
        except (OSError, ValueError) as e:
            logger.warning(
                f"RealtimeDetectionCWT: failed to load {load_path}: {e}")

    current_path = load_path if (load_path and not auto_load) else None

    dlg = RealtimeDetectionCWTDialog(
        signal, wavelet, sr_value, detector_params,
        sample_rate_source=sr_source, current_path=current_path,
    )
    if dlg.exec() != dlg.DialogCode.Accepted:
        raise ValueError("Realtime Detection (CWT) cancelled")

    accepted_params, result = dlg.result_state()
    if result is None:
        raise ValueError("Realtime Detection (CWT) produced no result")

    # Auto-save current params to temp.json AND mirror into block params
    # so the settings ride along with the workflow JSON.
    try:
        autosave_path = os.path.join(project_root(), _AUTOSAVE_RELPATH)
        os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
        save_state(accepted_params, autosave_path)
        from utils.lastsettings_capture import capture
        capture(block, autosave_path)
    except OSError as e:
        logger.warning(f"RealtimeDetectionCWT: auto-save failed: {e}")

    return {
        "detectedPulses": np.asarray(result["detectedPulses"], dtype=int),
        "score":          np.asarray(result["score"], dtype=float),
        "coincidentHint": np.asarray(result["coincidentHint"], dtype=np.int8),
    }
