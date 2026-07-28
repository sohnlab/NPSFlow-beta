"""Runner for MultizoneDetectionReview block - calls processing function."""

import json
import os
from utils.paths import project_root


def run(inputs, params, block):
    from processing.multizone_detection_review import multizone_detection_review

    data = inputs.get("dataIn")
    detected_pulses = inputs.get("detectedPulses")
    sample_rate = inputs.get("sampleRate", 1)

    if data is None:
        raise ValueError("No data provided to MultizoneDetectionReview.")
    if detected_pulses is None:
        raise ValueError("No detectedPulses provided to MultizoneDetectionReview.")

    # Handle dict output from PulseDetection block
    if isinstance(detected_pulses, dict):
        detected_pulses = detected_pulses.get('detected_pulses', detected_pulses)

    # Load pre-saved classification labels: `/lastSettings` → block's
    # per-instance settings (or autosave); path → that path; empty → fresh.
    from utils.settings_input import resolve_settings_input
    from utils.lastsettings_capture import capture
    init_labels = None
    init_regions = None
    init_cutoff = None
    init_zone_roles = None
    init_manual = None
    settings_file = inputs.get("settingsFile")
    root = project_root()
    autosave_abs = os.path.join(root, "SavedTemplates", "Review", "temp.json")
    path, _ = resolve_settings_input(settings_file, autosave_abs, block=block)

    if path and os.path.isfile(path):
        try:
            with open(path, "r") as f:
                settings = json.load(f)
            init_labels = settings.get("labels")
            init_regions = settings.get("regions")
            init_cutoff = settings.get("lpfCutoff")
            init_zone_roles = settings.get("zoneRoles")
            init_manual = settings.get("manual")
        except Exception as exc:
            print(f"[MultizoneDetectionReview] Failed to load settings file: {exc}")

    # If a previous session carries real work (any classification, or region
    # edits), ask whether to continue it or start fresh.
    n_detected = len(detected_pulses) if detected_pulses is not None else 0
    n_classified = sum(1 for lbl in (init_labels or [])
                       if lbl and lbl != "unclassified")
    has_edits = bool(init_regions) and len(init_regions) != n_detected
    if n_classified or has_edits:
        from utils.session_prompt import confirm_continue_previous
        detail = (f"{n_classified} pulse(s) already classified."
                  if n_classified else "Previous region edits found.")
        if not confirm_continue_previous(
                detail, title="Detection Review (mz)",
                allow_remember=getattr(block, "_auto_run", False)):
            init_labels = None
            init_regions = None

    result = multizone_detection_review(data, detected_pulses, sample_rate,
                                        init_labels=init_labels,
                                        init_regions=init_regions,
                                        init_cutoff=init_cutoff,
                                        init_zone_roles=init_zone_roles,
                                        init_manual=init_manual)
    # Mirror the just-written autosave into block params so settings
    # travel with the workflow JSON.
    capture(block, autosave_abs)
    return {
        "single": result["single"],
        "coincident": result["coincident"],
        "noise": result["noise"],
        "uncertain": result["uncertain"],
    }
