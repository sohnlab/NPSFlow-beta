"""Runner for PulseRegionReview block - calls processing function."""

import os


def run(inputs, params, block):
    from processing.pulse_region_review import pulse_region_review

    data = inputs.get("dataIn")
    detected_pulses = inputs.get("detectedPulses")
    sample_rate = inputs.get("sampleRate", 1)

    if data is None:
        raise ValueError("No data provided to PulseRegionReview.")
    if detected_pulses is None:
        raise ValueError("No detectedPulses provided to PulseRegionReview.")

    # Handle dict output from PulseDetection block
    if isinstance(detected_pulses, dict):
        detected_pulses = detected_pulses.get('detected_pulses', detected_pulses)

    # Load pre-saved classification labels: `/lastSettings` → block's
    # per-instance settings (or autosave); path → that path; empty → fresh.
    from utils.settings_input import load_settings_json
    from utils.lastsettings_capture import capture
    from utils.paths import saved_templates_dir
    init_labels = None
    init_regions = None
    init_cutoff = None
    autosave_abs = os.path.join(saved_templates_dir("Review"), "temp.json")
    settings = load_settings_json(inputs.get("settingsFile"), "Review",
                                  block=block)
    if settings is not None:
        init_labels = settings.get("labels")
        init_regions = settings.get("regions")
        init_cutoff = settings.get("lpfCutoff")

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
                detail, title="Detection Review",
                allow_remember=getattr(block, "_auto_run", False)):
            init_labels = None
            init_regions = None

    result = pulse_region_review(data, detected_pulses, sample_rate,
                                 init_labels=init_labels,
                                 init_regions=init_regions,
                                 init_cutoff=init_cutoff)
    # Mirror the just-written autosave into block params so settings
    # travel with the workflow JSON.
    capture(block, autosave_abs)
    return {
        "single": result["single"],
        "coincident": result["coincident"],
        "noise": result["noise"],
        "uncertain": result["uncertain"],
    }
