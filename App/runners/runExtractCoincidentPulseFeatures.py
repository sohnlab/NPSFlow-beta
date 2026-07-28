"""Runner for ExtractCoincidentPulseFeatures (Coincident Processing) block."""


def run(inputs, params, block):
    from processing.extract_coincident_pulse_features import (
        extract_coincident_pulse_features, SavedForLater,
    )
    import os
    from utils.paths import project_root
    from utils.settings_input import resolve_settings_input
    from utils.lastsettings_capture import capture
    from utils.tpsettings_io import get_zone

    data_in = inputs.get("dataIn")
    if data_in is None:
        raise ValueError("No dataIn provided to ExtractCoincidentPulseFeatures.")
    pulse_indices = inputs.get("pulseLabels")
    if pulse_indices is None:
        raise ValueError("No pulseLabels provided to ExtractCoincidentPulseFeatures.")
    tp = inputs.get("TPsettings")
    if tp is None or not isinstance(tp, dict):
        raise ValueError("No TPsettings provided to ExtractCoincidentPulseFeatures.")

    sample_rate = inputs.get("sampleRate", 1)
    trendline = inputs.get("trendline")
    pulse_features = inputs.get("pulseFeatures")

    root = project_root()
    autosave_abs = os.path.join(root, "SavedTemplates", "Coincident", "temp.json")
    settings_file, _ = resolve_settings_input(
        inputs.get("settingsFile"), autosave_abs, block=block)

    # If the resolved file carries prior review work, ask continue vs. fresh.
    from utils.session_prompt import review_state_summary, confirm_continue_previous
    has_work, detail = review_state_summary(settings_file)
    if has_work and not confirm_continue_previous(
            detail, title="Coincident Pulse Feature Review",
            allow_remember=getattr(block, "_auto_run", False)):
        settings_file = None

    methods = get_zone(tp, 0).get("methods")
    if pulse_features is not None and isinstance(pulse_features, dict):
        if methods is not None and "methods" not in pulse_features:
            pulse_features = dict(pulse_features)
            pulse_features["methods"] = methods
    elif methods is not None:
        pulse_features = {"methods": methods}

    try:
        result = extract_coincident_pulse_features(
            data_in=data_in, tp=tp, sample_rate=sample_rate,
            pulse_indices=pulse_indices, trendline=trendline,
            pulse_features=pulse_features, settings_file=settings_file)
    except SavedForLater:
        # Save-for-later wrote the autosave file; stash it on the block NOW —
        # otherwise the next run's restore() rolls temp.json back to the
        # workflow's stale lastSettings and the saved progress is lost.
        capture(block, autosave_abs)
        raise

    capture(block, autosave_abs)

    return {
        "pulsePeakLocations": result["pulse_peak_locations"],
        "rectangularizedPulses": result["rectangularized_pulses"],
        "acceptanceID": result["acceptance_id"],
        "pulseStartIndices": result["pulse_start_indices"],
        "pulseFeatures": pulse_features,
        "coincidenceMap": result.get("coincidence_map", []),
    }
