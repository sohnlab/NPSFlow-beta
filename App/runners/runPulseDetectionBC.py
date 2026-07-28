"""Runner for PulseDetectionBC block — pulse detection with baseline correction."""


def run(inputs, params, block):
    from processing.pulse_detection_bc import pulse_detection_bc

    data = inputs.get("dataIn")
    # sampleRate + downsampleFactor come from the Global Sample Rate block via
    # the engine's global-var injection (effective rate + N). base = eff × N.
    sample_rate = inputs.get("sampleRate", 1)
    downsample_factor = inputs.get("downsampleFactor", 1)
    pulse_template = inputs.get("pulseTemplate")
    filter_config = inputs.get("filterConfig")
    load_file = inputs.get("loadFile")

    if data is None:
        raise ValueError("No data provided to PulseDetectionBC.")

    # Load saved settings: `/lastSettings` → block's per-instance settings
    # (or autosave temp.json if none), path → that path, empty → start fresh.
    import os
    from utils.settings_input import load_settings_json
    from utils.lastsettings_capture import capture
    from utils.paths import saved_templates_dir
    autosave_abs = os.path.join(saved_templates_dir("Detection"), "temp.json")
    saved_settings = load_settings_json(load_file, "Detection", block=block)

    result = pulse_detection_bc(data, sample_rate, pulse_template,
                                filter_config, saved_settings,
                                downsample_factor=downsample_factor)
    # Mirror the just-written autosave into block params so settings
    # travel with the workflow JSON.
    capture(block, autosave_abs)
    return {
        "trendline": result["trendline"],
        "detectedPulses": result["detectedPulses"],
        "detectedPerZone": result["detectedPerZone"],
    }
