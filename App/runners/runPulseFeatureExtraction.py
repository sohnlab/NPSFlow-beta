"""Runner for PulseFeatureExtraction block - calls processing function."""


def run(inputs, params, block):
    from processing.pulse_feature_extraction import pulse_feature_extraction

    tp_settings = inputs.get("TPsettings")
    sample_rate = inputs.get("sampleRate", 1)

    if not isinstance(tp_settings, dict):
        raise ValueError("TPsettings must be a valid struct dict.")

    # TPsettings is zone-grouped; the entry processes every zone and emits a
    # zone-grouped pulseFeatures (bare segment dict for a single zone).
    result = pulse_feature_extraction(tp_settings, sample_rate)
    return {"pulseFeatures": result}
