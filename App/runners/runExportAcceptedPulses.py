"""Runner for ExportAcceptedPulses block - calls processing function."""


def run(inputs, params, block):
    from processing.export_accepted_pulses import export_accepted_pulses

    accepted_pulses = inputs.get("accepted_pulses")
    sample_rate = inputs.get("sampleRate", None)

    if accepted_pulses is None:
        raise ValueError("No accepted pulses provided.")

    export_accepted_pulses(accepted_pulses, sample_rate)
    return {}
