"""Runner for PulseTemplateDetrend block - calls processing function."""


def run(inputs, params, block):
    from processing.pulse_template_detrend import pulse_template_detrend

    template_data = inputs.get("dataIn")

    if template_data is None:
        raise ValueError("No template data provided to PulseTemplateDetrend.")

    result = pulse_template_detrend(template_data)
    return {"dataDetrend": result}
