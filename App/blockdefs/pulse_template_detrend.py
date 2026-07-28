def get_definition():
    return {
        "name": "PulseTemplateDetrend",
        "displayName": "Linear Detrending",
        "category": "Template",
        "color": [0.07, 0.53, 0.66],
        "inputs": [
            {"name": "dataIn", "type": "struct"},
        ],
        "outputs": [
            {"name": "dataDetrend", "type": "struct"},
        ],
        "isInteractive": True,
        "runner": "runPulseTemplateDetrend",
    }
