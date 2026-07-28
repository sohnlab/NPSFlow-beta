def get_definition():
    return {
        "name": "FilterBandpass",
        "displayName": "Filter Bandpass",
        "category": "Filters",
        "color": [0.78, 0.40, 0.58],
        "inputs": [],
        "outputs": [
            {"name": "filterConfig", "type": "struct"},
        ],
        "runner": "runFilterBandpass",
        "hidePortLabels": True,
        "defaultSize": [160, 40],
        "defaultParameters": {
            "lower": 10.0,
            "upper": 1000.0,
            "displayText": "10-1000 Hz",
        },
        "parameterDefinitions": [
            {"name": "lower", "displayName": "Lower (Hz)", "type": "numeric"},
            {"name": "upper", "displayName": "Upper (Hz)", "type": "numeric"},
        ],
    }
