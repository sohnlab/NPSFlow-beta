def get_definition():
    return {
        "name": "FilterHighpass",
        "displayName": "Highpass",
        "category": "Filters",
        "color": [0.85, 0.50, 0.66],
        "inputs": [],
        "outputs": [
            {"name": "filterConfig", "type": "struct"},
        ],
        "runner": "runFilterHighpass",
        "hidePortLabels": True,
        "defaultSize": [120, 40],
        "defaultParameters": {
            "cutoff": 10.0,
            "displayText": "10 Hz",
        },
        "parameterDefinitions": [
            {"name": "cutoff", "displayName": "Cutoff (Hz)", "type": "numeric"},
        ],
    }
