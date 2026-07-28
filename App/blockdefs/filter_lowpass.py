def get_definition():
    return {
        "name": "FilterLowpass",
        "displayName": "Lowpass",
        "category": "Filters",
        "color": [0.88, 0.55, 0.70],
        "inputs": [],
        "outputs": [
            {"name": "filterConfig", "type": "struct"},
        ],
        "runner": "runFilterLowpass",
        "hidePortLabels": True,
        "defaultSize": [120, 40],
        "defaultParameters": {
            "cutoff": 1000.0,
            "displayText": "1000 Hz",
        },
        "parameterDefinitions": [
            {"name": "cutoff", "displayName": "Cutoff (Hz)", "type": "numeric"},
        ],
    }
