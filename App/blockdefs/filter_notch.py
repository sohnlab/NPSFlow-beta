def get_definition():
    return {
        "name": "FilterNotch",
        "displayName": "Notch",
        "category": "Filters",
        "color": [0.91, 0.60, 0.74],
        "inputs": [],
        "outputs": [
            {"name": "filterConfig", "type": "struct"},
        ],
        "runner": "runFilterNotch",
        "hidePortLabels": True,
        "defaultSize": [120, 40],
        "defaultParameters": {
            "mode": "freqBW",
            "freq": 60.0,
            "bw": 2.0,
            "lower": 59.0,
            "upper": 61.0,
            "harmonics": [1],
            "displayText": "60-2 Hz",
        },
        "parameterDefinitions": [
            {"name": "mode", "displayName": "Mode", "type": "string"},
            {"name": "freq", "displayName": "Freq (Hz)", "type": "numeric"},
            {"name": "bw", "displayName": "BW (Hz)", "type": "numeric"},
            {"name": "lower", "displayName": "Lower (Hz)", "type": "numeric"},
            {"name": "upper", "displayName": "Upper (Hz)", "type": "numeric"},
            {"name": "harmonics", "displayName": "Harmonics", "type": "list"},
        ],
        "isNotchFilter": True,
    }
