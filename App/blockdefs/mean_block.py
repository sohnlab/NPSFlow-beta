def get_definition():
    return {
        "name": "MeanBlock",
        "displayName": "Mean",
        "category": "Analysis",
        "color": [0.62, 0.55, 0.72],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "mean", "type": "numeric",
             "description": "Mean value(s)"},
        ],
        "isDynamic": True,
        "isMeanBlock": True,
        "runner": "runMeanBlock",
        "defaultParameters": {
            "mode": "column",
        },
    }
