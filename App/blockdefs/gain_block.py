def get_definition():
    return {
        "name": "GainBlock",
        "displayName": "Gain",
        "category": "Utility",
        "color": [0.58, 0.51, 0.68],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [],
        "runner": "runGain",
        "defaultParameters": {
            "gain": 1.0,
            "displayText": "\u00d71",
        },
        "parameterDefinitions": [
            {"name": "gain", "displayName": "Gain", "type": "numeric"},
        ],
        "isDynamic": True,
        "isMirrorPorts": True,
        "showPortLabels": True,
    }
