def get_definition():
    return {
        "name": "DefineParameters",
        "displayName": "Define Parameters",
        "category": "Variables",
        "color": [0.6, 0.52, 0.81],
        "inputs": [
            {"name": "run", "type": "any"},
        ],
        "outputs": [
            {"name": "run", "type": "any"},
        ],
        "runner": "runDefineParameters",
        "isDefineParameters": True,
        "defaultParameters": {
            "variables": [],
            "globalSampleRate": None,
        },
    }
