def get_definition():
    return {
        "name": "InvertData",
        "displayName": "Invert Data",
        "category": "Preprocessing",
        "color": [0.11, 0.57, 0.70],
        "inputs": [
            {"name": "data", "type": "any", "required": True},
        ],
        "outputs": [
            {"name": "data", "type": "any", "displayName": "-data"},
        ],
        "runner": "runInvertData",
    }
