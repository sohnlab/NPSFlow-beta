def get_definition():
    return {
        "name": "PackVariable",
        "displayName": "Pack Variable",
        "category": "DataIO",
        "color": [0.55, 0.47, 0.76],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "structOut", "type": "struct", "displayName": "VarOut"},
        ],
        "isDynamic": True,
        "isPacker": True,
        "runner": "runPackVariable",
    }
