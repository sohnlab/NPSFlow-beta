def get_definition():
    return {
        "name": "DataBus",
        "displayName": "Data Bus",
        "category": "Variables",
        "color": [0.47, 0.39, 0.68],
        "inputs": [
            {"name": "Run", "type": "any", "required": False, "displayName": "Run"},
            {"name": "addInput", "type": "any", "required": False, "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "Run", "type": "any", "required": False, "displayName": "Run"},
        ],
        "isDynamic": True,
        "isDataBus": True,
        "defaultParameters": {
            "pool": "Reference",
        },
    }
