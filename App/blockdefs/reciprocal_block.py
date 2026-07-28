def get_definition():
    return {
        "name": "ReciprocalBlock",
        "displayName": "Reciprocal",
        "category": "Utility",
        "color": [0.54, 0.47, 0.64],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [],
        "runner": "runReciprocal",
        "isDynamic": True,
        "isMirrorPorts": True,
        "showPortLabels": True,
    }
