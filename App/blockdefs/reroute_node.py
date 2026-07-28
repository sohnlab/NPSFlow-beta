def get_definition():
    return {
        "name": "RerouteNode",
        "displayName": "",
        "category": "Utility",
        "color": [0.55, 0.55, 0.55],
        "inputs": [
            {"name": "in", "type": "any", "required": True},
        ],
        "outputs": [
            {"name": "out", "type": "any"},
        ],
        "runner": "runRerouteNode",
        "isCompact": True,
        "isReroute": True,
        "hidePortLabels": True,
        "defaultSize": [12, 12],
    }
