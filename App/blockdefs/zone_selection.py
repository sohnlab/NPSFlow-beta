def get_definition():
    return {
        "name": "ZoneSelection",
        "displayName": "Zone Selection",
        "category": "Preprocessing",
        "color": [0.89, 0.58, 0.27],
        "inputs": [],
        "outputs": [
            {"name": "zones", "type": "any",
             "description": "List of 1-based zone indices to keep"},
        ],
        "runner": "runZoneSelection",
        "isParameterized": True,
        "isDisplay": True,
        "defaultSize": [180, 60],
        "defaultParameters": {
            "selectedZones": [],
            "displayText": "(all)",
        },
    }
