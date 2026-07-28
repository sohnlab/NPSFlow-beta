def get_definition():
    return {
        "name": "LoadGeometry",
        "displayName": "Load Geometry",
        "category": "Variables",
        "color": [0.36, 0.7, 0.66],
        "inputs": [],
        "outputs": [],
        "isDynamic": True,
        "isLoadGeometry": True,
        "runner": "runLoadGeometry",
        "defaultParameters": {
            "refVariable": "geometry",
        },
    }
