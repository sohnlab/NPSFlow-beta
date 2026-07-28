def get_definition():
    return {
        "name": "LoadNPSData",
        "displayName": "Load NPS Data",
        "category": "DataIO",
        "color": [0.26, 0.55, 0.84],
        "isInteractive": True,
        "inputs": [
            {"name": "Run", "type": "any", "required": False},
            {"name": "filePath", "type": "string", "required": False},
        ],
        "outputs": [
            {"name": "filename", "type": "string"},
            {"name": "data", "type": "any"},
        ],
        "runner": "runLoadNPSData",
    }
