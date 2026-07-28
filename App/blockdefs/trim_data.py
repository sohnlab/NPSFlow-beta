def get_definition():
    return {
        "name": "TrimData",
        "displayName": "Trim Data",
        "category": "Preprocessing",
        "color": [0.05, 0.51, 0.64],
        "inputs": [
            {"name": "data", "type": "any"},
            {"name": "info", "type": "any", "required": False,
             "description": "Metadata bundle: {filename, platform}"},
            {"name": "zones", "type": "any", "required": False,
             "description": "List of 1-based zone indices to keep (mzNPS only)"},
            {"name": "settingsFile", "type": "string", "required": False,
             "description": "Path to saved trim settings JSON"},
        ],
        "outputs": [
            {"name": "data", "type": "any"},
            {"name": "trimIndex", "type": "numeric",
             "description": "Kept window [start, end] sample indices in the "
                            "original data (0-based, end exclusive)"},
        ],
        "isInteractive": True,
        "runner": "runTrimData",
    }
