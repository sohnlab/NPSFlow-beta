def get_definition():
    return {
        "name": "DataSummary",
        "displayName": "Data Summary",
        "category": "DataIO",
        "color": [0.3, 0.59, 0.88],
        "inputs": [
            {"name": "filename", "type": "string", "required": False,
             "description": "Name of loaded file"},
            {"name": "data", "type": "any",
             "description": "Loaded data struct or vector"},
            {"name": "platform", "type": "string", "required": False,
             "description": "NPS platform name"},
            {"name": "ampsPerVolt", "type": "numeric", "required": False,
             "description": "Current conversion factor (for mzNPS)"},
        ],
        "outputs": [
            {"name": "data", "type": "any",
             "description": "Data (pass-through)"},
            {"name": "info", "type": "any",
             "description": "Metadata bundle: {filename, platform}"},
        ],
        "runner": "runDataSummary",
        "isInteractive": True,
        "isParameterized": False,
    }
