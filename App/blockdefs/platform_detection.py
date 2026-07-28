def get_definition():
    return {
        "name": "PlatformDetection",
        "displayName": "Platform Detection",
        "category": "Preprocessing",
        "color": [0.44, 0.78, 0.74],
        "inputs": [
            {"name": "ampsPerVolt", "type": "numeric", "required": False,
             "description": "Current conversion factor (required for mzNPS)"},
            {"name": "data", "type": "any",
             "description": "Raw signal data (1-D vector)"},
        ],
        "outputs": [
            {"name": "data", "type": "any",
             "description": "Output data (resistance for mzNPS or passthrough)"},
            {"name": "platform", "type": "any",
             "description": "Detected platform name"},
        ],
        "runner": "runPlatformDetection",
        "isParameterized": True,
        "isPlatformDetection": True,
        "defaultParameters": {
            "mzBatchSize": 7,
            "useCurrentCol": True,
            "currentCol": 1,
        },
    }
