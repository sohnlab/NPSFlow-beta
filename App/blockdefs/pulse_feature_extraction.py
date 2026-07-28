def get_definition():
    return {
        "name": "PulseFeatureExtraction",
        "displayName": "Template Slicing",
        "category": "FeatureExtraction",
        "color": [0.14, 0.6, 0.73],
        "inputs": [
            {"name": "TPsettings", "type": "struct",
             "required": True, "description": "Template settings struct from Template Settings block"},
        ],
        "outputs": [
            {"name": "pulseFeatures", "type": "struct",
             "description": "Struct with bounds, labels, segmentNumbers, signal, etc."},
        ],
        "isInteractive": True,
        "runner": "runPulseFeatureExtraction",
    }
