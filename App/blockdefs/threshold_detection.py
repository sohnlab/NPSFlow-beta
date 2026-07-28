def get_definition():
    return {
        "name": "ThresholdDetection",
        "displayName": "RT Detection (Threshold)",
        "category": "Sorting",
        "color": [0.78, 0.50, 0.40],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True,
             "description": "Raw 1-D signal to scan for pulses."},
            {"name": "loadFile", "type": "string", "required": False,
             "description":
               "Path to a saved detector-settings JSON. Relative paths "
               "resolve from project root; bare names resolve to "
               "SavedTemplates/Sorting/RealtimeThreshold/<name>.json."},
        ],
        "outputs": [
            {"name": "detectedPulses", "type": "numeric",
             "description": "Nx2 array of [start, end] indices into dataIn."},
            {"name": "score", "type": "numeric",
             "description": "|signal − baseline| envelope, same length as "
                            "dataIn."},
            {"name": "coincidentHint", "type": "numeric",
             "description": "Length-N int8 column; always 0 for this "
                            "detector (no coincident-split logic)."},
        ],
        "isInteractive": True,
        "runner": "runThresholdDetection",
    }
