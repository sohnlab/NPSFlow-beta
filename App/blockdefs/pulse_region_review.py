def get_definition():
    return {
        "name": "PulseRegionReview",
        "displayName": "Detection Review",
        "category": "Detection",
        "color": [0.72, 0.28, 0.28],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True, "description": "Input signal"},
            {"name": "detectedPulses", "type": "numeric", "required": True, "description": "Nx2 pulse region indices"},
            {"name": "settingsFile", "type": "string", "required": False, "description": "Path to saved classification settings JSON"},
        ],
        "outputs": [
            {"name": "single", "type": "numeric", "description": "Nx2 indices of single pulses"},
            {"name": "coincident", "type": "numeric", "description": "Nx2 indices of coincident pulses"},
            {"name": "noise", "type": "numeric", "description": "Nx2 indices of noise pulses"},
            {"name": "uncertain", "type": "numeric", "description": "Nx2 indices of uncertain pulses"},
        ],
        "isInteractive": True,
        "runner": "runPulseRegionReview",
    }
