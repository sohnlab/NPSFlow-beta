def get_definition():
    return {
        "name": "PulseDetectionBC",
        "displayName": "Pulse Detection BC",
        "category": "Detection",
        "color": [0.88, 0.44, 0.44],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True, "description": "Input signal (time x channels)"},
            {"name": "pulseTemplate", "type": "numeric", "required": False, "description": "Pulse template for reference (optional)"},
            {"name": "loadFile", "type": "string", "required": False,
             "description": "Path to saved detection settings JSON file"},
        ],
        "outputs": [
            {"name": "trendline", "type": "numeric", "description": "Estimated baseline trendline (same size as input; N x Z for multi-zone)"},
            {"name": "detectedPulses", "type": "numeric", "description": "Nx2 detected pulse regions [start, end]; merged across zones for multi-zone input"},
            {"name": "detectedPerZone", "type": "any", "description": "Per-zone detected regions, zone-grouped {num_zones, zones:[Nx2,...]} (bare Nx2 for single zone)"},
        ],
        "isInteractive": True,
        "runner": "runPulseDetectionBC",
    }
