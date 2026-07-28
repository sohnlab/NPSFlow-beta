def get_definition():
    return {
        "name": "PulseSlicing",
        "displayName": "Pulse Slicing",
        "category": "FeatureExtraction",
        "color": [0.16, 0.62, 0.75],
        "inputs": [
            {"name": "pulseSegments", "type": "struct", "required": True,
             "description": "Segment bounds/labels from PulseFeatureExtraction"},
            {"name": "pulsePeakLocations", "type": "any", "required": False,
             "description": "Peak indices per pulse (list of arrays)"},
            {"name": "rectangularizedPulses", "type": "any", "required": True,
             "description": "Rectangularized pulse waveforms (list of arrays)"},
            {"name": "acceptanceID", "type": "any", "required": False,
             "description": "Boolean acceptance array"},
            {"name": "pulseStartIndices", "type": "numeric", "required": False,
             "description": "Global start index per pulse"},
        ],
        "outputs": [],
        "isDynamic": True,
        "isPulseSlicing": True,
        "isInteractive": True,
        "runner": "runPulseSlicing",
    }
