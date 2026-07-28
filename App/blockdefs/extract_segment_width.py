def get_definition():
    return {
        "name": "ExtractSegmentWidth",
        "displayName": "Extract Segment Width",
        "category": "FeatureExtraction",
        "color": [0.48, 0.73, 0.48],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "widthMatrix", "type": "numeric",
             "description": "Width matrix (N_pulses x N_segments)"},
        ],
        "runner": "runExtractSegmentWidth",
    }
