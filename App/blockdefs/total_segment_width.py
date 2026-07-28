def get_definition():
    return {
        "name": "TotalSegmentWidth",
        "displayName": "Total Segment Width",
        "category": "FeatureExtraction",
        "color": [0.94, 0.63, 0.32],
        "inputs": [
            {"name": "in1", "type": "any", "required": True,
             "displayName": "Start Segment",
             "description": "Segment whose startGlobal defines the start boundary"},
            {"name": "in2", "type": "any", "required": True,
             "displayName": "End Segment",
             "description": "Segment whose endGlobal defines the end boundary"},
        ],
        "outputs": [
            {"name": "width", "type": "numeric",
             "description": "Width in samples per pulse (endGlobal[in2] - startGlobal[in1])"},
        ],
        "runner": "runTotalSegmentWidth",
    }
