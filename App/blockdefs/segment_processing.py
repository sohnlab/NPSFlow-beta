def get_definition():
    return {
        "name": "SegmentProcessing",
        "displayName": "Segment Processing",
        "category": "Analysis",
        "color": [0.21, 0.67, 0.8],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input",
             "description": "Connect a segment struct"},
        ],
        "outputs": [],
        "isDynamic": True,
        "isSegmentProcessing": True,
        "runner": "runSegmentProcessing",
    }
