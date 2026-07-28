def get_definition():
    return {
        "name": "DebugLabeling",
        "displayName": "Debug - Labeling",
        "category": "Utility",
        "color": [0.5, 0.5, 0.55],
        "inputs": [
            {"name": "labeledData", "type": "any", "required": True,
             "description": "Output from Data Labeling (struct with data + labels)"},
            {"name": "sampleRate", "type": "numeric", "required": False,
             "description": "Sample rate (Hz); x-axis is sample index if omitted"},
        ],
        "outputs": [],
        "isInteractive": True,
        "runner": "runDebugLabeling",
    }
