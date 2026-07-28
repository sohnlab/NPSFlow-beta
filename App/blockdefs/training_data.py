def get_definition():
    return {
        "name": "TrainingData",
        "displayName": "Data Labeling",
        "category": "Preprocessing",
        "color": [0.45, 0.55, 0.75],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True, "description": "Raw input signal vector"},
            {"name": "single", "type": "numeric", "required": False, "description": "Nx2 indices of single pulses"},
            {"name": "coincident", "type": "numeric", "required": False, "description": "Nx2 indices of coincident pulses"},
            {"name": "noise", "type": "numeric", "required": False, "description": "Nx2 indices of noise pulses"},
            {"name": "uncertain", "type": "numeric", "required": False, "description": "Nx2 indices of uncertain pulses"},
            {"name": "settingsFile", "type": "string", "required": False, "description": "Path to saved training data settings JSON"},
        ],
        "outputs": [
            {"name": "labeledData", "type": "struct", "description": "Dict with keys 'data' (passthrough of dataIn) and 'labels' (per-sample int array). Wire to Save File for a two-column CSV."},
        ],
        "isInteractive": True,
        "runner": "runTrainingData",
    }
