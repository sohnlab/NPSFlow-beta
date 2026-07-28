def get_definition():
    return {
        "name": "AnalysisMethodSelector",
        "displayName": "Analysis Method Selector",
        "category": "Analysis",
        "color": [0.78, 0.47, 0.16],
        "inputs": [
            {"name": "templateData", "type": "struct"},
            {"name": "pulseFeatures", "type": "struct", "required": False},
        ],
        "outputs": [
            {"name": "analysisResult", "type": "struct"},
        ],
        "isInteractive": True,
        "runner": "runAnalysisMethodSelector",
    }
