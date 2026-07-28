def get_definition():
    return {
        "name": "MLClassify",
        "displayName": "ML Classify",
        "category": "Sorting",
        "color": [0.47, 0.55, 0.75],
        "showPortLabels": True,
        "inputs": [
            {"name": "modelFile", "type": "string", "required": True,
             "description": "Path to a trained classifier bundle (.joblib) "
                            "from the ML Training block."},
            {"name": "dataIn", "type": "any", "required": True,
             "description": "Signal to classify: raw array, a Load File "
                            "struct/entry, or Data Labeling output. Labels "
                            "found inside are used as ground truth for the "
                            "review plot."},
            {"name": "trueLabels", "type": "numeric", "required": False,
             "description": "Optional per-sample true labels; overrides any "
                            "labels found in dataIn."},
        ],
        "outputs": [
            {"name": "predLabels", "type": "numeric",
             "description": "Per-sample predicted labels (0 baseline, "
                            "1 single, 2 coincident), same length as the "
                            "signal."},
            {"name": "events", "type": "any",
             "description": "Predicted event dicts (onset, end, label, "
                            "proba, per-horizon updates)."},
            {"name": "summary", "type": "struct",
             "description": "Event-level score vs true labels (None when no "
                            "labels were provided)."},
        ],
        "defaultParameters": {
            "fileIndex": 0,
        },
        "parameterDefinitions": [
            {"name": "fileIndex", "displayName": "File index",
             "type": "numeric"},
        ],
        "isInteractive": True,
        "runner": "runMLClassify",
    }
