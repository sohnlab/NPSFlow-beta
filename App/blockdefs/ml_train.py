def get_definition():
    return {
        "name": "MLTrain",
        "displayName": "ML Train",
        "category": "Sorting",
        "color": [0.55, 0.45, 0.72],
        "showPortLabels": True,
        "inputs": [
            {"name": "trainSet", "type": "struct", "required": True,
             "description": "Training files ({fileNames, files} struct, "
                            "e.g. from ML Split)."},
            {"name": "sampleRate", "type": "numeric", "required": False,
             "description": "Sample rate of the recordings (Hz); overrides "
                            "the parameter when wired."},
        ],
        "outputs": [
            {"name": "modelFile", "type": "string",
             "description": "Path to the trained classifier bundle "
                            "(.joblib) for ML Evaluate / ML Classify."},
            {"name": "trainInfo", "type": "struct",
             "description": "Files, event counts, row count, top feature "
                            "importances."},
        ],
        "defaultParameters": {
            "model": "rf",
            "strategy": "two_stage",
            "oversample": "smote",
            "oversampleRatio": 1.0,
            "noiseRatio": 1.0,
            "hardNegativeRatio": 1.0,
            "horizonsMs": "1, 2, 5, 10, 20, 40",
            "seed": 0,
            "sampleRate": 50000,
            "modelPath": "Output/EventClassifier/event_classifier.joblib",
        },
        "parameterDefinitions": [
            {"name": "model", "displayName": "Model", "type": "choice",
             "options": ["rf", "gb"]},
            {"name": "strategy", "displayName": "Strategy", "type": "choice",
             "options": ["two_stage", "flat"]},
            {"name": "oversample", "displayName": "Oversample",
             "type": "choice", "options": ["smote", "random", "none"]},
            {"name": "oversampleRatio", "displayName": "Oversample ratio",
             "type": "numeric"},
            {"name": "noiseRatio", "displayName": "Baseline ratio",
             "type": "numeric"},
            {"name": "hardNegativeRatio", "displayName": "Hard negatives",
             "type": "numeric"},
            {"name": "horizonsMs", "displayName": "Horizons (ms)",
             "type": "text"},
            {"name": "seed", "displayName": "Seed", "type": "numeric"},
            {"name": "sampleRate", "displayName": "Sample rate (Hz)",
             "type": "numeric"},
            {"name": "modelPath", "displayName": "Model path",
             "type": "text"},
        ],
        "runner": "runMLTrain",
    }
