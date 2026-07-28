def get_definition():
    return {
        "name": "MLEvaluate",
        "displayName": "ML Evaluate",
        "category": "Sorting",
        "color": [0.5, 0.5, 0.74],
        "showPortLabels": True,
        "inputs": [
            {"name": "modelFile", "type": "string", "required": True,
             "description": "Trained classifier bundle path (from "
                            "ML Train / ML Training)."},
            {"name": "dataSet", "type": "struct", "required": True,
             "description": "Labeled files to score ({fileNames, files} "
                            "struct, e.g. ML Split's valSet or testSet)."},
        ],
        "outputs": [
            {"name": "metrics", "type": "struct",
             "description": "Per-horizon accuracy / macro-F1 / per-class "
                            "recalls / confusion (windows rebuilt with the "
                            "bundle's training config)."},
            {"name": "streamScores", "type": "struct",
             "description": "Per-file real-time stream simulation scores "
                            "(None unless Stream eval = yes)."},
        ],
        "defaultParameters": {
            "streamEval": "no",
        },
        "parameterDefinitions": [
            {"name": "streamEval", "displayName": "Stream eval",
             "type": "choice", "options": ["no", "yes"]},
        ],
        "runner": "runMLEvaluate",
    }
