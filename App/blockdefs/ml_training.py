def get_definition():
    return {
        "name": "MLTraining",
        "displayName": "ML Training",
        "category": "Sorting",
        "color": [0.55, 0.45, 0.72],
        "inputs": [
            {"name": "run", "type": "any", "required": False,
             "description": "Optional sequencing trigger."},
            {"name": "folder", "type": "string", "required": False,
             "description":
               "Folder of labeled files (.npz/.csv from Data Labeling / "
               "Save File). Scanned recursively; relative paths resolve "
               "from project root."},
            {"name": "sampleRate", "type": "numeric", "required": False,
             "description": "Sample rate of the labeled recordings (Hz). "
                            "Overrides the dialog value when wired."},
            {"name": "settingsFile", "type": "string", "required": False,
             "description": "Path to saved ML Training settings JSON."},
        ],
        "outputs": [
            {"name": "modelFile", "type": "string",
             "description":
               "Path to the trained classifier bundle (.joblib). Load with "
               "ml_training_core.load_bundle and stream through "
               "StreamEventClassifier for real-time baseline/single/"
               "coincident classification."},
            {"name": "metrics", "type": "struct",
             "description": "Train/val/test split, per-horizon metrics, "
                            "stream-simulation results, feature "
                            "importances."},
        ],
        "isInteractive": True,
        "runner": "runMLTraining",
    }
