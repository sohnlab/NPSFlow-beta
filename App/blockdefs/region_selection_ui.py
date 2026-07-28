def get_definition():
    return {
        "name": "RegionSelectionUI",
        "displayName": "Region Selection UI",
        "category": "FeatureExtraction",
        "color": [0.19, 0.65, 0.78],
        "inputs": [
            {"name": "templateData", "type": "struct"},
        ],
        "outputs": [
            {"name": "selectedSegments", "type": "struct"},
            {"name": "segmentSlices", "type": "any"},
        ],
        "isInteractive": True,
        "runner": "runRegionSelectionUI",
    }
