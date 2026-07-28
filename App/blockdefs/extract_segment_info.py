def get_definition():
    return {
        "name": "ExtractSegmentInfo",
        "displayName": "Extract Segment Info",
        "category": "FeatureExtraction",
        "color": [0.17, 0.63, 0.76],
        "inputs": [
            {"name": "segmentSlices", "type": "any"},
        ],
        "outputs": [
            {"name": "segmentInfo", "type": "struct"},
        ],
        "runner": "runExtractSegmentInfo",
    }
