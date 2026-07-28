def get_definition():
    return {
        "name": "WeightedMean",
        "displayName": "Weighted Mean",
        "category": "Analysis",
        "color": [0.7, 0.63, 0.8],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False, "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "weightedMean", "type": "numeric"},
        ],
        "isWeightedMean": True,
        "runner": "runWeightedMean",
    }
