def get_definition():
    return {
        "name": "StatisticsBlock",
        "displayName": "Statistics",
        "category": "Analysis",
        "color": [0.68, 0.44, 0.2],
        "inputs": [
            {"name": "data", "type": "numeric", "required": True,
             "description": "Input vector"},
        ],
        "outputs": [
            {"name": "mean", "type": "numeric"},
            {"name": "std", "type": "numeric"},
            {"name": "median", "type": "numeric"},
            {"name": "min", "type": "numeric"},
            {"name": "max", "type": "numeric"},
            {"name": "count", "type": "numeric"},
        ],
        "runner": "runStatistics",
        "showPortLabels": True,
    }
