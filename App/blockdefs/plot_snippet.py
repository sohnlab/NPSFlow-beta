def get_definition():
    return {
        "name": "PlotSnippet",
        "displayName": "Plot Snippet",
        "category": "Utility",
        "color": [0.34, 0.61, 0.7],
        "inputs": [
            {"name": "dataIn", "type": "any", "displayName": "data:"},
            {"name": "index", "type": "any", "displayName": "index:"},
            {"name": "nPoints", "type": "any", "required": False, "displayName": "n:"},
        ],
        "outputs": [],
        "isInteractive": True,
        "runner": "runPlotSnippet",
        "defaultSize": [180, 80],
    }
