def get_definition():
    return {
        "name": "PlotData",
        "displayName": "Plot Data",
        "category": "Utility",
        "color": [0.18, 0.45, 0.54],
        "inputs": [
            {"name": "x1", "type": "any", "required": False,
             "displayName": "x1:"},
            {"name": "y1", "type": "any", "required": False,
             "displayName": "y1:"},
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add series"},
        ],
        "outputs": [],
        "isDynamic": True,
        "isPlotData": True,
        "isInteractive": True,
        "runner": "runPlotData",
        "defaultSize": [180, 100],
    }
