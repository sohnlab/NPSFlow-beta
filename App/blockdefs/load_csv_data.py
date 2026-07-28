def get_definition():
    return {
        "name": "LoadCSVData",
        "displayName": "Load Data File",
        "category": "DataIO",
        "color": [0.22, 0.51, 0.8],
        "inputs": [
            {"name": "run", "type": "any"},
            {"name": "dataSource", "type": "string", "required": False, "displayName": "Data Source"},
            {"name": "addInput", "type": "string", "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "data", "type": "any"},
            {"name": "fileNames", "type": "any"},
        ],
        "isDynamic": True,
        "runner": "runLoadCSVData",
    }
