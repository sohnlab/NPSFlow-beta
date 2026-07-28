def get_definition():
    return {
        "name": "LoadData",
        "displayName": "Load File",
        "category": "DataIO",
        "color": [0.22, 0.51, 0.8],
        "isInteractive": True,
        "showPortLabels": True,
        "inputs": [
            {"name": "run", "type": "any", "required": False},
            {"name": "path", "type": "string", "required": False,
             "displayName": "Path"},
        ],
        "outputs": [
            {"name": "data", "type": "any",
             "description": "Unified struct. Keys: fileNames, files. Feed "
                            "into an Unpack block to expose individual "
                            "fields, or a Combine Data block to concatenate "
                            "all files."},
        ],
        "defaultParameters": {
            "lastPath": "",
            "displayText": "",
        },
        "parameterDefinitions": [
            {"name": "lastPath", "type": "text", "displayName": "Last path"},
        ],
        "resetOnReload": ["displayText"],
        "runner": "runLoadData",
    }
