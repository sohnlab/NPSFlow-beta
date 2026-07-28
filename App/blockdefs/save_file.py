def get_definition():
    return {
        "name": "SaveFile",
        "displayName": "Save File",
        "category": "DataIO",
        "subcategory": "Output",
        "color": [0.34, 0.63, 0.92],
        "inputs": [
            {"name": "filename", "type": "string", "required": False,
             "displayName": "Filename",
             "description": "Suggested output filename (without extension)"},
            {"name": "trimIndex", "type": "numeric", "required": False,
             "displayName": "trimIndex", "preserveCase": True,
             "description": "Trim window [start, end] to record as metadata "
                            "(2 values, not padded to the data length)"},
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [],
        "isDynamic": True,
        "isInteractive": True,
        "runner": "runSaveFile",
    }
