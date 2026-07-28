def get_definition():
    return {
        "name": "CombineData",
        "displayName": "Combine Data",
        "category": "DataIO",
        "color": [0.55, 0.47, 0.76],
        "showPortLabels": True,
        "inputs": [
            {"name": "dataIn", "type": "struct", "required": True,
             "description": "Load File struct ({files: [...]}) or a list of "
                            "file entries to concatenate in order."},
        ],
        "outputs": [
            {"name": "data", "type": "numeric",
             "description": "All files' data concatenated along axis 0."},
            {"name": "labels", "type": "numeric",
             "description": "Concatenated labels; None when any file lacks "
                            "them."},
            {"name": "boundaries", "type": "numeric",
             "description": "Per-file start indices into data (length N+1; "
                            "file k spans boundaries[k]:boundaries[k+1]). "
                            "Use to mask the artificial seams between "
                            "files."},
            {"name": "fileNames", "type": "any",
             "description": "File names in concatenation order."},
        ],
        "runner": "runCombineData",
    }
