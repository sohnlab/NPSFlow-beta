def get_definition():
    return {
        "name": "LoadLabeledData",
        "displayName": "Load Labeled Data",
        "category": "DataIO",
        "color": [0.22, 0.51, 0.8],
        "inputs": [
            {"name": "run", "type": "any"},
            {"name": "folder", "type": "string", "required": False, "displayName": "Folder"},
        ],
        "outputs": [
            {"name": "labeledDataList", "type": "any",
             "description": "List of dicts {data: float64, labels: int32, fileName: str} — one per labeled CSV found (recursive)."},
        ],
        "runner": "runLoadLabeledData",
    }
