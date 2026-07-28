def get_definition():
    return {
        "name": "DisplayBlock",
        "displayName": "Display",
        "category": "DataIO",
        "color": [0.55, 0.55, 0.55],
        "inputs": [
            {"name": "dataIn", "type": "any"},
        ],
        "outputs": [],
        "isDisplay": True,
        "runner": "runDisplayBlock",
    }
