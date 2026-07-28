def get_definition():
    return {
        "name": "FilterUI",
        "displayName": "Filter UI",
        "category": "Filters",
        "color": [0.80, 0.42, 0.60],
        "inputs": [
            {"name": "data", "type": "any"},
            {"name": "filterConfigIn", "type": "struct", "required": False},
        ],
        "outputs": [
            {"name": "filteredData", "type": "any"},
            {"name": "filterConfig", "type": "struct"},
        ],
        "isInteractive": True,
        "runner": "runFilterUI",
    }
