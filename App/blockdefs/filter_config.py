def get_definition():
    return {
        "name": "FilterConfig",
        "displayName": "Filter Config",
        "category": "Filters",
        "color": [0.82, 0.45, 0.62],
        "inputs": [
            {"name": "addInput", "type": "struct", "required": False,
             "description": "Connect to add", "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "filterConfig", "type": "struct",
             "description": "Compiled filter chain (list of filter dicts)"},
        ],
        "runner": "runFilterConfig",
        "isDynamic": True,
        "isFilterConfig": True,
    }
