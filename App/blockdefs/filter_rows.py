def get_definition():
    return {
        "name": "FilterRows",
        "displayName": "Filter Rows",
        "category": "Analysis",
        "color": [0.72, 0.48, 0.24],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [],
        "runner": "runFilterRows",
        "defaultParameters": {
            "column": "",
            "operator": ">",
            "value": 0,
            "displayText": "Filter Rows",
        },
        "parameterDefinitions": [
            {"name": "column", "displayName": "Column (port name)",
             "type": "text"},
            {"name": "operator", "displayName": "Operator",
             "type": "choice", "options": [">", "<", ">=", "<=",
                                           "=", "!="]},
            {"name": "value", "displayName": "Threshold", "type": "numeric"},
        ],
        "isDynamic": True,
        "isFilterRows": True,
        "isMirrorPorts": True,
        "showPortLabels": True,
    }
