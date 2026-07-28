def get_definition():
    return {
        "name": "Priority",
        "displayName": "Sequence",
        "category": "FlowControl",
        "color": [0.22, 0.61, 0.23],
        "inputs": [
            {"name": "Run", "type": "any", "required": False,
             "displayName": "Run",
             "description": "Run signal to distribute to ordered outputs"},
        ],
        "outputs": [
            {"name": "addOutput", "type": "any", "required": False,
             "displayName": "Add output",
             "description": "Connect to add an ordered output"},
        ],
        "isPriority": True,
        "isDynamic": True,
        "defaultSize": [120, 53],
    }
