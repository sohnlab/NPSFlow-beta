def get_definition():
    return {
        "name": "DifferenceBlock",
        "displayName": "Difference",
        "category": "Analysis",
        "color": [0.66, 0.59, 0.76],
        "inputs": [
            {"name": "input1", "type": "numeric", "required": True,
             "displayName": "in 1",
             "description": "First operand (scalar or vector)"},
            {"name": "input2", "type": "numeric", "required": True,
             "displayName": "in 2",
             "description": "Second operand (scalar or vector)"},
        ],
        "outputs": [
            {"name": "difference", "type": "numeric",
             "description": "in 1 - in 2"},
        ],
        "runner": "runDifferenceBlock",
        "defaultParameters": {
            "formula": "input1 - input2",
        },
    }
