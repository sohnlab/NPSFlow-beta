def get_definition():
    return {
        "name": "ConstantBlock",
        "displayName": "Constant",
        "category": "DataIO",
        "color": [0.5, 0.42, 0.71],
        "isCompact": True,
        "defaultSize": [100, 20],
        "inputs": [],
        "outputs": [
            {"name": "value", "type": "numeric"},
        ],
        "runner": "runConstant",
        "defaultParameters": {
            "value": "0",
        },
    }
