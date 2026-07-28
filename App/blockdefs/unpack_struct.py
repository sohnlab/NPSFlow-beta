def get_definition():
    return {
        "name": "UnpackStruct",
        "displayName": "Unpack Variable",
        "category": "DataIO",
        "color": [0.55, 0.47, 0.76],
        "inputs": [
            {"name": "structIn", "type": "any", "displayName": "VarIn"},
        ],
        "outputs": [],
        "isDynamic": True,
        "isUnpacker": True,
        "defaultParameters": {
            "unpackDepth": 1,
        },
    }
