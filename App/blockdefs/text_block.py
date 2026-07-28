def get_definition():
    return {
        "name": "TextBlock",
        "displayName": "Text",
        "category": "DataIO",
        "color": [0.52, 0.44, 0.73],
        "inputs": [],
        "outputs": [
            {"name": "text", "type": "string"},
        ],
        "runner": "runText",
        "isParameterized": True,
        "isDisplay": True,
        "defaultParameters": {
            "valueString": "",
            "displayText": "",
        },
    }
