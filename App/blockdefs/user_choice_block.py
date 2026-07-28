def get_definition():
    return {
        "name": "UserChoiceBlock",
        "displayName": "User Choice",
        "category": "Utility",
        "color": [0.63, 0.63, 0.63],
        "inputs": [
            {"name": "trigger", "type": "any", "required": False},
        ],
        "outputs": [
            {"name": "choice", "type": "any"},
        ],
        "isInteractive": True,
        "runner": "runUserChoice",
        "defaultParameters": {
            "question": "Continue?",
            "options": ["Yes", "No"],
        },
    }
