def get_definition():
    return {
        "name": "DROverR",
        "displayName": "ΔR/R",
        "category": "Analysis",
        "color": [0.77, 0.53, 0.29],
        "inputs": [
            {"name": "R", "type": "numeric", "required": True,
             "preserveCase": True,
             "description": "R values (array, one per pulse)"},
            {"name": "deltaR", "type": "numeric", "required": True,
             "displayName": "ΔR", "preserveCase": True,
             "description": "ΔR values (array, one per pulse)"},
        ],
        "outputs": [
            {"name": "dROverR", "type": "numeric", "required": True,
             "displayName": "ΔR/R", "preserveCase": True,
             "description": "ΔR / R per pulse"},
        ],
        "runner": "runDROverR",
        "defaultParameters": {
            "formula": "deltaR / R",
        },
    }
