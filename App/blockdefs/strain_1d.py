def get_definition():
    return {
        "name": "Strain1D",
        "displayName": "Strain 1D",
        "category": "Analysis",
        "color": [0.82, 0.58, 0.34],
        "inputs": [
            {"name": "diameter", "type": "numeric", "required": True,
             "description": "Free cell diameter (d)"},
            {"name": "Wc", "type": "numeric", "required": True,
             "description": "Contraction width"},
        ],
        "outputs": [
            {"name": "strain", "type": "numeric", "required": True,
             "displayName": "Strain 1D",
             "description": "1D strain = (d - Wc) / d"},
        ],
        "runner": "runStrain1D",
        "defaultParameters": {
            "formula": "(diameter - Wc) / diameter",
        },
    }
