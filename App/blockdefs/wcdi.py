def get_definition():
    return {
        "name": "WCDI",
        "displayName": "wCDI",
        "category": "Analysis",
        "color": [0.66, 0.42, 0.18],
        "inputs": [
            {"name": "Vc", "type": "numeric", "required": True,
             "description": "Contraction (squeeze) velocity"},
            {"name": "Vnp", "type": "numeric", "required": True,
             "description": "Average node-pore (sizing) velocity"},
            {"name": "diameter", "type": "numeric", "required": True,
             "description": "Free cell diameter (um)"},
            {"name": "H", "type": "numeric", "required": True,
             "description": "Channel height (um)"},
        ],
        "outputs": [
            {"name": "wCDI", "type": "numeric",
             "description": "Whole-cell deformability index (dimensionless)"},
        ],
        "runner": "runWCDI",
        "defaultParameters": {
            "formula": "Vc / Vnp * diameter / H",
        },
    }
