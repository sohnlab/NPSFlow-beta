def get_definition():
    return {
        "name": "PoreDiameter",
        "displayName": "Pore Diameter",
        "category": "Analysis",
        "color": [0.74, 0.5, 0.26],
        "inputs": [
            {"name": "dROverR", "type": "numeric", "required": True,
             "displayName": "ΔR/R", "preserveCase": True,
             "description": "Relative resistance change"},
            {"name": "d", "type": "numeric", "required": True,
             "displayName": "d", "description": "Particle diameter"},
            {"name": "L", "type": "numeric", "required": True,
             "displayName": "L", "description": "Pore length"},
        ],
        "outputs": [
            {"name": "D", "type": "numeric",
             "displayName": "D", "description": "Pore diameter"},
        ],
        "runner": "runPoreDiameter",
    }
