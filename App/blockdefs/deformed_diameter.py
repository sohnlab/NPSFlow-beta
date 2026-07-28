def get_definition():
    return {
        "name": "DeformedDiameter",
        "displayName": "Deformed Diameter",
        "category": "Analysis",
        "color": [0.76, 0.52, 0.28],
        "inputs": [
            {"name": "d_c", "type": "numeric", "required": True,
             "description": "Equivalent spherical diameter in contraction (um)"},
            {"name": "Wc", "type": "numeric", "required": True,
             "description": "Contraction channel width (um)"},
        ],
        "outputs": [
            {"name": "deformedDiameter", "type": "numeric",
             "description": "Deformed diameter (um)"},
        ],
        "runner": "runDeformedDiameter",
        "defaultParameters": {
            "formula": "0.01 * (np.pi / 4 * Wc) * d_c ** 2",
        },
    }
