def get_definition():
    return {
        "name": "SizeCalculation",
        "displayName": "Size Calculation",
        "category": "Analysis",
        "color": [0.79, 0.55, 0.31],
        "inputs": [
            {"name": "De", "type": "numeric", "required": True,
             "displayName": "D_eff", "preserveCase": True,
             "description": "Effective diameter (nm)"},
            {"name": "L", "type": "numeric", "required": True,
             "preserveCase": True,
             "description": "Effective length (nm)"},
            {"name": "dROverR", "type": "numeric", "required": True,
             "displayName": "ΔR/R", "preserveCase": True,
             "description": "Resistance ratio (scalar or array)"},
        ],
        "outputs": [
            {"name": "diameter", "type": "numeric", "required": True,
             "description": "Calculated particle diameter"},
        ],
        "runner": "runSizeCalculation",
        "defaultParameters": {
            "formula": "np.cbrt((dR_R * D**3 * L) / (D + 0.8 * dR_R * L))",
        },
    }
