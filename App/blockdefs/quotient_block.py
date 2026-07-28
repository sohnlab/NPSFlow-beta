def get_definition():
    return {
        "name": "QuotientBlock",
        "displayName": "Quotient",
        "category": "Analysis",
        "color": [0.66, 0.59, 0.76],
        "inputs": [
            {"name": "input1", "type": "numeric", "required": True,
             "displayName": "dividend",
             "description": "Numerator (scalar or 1D vector)"},
            {"name": "input2", "type": "numeric", "required": True,
             "displayName": "divisor",
             "description": "Denominator (scalar or 1D vector); 0 -> inf"},
        ],
        "outputs": [
            {"name": "quotient", "type": "numeric",
             "description": "dividend / divisor (element-wise; scalar broadcasts)"},
        ],
        "runner": "runQuotientBlock",
    }
