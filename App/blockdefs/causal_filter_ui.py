def get_definition():
    return {
        "name": "CausalFilterUI",
        "displayName": "Causal Filter",
        "category": "Filters",
        "color": [0.80, 0.42, 0.60],
        "inputs": [
            {
                "name": "data",
                "type": "any",
                "description": "Load Data struct ({fileNames, files}) or a bare array",
            },
            {
                "name": "sampleRate",
                "type": "numeric",
                "required": False,
                "displayName": "sample rate",
                "description": "Sample rate in Hz; falls back to 10 kHz when "
                "unconnected",
            },
            {"name": "filterConfigIn", "type": "struct", "required": False},
        ],
        "outputs": [
            {"name": "filtered", "type": "any"},
            {"name": "filterConfig", "type": "struct"},
        ],
        "isInteractive": True,
        "runner": "runCausalFilterUI",
    }
