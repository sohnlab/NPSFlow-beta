def get_definition():
    return {
        "name": "GlobalSampleRate",
        "displayName": "Global Sample Rate",
        "category": "Variables",
        "color": [0.63, 0.55, 0.84],
        "inputs": [
            {"name": "sampleRate", "type": "numeric", "required": False,
             "displayName": "sample rate"},
            {"name": "downsampleFactor", "type": "numeric", "required": False,
             "displayName": "DS factor"},
        ],
        "outputs": [],
        "isGlobalSampleRate": True,
        "defaultSize": [170, 70],
        # Publishes two globals: sampleRate = base / N (effective) and
        # downsampleFactor = N. base defaults to 10 kHz, N to 1; a wired input
        # supersedes. Non-editable (no defaultParameters) by design — see
        # workflow_engine._execute_block.
    }
