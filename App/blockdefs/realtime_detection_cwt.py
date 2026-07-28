"""Block definition for Realtime Detection (CWT)."""


def get_definition():
    return {
        "name": "RealtimeDetectionCWT",
        "displayName": "RT Detection (CWT)",
        "category": "Sorting",
        "color": [0.78, 0.50, 0.40],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True,
             "description": "Raw 1-D signal to scan for pulses."},
            {"name": "wavelet", "type": "numeric", "required": True,
             "description": "Mother wavelet (e.g. from Wavelet Editor)."},
            {"name": "loadFile", "type": "string", "required": False,
             "description":
               "Path to a saved detector-settings JSON. Relative paths "
               "resolve from project root; bare names resolve to "
               "SavedTemplates/Detection/RealtimeCWT/<name>.json."},
        ],
        "outputs": [
            {"name": "detectedPulses", "type": "numeric",
             "description": "Nx2 array of [start, end] indices into dataIn."},
            {"name": "score", "type": "numeric",
             "description": "Detection score (max- or rss-combined CWT "
                            "coefficients), same length as dataIn."},
            {"name": "coincidentHint", "type": "numeric",
             "description": "Length-N int8 column; 1 = candidate coincident "
                            "region (≥2 peaks merged or width > 1.3×template)."},
        ],
        "isInteractive": True,
        "runner": "runRealtimeDetectionCWT",
    }
