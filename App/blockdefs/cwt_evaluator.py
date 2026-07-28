"""Block definition for CWT Evaluator: compare wavelets on the same signal."""


def get_definition():
    return {
        "name": "CwtEvaluator",
        "displayName": "CWT Evaluator",
        "category": "Detection",
        "color": [0.55, 0.40, 0.78],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True,
             "description": "Raw 1-D signal to scan for pulses."},
            {"name": "templateWavelet", "type": "numeric", "required": False,
             "description":
               "User template wavelet (e.g. from Wavelet Editor). If "
               "omitted, only standard wavelets are evaluated."},
            {"name": "groundTruth", "type": "numeric", "required": False,
             "description":
               "Optional Nx2 [start, end] array of true pulse regions. "
               "When present the metrics include precision / recall / "
               "F1 (greedy IoU matching)."},
            {"name": "loadFile", "type": "string", "required": False,
             "description":
               "Path to a saved evaluator-settings JSON. Relative paths "
               "resolve from project root; bare names resolve to "
               "SavedTemplates/Detection/CwtEvaluator/<name>.json."},
        ],
        "outputs": [
            {"name": "detectedPulses", "type": "numeric",
             "description":
               "Nx2 array from the chosen winner wavelet."},
            {"name": "score", "type": "numeric",
             "description": "Winner wavelet's CWT score (length matches dataIn)."},
            {"name": "metrics", "type": "numeric",
             "description":
               "Dict {wavelet_name: {count, peakSnrMean, threshold, "
               "tp, fp, fn, precision, recall, f1, meanIoU}}."},
            {"name": "winnerName", "type": "string",
             "description": "Name of the wavelet selected as winner."},
        ],
        "isInteractive": True,
        "runner": "runCwtEvaluator",
    }
