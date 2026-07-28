"""Runner for FilterHighpass block - produces a filter config dict."""


def run(inputs, params, block):
    cutoff = float(params.get("cutoff", 10.0))
    _v = int(cutoff) if cutoff == int(cutoff) else cutoff
    block.parameters["displayText"] = f"{_v} Hz"
    return {
        "filterConfig": {
            "type": "highpass",
            "cutoff1": cutoff,
            "cutoff2": 0,
            "notch_mode": "freqBW",
            "notch_bw": 0,
            "bp_mode": "lohi",
            "bp_center": 0,
            "bp_bw": 0,
            "description": f"Highpass cutoff: {cutoff:.3f} Hz",
        }
    }
