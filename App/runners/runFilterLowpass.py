"""Runner for FilterLowpass block - produces a filter config dict."""


def run(inputs, params, block):
    cutoff = float(params.get("cutoff", 1000.0))
    _v = int(cutoff) if cutoff == int(cutoff) else cutoff
    block.parameters["displayText"] = f"{_v} Hz"
    return {
        "filterConfig": {
            "type": "lowpass",
            "cutoff1": cutoff,
            "cutoff2": 0,
            "notch_mode": "freqBW",
            "notch_bw": 0,
            "bp_mode": "lohi",
            "bp_center": 0,
            "bp_bw": 0,
            "description": f"Lowpass cutoff: {cutoff:.3f} Hz",
        }
    }
