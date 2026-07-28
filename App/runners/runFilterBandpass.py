"""Runner for FilterBandpass block - produces a filter config dict."""


def run(inputs, params, block):
    lower = float(params.get("lower", 10.0))
    upper = float(params.get("upper", 1000.0))
    _lo = int(lower) if lower == int(lower) else lower
    _hi = int(upper) if upper == int(upper) else upper
    block.parameters["displayText"] = f"{_lo}-{_hi} Hz"
    return {
        "filterConfig": {
            "type": "bandpass",
            "cutoff1": lower,
            "cutoff2": upper,
            "notch_mode": "freqBW",
            "notch_bw": 0,
            "bp_mode": "lohi",
            "bp_center": (lower + upper) / 2,
            "bp_bw": upper - lower,
            "description": f"Bandpass: {lower:.3f} - {upper:.3f} Hz",
        }
    }
