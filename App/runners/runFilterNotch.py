"""Runner for FilterNotch block - produces filter config dict(s).

Supports two modes:
  - freqBW: center frequency + bandwidth (supports harmonics)
  - band: lower/upper frequency bounds
"""


def run(inputs, params, block):
    mode = params.get("mode", "freqBW")
    harmonics = params.get("harmonics", [1])
    if not harmonics:
        harmonics = [1]

    if mode == "band":
        lower = float(params.get("lower", 59.0))
        upper = float(params.get("upper", 61.0))
        _lo = int(lower) if lower == int(lower) else lower
        _hi = int(upper) if upper == int(upper) else upper
        block.parameters["displayText"] = f"{_lo}-{_hi} Hz"
        return {
            "filterConfig": {
                "type": "notch",
                "cutoff1": lower,
                "cutoff2": upper,
                "notch_mode": "band",
                "notch_bw": 0,
                "bp_mode": "lohi",
                "bp_center": 0,
                "bp_bw": 0,
                "description": f"Notch: {lower:.3f} - {upper:.3f} Hz",
            }
        }

    # freqBW mode
    freq = float(params.get("freq", 60.0))
    bw = float(params.get("bw", 2.0))
    _f = int(freq) if freq == int(freq) else freq
    _b = int(bw) if bw == int(bw) else bw

    if len(harmonics) == 1 and harmonics[0] == 1:
        block.parameters["displayText"] = f"{_f}-{_b} Hz"
    else:
        h_str = ",".join(str(h) for h in sorted(harmonics))
        block.parameters["displayText"] = f"{_f}-{_b} Hz x{h_str}"

    def _make_config(f, bandwidth):
        return {
            "type": "notch",
            "cutoff1": f,
            "cutoff2": f,
            "notch_mode": "freqBW",
            "notch_bw": bandwidth,
            "bp_mode": "lohi",
            "bp_center": 0,
            "bp_bw": 0,
            "description": f"Notch: {f:.3f} Hz (BW={bandwidth:.1f} Hz)",
        }

    if len(harmonics) == 1 and harmonics[0] == 1:
        return {"filterConfig": _make_config(freq, bw)}

    configs = []
    for h in sorted(harmonics):
        configs.append(_make_config(freq * h, bw))
    return {"filterConfig": configs}
