"""Runner for FilterConfig block - compiles individual filter configs into a chain.

The output format matches the filterConfig output from the Filter UI block:
a list of filter config dicts, each with keys: type, cutoff1, cutoff2,
notch_mode, notch_bw, bp_mode, bp_center, bp_bw, description.
"""


def run(inputs, params, block):
    chain = []
    # Dynamic ports are named in1, in2, ...
    for key in sorted(inputs.keys()):
        cfg = inputs.get(key)
        if cfg is None:
            continue
        # Single filter config dict
        if isinstance(cfg, dict) and "type" in cfg:
            chain.append(cfg)
        # List of configs (e.g. from Notch block with harmonics)
        elif isinstance(cfg, list):
            for item in cfg:
                if isinstance(item, dict) and "type" in item:
                    chain.append(item)
    return {"filterConfig": chain}
