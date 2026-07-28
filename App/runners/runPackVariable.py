"""Runner for PackVariable — collect all connected inputs into one dict.

Mirrors Unpack Variable in reverse: every connected input port contributes one
key/value pair to the single struct output. The key is the port's display label
(defaults to the upstream variable's name, editable via "Rename Inputs"); the
value is whatever flowed into that port. Colliding keys get a _2, _3 suffix.
"""


def run(inputs, params, block):
    # No block handle (rare): pack the raw inputs minus framework ports.
    if block is None:
        skip = {"addInput", "Run"}
        return {"structOut": {k: v for k, v in inputs.items() if k not in skip}}

    packed = {}
    used = set()
    # Iterate ports (not inputs) so injected globals and the addInput
    # placeholder are naturally excluded, and port order is preserved.
    for port in block.input_ports:
        if port.name == "addInput" or port.name not in inputs:
            continue
        key = (port.display_name or port.name).strip() or port.name
        base, n = key, 2
        while key in used:
            key = f"{base}_{n}"
            n += 1
        used.add(key)
        packed[key] = inputs[port.name]

    return {"structOut": packed}
