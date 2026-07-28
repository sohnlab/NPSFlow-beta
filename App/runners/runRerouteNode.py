"""Reroute node — pass-through for wire routing."""


def run(inputs, params, block):
    return {"out": inputs.get("in")}
