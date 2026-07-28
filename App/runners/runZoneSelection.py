"""Runner for ZoneSelection block - outputs saved zone indices."""


def run(inputs, params, block):
    zones = params.get("selectedZones", [])
    if zones:
        block.parameters["displayText"] = ", ".join(str(z) for z in zones)
    else:
        block.parameters["displayText"] = "(all)"
    return {"zones": zones if zones else None}
