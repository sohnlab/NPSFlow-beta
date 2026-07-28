"""Runner for LoadGeometry block — fallback (execution handled by engine)."""


def run(inputs, params, block):
    raise RuntimeError("LoadGeometry should be handled by the engine. Please reload.")
