"""Runner for DefineParameters block — applies saved variables and global parameters."""


def run(inputs, params, block):
    # Just pass through — variables are already in block.parameters,
    # and the engine handles storing them in reference_store/global_vars
    return {"run": True}
