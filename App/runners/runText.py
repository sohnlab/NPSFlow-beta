"""Runner for Text block - returns a user-defined text string."""


def run(inputs, params, block):
    return {"text": params.get("valueString", "")}
