def get_definition():
    return {
        "name": "CompileTable",
        "displayName": "Compile Table",
        "category": "DataIO",
        "color": [0.58, 0.5, 0.79],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "table", "type": "struct",
             "description": "Compiled table (dict of columns)"},
        ],
        "isCompile": True,
        "isInteractive": True,  # keep output_data across runs so View Table works
        "runner": "runCompileTable",
    }
