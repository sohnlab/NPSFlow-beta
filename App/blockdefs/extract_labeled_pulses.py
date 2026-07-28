def get_definition():
    return {
        "name": "ExtractLabeledPulses",
        "displayName": "Extract Labeled Pulses",
        "category": "Sorting",
        "color": [0.45, 0.6, 0.48],
        "isInteractive": True,
        "showPortLabels": True,
        "inputs": [
            {
                "name": "labeledData",
                "type": "any",
                "required": True,
                "description": "Labeled signal: Load File struct ({files: [...]}), "
                "a single {data, labels} entry, a list of entries, "
                "or a bare data array (wire labels separately).",
            },
            {
                "name": "labels",
                "type": "numeric",
                "required": False,
                "description": "Per-sample label array, only needed when "
                "labeledData is a bare data array.",
            },
        ],
        "outputs": [
            {
                "name": "pulseList",
                "type": "any",
                "description": "List of dicts {fileName, data, single, coincident, "
                "noise, uncertain} — one per file. Each class field "
                "is an Nx2 [start, end] index array (inclusive ends, "
                "indices local to that file's data).",
            },
        ],
        "resetOnReload": ["displayText"],
        "runner": "runExtractLabeledPulses",
    }
