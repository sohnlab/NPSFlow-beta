def get_definition():
    return {
        "name": "PulseTemplateExtractor",
        "displayName": "Template Extractor",
        "category": "Template",
        "color": [0.09, 0.55, 0.68],
        "inputs": [
            {"name": "data", "type": "any",
             "description": "Signal array, or a Load File struct "
                            "({fileNames, files}) — multiple files get a "
                            "File selector in the dialog."},
            {"name": "loadFile", "type": "string", "required": False,
             "description": "Path to saved extractor settings JSON; "
                            "/lastSettings reuses the last run, empty starts fresh"},
        ],
        "outputs": [
            {"name": "dataSegment", "type": "struct"},
            {"name": "indexVector", "type": "numeric",
             "displayName": "index",
             "description": "Sample indices of the extracted template"},
            {"name": "timeVector", "type": "numeric",
             "displayName": "time",
             "description": "Time (s) of the extracted template = index / sampleRate"},
        ],
        "isInteractive": True,
        "runner": "runPulseTemplateExtractor",
    }
