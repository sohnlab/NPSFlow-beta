def get_definition():
    return {
        "name": "PulseTemplateProcessing",
        "displayName": "Template Processing",
        "category": "Template",
        "color": [0.1, 0.56, 0.69],
        "inputs": [
            {"name": "templateIn", "type": "any",
             "description": "Pulse template: numeric array, or a struct with a template/pulse_template_rec field (extractor/detrend output)."},
            {"name": "filterConfigIn", "type": "struct", "required": False,
             "description": "Optional filter chain config (from FilterConfig/Filter UI block), applied to the template before processing."},
            {"name": "geometry", "type": "any", "required": False,
             "description": "Optional Device Geometry struct. Enables 'Match Geometry' auto-threshold by counting expected pores."},
            {"name": "loadFile", "type": "string", "required": False,
             "description": "Path to a saved TPsettings JSON file. When provided, pre-populates the interactive dialog with the saved settings."},
        ],
        "outputs": [
            {"name": "TPsettings", "type": "struct"},
        ],
        "isInteractive": True,
        "runner": "runPulseTemplateProcessing",
    }
