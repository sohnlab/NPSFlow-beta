def get_definition():
    return {
        "name": "ExtractSinglePulseFeatures",
        "displayName": "Pulse Processing",
        "category": "FeatureExtraction",
        "color": [0.12, 0.58, 0.71],
        "inputs": [
            {
                "name": "dataIn",
                "type": "any",
                "required": True,
                "description": "Current signal (unfiltered), or Extract Labeled "
                "Pulses' pulseList (per-file {fileName, data, "
                "single, ...} entries; pulseLabels then unneeded)",
            },
            {
                "name": "trendline",
                "type": "numeric",
                "required": False,
                "description": "Baseline trend from BaselineCorrection",
            },
            {
                "name": "pulseLabels",
                "type": "numeric",
                "required": False,
                "description": "Nx2 pulse region indices (e.g. single pulses from "
                "Detection Review); required unless dataIn is a "
                "pulseList",
            },
            {
                "name": "TPsettings",
                "type": "struct",
                "required": True,
                "description": "Template settings struct (threshold, peak_counts, edge_margin, peak_sequence, filter_padding, filter_config, etc.)",
            },
            {
                "name": "pulseFeatures",
                "type": "struct",
                "required": False,
                "description": "Segment definitions from PulseFeatureExtraction (bounds, labels, methods, etc.)",
            },
            {
                "name": "settingsFile",
                "type": "string",
                "required": False,
                "description": "Path to previously saved review settings JSON (loads acceptance, peaks, regions)",
            },
        ],
        "outputs": [
            {
                "name": "pulsePeakLocations",
                "type": "any",
                "description": "Peak indices per pulse (list of arrays)",
            },
            {
                "name": "rectangularizedPulses",
                "type": "any",
                "description": "Rectangularized pulse segments (list of arrays)",
            },
            {
                "name": "acceptanceID",
                "type": "any",
                "description": "Boolean acceptance array",
            },
            {
                "name": "pulseStartIndices",
                "type": "numeric",
                "description": "Start index of each pulse segment",
            },
            {
                "name": "pulseFeatures",
                "type": "struct",
                "description": "Segment definitions with methods (passed through from input, enriched with TPsettings methods)",
            },
        ],
        "isInteractive": True,
        "runner": "runExtractSinglePulseFeatures",
    }
