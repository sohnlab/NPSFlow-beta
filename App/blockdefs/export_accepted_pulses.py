def get_definition():
    return {
        "name": "ExportAcceptedPulses",
        "displayName": "Export Accepted Pulses",
        "category": "DataIO",
        "color": [0.38, 0.67, 0.95],
        "inputs": [
            {"name": "accepted_pulses", "type": "any"},
        ],
        "outputs": [],
        "runner": "runExportAcceptedPulses",
    }
