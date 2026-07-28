def get_definition():
    return {
        "name": "SegmentWidthToTime",
        "displayName": "Width → Time",
        "category": "Analysis",
        "color": [0.8, 0.56, 0.32],
        "inputs": [
            {"name": "widthMatrix", "type": "numeric", "required": True,
             "description": "Width matrix (N_pulses x N_segments) in samples"},
        ],
        "outputs": [
            {"name": "timeMatrix", "type": "numeric",
             "description": "Transit time per segment (N_pulses x N_segments)"},
        ],
        "runner": "runSegmentWidthToTime",
        "defaultParameters": {
            "units": "ms",
        },
        "parameterDefinitions": [
            {"name": "units", "displayName": "Output unit",
             "type": "choice", "options": ["ms", "s"]},
        ],
    }
