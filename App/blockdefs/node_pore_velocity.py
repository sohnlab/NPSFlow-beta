def get_definition():
    return {
        "name": "NodePoreVelocity",
        "displayName": "Velocity",
        "category": "Analysis",
        "color": [0.8, 0.56, 0.32],
        "inputs": [
            {"name": "Time", "type": "numeric", "required": True,
             "description": "Transit time matrix (N_pulses x N_segments)"},
            {"name": "ChannelLength", "type": "numeric", "required": True,
             "description": "Channel lengths array"},
        ],
        "outputs": [
            {"name": "velocity", "type": "numeric",
             "description": "Velocity per segment (selected output unit)"},
        ],
        "runner": "runNodePoreVelocity",
        "defaultParameters": {
            # Inputs are normalised to (mm, s) using the units below, so the
            # formula yields mm/s; the result is then scaled to outputUnit.
            # The formula sees ChannelLength in mm and Time in s.
            "formula": "ChannelLength / Time",
            "timeUnit": "ms",
            "lengthUnit": "µm",
            "outputUnit": "mm/s",
        },
        "parameterDefinitions": [
            {"name": "timeUnit", "displayName": "Time input unit",
             "type": "choice", "options": ["ms", "s"], "group": "Input Units"},
            {"name": "lengthUnit", "displayName": "Length input unit",
             "type": "choice", "options": ["µm", "mm", "m"],
             "group": "Input Units"},
            {"name": "outputUnit", "displayName": "Velocity",
             "type": "choice",
             "options": ["m/s", "mm/s", "µm/s", "m/ms", "mm/ms", "µm/ms"],
             "group": "Output Unit"},
        ],
    }
