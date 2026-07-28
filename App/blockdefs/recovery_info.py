def get_definition():
    return {
        "name": "RecoveryInfo",
        "displayName": "Recovery Info",
        "category": "Analysis",
        "color": [0.69, 0.45, 0.21],
        "inputs": [
            {"name": "addInput", "type": "any", "required": False,
             "displayName": "Add input"},
        ],
        "outputs": [
            {"name": "Rprime", "type": "numeric",
             "displayName": "R'",
             "description": "R' matrix (N_segments x M_pulses): midpoint of each segment per pulse"},
            {"name": "segmentTimes", "type": "numeric",
             "displayName": "segmentTimes",
             "description": "Time of each segment start relative to sample 0 (N_segments x M_pulses, ms)"},
        ],
        "isDynamic": True,
        "runner": "runRecoveryInfo",
    }
