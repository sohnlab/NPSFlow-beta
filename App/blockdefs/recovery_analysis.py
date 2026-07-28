def get_definition():
    return {
        "name": "RecoveryAnalysis",
        "displayName": "Recovery Analysis",
        "category": "Analysis",
        "color": [0.71, 0.47, 0.23],
        "inputs": [
            {"name": "refAmplitude", "type": "numeric", "required": True,
             "displayName": "\u0394ref", "preserveCase": True,
             "description": "Reference (sizing) amplitude per pulse (M,)"},
            {"name": "recAmplitude", "type": "numeric", "required": True,
             "displayName": "\u0394rec", "preserveCase": True,
             "description": "Recovery R' matrix (N_segments x M_pulses) from Recovery Info"},
            {"name": "segmentTimes", "type": "numeric", "required": True,
             "description": "Absolute segment start times (N_segments x M_pulses, ms) from Recovery Info"},
            {"name": "squeezeSegment", "type": "any", "required": False,
             "displayName": "squeeze",
             "description": "(Optional) Squeeze segment dict — if connected, recovery time is relative to squeeze end"},
        ],
        "outputs": [
            {"name": "recoveryTime", "type": "numeric",
             "description": "Time to recovery (ms) per pulse, Inf if not recovered (M,)"},
            {"name": "recoveryCategory", "type": "numeric",
             "description": "0=instant, 1..N-1=transient, N=prolonged (M,)"},
            {"name": "recoveryRate", "type": "numeric",
             "description": "Slope of linear fit to recovery dI vs time (dI/ms) (M,)"},
            {"name": "rSquared", "type": "numeric",
             "description": "R-squared of the linear fit (M,)"},
        ],
        "runner": "runRecoveryAnalysis",
        "defaultParameters": {
            "tolerance": 0.08,
        },
    }
