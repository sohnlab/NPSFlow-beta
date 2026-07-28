def get_definition():
    return {
        "name": "DetectionAgreement",
        "displayName": "Detection Agreement",
        "category": "Analysis",
        "color": [0.73, 0.50, 0.25],
        "inputs": [
            {"name": "dataIn", "type": "numeric", "required": True,
             "description": "Underlying 1-D signal. Multi-channel input "
                            "uses column 0."},
            {"name": "pulsesA", "type": "numeric", "required": True,
             "description": "Nx2 [start, end] indices from detector A "
                            "(e.g. Pulse Detection BC)."},
            {"name": "pulsesB", "type": "numeric", "required": True,
             "description": "Mx2 [start, end] indices from detector B "
                            "(e.g. RT Detection Combined)."},
        ],
        "outputs": [
            {"name": "score", "type": "numeric",
             "description": "Scalar in [0,1]. Formula (F1 / Jaccard / "
                            "Mean IoU) selected in the dialog."},
            {"name": "matchTable", "type": "numeric",
             "description": "Kx5 array: [side, idx, partner_idx, iou, "
                            "midpoint_offset_samples]. side 0=A, 1=B; "
                            "partner_idx -1 if unmatched."},
        ],
        "isInteractive": True,
        "runner": "runDetectionAgreement",
    }
