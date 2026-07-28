def get_definition():
    return {
        "name": "Downsample",
        "displayName": "Preprocess",
        "category": "Preprocessing",
        "color": [0.11, 0.57, 0.70],
        "isInteractive": True,
        "inputs": [
            {"name": "data", "type": "any", "required": True},
            {"name": "sampleRate", "type": "numeric", "required": False,
             "displayName": "sample rate",
             "description": "Base sample rate (Hz). Wired value wins; falls back "
                            "to the global sample rate when unconnected."},
            {"name": "settingsFile", "type": "string", "required": False,
             "displayName": "settings",
             "description": "Path to saved settings JSON, or /lastSettings"},
        ],
        "outputs": [
            {"name": "dataSmoothed", "type": "any", "displayName": "smoothed"},
            {"name": "dataDownsampled", "type": "any", "displayName": "downsampled"},
            {"name": "dataFiltered", "type": "any", "displayName": "filtered"},
            {"name": "dataDetrended", "type": "any", "displayName": "detrended"},
            {"name": "trendline", "type": "any", "displayName": "trendline"},
            {"name": "DSfactor", "type": "numeric", "displayName": "DSfactor"},
        ],
        # JOVE preprocessing chain — see processing/jove_preprocess.DEFAULT_PARAMS.
        "defaultParameters": {
            "smooth_enable": True, "smooth_width": 200, "smooth_type": 1,
            "ds_enable": True, "ds_factor": 20,
            "lp_enable": True, "lp_cutoff": 50.0, "lp_pad": True,
            "asls_enable": True, "asls_lambda": 1e9, "asls_p": 3e-3,
            "asls_margin": 1e-4, "asls_iter": 20,
        },
        "runner": "runDownsample",
    }
