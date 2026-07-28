"""Runner for PulseProcessing block - calls PulseTemplateProcessing.

Wraps all outputs into a single struct matching the MATLAB version:
  pulseTemplate_rec, threshold, EdgeMargin, peakCounts,
  peakLocations, peakSequence, filterPadding, filterConfig
"""


def run(inputs, params, block):
    from processing.pulse_template_processing import pulse_template_processing

    load_file = inputs.get("loadFile")
    loaded_settings = None   # full zone-grouped struct (restores EVERY zone)
    loaded_zone0 = None      # zone 0 flat view, for template/filter defaults

    import os
    from utils.paths import project_root
    from utils.tpsettings_io import load_tpsettings, get_zone
    from utils.app_logger import logger

    root = project_root()

    # Resolution via the shared helper: explicit path → that path; /lastSettings
    # (or an empty port under reuse-mode — the Load File "Most Recent" shortcut)
    # → autosave; empty otherwise → fresh.  ``auto_load`` is True only for the
    # autosave/reuse case, which keeps the filter chain from being silently
    # restored (see below).
    from utils.settings_input import resolve_settings_input
    autosave = os.path.join(root, "SavedTemplates", "PulseShape", "temp.json")
    path, auto_load = resolve_settings_input(load_file, autosave, block=block)

    if path and os.path.isfile(path):
        if not auto_load:
            logger.info(f"Loading TPsettings from: {path}")
        try:
            # Keep the FULL zone-grouped struct so every zone's settings are
            # restored (pulse_template_processing applies get_zone(.., ch) per
            # channel). Zone 0's flat view drives the template/filter defaults.
            loaded_settings = load_tpsettings(path)
            loaded_zone0 = get_zone(loaded_settings, 0)
        except Exception as e:
            logger.warning(f"Failed to load TPsettings: {e}")

    pulse_template = inputs.get("templateIn")
    sample_rate = inputs.get("sampleRate", 1)
    filter_config_in = inputs.get("filterConfigIn")

    # When loading from file, use the saved template and filter config
    # as defaults if no live inputs are connected
    if loaded_zone0:
        if pulse_template is None:
            pulse_template = loaded_zone0.get("template_original")
        # Only restore the filter chain from explicit loadFile inputs.
        # Autosave shouldn't silently bypass the interactive Filter UI when
        # the filterConfigIn port is left disconnected.
        if filter_config_in is None and not auto_load:
            # Convert stored filter_config to filter_config_in list format
            fc = loaded_zone0.get("filter_config", {})
            if fc and fc.get("filter_types"):
                filter_config_in = []
                ftypes = fc["filter_types"]
                cutoffs = fc.get("cutoff_frequencies", [])
                for i, ft in enumerate(ftypes):
                    entry = {"type": ft}
                    if i < len(cutoffs):
                        entry["cutoff1"] = cutoffs[i][0] if len(cutoffs[i]) > 0 else 0
                        entry["cutoff2"] = cutoffs[i][1] if len(cutoffs[i]) > 1 else 0
                    filter_config_in.append(entry)

    if pulse_template is None:
        raise ValueError("No pulse template provided to PulseProcessing.")

    geometry = inputs.get("geometry")

    result = pulse_template_processing(
        pulse_template, sample_rate,
        exclude_end_zones=False,
        end_zone_percent=5,
        filter_config_in=filter_config_in,
        loaded_settings=loaded_settings,
        geometry=geometry)

    # Auto-save to SavedTemplates/PulseShape/temp.json AND mirror into block
    # params so the settings travel with the workflow JSON.
    try:
        import os
        from utils.paths import project_root
        from utils.tpsettings_io import save_tpsettings
        root = project_root()
        folder = os.path.join(root, "SavedTemplates", "PulseShape")
        os.makedirs(folder, exist_ok=True)
        autosave_path = os.path.join(folder, "temp.json")
        save_tpsettings(result, autosave_path)
        from utils.lastsettings_capture import capture
        capture(block, autosave_path)
    except Exception:
        pass

    return {"TPsettings": result}
