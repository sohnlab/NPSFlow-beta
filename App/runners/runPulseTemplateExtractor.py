"""Runner for PulseTemplateExtractor block - calls processing function."""


def run(inputs, params, block):
    from processing.pulse_template_extractor import pulse_template_extractor

    import numpy as np

    pulse_data = inputs.get("data")
    # Global sample rate when wired; 10 kHz fallback (as the RT detection
    # blocks) — a 1 Hz default collapses the 2 s view window to 2 samples.
    sr = inputs.get("sampleRate")
    try:
        sample_rate = float(np.ravel(sr)[0]) if sr is not None else 10000.0
    except (TypeError, ValueError, IndexError):
        sample_rate = 10000.0

    if pulse_data is None:
        raise ValueError("No pulse data provided to PulseTemplateExtractor.")

    # Settings come only from the loadFile port: `/lastSettings` → the block's
    # last-saved settings (or autosave), path → that path, empty → start fresh.
    import os
    from utils.paths import project_root
    from utils.settings_input import resolve_settings_input
    from utils.lastsettings_capture import capture
    root = project_root()
    autosave_abs = os.path.join(
        root, "SavedTemplates", "PulseShape", "extractor_temp.json")
    settings_file, _ = resolve_settings_input(
        inputs.get("loadFile"), autosave_abs, block=block)

    result = pulse_template_extractor(
        pulse_data, sample_rate, settings_file=settings_file)
    # Mirror the just-written autosave into block params so /lastSettings
    # travels with the workflow JSON.
    capture(block, autosave_abs)

    # Convenience vectors for plotting the extracted template against the
    # original signal's sample index / time base.
    start = result.get("start_idx")
    end = result.get("end_idx")
    index_vector = None
    time_vector = None
    if start is not None and end is not None:
        index_vector = np.arange(start, end + 1)
        time_vector = index_vector / float(sample_rate)

    return {
        "dataSegment": result,
        "indexVector": index_vector,
        "timeVector": time_vector,
    }
