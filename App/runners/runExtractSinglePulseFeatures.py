"""Runner for ExtractSinglePulseFeatures (Batch Processing) block.

Unpacks TPsettings struct into individual parameters, then calls the
processing function for per-pulse filtering, peak detection,
rectangularization, sequence validation, and interactive review.

``dataIn`` accepts either a plain signal array (with ``pulseLabels`` wired
separately) or Extract Labeled Pulses' ``pulseList`` — a list of
{fileName, data, single, ...} entries. Entries are concatenated in order,
their per-file ``single`` regions offset into the combined signal, and the
file names/boundaries drive the review dialog's per-file LP Filter group.
"""

import numpy as np


def _pulse_entries(data_in):
    """List of per-file dicts from a pulseList input, else None."""
    if isinstance(data_in, dict) and isinstance(data_in.get("files"), (list, tuple)):
        data_in = data_in["files"]
    if (
        isinstance(data_in, (list, tuple))
        and len(data_in)
        and all(isinstance(e, dict) and "data" in e for e in data_in)
    ):
        return list(data_in)
    return None


def run(inputs, params, block):
    from processing.extract_single_pulse_features import (
        extract_single_pulse_features,
    )

    data_in = inputs.get("dataIn")
    if data_in is None:
        raise ValueError("No dataIn provided to ExtractSinglePulseFeatures.")

    entries = _pulse_entries(data_in)
    file_names = file_boundaries = None
    if entries is not None:
        if not any("single" in e for e in entries):
            raise ValueError(
                "dataIn entries carry no 'single' regions; wire Extract "
                "Labeled Pulses' pulseList output (or a plain signal plus "
                "pulseLabels)."
            )
        datas, file_names, regions = [], [], []
        offset = 0
        for i, e in enumerate(entries):
            d = np.asarray(e["data"])
            r = np.asarray(
                e.get("single") if e.get("single") is not None else np.empty((0, 2)),
                dtype=int,
            ).reshape(-1, 2)
            datas.append(d)
            file_names.append(str(e.get("fileName") or f"file {i + 1}"))
            regions.append(r + offset)
            offset += len(d)
        try:
            data_in = np.concatenate(datas, axis=0) if len(datas) > 1 else datas[0]
        except ValueError as exc:
            raise ValueError(
                f"Cannot concatenate data across files: shapes "
                f"{[d.shape for d in datas]}"
            ) from exc
        pulse_indices = np.vstack(regions)
        file_boundaries = np.concatenate(
            [[0], np.cumsum([len(d) for d in datas])]
        ).astype(np.int64)
    else:
        pulse_indices = inputs.get("pulseLabels")
        if pulse_indices is None:
            raise ValueError("No pulseLabels provided to ExtractSinglePulseFeatures.")

    tp = inputs.get("TPsettings")
    if tp is None or not isinstance(tp, dict):
        raise ValueError("No TPsettings provided to ExtractSinglePulseFeatures.")
    # TPsettings is zone-grouped; the entry now processes every zone and
    # unpacks the per-zone fields itself. The methods enrichment below stays
    # zone-0 (segment methods are shared).
    from utils.tpsettings_io import get_zone

    sample_rate = inputs.get("sampleRate", 1)
    trendline = inputs.get("trendline")
    pulse_features = inputs.get("pulseFeatures")

    # `/lastSettings` → block's per-instance settings (or autosave);
    # path → that path; empty → start fresh.
    import os
    from utils.paths import project_root
    from utils.settings_input import resolve_settings_input
    from utils.lastsettings_capture import capture

    root = project_root()
    autosave_abs = os.path.join(root, "SavedTemplates", "Processing", "temp.json")
    settings_file, _ = resolve_settings_input(
        inputs.get("settingsFile"), autosave_abs, block=block
    )

    # If the resolved file carries prior review work, ask continue vs. fresh.
    from utils.session_prompt import review_state_summary, confirm_continue_previous

    has_work, detail = review_state_summary(settings_file)
    if has_work and not confirm_continue_previous(
        detail,
        title="Single Pulse Feature Review",
        allow_remember=getattr(block, "_auto_run", False),
    ):
        settings_file = None

    # Merge methods from TPsettings (zone 0) into pulseFeatures if available
    methods = get_zone(tp, 0).get("methods")
    if pulse_features is not None and isinstance(pulse_features, dict):
        if methods is not None and "methods" not in pulse_features:
            pulse_features = dict(pulse_features)
            pulse_features["methods"] = methods
    elif methods is not None:
        # No pulseFeatures input - create a minimal one with methods
        pulse_features = {"methods": methods}

    try:
        ds_factor = float(np.ravel(inputs.get("downsampleFactor"))[0])
        if ds_factor <= 0:
            ds_factor = None
    except (TypeError, ValueError, IndexError):
        ds_factor = None

    result = extract_single_pulse_features(
        data_in=data_in,
        tp=tp,
        sample_rate=sample_rate,
        pulse_indices=pulse_indices,
        trendline=trendline,
        pulse_features=pulse_features,
        settings_file=settings_file,
        ds_factor=ds_factor,
        file_names=file_names,
        file_boundaries=file_boundaries,
    )

    # Mirror the just-written autosave into block params so settings
    # travel with the workflow JSON.
    capture(block, autosave_abs)

    return {
        "pulsePeakLocations": result["pulse_peak_locations"],
        "rectangularizedPulses": result["rectangularized_pulses"],
        "acceptanceID": result["acceptance_id"],
        "pulseStartIndices": result["pulse_start_indices"],
        "pulseFeatures": pulse_features,
    }
