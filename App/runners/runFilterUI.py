"""Runner for FilterUI block — zero-phase filtering via
processing.filter_ui.

Accepts a bare array (legacy behavior: per-zone interactive dialogs) or a
Load Data multi-file struct ({fileNames, files:[{data, ...}]}). Struct
input opens the multi-file preview dialog (one chain applied to all files
and zones) and the output preserves the structure: only each entry's data
array is replaced, all other keys (labels, fileName, ...) pass through
untouched.

When the filterConfigIn port is connected, filters are applied
programmatically without opening the interactive UI.
"""


def _is_port_connected(block, port_name):
    """Check if a specific input port has a wire connected."""
    for port in block.input_ports:
        if port.name == port_name:
            return port.is_connected
    return False


def run(inputs, params, block):
    import numpy as np
    from processing.filter_ui import apply_filter_chain, filter_ui
    from utils.app_logger import logger

    data = inputs.get("data")
    if data is None:
        raise ValueError("No data provided to FilterUI.")

    chain = None
    if _is_port_connected(block, "filterConfigIn"):
        chain = inputs.get("filterConfigIn")
        if not isinstance(chain, list):
            chain = []

    # ── Multi-file struct path ──
    if isinstance(data, dict):
        from processing.causal_filter_ui import multifile_filter_ui
        from runners.runCausalFilterUI import resolve_fs
        from utils.multifile import (
            collect_previews,
            map_entry_data,
            normalize_entries,
            rebuild_output,
        )

        fs = resolve_fs(inputs, logger, "FilterUI")
        entries, single = normalize_entries(data)
        previews = collect_previews(entries)
        if not previews:
            raise ValueError("No numeric data arrays found in FilterUI input.")
        if chain is None:
            chain = multifile_filter_ui(previews, fs, causal=False)

        if chain:
            new_entries, n = map_entry_data(
                entries, lambda a: apply_filter_chain(a, fs, chain)
            )
            filtered = rebuild_output(data, new_entries, single)
            descs = ", ".join(fi.get("description", fi["type"]) for fi in chain)
            logger.info(
                f"[FilterUI] Applied {len(chain)} filter(s) to "
                f"{n} file(s) at {fs:g} Hz: {descs}"
            )
        else:
            filtered = rebuild_output(data, entries, single)
            logger.info(
                "[FilterUI] No filters in config — data passed through unchanged."
            )
        return {"filteredData": filtered, "filterConfig": list(chain)}

    # ── Legacy bare-array path ──
    sample_rate = inputs.get("sampleRate", 1)

    if chain is not None:
        filtered_data = apply_filter_chain(data, sample_rate, chain)
        if chain:
            descs = ", ".join(fi.get("description", fi["type"]) for fi in chain)
            logger.info(f"Applied {len(chain)} filter(s) from config: {descs}")
        else:
            logger.info("No filters in config — no filtering applied.")
        return {"filteredData": filtered_data, "filterConfig": list(chain)}

    # No config connected - open interactive UI (per-zone dialogs)
    try:
        ds_factor = float(np.ravel(inputs.get("downsampleFactor"))[0])
        if ds_factor <= 0:
            ds_factor = None
    except (TypeError, ValueError, IndexError):
        ds_factor = None

    filtered, config = filter_ui(data, sample_rate, ds_factor=ds_factor)
    return {"filteredData": filtered, "filterConfig": config}
