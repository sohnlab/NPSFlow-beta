"""Runner for CausalFilterUI — designs a causal filter chain on one
preview file and applies it to every file in a Load Data struct
({fileNames, files:[{data, ...}]}), preserving the structure: only each
entry's data array is replaced, all other keys (labels, fileName, ...)
pass through untouched. Bare arrays are filtered directly.

When the filterConfigIn port is connected, the chain is applied
programmatically without opening the UI (entries without an 'order' key
default to order 2).
"""

import numpy as np

from utils.multifile import (
    collect_previews,
    map_entry_data,
    normalize_entries,
    rebuild_output,
)


def _is_port_connected(block, port_name):
    for port in block.input_ports:
        if port.name == port_name:
            return port.is_connected
    return False


def resolve_fs(inputs, logger, tag):
    """Sample rate from the wired/global sampleRate input, else the 10 kHz
    labeled-data convention."""
    sr = inputs.get("sampleRate")
    try:
        fs = float(np.ravel(sr)[0]) if sr is not None else 0.0
    except (TypeError, ValueError, IndexError):
        fs = 0.0
    if fs <= 0:
        fs = 10000.0
        logger.info(f"[{tag}] No sample rate wired — assuming 10 kHz.")
    return fs


def run(inputs, params, block):
    from processing.causal_filter_ui import apply_causal_chain, causal_filter_ui
    from utils.app_logger import logger

    data_in = inputs.get("data")
    if data_in is None:
        raise ValueError("No data provided to Causal Filter.")

    fs = resolve_fs(inputs, logger, "CausalFilter")

    entries, single = normalize_entries(data_in)
    previews = collect_previews(entries)
    if not previews:
        raise ValueError("No numeric data arrays found in Causal Filter input.")

    # Chain source: wired config (headless) or the interactive UI.
    if _is_port_connected(block, "filterConfigIn"):
        chain = inputs.get("filterConfigIn")
        if not isinstance(chain, list):
            chain = []
    else:
        chain = causal_filter_ui(previews, fs)

    if chain:
        new_entries, n = map_entry_data(
            entries, lambda a: apply_causal_chain(a, fs, chain)
        )
        filtered = rebuild_output(data_in, new_entries, single)
        descs = ", ".join(fi.get("description", fi["type"]) for fi in chain)
        logger.info(
            f"[CausalFilter] Applied {len(chain)} causal filter(s) "
            f"to {n} file(s) at {fs:g} Hz: {descs}"
        )
    else:
        filtered = rebuild_output(data_in, entries, single)
        logger.info(
            "[CausalFilter] No filters in config — data passed through unchanged."
        )

    return {"filtered": filtered, "filterConfig": list(chain)}
