"""Helpers for the Load Data multi-file struct ({fileNames, files:[...]}).

Used by filter runners to normalize flexible data inputs (struct, single
entry dict, or bare array) and to map a function over each entry's data
array while passing every other key (labels, fileName, ...) through
untouched.
"""

import numpy as np

DATA_KEYS = ("data", "labeledData.data")


def entry_data_key(entry):
    if isinstance(entry, dict):
        for k in DATA_KEYS:
            if k in entry:
                return k
    return None


def is_filterable(arr):
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim in (1, 2)
        and np.issubdtype(arr.dtype, np.number)
    )


def normalize_entries(data_in):
    """Normalize a flexible data input into (entries, single).

    Accepts a Load Data struct ({fileNames, files}), a single entry dict,
    or a bare array. `single` is True when the output should collapse back
    to the input's non-list shape (see rebuild_output).
    """
    if isinstance(data_in, dict) and "files" in data_in:
        return list(data_in["files"]), False
    if entry_data_key(data_in):
        return [data_in], True
    return [{"data": np.asarray(data_in), "fileName": "input"}], True


def collect_previews(entries):
    """[(name, ndarray)] of the filterable entries, for preview UIs."""
    previews = []
    for i, entry in enumerate(entries):
        key = entry_data_key(entry)
        if key is None:
            continue
        arr = np.asarray(entry[key])
        if is_filterable(arr):
            previews.append((str(entry.get("fileName") or f"file {i + 1}"), arr))
    return previews


def map_entry_data(entries, fn):
    """Apply fn to each filterable entry's data array; other entries pass
    through untouched. Returns (new_entries, n_mapped)."""
    out, n = [], 0
    for entry in entries:
        key = entry_data_key(entry)
        arr = np.asarray(entry[key]) if key else None
        if arr is not None and is_filterable(arr):
            e = dict(entry)
            e[key] = fn(arr)
            out.append(e)
            n += 1
        else:
            out.append(entry)
    return out, n


def rebuild_output(data_in, new_entries, single):
    """Reassemble mapped entries in the input's shape: bare array in, bare
    array out; entry dict in, entry dict out; struct in, struct out."""
    if single:
        entry = new_entries[0]
        return entry if isinstance(data_in, dict) else entry["data"]
    out = dict(data_in)
    out["files"] = new_entries
    return out
