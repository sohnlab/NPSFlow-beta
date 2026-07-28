"""Runner for CombineData block — concatenates the per-file entries of a
Load File struct into single arrays, with per-file boundary indices."""

import numpy as np


def _numeric_arrays(entry):
    out = {}
    for k, v in entry.items():
        if k == "fileName":
            continue
        arr = np.asarray(v)
        if arr.dtype.kind in "iufb" and arr.ndim >= 1 and arr.size:
            out[k] = arr
    return out


def run(inputs, params, block):
    from processing.ml_training_core import extract_entries

    entries = extract_entries(inputs.get("dataIn"))
    if not entries:
        raise ValueError("No file entries found to combine.")

    per = [_numeric_arrays(e) for e in entries]
    common = set(per[0])
    for p in per[1:]:
        common &= set(p)
    if not common:
        raise ValueError("File entries share no numeric array keys.")

    if "data" in common:
        data_key = "data"
    elif len(common - {"labels"}) == 1:
        data_key = next(iter(common - {"labels"}))
    else:
        raise ValueError(
            f"Ambiguous data key; files share {sorted(common)} — expected "
            f"a 'data' key or a single common array.")

    def cat(key):
        parts = [p[key] for p in per]
        try:
            return np.concatenate(parts, axis=0)
        except ValueError as exc:
            shapes = [p[key].shape for p in per]
            raise ValueError(
                f"Cannot concatenate '{key}': shapes {shapes}") from exc

    data = cat(data_key)
    labels = cat("labels") if "labels" in common else None

    lengths = [len(p[data_key]) for p in per]
    boundaries = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    file_names = [e.get("fileName", f"file {i}")
                  for i, e in enumerate(entries)]

    block.parameters["displayText"] = \
        f"{len(entries)} file(s), {len(data):,} samples"
    return {
        "data": data,
        "labels": labels,
        "boundaries": boundaries,
        "fileNames": file_names,
    }
