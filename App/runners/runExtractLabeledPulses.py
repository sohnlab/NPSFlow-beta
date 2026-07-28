"""Runner for ExtractLabeledPulses block.

Turns a per-sample labeled signal into Detection-Review-style Nx2 pulse
regions grouped by class id (1=single, 2=coincident, 3=noise,
4=uncertain), kept separate per file: the output is a list of
{fileName, data, single, coincident, noise, uncertain} entries with
indices local to each file. Regions are contiguous runs of the same
label; end indices are inclusive, matching Pulse Processing's
``data[s:e+1]`` slicing. Opens a paginated 3x3 grid viewer of the
extracted pulses, filterable by file and class.
"""

import numpy as np

from utils.app_logger import logger
from processing.ml_training_core import entry_data_labels


CLASS_PORTS = ((1, "single"), (2, "coincident"), (3, "noise"), (4, "uncertain"))


def _entry_triple(entry, i):
    """(name, data, labels) from a file entry, tolerating dot-prefixes."""
    name = str(entry.get("fileName") or f"file {i + 1}")
    data, labels = entry_data_labels(entry)
    return name, data, labels


def _resolve_files(labeled, labels_in):
    """Normalize the flexible input into a list of (name, data, labels)."""
    if isinstance(labeled, dict) and "files" in labeled:
        labeled = labeled["files"]
    if isinstance(labeled, dict):
        labeled = [labeled]
    if isinstance(labeled, (list, tuple)):
        files = []
        for i, entry in enumerate(labeled):
            if not isinstance(entry, dict):
                continue
            name, data, labels = _entry_triple(entry, i)
            if data is None or labels is None:
                continue
            files.append((name, np.asarray(data), np.asarray(labels).ravel()))
        return files
    if labels_in is None:
        raise ValueError(
            "labeledData is a bare array; wire the labels input as well "
            "(or connect a {data, labels} struct)."
        )
    return [("input", np.asarray(labeled), np.asarray(labels_in).ravel())]


def _label_runs(labels):
    """{class id: [(start, end inclusive)]} of contiguous label runs."""
    by_id = {cid: [] for cid, _ in CLASS_PORTS}
    if len(labels):
        change = np.nonzero(np.diff(labels))[0]
        starts = np.concatenate(([0], change + 1))
        ends = np.concatenate((change, [len(labels) - 1]))
        for s, e in zip(starts, ends):
            runs = by_id.get(int(labels[s]))
            if runs is not None:
                runs.append((int(s), int(e)))
    return by_id


def run(inputs, params, block):
    files = _resolve_files(inputs.get("labeledData"), inputs.get("labels"))
    if not files:
        raise ValueError("No {data, labels} entries found in labeledData.")
    for name, data, labels in files:
        if len(labels) != len(data):
            raise ValueError(
                f"data/labels length mismatch in {name or 'input'}: "
                f"{len(data)} vs {len(labels)}"
            )

    entries = []
    regions = []  # (start, end, class id) in concatenated space, viewer only
    totals = {cid: 0 for cid, _ in CLASS_PORTS}
    offset = 0
    for name, data, labels in files:
        by_id = _label_runs(labels.astype(np.int64))
        entry = {"fileName": name, "data": data}
        for cid, cname in CLASS_PORTS:
            entry[cname] = np.asarray(by_id[cid], dtype=np.int64).reshape(-1, 2)
            totals[cid] += len(by_id[cid])
            regions.extend((s + offset, e + offset, cid) for s, e in by_id[cid])
        entries.append(entry)
        offset += len(data)

    block.parameters.pop("displayText", None)
    counts = ", ".join(f"{totals[cid]} {cname}" for cid, cname in CLASS_PORTS)
    logger.info(f"[ExtractLabeledPulses] {len(files)} file(s): {counts}")

    boundaries = np.concatenate([[0], np.cumsum([len(f[1]) for f in files])]).astype(
        np.int64
    )
    if len(files) > 1:
        try:
            view_data = np.concatenate([f[1] for f in files], axis=0)
        except ValueError as exc:
            shapes = [f[1].shape for f in files]
            raise ValueError(
                f"Cannot concatenate data across files: shapes {shapes}"
            ) from exc
    else:
        view_data = files[0][1]

    from processing.extract_labeled_pulses import extracted_pulses_view

    extracted_pulses_view(
        view_data, sorted(regions), boundaries, [f[0] for f in files], sample_rate=1
    )
    return {"pulseList": entries}
