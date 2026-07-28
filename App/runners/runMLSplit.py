"""Runner for MLSplit block — file-level train/val/test split of labeled
recordings (no leakage: windows from one file never cross splits)."""


def run(inputs, params, block):
    from processing.ml_training_core import (
        auto_split, canonicalize_labels, entry_data_labels, event_counts,
        extract_entries, segment_entries,
    )

    entries = extract_entries(inputs.get("dataIn"))
    if not entries:
        raise ValueError("ML Split: no file entries on dataIn.")

    by_event = str(params.get("splitMode", "by file")) == "by event"
    if by_event:
        entries = segment_entries(entries)

    fractions = {
        "train": float(params.get("trainFrac", 0.6)),
        "val": float(params.get("valFrac", 0.2)),
        "test": float(params.get("testFrac", 0.2)),
    }
    n_splits = sum(1 for v in fractions.values() if v > 0)
    if len(entries) < n_splits:
        raise ValueError(
            f"ML Split: {len(entries)} file(s) cannot fill {n_splits} "
            f"nonzero splits — label more recordings or zero a fraction.")

    by_name, counts = {}, {}
    for k, e in enumerate(entries):
        name = str(e.get("fileName") or f"file{k}")
        by_name[name] = e
        _, labels = entry_data_labels(e)
        counts[name] = event_counts(canonicalize_labels(labels)) \
            if labels is not None else \
            {"single": 0, "coincident": 0, "uncertain": 0}

    split = auto_split(counts, fractions, int(params.get("seed", 0)))

    def subset(kind):
        names = sorted(n for n, a in split.items() if a == kind)
        return {"fileNames": names, "files": [by_name[n] for n in names]}

    train, val, test = subset("train"), subset("val"), subset("test")
    unit = "segments" if by_event else "files"
    block.parameters["displayText"] = (
        f"{len(train['files'])} / {len(val['files'])} / "
        f"{len(test['files'])} {unit}")
    return {
        "trainSet": train,
        "valSet": val,
        "testSet": test,
        "splitInfo": {"split": split, "counts": counts},
    }
