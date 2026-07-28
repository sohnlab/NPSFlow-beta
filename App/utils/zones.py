"""Zone-compatibility helpers for multi-column (mzNPS) data.

Convention: one zone behaves exactly as the legacy single-signal case.
``wrap_zones([x]) is x`` and ``zone_list(x) == [x]`` are inverses for a
single zone, so producers emit the bare value for 1 zone and a
``{num_zones, zones:[...]}`` struct only for >1. Pure functions only — safe
to hot-reload (no module-level state).
"""
import numpy as np


def is_multizone(arr):
    """True if *arr* is a 2-D array with more than one column."""
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] > 1


def zone_columns(arr):
    """Return a list of 1-D zone columns. 1-D or single-column → one zone."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] > 1:
        return [a[:, i] for i in range(a.shape[1])]
    return [a.ravel()]


def zone_list(obj):
    """Per-zone payloads. Wrapped ``{zones:[...]}`` → its list; else ``[obj]``."""
    if isinstance(obj, dict) and isinstance(obj.get("zones"), list):
        return obj["zones"]
    return [obj]


def n_zones(obj):
    """1 for a bare/legacy value; ``len(zones)`` for a wrapped struct."""
    if isinstance(obj, dict) and isinstance(obj.get("zones"), list):
        return len(obj["zones"])
    return 1


def wrap_zones(items):
    """1 item → return it bare (backward compatible); >1 → wrapped struct."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return {"num_zones": len(items), "zones": items}
