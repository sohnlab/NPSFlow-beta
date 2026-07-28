"""Save and load TPsettings structs as JSON files.

Numpy arrays are stored as lists; tuples (e.g. exclusion_zones) are
restored from the nested list representation.
"""

import json
import numpy as np


def _encode(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(),
                "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_encode(v) for v in obj]
        if isinstance(obj, tuple):
            return {"__tuple__": True, "data": converted}
        return converted
    return obj


def _decode(obj):
    """Recursively restore numpy arrays and tuples from JSON dicts."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
        if obj.get("__tuple__"):
            return tuple(_decode(v) for v in obj["data"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


# Fields kept once at the top level rather than per zone.
_SHARED_KEYS = ("filter_config", "filter_padding")


def normalize_tpsettings(tp):
    """Return *tp* in the canonical zone-grouped shape.

    Canonical shape::

        {num_zones, filter_config, filter_padding, zones: [<per-zone dict>, ...]}

    A legacy flat dict (threshold/template_original at the top level, no
    ``zones`` key) is wrapped as a single zone so old saved settings and
    old workflows keep loading.
    """
    if not isinstance(tp, dict):
        return {"num_zones": 0, "filter_config": {}, "filter_padding": {},
                "zones": []}
    if isinstance(tp.get("zones"), list):
        return tp
    flat = dict(tp)
    shared = {k: flat.pop(k, {}) for k in _SHARED_KEYS}
    return {"num_zones": 1, **shared, "zones": [flat]}


def get_zone(tp, k=0):
    """Return zone *k* as a flat dict matching the legacy single-zone shape.

    The per-zone fields are merged with the top-level shared fields
    (``filter_config``/``filter_padding``), so downstream consumers that
    expect the old flat TPsettings struct work unchanged. Falls back to
    zone 0 when *k* is out of range.
    """
    norm = normalize_tpsettings(tp)
    zones = norm["zones"]
    if not zones:
        return {}
    zone = dict(zones[k] if 0 <= k < len(zones) else zones[0])
    for key in _SHARED_KEYS:
        zone.setdefault(key, norm.get(key, {}))
    return zone


def save_tpsettings(tp_settings, filepath):
    """Save a TPsettings dict to *filepath* (JSON)."""
    with open(filepath, "w") as f:
        json.dump(_encode(tp_settings), f, indent=2)


def load_tpsettings(filepath):
    """Load a TPsettings dict from *filepath* (JSON)."""
    with open(filepath, "r") as f:
        return _decode(json.load(f))
