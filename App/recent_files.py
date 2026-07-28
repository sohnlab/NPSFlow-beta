"""Persisted application state (recent files + arbitrary preferences).

Backed by a single JSON file (``app_state.json``) in the project root.
Reads/writes preserve unrelated keys, so multiple subsystems can share it
without clobbering each other.
"""

import json
import os

MAX_RECENT = 5
STATE_FILENAME = "app_state.json"


def _state_path(root_dir):
    return os.path.join(root_dir, STATE_FILENAME)


def _read_all(root_dir):
    path = _state_path(root_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_all(root_dir, data):
    path = _state_path(root_dir)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def load(root_dir):
    """Return the list of recent workflow file paths (absolute)."""
    items = [
        p for p in _read_all(root_dir).get("recentFiles", []) if isinstance(p, str)
    ]
    return [p if os.path.isabs(p) else os.path.join(root_dir, p) for p in items]


def _save_recent(root_dir, items):
    # Store paths inside the project relative to root_dir so the state file
    # carries no machine-specific prefixes; outside paths stay absolute.
    def _rel(p):
        rel = os.path.relpath(p, root_dir)
        return p if rel.startswith("..") else rel

    data = _read_all(root_dir)
    data["recentFiles"] = [_rel(p) for p in items]
    _write_all(root_dir, data)


def push(root_dir, filepath):
    if not filepath:
        return
    abs_path = os.path.abspath(filepath)
    items = [p for p in load(root_dir) if os.path.abspath(p) != abs_path]
    items.insert(0, abs_path)
    _save_recent(root_dir, items[:MAX_RECENT])


def clear(root_dir):
    _save_recent(root_dir, [])


def remove(root_dir, filepath):
    if not filepath:
        return
    abs_path = os.path.abspath(filepath)
    items = [p for p in load(root_dir) if os.path.abspath(p) != abs_path]
    _save_recent(root_dir, items)


def get_setting(root_dir, key, default=None):
    """Return a persisted preference value, or ``default`` if unset."""
    return _read_all(root_dir).get(key, default)


def set_setting(root_dir, key, value):
    """Persist a preference value, preserving other keys."""
    data = _read_all(root_dir)
    data[key] = value
    _write_all(root_dir, data)
