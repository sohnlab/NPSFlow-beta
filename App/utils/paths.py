"""Repo-root and SavedTemplates path helpers."""

import os


def project_root():
    """Absolute path of the repo root (the directory containing App/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def saved_templates_dir(subfolder):
    """Absolute path of SavedTemplates/<subfolder>, created if missing."""
    d = os.path.join(project_root(), "SavedTemplates", subfolder)
    os.makedirs(d, exist_ok=True)
    return d


def to_project_relative(path):
    """Project-relative form of *path* if it lies inside the repo root;
    paths outside the repo (or already relative) pass through unchanged."""
    if isinstance(path, str) and os.path.isabs(path):
        rel = os.path.relpath(path, project_root())
        if not rel.startswith(".."):
            return rel
    return path


def resolve_project_path(path):
    """Absolute form of a possibly project-relative path ('' passes through)."""
    if isinstance(path, str) and path and not os.path.isabs(path):
        return os.path.join(project_root(), path)
    return path


def map_state_paths(data, fn):
    """Apply *fn* to every persisted file path in a workflow state dict
    (Load Data ``lastPath``, ``recentLoadPaths``), recursing into
    version-history states. Mutates *data* in place."""
    for b in data.get("blocks", []):
        params = b.get("parameters") or {}
        if isinstance(params.get("lastPath"), str):
            params["lastPath"] = fn(params["lastPath"])
    if isinstance(data.get("recentLoadPaths"), list):
        data["recentLoadPaths"] = [
            fn(p) for p in data["recentLoadPaths"] if isinstance(p, str)
        ]
    for entry in data.get("history") or []:
        state = entry.get("state")
        if isinstance(state, dict):
            map_state_paths(state, fn)
