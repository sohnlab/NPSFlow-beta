"""Per-workflow "recent files" list for the Load Data block.

The list ties to the workflow file: it is serialized into the workflow JSON
(``recentLoadPaths``), restored on load, and reset on new/open so it never
spans workflows. All Load Data blocks in a workflow share the one list.

The Load Data runner has no handle on the app, so the app keeps this module
bound to the *active* workflow canvas's list (``bind`` on tab switch / open /
new). The runner reads it via ``get_recent`` and promotes a freshly-loaded
path with ``add_recent``; both operate on the bound list in place, so the
canvas keeps the change and saves it with the workflow.
"""

MAX_RECENT = 10

# Survive hot reload: run_all reloads utils.* before every run, which re-executes
# this module. importlib.reload keeps the module dict, so guard the init to
# preserve the active workflow's bound list across reloads (otherwise the
# binding is orphaned and the recent list silently empties on every run).
try:
    _current
except NameError:
    _current = []


def bind(paths_list):
    """Point the bridge at the active workflow's list (the app owns it)."""
    global _current
    _current = paths_list


def get_recent():
    """Return a copy of the active workflow's recent paths (most-recent first)."""
    return list(_current)


def add_recent(path):
    """Promote a freshly-loaded path to the front of the active list."""
    if not isinstance(path, str) or not path:
        return
    lst = _current
    if path in lst:
        lst.remove(path)
    lst.insert(0, path)
    del lst[MAX_RECENT:]
