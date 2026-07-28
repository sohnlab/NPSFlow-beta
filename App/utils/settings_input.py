"""Shared resolution for the *settingsFile* / *loadFile* input ports of
interactive blocks.

The same convention applies to every block that can load saved settings:

* `/lastSettings` (exact match): use the block's autosaved settings file
  (typically `SavedTemplates/<block-folder>/temp.json`). Falls through to a
  fresh start if no autosave exists.
* Non-empty path: treat as a settings file path; resolved against the
  project root if relative.
* Empty / not a string: start fresh — unless global reuse-mode is on (the
  Load File path port is supplied), in which case empty is treated as
  `/lastSettings` so the block reuses its last-saved settings.  See
  `set_reuse_when_empty`.
"""

import json
import os

from utils.paths import project_root, saved_templates_dir


LAST_SETTINGS_SENTINEL = "/lastSettings"


# Global "reuse last settings" switch, set by the engine at the start of each
# run from the Load File block's path port (see WorkflowEngine).  When on, an
# *empty* settings port is treated as ``/lastSettings`` so every block reuses
# its last-saved settings; an explicitly-set port always wins regardless.
_REUSE_WHEN_EMPTY = False


def set_reuse_when_empty(value):
    """Set the global reuse-on-empty switch for this run."""
    global _REUSE_WHEN_EMPTY
    _REUSE_WHEN_EMPTY = bool(value)


def reuse_when_empty():
    """True if empty settings ports should reuse the last settings this run."""
    return _REUSE_WHEN_EMPTY


def load_settings_json(raw, folder, block=None, filename="temp.json"):
    """Resolve a settings port value and load its JSON dict, or ``None``.

    ``folder`` is the block's SavedTemplates subfolder (autosave location).
    Returns ``None`` on an empty port (fresh start) or unreadable file; the
    latter is logged as a warning.
    """
    autosave = os.path.join(saved_templates_dir(folder), filename)
    path, _ = resolve_settings_input(raw, autosave, block)
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        from utils.app_logger import logger
        logger.warning(f"[{folder}] failed to load settings {path}: {e}")
        return None


def resolve_settings_input(raw, autosave_path, block=None):
    """Resolve a settings-file port value to an absolute path or ``None``.

    Parameters
    ----------
    raw : Any
        Value received on the ``settingsFile`` / ``loadFile`` port.
    autosave_path : str
        Absolute path to the block's autosaved settings file. May or may
        not exist on disk.
    block : BlockNode | None
        Block instance whose ``parameters["lastSettings"]`` may hold a
        per-instance snapshot. When the ``/lastSettings`` sentinel is used
        and the block has a stored snapshot, the snapshot is written to
        ``autosave_path`` so the dialog's autoload picks it up — this is
        how per-instance settings travel with the workflow JSON.

    Returns
    -------
    (path, is_autosave) : (str | None, bool)
        ``path`` is the resolved absolute path, or ``None`` if the caller
        should start fresh. ``is_autosave`` is True only when the
        ``/lastSettings`` sentinel resolved to an existing autosave —
        useful for runners that log differently in the two cases.
    """
    s = raw.strip() if isinstance(raw, str) else ""
    if not s:
        # Empty port: fresh by default.  When global reuse-mode is on (the
        # Load File path is supplied), reuse the block's last settings.
        if not _REUSE_WHEN_EMPTY:
            return None, False
        s = LAST_SETTINGS_SENTINEL
    if s == LAST_SETTINGS_SENTINEL:
        if block is not None:
            from utils.lastsettings_capture import restore
            if restore(block, autosave_path):
                return autosave_path, True
        if autosave_path and os.path.isfile(autosave_path):
            return autosave_path, True
        return None, False
    path = s.replace("\\", "/")
    if not os.path.isabs(path):
        path = os.path.join(project_root(), path)
    return path, False
