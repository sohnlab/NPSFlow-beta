"""Mirror interactive blocks' autosaved settings into block.parameters so
they ride along with the workflow JSON.

After a successful interactive run, each runner calls ``capture(block, path)``
to copy the freshly autosaved temp.json into ``block.parameters["lastSettings"]``.
On the next run, ``restore(block, path)`` writes the per-instance copy back
into the autosave path so the dialog's autoload picks it up.

This makes ``/lastSettings`` travel with the workflow: closing and reopening
the app no longer drops the block's last working configuration.
"""

import json
import os


def capture(block, autosave_path):
    """Read the autosave file and stash it on the block. No-op if the file
    is missing or unreadable — silent because autosave is best-effort."""
    if not autosave_path or not os.path.isfile(autosave_path):
        return
    try:
        with open(autosave_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, (dict, list)):
        block.parameters["lastSettings"] = data


def restore(block, autosave_path):
    """Write the block's stashed settings to the autosave path. Returns
    True if the file was written (so the caller knows the dialog will see
    per-instance state on autoload). False if there's nothing to restore."""
    data = block.parameters.get("lastSettings")
    if not isinstance(data, (dict, list)):
        return False
    try:
        os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
        with open(autosave_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False
