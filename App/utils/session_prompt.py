"""Themed 'continue previous session vs. start fresh' prompt.

Interactive review blocks (Detection Review, Single/Coincident Feature
Review, Wavelet Editor) can silently restore a prior session's classifications
and manual edits. When they do, this prompt lets the user choose to continue
that work or start fresh. Uses the app-themed QDialog (not the native
QMessageBox) so it matches the rest of the UI.
"""

import json
import os


# Per-run memory of the user's continue/restart choice. Module-level so it
# persists across blocks within a single "Run All" (utils modules reload once
# before each run), then resets on the next run. Only consulted during an
# auto (multi-block) run and only when the user ticked "Remember my settings".
_REMEMBERED = {"choice": None}


def load_json(path):
    """Load a settings JSON file, returning the dict or None on any failure."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _acceptance_detail(data):
    """A 'N of M pulses accepted' string from a saved review state, or None."""
    accs = []
    if isinstance(data, dict):
        zones = data.get("zones")
        if isinstance(zones, list) and zones:
            for z in zones:
                a = z.get("acceptance") if isinstance(z, dict) else None
                if isinstance(a, list):
                    accs.extend(a)
        elif isinstance(data.get("acceptance"), list):
            accs.extend(data["acceptance"])
    if not accs:
        return None
    n_acc = sum(1 for v in accs if v == 1)
    return f"{n_acc} of {len(accs)} pulses currently accepted."


def review_state_summary(path):
    """Inspect a saved single/coincident review state.

    Returns ``(has_work, detail)`` — ``has_work`` True when the file holds
    accept/reject choices or manual edits worth prompting about, and
    ``detail`` a short human string (or None).
    """
    data = load_json(path)
    if not data:
        return False, None
    detail = _acceptance_detail(data)
    done = data.get("completed_events")
    if isinstance(done, list) and done:
        extra = f"{len(done)} event(s) marked complete."
        detail = f"{detail} {extra}" if detail else extra
    edit_keys = ("peak_locs", "peak_roles", "edges", "levels", "tpl_edit",
                 "pulse_excl_ranges", "pulse_key_peaks")
    has_edits = any(data.get(k) for k in edit_keys)
    if not has_edits and isinstance(data.get("zones"), list):
        has_edits = any(isinstance(z, dict) and z.get("peak_locs")
                        for z in data["zones"])
    return (detail is not None or has_edits), detail


def confirm_continue_previous(detail=None, title="Previous Session Found",
                              allow_remember=False):
    """Ask whether to continue a previous review session or restart.

    Returns True to continue (keep loaded state), False to restart fresh.
    Falls back to True when no GUI is available (batch/headless runs).

    When ``allow_remember`` is True (an auto, multi-block run) the dialog
    shows a "Remember my settings" checkbox; if ticked, the choice is stored
    and reused for every later block in the same run without prompting again.
    An individual block run (``allow_remember`` False) always prompts and
    never consults the remembered choice.
    """
    if allow_remember and _REMEMBERED["choice"] is not None:
        return _REMEMBERED["choice"]

    try:
        from PySide6.QtWidgets import (
            QApplication, QDialog, QLabel, QPushButton, QCheckBox,
            QVBoxLayout, QHBoxLayout,
        )
    except Exception:
        return True
    if QApplication.instance() is None:
        return True

    from utils.dialog_style import apply_dialog_style

    dlg = QDialog()
    apply_dialog_style(dlg)
    dlg.setWindowTitle(title)
    dlg.setModal(True)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(24, 20, 24, 18)
    layout.setSpacing(12)

    heading = QLabel("A previous session was found.")
    hf = heading.font()
    hf.setPointSize(hf.pointSize() + 1)
    heading.setFont(hf)
    layout.addWidget(heading)

    lines = []
    if detail:
        lines.append(detail)
    lines.append("Continue from where you left off, or restart?")
    body = QLabel("\n\n".join(lines))
    body.setWordWrap(True)
    layout.addWidget(body)

    remember_chk = None
    if allow_remember:
        remember_chk = QCheckBox("Remember my settings for the rest of this run")
        layout.addWidget(remember_chk)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    restart_btn = QPushButton("Restart")
    cont_btn = QPushButton("Continue")
    cont_btn.setObjectName("confirmBtn")
    cont_btn.setDefault(True)
    for b in (restart_btn, cont_btn):
        b.setMinimumWidth(130)  # equal-size buttons
    btn_row.addWidget(restart_btn)
    btn_row.addWidget(cont_btn)
    layout.addLayout(btn_row)

    choice = {"cont": True}
    cont_btn.clicked.connect(lambda: (choice.update(cont=True), dlg.accept()))
    restart_btn.clicked.connect(lambda: (choice.update(cont=False), dlg.accept()))

    dlg.exec()

    if allow_remember and remember_chk is not None and remember_chk.isChecked():
        _REMEMBERED["choice"] = choice["cont"]
    return choice["cont"]
