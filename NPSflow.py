#!/usr/bin/env python3
"""NPSflow - Visual Node-Based Workflow Editor for NPS Signal Processing. 

Launch the NPSflow application.

Usage:
    python NPSflow.py
"""

import sys
import os
import tempfile
import time

# Ensure the App directory (containing all application modules) is in the path
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "App")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

def _user_tag():
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return str(os.getuid()) if hasattr(os, "getuid") else "default"


# Per-user names so two users on a shared machine don't collide.
LOCK_FILE = os.path.join(tempfile.gettempdir(), f"npsflow-{_user_tag()}.lock")
QUIT_FILE = os.path.join(tempfile.gettempdir(), f"npsflow-{_user_tag()}.quit")


def _request_quit_and_wait(lock):
    """Ask the running instance to quit and wait for its lock to release.

    The old instance closes via its window (save prompts run there), so this
    can legitimately take a while; returns False on timeout or if the user
    cancels the close in the old instance."""
    try:
        with open(QUIT_FILE, "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        return False

    acquired = False
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if lock.tryLock(200):
            acquired = True
            break

    try:
        os.remove(QUIT_FILE)
    except OSError:
        pass
    return acquired


def _install_quit_watcher(app, window):
    """Poll for a quit request from a newer instance. Close the main window
    (so closeEvent runs its unsaved-changes prompts); if the close is
    accepted, quit explicitly — modal block dialogs are their own top-level
    windows and would otherwise keep the app (and the lock) alive. If the
    user cancels the close, nothing happens and the new instance times out."""
    from PySide6.QtCore import QTimer

    def _check():
        if not os.path.exists(QUIT_FILE):
            return
        try:
            os.remove(QUIT_FILE)
        except OSError:
            pass
        if window.close():
            app.quit()

    timer = QTimer(app)
    timer.timeout.connect(_check)
    timer.start(200)  # check every 200ms
    return timer  # prevent GC


def _check_dependencies():
    """Check if all required packages are installed. Returns list of missing packages."""
    required_packages = {
        'PySide6': 'PySide6',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'h5py': 'h5py',
    }

    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    return missing


def _stream_pip_fancy(proc, total):
    """Compact pip display: FIFO of the last 4 finished packages, the package
    currently being fetched, and an overall progress bar. TTY only (redraws a
    fixed 6-line region with ANSI escapes). Returns (returncode, raw tail)."""
    import re
    from collections import deque

    fifo = deque(maxlen=4)     # last finished packages
    tail = deque(maxlen=40)    # raw pip output, shown only on failure
    cur_name = cur_note = None
    done, pct, barw = 0, 0, 40

    def name_of(spec):
        return re.split(r"[><=!~\[;(\s]", spec, maxsplit=1)[0]

    def draw():
        rows = [""] * (4 - len(fifo)) + list(fifo)
        rows.append(("→ " + cur_name + (f" — {cur_note}" if cur_note else ""))
                    if cur_name else "")
        filled = barw * pct // 100
        rows.append("[" + "█" * filled + "░" * (barw - filled) + f"] {pct:3d}%")
        out = "\x1b[6A"
        for r in rows:
            out += "\x1b[2K  " + r[:76] + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()

    sys.stdout.write("\n" * 6)
    draw()
    try:
        for raw in proc.stdout:
            raw = raw.rstrip()
            tail.append(raw)
            line = raw.strip()
            low = line.lower()
            if low.startswith("requirement already satisfied:"):
                done += 1
                fifo.append(f"✓ {name_of(line.split(':', 1)[1].strip())}"
                            " (already installed)")
            elif low.startswith("collecting "):
                if cur_name:
                    fifo.append(f"✓ {cur_name}")
                done += 1
                cur_name = name_of(line.split(None, 1)[1])
                cur_note = "collecting"
            elif low.startswith(("downloading ", "using cached ")):
                m = re.search(r"\(([^)]+)\)\s*$", line)
                cur_note = "cached" if low.startswith("using") else "downloading"
                if m:
                    cur_note += f" {m.group(1)}"
            elif low.startswith("installing collected packages"):
                if cur_name:
                    fifo.append(f"✓ {cur_name}")
                cur_name, cur_note = "installing collected packages…", None
                pct = 95
            elif low.startswith("successfully installed"):
                cur_name, cur_note = None, None
                pct = 100
            else:
                continue
            if pct < 95:
                pct = min(done * 100 // max(total, 1), 90)
            draw()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n  Cancelled. NPSflow can't start without these packages.")
        sys.exit(1)
    proc.wait()
    if proc.returncode == 0:
        pct, cur_name = 100, None
        draw()
    return proc.returncode, tail


def _install_missing_packages(missing_packages):
    """Confirm in the TERMINAL, then install requirements with live pip output.

    Runs before any GUI toolkit is available (PySide6/tkinter may be absent in
    a fresh venv), so it deliberately uses stdin/stdout only. On success it
    re-verifies the imports in-process and returns; if the freshly-installed
    packages still can't be imported, it asks the user to relaunch and exits.
    """
    import subprocess

    req_file = os.path.join(app_dir, "requirements.txt")
    bar = "=" * 64

    print()
    print(bar)
    print("  NPSflow — missing Python packages")
    print(bar)
    print("  Not installed: " + ", ".join(missing_packages))
    print(f"  Interpreter:   {sys.executable}")
    print(f"  Requirements:  {req_file}")
    print()

    # Confirm. Default to yes on a bare Enter; auto-proceed when stdin isn't a
    # terminal (e.g. launched from a script) so the app isn't left hanging.
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            print("  Non-interactive shell detected — proceeding with install.")
            answer = "y"
        else:
            answer = input("  Install these now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled. NPSflow can't start without these packages.")
        sys.exit(1)
    if answer not in ("", "y", "yes"):
        print("  Cancelled. NPSflow can't start without these packages.")
        sys.exit(1)

    # Count requirement lines for a coarse overall-progress readout.
    try:
        with open(req_file) as f:
            total = sum(1 for ln in f
                        if ln.strip() and not ln.strip().startswith("#"))
    except OSError:
        total = len(missing_packages)
    total = max(total, 1)

    print()
    print(f"  Installing {total} package(s)...")
    print("-" * 64)

    # Stream pip output so the user sees real progress (stderr merged in).
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
    except Exception as exc:
        print(f"  Could not launch pip: {exc}")
        sys.exit(1)

    if sys.stdout.isatty():
        returncode, tail = _stream_pip_fancy(proc, total)
    else:
        # No TTY (log file / IDE console): plain line-by-line output.
        done = 0
        for line in proc.stdout:
            line = line.rstrip()
            low = line.lower()
            if low.startswith("collecting ") or low.startswith("requirement already satisfied"):
                done = min(done + 1, total)
            elif low.startswith("successfully installed"):
                done = total
            print(f"  [{done:>2}/{total}] {line}")
        proc.wait()
        returncode, tail = proc.returncode, None

    print("-" * 64)
    if returncode != 0:
        if tail:
            print("  Last pip output:")
            for ln in tail:
                print(f"    {ln}")
        print(f"  ✗ pip install failed (exit code {returncode}).")
        print("    Fix the errors above, or install manually with:")
        print(f"      {sys.executable} -m pip install -r {req_file}")
        sys.exit(1)
    print("  ✓ All packages installed.")

    # Re-verify in this process (invalidate import caches first so freshly
    # written packages are discoverable).
    import importlib
    importlib.invalidate_caches()
    still_missing = _check_dependencies()
    if still_missing:
        print()
        print("  Installed, but this process can't import them yet "
              f"({', '.join(still_missing)}).")
        print("  Please re-run NPSflow:")
        print(f"      {sys.executable} {os.path.join(root_dir, 'NPSflow.py')}")
        sys.exit(0)
    print("  Starting NPSflow...")
    print()


def _install_qt_message_filter():
    """Suppress harmless Qt accessibility warnings that fire when clearing
    tree/list widgets (e.g. during Reload).  The warnings are cosmetic —
    Qt's accessibility bridge queries rows on a model that was just cleared."""
    from PySide6.QtCore import qInstallMessageHandler, QtMsgType

    _default_handler = None

    def _filter(msg_type, context, message):
        if (msg_type == QtMsgType.QtWarningMsg
                and "qt.accessibility.table" in message
                and "out of bounds" in message):
            return  # suppress
        if _default_handler is not None:
            _default_handler(msg_type, context, message)

    _default_handler = qInstallMessageHandler(_filter)


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Ensure required packages are present before importing Qt. Any missing ones
    # are confirmed + installed in the terminal — no GUI toolkit exists yet.
    missing_packages = _check_dependencies()
    if missing_packages:
        _install_missing_packages(missing_packages)

    _install_qt_message_filter()

    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName("NPSflow")
    app.setStyle("Fusion")

    # Activate persisted theme (defaults to light) and apply its QPalette.
    import recent_files
    import theme as theme_module
    saved = recent_files.get_setting(root_dir, "theme", "light")
    if saved not in ("light", "dark"):
        saved = "light"
    theme_module.set_theme(saved)
    app.setPalette(theme_module.build_qpalette())

    # Single instance check. QLockFile is cross-platform and automatically
    # removes locks whose owning process has died.
    from PySide6.QtCore import QLockFile
    lock = QLockFile(LOCK_FILE)
    lock.setStaleLockTime(0)  # held for the app's lifetime; never stale by age
    if not lock.tryLock(1):
        msg = QMessageBox()
        msg.setWindowTitle("NPSflow Already Running")
        msg.setText("Another instance of NPSflow is already running.")
        msg.setInformativeText(
            "Would you like to close the previous instance and start a new one?")
        msg.setIcon(QMessageBox.Icon.Question)
        btn_close = msg.addButton(
            "Close Previous && Start New",
            QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = msg.addButton(
            "Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_cancel)
        msg.exec()

        if msg.clickedButton() == btn_close:
            if not _request_quit_and_wait(lock):
                QMessageBox.critical(
                    None, "Error",
                    "Could not close the previous instance (it may be "
                    "waiting on an unsaved-changes prompt).\n"
                    "Please close it manually and try again.")
                os._exit(0)
        else:
            os._exit(0)

    from workflow_app import NPSWorkflowApp
    window = NPSWorkflowApp()
    window.show()

    # Watch for quit requests from future instances
    _quit_timer = _install_quit_watcher(app, window)

    exit_code = app.exec()

    lock.unlock()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
