"""Two-process regression tests for the QLockFile single-instance guard.

The old fcntl/msvcrt implementation opened the lock file with mode "w"
before locking, truncating the holder's PID and letting a second launch
proceed alongside the first. These tests exercise the real NPSflow module
functions across process boundaries.
"""
import os
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtCore import QLockFile

TAG = uuid.uuid4().hex[:8]
LOCK = os.path.join(tempfile.gettempdir(), f"npsflow-test-{TAG}.lock")
QUIT = os.path.join(tempfile.gettempdir(), f"npsflow-test-{TAG}.quit")

HOLDER = r"""
import sys, os, time
from PySide6.QtCore import QLockFile
lock = QLockFile(sys.argv[1])
lock.setStaleLockTime(0)
ok = lock.tryLock(1)
print("locked" if ok else "failed", flush=True)
if not ok:
    sys.exit(1)
quit_file = sys.argv[2] if len(sys.argv) > 2 else None
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    if quit_file and os.path.exists(quit_file):
        os.remove(quit_file)
        lock.unlock()
        sys.exit(0)
    time.sleep(0.05)
sys.exit(0)
"""


def _spawn_holder(*args):
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, *args],
                            stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "locked", f"holder failed to lock: {line!r}"
    return proc


def test_second_instance_blocked_and_lock_survives():
    proc = _spawn_holder(LOCK)
    try:
        lock = QLockFile(LOCK)
        lock.setStaleLockTime(0)
        assert not lock.tryLock(200), "second instance acquired a held lock"
        # the old bug: the failed attempt destroyed the holder's lock info
        assert not lock.tryLock(200), "lock did not survive a failed attempt"
    finally:
        proc.kill()
        proc.wait()
        QLockFile(LOCK).removeStaleLockFile()


def test_stale_lock_of_dead_process_is_reclaimed():
    proc = _spawn_holder(LOCK)
    proc.kill()
    proc.wait()
    lock = QLockFile(LOCK)
    lock.setStaleLockTime(0)
    assert lock.tryLock(2000), "stale lock of dead process not reclaimed"
    lock.unlock()


def test_quit_request_hands_over_lock():
    import NPSflow
    NPSflow.QUIT_FILE = QUIT  # isolate from any real running instance
    proc = _spawn_holder(LOCK, QUIT)
    try:
        lock = QLockFile(LOCK)
        lock.setStaleLockTime(0)
        assert not lock.tryLock(100), "lock unexpectedly free"
        assert NPSflow._request_quit_and_wait(lock), "takeover failed"
        assert not os.path.exists(QUIT), "quit file left behind"
        lock.unlock()
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except Exception as e:
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
