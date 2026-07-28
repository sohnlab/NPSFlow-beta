"""Evidence: does _get_state/_apply_state round-trip a WARNING pulse's peak
edits the same as an accepted pulse's?  (no Qt event loop)

Run:  python App/test_review_peak_roundtrip.py
"""
import os, sys, json, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from processing.extract_single_pulse_features import _PulseFeatureReviewDialog as D


class _W:
    """Fake Qt widget supporting value()/setValue()/isChecked()/setChecked()."""
    def __init__(self, v=0, checked=False):
        self._v, self._c = v, checked
    def value(self): return self._v
    def setValue(self, v): self._v = int(v)
    def isChecked(self): return self._c
    def setChecked(self, b): self._c = bool(b)


def _make(n_pulses, acceptance):
    d = D.__new__(D)
    d.acceptance = np.array(acceptance, dtype=int)
    d._rect_method = "median"
    d._detrend = False
    d._page = 0
    d._total_pages = 1
    d._pulse_indices = np.array([[i * 100, i * 100 + 50] for i in range(n_pulses)])
    d._peak_locs = [np.array([i * 100 + 10, i * 100 + 20], dtype=int)
                    for i in range(n_pulses)]
    d._pulse_key_peaks = [[10, 20] for _ in range(n_pulses)]
    d._pulse_key_peak_pols = [[1, -1] for _ in range(n_pulses)]
    d._pulse_excl_ranges = [[] for _ in range(n_pulses)]
    # fake widgets
    d._chk_show_trend = _W(checked=False)
    d._chk_detrend = _W(checked=False)
    d._rb_starting = _W(checked=True)
    d._rb_midpoint = _W()
    d._rb_rect_mean = _W()
    d._rb_rect_median = _W()
    d._slider_expand = _W(5)
    d._slider_cursor_radius = _W(15)
    d._spin_ref_samples = _W(1500)
    # stub heavy methods
    d._recompute_rect = lambda i: None
    d._render_page = lambda: None
    return d


def test_warning_vs_accepted_peak_roundtrip():
    # pulse 0 accepted (1), pulse 1 accepted (1), pulse 2 WARNING (2)
    src = _make(3, [1, 1, 2])
    # Edit peaks on the accepted pulse (0) and the warning pulse (2)
    src._peak_locs[0] = np.array([10, 20, 30], dtype=int)     # added a peak
    src._peak_locs[2] = np.array([210, 240], dtype=int)       # moved peaks
    state = src._get_state()

    # round-trip through JSON (mirrors autosave -> lastSettings -> reload)
    state = json.loads(json.dumps(state))

    dst = _make(3, [1, 1, 1])           # fresh detection: different peaks, no warning
    dst._apply_state(state)

    assert list(dst._peak_locs[0]) == [10, 20, 30], \
        f"accepted-pulse peaks lost: {list(dst._peak_locs[0])}"
    assert list(dst._peak_locs[2]) == [210, 240], \
        f"WARNING-pulse peaks lost: {list(dst._peak_locs[2])}"
    assert int(dst.acceptance[2]) == 2, \
        f"warning status lost: {int(dst.acceptance[2])}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
