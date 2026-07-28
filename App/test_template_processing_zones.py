"""Template Processing restores EVERY zone's saved settings, not just zone 0.

The reported bug: a multi-zone (N-channel) template reopened with only the
first zone's settings restored; zones 1+ started fresh. Mocks the Qt dialog.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import processing.pulse_template_processing as mod


def _min_zone(threshold_upper):
    return {"threshold": {"upper": float(threshold_upper)},
            "template_original": np.zeros(10),
            "peak_locations": [], "peak_sequence": [], "key_peaks": [],
            "methods": [], "exclusion_zones": [], "end_zones": {},
            "edge_margin": {}, "peak_counts": {}}


def _run_with_fake_dialog(template_data, loaded_settings):
    """Run pulse_template_processing with the threshold dialog mocked to
    record the loaded_settings each channel receives and echo it back."""
    captured = []

    class FakeDlg:
        def __init__(self, template, filtered, fs, **kw):
            self.ls = kw.get("loaded_settings")
            captured.append(self.ls)
            self.confirmed = True
            self.restart_requested = False

        def exec(self):
            return 0

        def get_result(self, filter_config, padding_info):
            thr = self.ls.get("threshold") if isinstance(self.ls, dict) else None
            return _min_zone(thr.get("upper", 0) if isinstance(thr, dict) else 0)

    saved = mod._ThresholdDialog
    mod._ThresholdDialog = FakeDlg
    try:
        fc_in = [{"type": "lowpass", "cutoff1": 300, "cutoff2": 0}]
        res = mod.pulse_template_processing(
            template_data, 1000.0, filter_config_in=fc_in,
            loaded_settings=loaded_settings)
    finally:
        mod._ThresholdDialog = saved
    return res, captured


def test_each_channel_restores_its_own_zone():
    full = mod._wrap_zones([_min_zone(10), _min_zone(20), _min_zone(30)])
    template = np.ones((100, 3))   # 3-channel template
    res, captured = _run_with_fake_dialog(template, full)

    got = [ls.get("threshold", {}).get("upper") for ls in captured]
    assert got == [10.0, 20.0, 30.0], got        # each zone got its OWN settings
    assert res.get("num_zones") == 3


def test_single_channel_uses_flattened_zone0():
    # A full zone-grouped struct reaching a 1-D template flattens to zone 0.
    full = mod._wrap_zones([_min_zone(42), _min_zone(99)])
    template = np.ones(100)
    res, captured = _run_with_fake_dialog(template, full)
    assert captured[0].get("threshold", {}).get("upper") == 42.0
    assert res.get("num_zones") == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("ok ", fn.__name__)
        except Exception as e:
            failed += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
