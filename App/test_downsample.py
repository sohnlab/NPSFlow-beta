"""Preprocess (Downsample) block: pipeline outputs, blockdef shape, runner
port-mapping (dialog monkeypatched), and registry."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


class _Block:
    """Minimal block stand-in the runner writes confirmed params onto."""
    def __init__(self):
        self.parameters = {}


def _run_with_stub_dialog(inputs, params):
    """Call the runner with the interactive dialog replaced by a stub that
    returns the given params unchanged (no Qt widgets constructed)."""
    import processing.downsample_ui as ui
    orig = ui.downsample_ui
    ui.downsample_ui = lambda data, fs, init_params=None: dict(init_params or {})
    try:
        from runners.runDownsample import run as ds_run
        return ds_run(inputs, params, _Block())
    finally:
        ui.downsample_ui = orig


def test_runner_emits_six_ports():
    x = np.arange(40, dtype=float)
    params = {"smooth_enable": False, "ds_enable": True, "ds_factor": 4,
              "lp_enable": False, "asls_enable": False}
    out = _run_with_stub_dialog({"data": x, "sampleRate": 1000}, params)
    assert set(out) == {"dataSmoothed", "dataDownsampled", "dataFiltered",
                        "dataDetrended", "trendline", "DSfactor"}
    assert out["DSfactor"] == 4
    assert np.array_equal(np.asarray(out["dataDownsampled"]), x[::4])
    assert len(out["dataSmoothed"]) == 40
    assert "sampleRate" not in out          # GSR owns the effective rate now


def test_runner_missing_data_raises():
    try:
        _run_with_stub_dialog({"sampleRate": 1000}, {"ds_factor": 4})
    except ValueError:
        return
    assert False, "expected ValueError on missing data"


def test_definition_shape():
    from blockdefs.downsample import get_definition
    d = get_definition()
    assert d["name"] == "Downsample"
    assert d["displayName"] == "Preprocess"
    assert d["category"] == "Preprocessing"
    assert "updatesGlobalSampleRate" not in d
    in_names = [p["name"] for p in d["inputs"]]
    out_names = [p["name"] for p in d["outputs"]]
    assert in_names == ["data", "sampleRate", "settingsFile"]
    assert out_names == ["dataSmoothed", "dataDownsampled", "dataFiltered",
                         "dataDetrended", "trendline", "DSfactor"]
    assert d["defaultParameters"]["ds_factor"] == 20


def test_registry_registers_downsample():
    from block_registry import BlockRegistry
    reg = BlockRegistry()
    reg.scan_block_defs()
    assert "Downsample" in reg.definitions
    assert reg.definitions["Downsample"]["runner"] == "runDownsample"


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
