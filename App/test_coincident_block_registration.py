"""Block auto-registers with the expected ports."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_definition_ports_and_metadata():
    from blockdefs.extract_coincident_pulse_features import get_definition
    d = get_definition()
    assert d["name"] == "ExtractCoincidentPulseFeatures"
    assert d["displayName"] == "Coincident Processing"
    assert d["category"] == "FeatureExtraction"
    assert d["isInteractive"] is True
    assert d["runner"] == "runExtractCoincidentPulseFeatures"
    ins = {p["name"] for p in d["inputs"]}
    assert {"dataIn", "pulseLabels", "TPsettings"} <= ins
    outs = {p["name"] for p in d["outputs"]}
    assert {"pulsePeakLocations", "rectangularizedPulses", "acceptanceID",
            "pulseStartIndices", "coincidenceMap"} <= outs


def test_runner_imports_and_is_callable():
    import runners.runExtractCoincidentPulseFeatures as r
    assert hasattr(r, "run")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("ok ", fn.__name__)
        except Exception as e:
            failed += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
