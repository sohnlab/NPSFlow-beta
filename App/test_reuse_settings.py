"""Tests for the Load-File-driven "reuse settings" scheme (no Qt).

Run:  python App/test_reuse_settings.py
"""
import os, sys, json, tempfile, types

sys.path.insert(0, os.path.dirname(__file__))

from utils import settings_input as si
from port import Port
from workflow_engine import WorkflowEngine


# ---- resolve_settings_input: empty-port reuse gating -----------------------

def test_empty_port_fresh_when_reuse_off():
    si.set_reuse_when_empty(False)
    block = types.SimpleNamespace(parameters={"lastSettings": {"a": 1}})
    with tempfile.TemporaryDirectory() as d:
        autosave = os.path.join(d, "temp.json")
        path, is_auto = si.resolve_settings_input("", autosave, block=block)
    assert path is None and is_auto is False


def test_empty_port_reuses_when_reuse_on():
    si.set_reuse_when_empty(True)
    block = types.SimpleNamespace(parameters={"lastSettings": {"a": 1}})
    with tempfile.TemporaryDirectory() as d:
        autosave = os.path.join(d, "temp.json")
        path, is_auto = si.resolve_settings_input("", autosave, block=block)
        # restore() should have written lastSettings to the autosave path
        assert path == autosave and is_auto is True
        assert os.path.isfile(autosave)
        assert json.load(open(autosave)) == {"a": 1}
    si.set_reuse_when_empty(False)


def test_explicit_path_wins_regardless_of_mode():
    si.set_reuse_when_empty(True)
    with tempfile.TemporaryDirectory() as d:
        autosave = os.path.join(d, "temp.json")
        path, is_auto = si.resolve_settings_input(
            "/abs/custom.json", autosave, block=None)
        assert path == "/abs/custom.json" and is_auto is False
    si.set_reuse_when_empty(False)


def test_empty_reuse_but_nothing_to_restore():
    si.set_reuse_when_empty(True)
    block = types.SimpleNamespace(parameters={})  # no lastSettings
    with tempfile.TemporaryDirectory() as d:
        autosave = os.path.join(d, "temp.json")  # does not exist
        path, is_auto = si.resolve_settings_input("", autosave, block=block)
        assert path is None and is_auto is False
    si.set_reuse_when_empty(False)


# ---- engine detection: which Load File path states mean "reuse" -------------

def _load_block_with_path_source(src_block):
    """A LoadData block whose `path` port is wired from src_block.out (or
    unwired when src_block is None)."""
    blk = types.SimpleNamespace(definition_name="LoadData", input_ports=[])
    port = Port(name="path", direction="input")
    port.parent_block = blk
    if src_block is not None:
        out = Port(name="out", direction="output")
        out.parent_block = src_block
        port.connections.append(types.SimpleNamespace(source_port=out))
    blk.input_ports.append(port)
    return blk


def _text_block(value):
    return types.SimpleNamespace(definition_name="TextBlock",
                                 parameters={"valueString": value})


def _mode(blocks):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    return eng._reuse_settings_mode(blocks)


def test_mode_off_when_path_unwired():
    assert _mode([_load_block_with_path_source(None)]) is False


def test_mode_on_when_text_path_nonempty():
    blk = _load_block_with_path_source(_text_block("data/file.mat"))
    assert _mode([blk]) is True


def test_mode_off_when_text_path_empty():
    blk = _load_block_with_path_source(_text_block("   "))
    assert _mode([blk]) is False


def test_mode_on_when_computed_source():
    other = types.SimpleNamespace(definition_name="SomeUpstream", parameters={})
    assert _mode([_load_block_with_path_source(other)]) is True


def test_mode_off_when_no_load_block():
    assert _mode([_text_block("x")]) is False


def test_mode_from_persisted_recent_choice():
    """Unwired path port: the Load File picker's last recent-vs-new choice is
    persisted on the block, so reuse mode is known at run start even when the
    picker doesn't re-open (cached block / re-run from a downstream block)."""
    blk = _load_block_with_path_source(None)
    blk.parameters = {"reuseSettings": True}   # recent file last chosen
    assert _mode([blk]) is True

    blk.parameters = {"reuseSettings": False}  # new file last chosen
    assert _mode([blk]) is False

    blk.parameters = {}                         # never loaded yet -> fresh
    assert _mode([blk]) is False


# ---- runner honors reuse-mode on an empty settings port --------------------

def test_template_processing_runner_reuses_on_empty_when_mode_on():
    """Regression: Template Processing must load its autosave TPsettings when
    the Load File 'Most Recent' shortcut turns reuse-mode on and the loadFile
    port is left empty (previously it ignored reuse-mode and started fresh)."""
    import processing.pulse_template_processing as ptp_mod  # noqa: F401 (faked)
    import utils.tpsettings_io as tio
    import utils.lastsettings_capture as lsc

    SAVED = {"template_original": [1.0, 2.0, 3.0], "filter_config": {}}
    captured = {}

    def fake_processing(pulse_template, sample_rate, **kw):
        captured["loaded_settings"] = kw.get("loaded_settings")
        captured["template"] = pulse_template
        return {"ok": True}

    # Stub the heavy/interactive pieces so we test only the resolution path.
    fake_ptp = types.ModuleType("processing.pulse_template_processing")
    fake_ptp.pulse_template_processing = fake_processing
    saved_mods = sys.modules.get("processing.pulse_template_processing")
    sys.modules["processing.pulse_template_processing"] = fake_ptp

    orig = {"load": tio.load_tpsettings, "save": tio.save_tpsettings,
            "restore": lsc.restore, "capture": lsc.capture,
            "isfile": os.path.isfile}
    tio.load_tpsettings = lambda p: dict(SAVED)
    tio.save_tpsettings = lambda *a, **k: None
    lsc.restore = lambda block, path: True          # a snapshot exists
    lsc.capture = lambda *a, **k: None
    os.path.isfile = lambda p: (str(p).endswith(os.path.join("PulseShape", "temp.json"))
                                or orig["isfile"](p))

    si.set_reuse_when_empty(True)
    block = types.SimpleNamespace(parameters={})
    try:
        from runners.runPulseTemplateProcessing import run
        run({"loadFile": "", "templateIn": None, "sampleRate": 1}, {}, block)
    except Exception as e:
        raise AssertionError(
            f"runner started fresh instead of loading settings: {e!r}")
    finally:
        if saved_mods is not None:
            sys.modules["processing.pulse_template_processing"] = saved_mods
        else:
            sys.modules.pop("processing.pulse_template_processing", None)
        tio.load_tpsettings = orig["load"]
        tio.save_tpsettings = orig["save"]
        lsc.restore = orig["restore"]
        lsc.capture = orig["capture"]
        os.path.isfile = orig["isfile"]
        si.set_reuse_when_empty(False)

    # The runner flattens loaded TPsettings to a zone-0 view (which also
    # surfaces shared filter_* keys), so check the saved content survives
    # rather than byte-identical equality.
    ls = captured.get("loaded_settings")
    assert ls is not None \
        and ls.get("template_original") == SAVED["template_original"], \
        "empty loadFile under reuse-mode should load the autosave TPsettings"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
