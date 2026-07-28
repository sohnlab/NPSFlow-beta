"""Global Sample Rate block: fixed 10 kHz default, connected input supersedes,
and the block exposes no editable parameters (no Edit Node option)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow_engine import WorkflowEngine


class _Blk:
    def __init__(self, params=None):
        self.id = "gsr"
        self.definition_name = "GlobalSampleRate"
        self.parameters = params or {}
        self.input_ports = []
        self.output_ports = []
        self.status = "pending"
        self.display_name = "Global Sample Rate"
        self.error_message = ""
        self.output_data = {}


GSR_DEFN = {"isGlobalSampleRate": True}


def test_default_base_no_factor():
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk(), GSR_DEFN, {}, [])
    assert out == {"sampleRate": 10000, "downsampleFactor": 1}, out


def test_base_input_supersedes_default():
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk(), GSR_DEFN, {"sampleRate": 44100}, [])
    assert out == {"sampleRate": 44100, "downsampleFactor": 1}, out


def test_wired_base_beats_nondefault_legacy_param():
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk({"sampleRate": 200000}), GSR_DEFN,
                             {"sampleRate": 5000}, [])
    assert out == {"sampleRate": 5000, "downsampleFactor": 1}, out


def test_effective_is_base_over_n():
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk(), GSR_DEFN,
                             {"sampleRate": 50000, "downsampleFactor": 20}, [])
    assert out == {"sampleRate": 2500.0, "downsampleFactor": 20}, out


def test_factor_below_one_coerced():
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk(), GSR_DEFN,
                             {"sampleRate": 8000, "downsampleFactor": 0}, [])
    assert out == {"sampleRate": 8000, "downsampleFactor": 1}, out


def test_unwired_base_falls_back_to_current_global():
    eng = WorkflowEngine(registry=None)
    eng.global_vars = {"sampleRate": 50000}
    out = eng._execute_block(_Blk(), GSR_DEFN, {"downsampleFactor": 20}, [])
    assert out == {"sampleRate": 2500.0, "downsampleFactor": 20}, out


def test_legacy_saved_base_respected_without_input():
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk({"sampleRate": 200000}), GSR_DEFN, {}, [])
    assert out == {"sampleRate": 200000, "downsampleFactor": 1}, out


def test_apply_hook_propagates_both_globals():
    eng = WorkflowEngine(registry=None)
    eng.global_vars = {}

    class _OB:
        output_data = {"sampleRate": 2500.0, "downsampleFactor": 20}
    eng._apply_global_sample_rate(_OB(), {"isGlobalSampleRate": True})
    assert eng.global_vars["sampleRate"] == 2500.0
    assert eng.global_vars["downsampleFactor"] == 20


def test_definition_not_editable():
    """No defaultParameters -> the block is not parameterized, so the app's
    Edit Node gating (isParameterized or defaultParameters) excludes it."""
    from blockdefs.global_sample_rate import get_definition
    defn = get_definition()
    assert "defaultParameters" not in defn
    assert not defn.get("isParameterized")


def test_definition_two_inputs_non_compact():
    from blockdefs.global_sample_rate import get_definition
    d = get_definition()
    in_names = [p["name"] for p in d["inputs"]]
    assert in_names == ["sampleRate", "downsampleFactor"]
    assert not d.get("isCompact")
    assert d.get("isGlobalSampleRate") is True
    assert "defaultParameters" not in d          # stays non-editable


def test_apply_hook_updatesglobalsamplerate_flag():
    eng = WorkflowEngine(registry=None)
    eng.global_vars = {"sampleRate": 10000}

    class _OB:
        output_data = {"sampleRate": 500.0}
    eng._apply_global_sample_rate(_OB(), {"updatesGlobalSampleRate": True})
    assert eng.global_vars["sampleRate"] == 500.0


def test_apply_hook_isglobalsamplerate_sets_rate():
    eng = WorkflowEngine(registry=None)
    eng.global_vars = {}

    class _OB:
        output_data = {"sampleRate": 44100}
    eng._apply_global_sample_rate(_OB(), {"isGlobalSampleRate": True})
    assert eng.global_vars["sampleRate"] == 44100


def test_apply_hook_presence_based_preserves_zero():
    eng = WorkflowEngine(registry=None)
    eng.global_vars = {"sampleRate": 10000}

    class _OB:
        output_data = {"sampleRate": 0}
    eng._apply_global_sample_rate(_OB(), {"isGlobalSampleRate": True})
    assert eng.global_vars["sampleRate"] == 0


def test_apply_hook_ignores_unflagged_block():
    eng = WorkflowEngine(registry=None)
    eng.global_vars = {"sampleRate": 10000}

    class _OB:
        output_data = {"sampleRate": 999}
    eng._apply_global_sample_rate(_OB(), {})
    assert eng.global_vars["sampleRate"] == 10000


# ── hardening: coerce array inputs, never publish None, always run in Phase 1 ──

def test_coerces_array_sampleRate_to_scalar_float():
    import numpy as np
    eng = WorkflowEngine(registry=None)
    # File sample rates load as (1,1)/0-d/1-elem arrays; publish a plain float.
    for val in (np.array([[50000.0]]), np.squeeze(np.array([[50000.0]])),
                np.array([50000.0])):
        out = eng._execute_block(_Blk(), GSR_DEFN, {"sampleRate": val}, [])
        assert out["sampleRate"] == 50000.0, (val, out)
        assert type(out["sampleRate"]) is float          # not an ndarray/np scalar
    # Array base with a factor still divides to a plain float.
    out = eng._execute_block(_Blk(), GSR_DEFN,
                             {"sampleRate": np.array([[50000.0]]),
                              "downsampleFactor": 20}, [])
    assert out["sampleRate"] == 2500.0 and type(out["sampleRate"]) is float


def test_never_publishes_none_when_param_is_none():
    # A stale saved param of None must not shadow the 10 kHz fallback.
    eng = WorkflowEngine(registry=None)
    out = eng._execute_block(_Blk({"sampleRate": None}), GSR_DEFN, {}, [])
    assert out["sampleRate"] == 10000.0
    assert out["downsampleFactor"] == 1


def test_global_setter_ids_forces_gsr_into_phase1():
    # isGlobalSampleRate blocks must be identified so run_all forces them into
    # Phase 1 even when unwired (otherwise they publish too late in Phase 2).
    class _Reg:
        def has(self, n):
            return n in ("GlobalSampleRate", "DataSummary")

        def get(self, n):
            return {"isGlobalSampleRate": True} if n == "GlobalSampleRate" else {}

    class _B:
        def __init__(self, bid, dn):
            self.id = bid
            self.definition_name = dn

    eng = WorkflowEngine(registry=_Reg())
    blocks = [_B("g", "GlobalSampleRate"), _B("x", "DataSummary")]
    assert eng._global_setter_ids(blocks) == {"g"}


def test_wired_base_is_latched_then_reused_when_wire_silent():
    # Regression: a run where the wire delivers latches the base; a later run
    # with no wire data must reuse it — never the published effective global
    # (which would divide by N again: 50 kHz -> 2.5 kHz -> 125 Hz).
    eng = WorkflowEngine(registry=None)
    blk = _Blk()
    out = eng._execute_block(blk, GSR_DEFN,
                             {"sampleRate": 50000, "downsampleFactor": 20}, [])
    assert out == {"sampleRate": 2500.0, "downsampleFactor": 20}, out
    assert blk.parameters["sampleRate"] == 50000.0
    eng.global_vars = {"sampleRate": 2500.0, "downsampleFactor": 20}
    out = eng._execute_block(blk, GSR_DEFN, {}, [])
    assert out == {"sampleRate": 2500.0, "downsampleFactor": 20}, out


def test_run_block_never_injects_effective_global_as_base():
    # Regression: run_block injects globals into unfilled inputs; a GSR must
    # be exempt or its own published effective rate becomes next run's base.
    class _Reg:
        def has(self, n):
            return True

        def get(self, n):
            return GSR_DEFN

    eng = WorkflowEngine(registry=_Reg())
    blk = _Blk({"sampleRate": 50000, "downsampleFactor": 20})
    eng.global_vars = {"sampleRate": 2500.0, "downsampleFactor": 20}
    eng.run_block(blk, [])
    assert blk.output_data == {"sampleRate": 2500.0, "downsampleFactor": 20}, \
        blk.output_data
    assert eng.global_vars["sampleRate"] == 2500.0


def test_promote_global_setters_places_each_after_own_upstream():
    # Two GSRs: g_base isolated (base rate), g_eff fed by a late Preprocess block.
    # g_base must run before an early consumer (ds); g_eff must stay after pre.
    class _B:
        def __init__(self, bid): self.id = bid
    class _P:
        def __init__(self, b): self.parent_block = b
    class _W:
        def __init__(self, s, d): self.source_port = _P(s); self.dest_port = _P(d)

    ids = ["start", "load", "g_base", "ds", "pre", "g_eff"]
    blk = {i: _B(i) for i in ids}
    order = [blk[i] for i in ids]                 # natural topological order
    wires = [_W(blk["load"], blk["ds"]),          # load -> ds (early consumer)
             _W(blk["load"], blk["pre"]),         # load -> pre
             _W(blk["pre"], blk["g_eff"])]        # pre -> g_eff (effective GSR)
    # g_base has NO wires (isolated base-rate publisher)
    eng = WorkflowEngine(registry=None)
    out = [b.id for b in eng._promote_global_setters(order, wires, {"g_base", "g_eff"})]
    assert out.index("g_base") < out.index("ds"), out       # base GSR before consumer
    assert out.index("pre") < out.index("g_eff"), out       # effective GSR after pre
    assert out.index("g_eff") == out.index("pre") + 1, out  # right after pre


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
