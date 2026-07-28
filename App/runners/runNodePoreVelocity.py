"""Runner for NodePoreVelocity block.

Computes velocity = distance / time. For the current time-based block, the
``Time`` and ``ChannelLength`` inputs are first normalised to seconds and mm
using the block's ``timeUnit`` / ``lengthUnit`` settings, so the default
formula (`ChannelLength / Time`) yields mm/s regardless of the input units.
Shape is preserved (N_pulses x N_segments); a single-segment column yields a
per-pulse velocity vector.

The formula is evaluated against a namespace built from the connected input
ports, so older width-based blocks (`SegmentWidth` + `sampleRate`) keep working
with their saved formula — unit normalisation is skipped when there is no
``Time`` input.
"""

import numpy as np

DEFAULT_FORMULA = "ChannelLength / Time"

# Matrix inputs that define the segment count (first present wins).
_MATRIX_INPUTS = ("Time", "SegmentWidth")

# Conversion factors → canonical units (seconds, millimetres).
_TIME_TO_S = {"s": 1.0, "ms": 1e-3}
_LENGTH_TO_MM = {"m": 1e3, "mm": 1.0, "µm": 1e-3, "um": 1e-3}


def _output_factor(output_unit):
    """Scale a canonical mm/s velocity into *output_unit* (e.g. 'µm/ms')."""
    try:
        length, time = str(output_unit).split("/")
    except ValueError:
        return 1.0
    return _TIME_TO_S.get(time, 1.0) / _LENGTH_TO_MM.get(length, 1.0)


def run(inputs, params, block):
    safe_ns = {"np": np, "sampleRate": float(inputs.get("sampleRate", 1))}

    # Primary matrix input → drives the segment count for ChannelLength matching.
    matrix = None
    for name in _MATRIX_INPUTS:
        val = inputs.get(name)
        if val is None:
            continue
        m = np.asarray(val, dtype=float)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        safe_ns[name] = m
        if matrix is None:
            matrix = m

    # Match ChannelLength to the number of segments (pad with last, then trim).
    ChannelLength = inputs.get("ChannelLength")
    if ChannelLength is not None:
        ChannelLength = np.asarray(ChannelLength, dtype=float).ravel()
        if matrix is not None and matrix.shape[1] > 0:
            n_segments = matrix.shape[1]
            if len(ChannelLength) < n_segments:
                ChannelLength = np.concatenate([
                    ChannelLength,
                    np.full(n_segments - len(ChannelLength), ChannelLength[-1]),
                ])
            ChannelLength = ChannelLength[:n_segments]
        safe_ns["ChannelLength"] = ChannelLength

    # Any other connected inputs are exposed to the formula by their port name.
    for port in block.input_ports:
        if port.name in safe_ns or port.name == "addInput":
            continue
        val = inputs.get(port.name)
        if val is not None:
            safe_ns[port.name] = np.asarray(val, dtype=float)

    # Normalise units for the time-based block so the formula yields mm/s.
    # Skipped for legacy SegmentWidth blocks (no Time input), which carry their
    # own scaling in the saved formula.
    if "Time" in safe_ns:
        tf = _TIME_TO_S.get(params.get("timeUnit", "ms"), 1.0)
        if tf != 1.0:
            safe_ns["Time"] = safe_ns["Time"] * tf
        lf = _LENGTH_TO_MM.get(params.get("lengthUnit", "µm"), 1.0)
        if lf != 1.0 and "ChannelLength" in safe_ns:
            safe_ns["ChannelLength"] = safe_ns["ChannelLength"] * lf

    formula = params.get("formula", DEFAULT_FORMULA)
    # A time-based block carrying a legacy SegmentWidth formula is stale —
    # fall back to the default so it computes instead of raising NameError.
    if "Time" in safe_ns and "SegmentWidth" not in safe_ns \
            and "SegmentWidth" in formula:
        formula = DEFAULT_FORMULA
    try:
        result = eval(formula, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:
        raise RuntimeError(f"Formula evaluation failed: {exc}") from exc

    # Scale the canonical mm/s result into the selected output unit.
    out_factor = _output_factor(params.get("outputUnit", "mm/s"))
    if out_factor != 1.0:
        result = np.asarray(result, dtype=float) * out_factor

    return {"velocity": result.tolist() if hasattr(result, 'tolist') else result}
