"""Runner for the Wavelet Editor block."""

import os
import numpy as np

from utils.paths import project_root


_AUTOSAVE_RELPATH = os.path.join("SavedTemplates", "Wavelet", "temp.json")


def _resolve_load_path(raw):
    """Resolve a loadFile input string to an absolute path or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if os.path.isabs(s):
        return s
    if os.sep not in s and "/" not in s and "\\" not in s and "." not in s:
        return os.path.join(project_root(),
                            "SavedTemplates", "Wavelet", s + ".json")
    return os.path.join(project_root(), s)


def run(inputs, params, block):
    from processing.wavelet_editor import (
        derive_segments, reconstruct, normalize,
        save_state, load_state, WaveletEditorDialog,
    )
    from utils.app_logger import logger

    tp = inputs.get("TPsettings")
    if tp is None:
        raise ValueError("WaveletEditor: TPsettings input is required")

    # TPsettings is zone-grouped; this block edits a single zone.
    from utils.tpsettings_io import get_zone
    tp = get_zone(tp, 0)

    segments = derive_segments(tp)
    norm_mode = "wavelet"

    # Resolution: /lastSettings → autosave; path → that path; empty → fresh.
    from utils.settings_input import (
        resolve_settings_input, LAST_SETTINGS_SENTINEL, reuse_when_empty,
    )
    raw_load = inputs.get("loadFile")
    # Empty port + global reuse-mode → reuse this block's last settings.
    if (not isinstance(raw_load, str) or not raw_load.strip()) and reuse_when_empty():
        raw_load = LAST_SETTINGS_SENTINEL
    autosave_abs = os.path.join(project_root(), _AUTOSAVE_RELPATH)
    if isinstance(raw_load, str) and raw_load.strip() == LAST_SETTINGS_SENTINEL:
        load_path, auto_load = resolve_settings_input(
            raw_load, autosave_abs, block=block)
    else:
        load_path = _resolve_load_path(raw_load)
        auto_load = False

    loaded = None
    if load_path is not None:
        if os.path.isfile(load_path):
            try:
                loaded = load_state(load_path)
                logger.info(f"WaveletEditor: loaded state from {load_path}")
            except Exception as e:
                logger.warning(
                    f"WaveletEditor: failed to load {load_path}: {e}")
        elif not auto_load:
            logger.warning(
                f"WaveletEditor: loadFile not found: {load_path}")

    # If the loaded state carries restorable segment edits, ask continue vs.
    # fresh; starting fresh discards it and falls back to derived defaults.
    if loaded is not None:
        applies = loaded.get("n_segments_source") == len(segments)
        kept = loaded.get("kept_rows")
        if applies or kept is not None:
            from utils.session_prompt import confirm_continue_previous
            if kept is not None:
                detail = f"{len(kept)} of {len(segments)} segments kept."
            else:
                detail = "Saved segment width/amplitude edits found."
            if not confirm_continue_previous(
                    detail, title="Wavelet Editor",
                    allow_remember=getattr(block, "_auto_run", False)):
                loaded = None

    if loaded is not None:
        try:
            if "normalization" in loaded:
                norm_mode = loaded["normalization"]
            if loaded.get("n_segments_source") == len(segments):
                ls_segs = loaded["segments"]
                if len(ls_segs) != len(segments):
                    raise ValueError(
                        f"loaded segments list length {len(ls_segs)} "
                        f"!= n_segments_source {len(segments)}")
                for s, ls in zip(segments, ls_segs):
                    s["width"] = int(ls["width"])
                    s["amplitude"] = float(ls["amplitude"])
                # Honour kept-row indices if present in saved state
                kept = loaded.get("kept_rows")
                if kept is not None:
                    kept_set = set(int(k) for k in kept)
                    segments = [s for i, s in enumerate(segments)
                                if i in kept_set]
            else:
                logger.warning(
                    "WaveletEditor: loaded n_segments_source "
                    f"({loaded.get('n_segments_source')}) != current "
                    f"({len(segments)}); ignoring loaded segment edits.")
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"WaveletEditor: malformed loaded state, falling back to "
                f"derived defaults: {e}")
            segments = derive_segments(tp)

    # current_path is the file the dialog "owns" — used by Save (without
    # As). Auto-loaded temp.json is intentionally not set as current_path,
    # so Save prompts for a real filename instead of overwriting temp.
    current_path = load_path if (load_path and not auto_load) else None

    dlg = WaveletEditorDialog(segments, norm_mode=norm_mode,
                              current_path=current_path)
    if dlg.exec() != dlg.DialogCode.Accepted:
        raise ValueError("Wavelet Editor cancelled")

    segs_out, norm_mode_out = dlg.result_state()
    n_source = len(derive_segments(tp))
    kept_rows = _diff_kept_rows(derive_segments(tp), segs_out)

    try:
        from utils.paths import saved_templates_dir
        autosave_path = os.path.join(saved_templates_dir("Wavelet"),
                                     "temp.json")
        save_state(segs_out, norm_mode_out,
                   n_segments_source=n_source,
                   kept_rows=kept_rows,
                   filepath=autosave_path)
        from utils.lastsettings_capture import capture
        capture(block, autosave_path)
    except Exception as e:
        logger.warning(f"WaveletEditor: auto-save failed: {e}")

    full = reconstruct(segs_out)
    wavelet, applied = normalize(full, norm_mode_out)
    if not applied and norm_mode_out != "none":
        logger.warning(
            f"WaveletEditor: normalization '{norm_mode_out}' skipped "
            "(degenerate input); emitting raw vector.")

    widths = [int(s["width"]) for s in segs_out]
    amps = [float(s["amplitude"]) for s in segs_out]
    boundaries = list(map(int, np.cumsum(widths))) if widths else []
    segments_struct = {
        "widths":        widths,
        "amplitudes":    amps,
        "boundaries":    boundaries,
        "normalization": norm_mode_out,
        "n_segments":    len(widths),
    }
    return {"wavelet": np.asarray(wavelet, dtype=float),
            "segments": segments_struct}


def _diff_kept_rows(original_segments, kept_segments):
    """Return indices into original_segments that survive in kept_segments.

    Matching is by orig_width / orig_amplitude pair, in order, since the
    user can only delete (not reorder or insert) and edits stay bound to
    their orig_* fields.
    """
    kept_indices = []
    j = 0
    for i, o in enumerate(original_segments):
        if j >= len(kept_segments):
            break
        k = kept_segments[j]
        if (int(k.get("orig_width", -1)) == int(o["orig_width"])
                and float(k.get("orig_amplitude", float("nan")))
                == float(o["orig_amplitude"])):
            kept_indices.append(i)
            j += 1
    return kept_indices
