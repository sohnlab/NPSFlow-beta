"""Per-pulse feature extraction with interactive review.

Translates MATLAB ExtractSinglePulseFeatures.m: processes each detected
pulse region, applies template-derived filtering and thresholding to
extract peaks and rectangularized segments, then provides an interactive
review UI for accepting or rejecting individual pulses.
"""

import copy
import sys
import traceback
import numpy as np
from scipy.signal import find_peaks

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)
from PySide6.QtCore import Qt, QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QPainter, QPen, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from utils.edge_refine import refine_interior
from theme import theme


# ---------------------------------------------------------------------------
# PySide6 exception safety — prevent abort() on unhandled slot exceptions
# ---------------------------------------------------------------------------
def _qt_msg_handler(msg_type, context, message):
    """Custom Qt message handler that prints fatal messages instead of aborting."""
    if msg_type == QtMsgType.QtFatalMsg:
        print(f"[Qt Fatal – suppressed abort] {message}", file=sys.stderr)
    elif msg_type == QtMsgType.QtCriticalMsg:
        print(f"[Qt Critical] {message}", file=sys.stderr)
    elif msg_type == QtMsgType.QtWarningMsg:
        print(f"[Qt Warning] {message}", file=sys.stderr)


def _install_exception_hook():
    def _hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook
    qInstallMessageHandler(_qt_msg_handler)


_install_exception_hook()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_single_pulse_features(
    data_in,
    tp=None,
    sample_rate=1.0,
    pulse_indices=None,
    trendline=None,
    pulse_features=None,
    settings_file=None,
    ds_factor=None,
    file_names=None,
    file_boundaries=None,
):
    """Extract features from each detected pulse and provide review UI.

    Zone-aware: ``data_in`` may be 1-D (single zone) or N×Z (one column per
    zone), and ``tp`` is the FULL zone-grouped TPsettings struct. Every zone
    is computed; zone 0 is reviewed in the interactive dialog (the active-zone
    selector comes later), and the remaining zones use their computed
    accept/reject defaults. Outputs are zone-grouped via ``wrap_zones`` — for
    a single zone they are returned bare, byte-for-byte identical to the
    legacy single-signal behavior.

    Parameters
    ----------
    data_in : np.ndarray
        1-D signal or N×Z multi-zone array.
    tp : dict
        Zone-grouped TPsettings struct (see ``utils.tpsettings_io``).
    pulse_indices : np.ndarray
        Nx2 array of [start, end] indices for pulse regions (shared by zones).
    pulse_features : dict, optional
        Segment definitions from PulseFeatureExtraction.
    file_names : list of str, optional
        Source file per concatenated block (multi-file input); drives the
        per-file LP Filter group in the review dialog.
    file_boundaries : array-like, optional
        len(file_names)+1 cumulative sample offsets of each file within
        ``data_in``.

    Returns
    -------
    dict with keys (each bare for 1 zone, ``wrap_zones`` struct for >1):
        pulse_peak_locations, rectangularized_pulses,
        acceptance_id, pulse_start_indices
    """
    from utils.app_logger import logger
    from utils.zones import zone_columns, wrap_zones

    fs = float(sample_rate)

    # --- unpack pulse indices (shared across zones) -------------------------
    if pulse_indices is None:
        return _empty_result()
    pulse_indices = np.atleast_2d(np.asarray(pulse_indices, dtype=int))
    n_pulses = len(pulse_indices)
    if n_pulses == 0:
        return _empty_result()

    # --- split data / trendline into per-zone columns -----------------------
    data_cols = zone_columns(np.asarray(data_in))
    n_zones = len(data_cols)

    trend_arr = None
    if trendline is not None:
        trend_arr = np.asarray(trendline)
    if trend_arr is not None and trend_arr.ndim == 2 and trend_arr.shape[1] > 1:
        trend_cols = zone_columns(trend_arr)
    else:
        trend_cols = None  # shared (or None) for every zone

    def _trend_for(k):
        t = trend_cols[k] if trend_cols is not None else trendline
        if t is None:
            return None
        t = np.atleast_1d(t).ravel().astype(float)
        return t if len(t) else None

    from utils.tpsettings_io import get_zone

    # --- compute every zone --------------------------------------------------
    computes = []
    for k in range(n_zones):
        params = _unpack_tp_zone(get_zone(tp, k))
        computes.append(
            _compute_zone(
                data_cols[k].astype(float),
                fs,
                _trend_for(k),
                pulse_indices,
                **params,
            )
        )

    # Embed the shared pulse_indices + per-zone trendline into each compute
    # dict so the dialog can carry them per zone.
    for k in range(n_zones):
        computes[k]["pulse_indices"] = pulse_indices
        computes[k]["trendline"] = _trend_for(k)

    logger.info("Feature extraction complete. Launching review UI...")

    # --- interactive review (all zones, active-zone selector) ---------------
    per_zone = _review_all_zones(
        computes,
        pulse_indices,
        fs,
        get_zone(tp, 0),
        settings_file,
        ds_factor=ds_factor,
        file_names=file_names,
        file_boundaries=file_boundaries,
    )

    return {
        "pulse_peak_locations": wrap_zones(
            [z["pulse_peak_locations"] for z in per_zone]
        ),
        "rectangularized_pulses": wrap_zones(
            [z["rectangularized_pulses"] for z in per_zone]
        ),
        "acceptance_id": wrap_zones([z["acceptance_id"] for z in per_zone]),
        "pulse_start_indices": wrap_zones([z["pulse_start_indices"] for z in per_zone]),
    }


def _unpack_tp_zone(tpz):
    """Unpack one ``get_zone(tp, k)`` flat dict into ``_compute_zone`` kwargs.

    Mirrors the field extraction the runner used to do (single-zone), now
    applied per zone.
    """
    tpz = tpz or {}
    return dict(
        threshold=tpz.get("threshold"),
        edge_margin=tpz.get("edge_margin"),
        peak_counts=tpz.get("peak_counts"),
        peak_sequence=tpz.get("peak_sequence"),
        filter_padding=tpz.get("filter_padding"),
        filter_config=tpz.get("filter_config"),
        key_peaks=tpz.get("key_peaks"),
        key_peak_properties=tpz.get("key_peak_properties"),
        peak_locations=tpz.get("peak_locations"),
        peak_locations_apex=tpz.get("peak_locations_apex"),
        edge_method=tpz.get("edge_method", "peaks"),
        template_original=tpz.get("template_original"),
        exclusion_zones=tpz.get("exclusion_zones", []),
        end_zones=tpz.get("end_zones", {}),
        key_peak_seq_indices_in=tpz.get("key_peak_seq_indices"),
        exclusion_zone_seq_bounds_in=tpz.get("exclusion_zone_seq_bounds"),
    )


def _rectangularize_filtered(seg_filtered, bound_idx):
    """Median and mean rectangularization of *seg_filtered*, split at the
    interior boundaries *bound_idx* (plus [0, len] sentinels). Returns
    (rect_median, rect_mean). Shared by the initial compute and the Edge
    toggle so a toggle round-trip reproduces the original rects."""
    seg_rect_median = np.zeros_like(seg_filtered)
    seg_rect_mean = np.zeros_like(seg_filtered)
    if len(seg_filtered) == 0:
        return seg_rect_median, seg_rect_mean
    all_bounds = np.sort(
        np.unique(
            np.concatenate([[0], np.asarray(bound_idx, dtype=int), [len(seg_filtered)]])
        )
    )
    if len(all_bounds) < 2:
        seg_rect_median[:] = np.median(seg_filtered)
        seg_rect_mean[:] = np.mean(seg_filtered)
    else:
        for k in range(len(all_bounds) - 1):
            si = int(all_bounds[k])
            ei = int(all_bounds[k + 1])
            if ei <= si:
                ei = si + 1
            seg_rect_median[si:ei] = np.median(seg_filtered[si:ei])
            seg_rect_mean[si:ei] = np.mean(seg_filtered[si:ei])
    return seg_rect_median, seg_rect_mean


def _refine_pulse_bounds(global_peaks, filtered_seg, s, edge_method):
    """Map a pulse's global apex peak locations to FWHM-refined global
    segment boundaries using its filtered segment (count/order preserved).

    Returns *global_peaks* unchanged for ``edge_method != 'fwhm'`` or when the
    filtered segment / peaks are unavailable, so Peaks mode and legacy
    templates are byte-identical.
    """
    if global_peaks is None or len(global_peaks) == 0:
        return np.asarray([] if global_peaks is None else global_peaks, dtype=int)
    gp = np.asarray(global_peaks, dtype=int)
    if edge_method != "fwhm" or filtered_seg is None or len(filtered_seg) < 3:
        return gp
    n = len(filtered_seg)
    local = gp - int(s)
    valid = (local >= 0) & (local < n)
    if not valid.any():
        return gp
    ref_local = refine_interior(filtered_seg, local[valid], edge_method)
    out = gp.copy()
    out[valid] = ref_local + int(s)
    return out


def _review_all_zones(
    computes,
    pulse_indices,
    fs,
    tpz0,
    settings_file,
    ds_factor=None,
    file_names=None,
    file_boundaries=None,
):
    """Build one review dialog covering all zones; return per-zone results.

    The dialog is constructed from zone 0's compute arrays (positional, exactly
    as the legacy single-zone path) plus the full ``computes`` list via the
    ``zones`` kwarg. For a single zone this is byte-for-byte identical to the
    old single-zone dialog (no selector). The user reviews each zone via the
    active-zone selector; ``dlg.zone_results()`` collects every zone's edits
    (untouched zones keep their computed accept/reject defaults).
    """
    r0 = computes[0]
    single_indices = list(range(len(r0["pulse_peak_locs"])))
    filter_config = (tpz0 or {}).get("filter_config")
    edge_method = (tpz0 or {}).get("edge_method", "peaks")

    dlg = _PulseFeatureReviewDialog(
        r0["data"],
        pulse_indices,
        r0["filtered_pulses"],
        r0["rect_pulses"],
        r0["pulse_peak_locs"],
        r0["pulse_diffs"],
        r0["thresholds_used"],
        r0["counts_used"],
        r0["sequence_matches"],
        r0["template_seq"],
        single_indices,
        fs,
        pulse_excl_ranges=r0["pulse_excl_ranges_list"],
        pulse_key_peaks=r0["pulse_key_peaks_list"],
        pulse_key_peak_pols=r0["pulse_key_peak_pols_list"],
        pulse_key_peak_seqs=r0["pulse_key_peak_seqs_list"],
        template_key_seq_indices=r0["template_key_seq_indices"],
        template_key_polarities=r0["template_key_polarities"],
        trendline=r0.get("trendline"),
        filter_config=filter_config,
        pad_length=r0["pad_length"],
        pad_method=r0["pad_method"],
        rect_pulses_mean=r0["rect_pulses_mean"],
        zones=computes,
        active_zone=0,
        edge_method=edge_method,
        ds_factor=ds_factor,
        file_names=file_names,
        file_boundaries=file_boundaries,
    )

    # Load saved settings if provided
    if settings_file:
        dlg.load_state_from_file(settings_file)

    dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Batch Processing without confirming.")

    return dlg.zone_results()


# ---------------------------------------------------------------------------
# Per-pulse compute (no Qt) — extracted for reuse / future multi-zone support
# ---------------------------------------------------------------------------


def _compute_zone(
    data,
    sample_rate,
    trendline,
    pulse_indices,
    threshold=None,
    edge_margin=None,
    peak_counts=None,
    peak_sequence=None,
    filter_padding=None,
    filter_config=None,
    key_peaks=None,
    key_peak_properties=None,
    peak_locations=None,
    peak_locations_apex=None,
    edge_method="peaks",
    template_original=None,
    exclusion_zones=None,
    end_zones=None,
    key_peak_seq_indices_in=None,
    exclusion_zone_seq_bounds_in=None,
):
    """Per-pulse compute for a single zone (pure; no Qt/dialog).

    Takes the normalized 1-D ``data``, ``sample_rate``, ``trendline`` and the
    ``pulse_indices`` (Nx2) along with the template/threshold params, and
    returns a dict containing every per-pulse array and derived
    template-sequence / key-peak value consumed by
    ``_PulseFeatureReviewDialog``.
    """
    from utils.app_logger import logger

    fs = float(sample_rate)
    pulse_indices = np.atleast_2d(np.asarray(pulse_indices, dtype=int))
    n_pulses = len(pulse_indices)

    # --- unpack template params ---------------------------------------------
    base_pos_thresh = threshold.get("upper", 0) if threshold else 0
    base_neg_thresh = threshold.get("lower", 0) if threshold else 0

    target_pos = peak_counts.get("positive", 0) if peak_counts else 0
    target_neg = peak_counts.get("negative", 0) if peak_counts else 0

    pad_length = filter_padding.get("pad_length", 0) if filter_padding else 0
    pad_method = (
        filter_padding.get("method", "replicate") if filter_padding else "replicate"
    )

    first_peak_prop = 0.0
    last_peak_prop = 0.0
    if edge_margin and isinstance(edge_margin, dict):
        first_peak_prop = edge_margin.get("first_peak_prop", 0.0)
        last_peak_prop = edge_margin.get("last_peak_prop", 0.0)

    # Resolve template sequence
    template_seq = _resolve_sequence(peak_sequence)

    # Compute key peak positions within the template sequence.
    # Prefer matching saved key-peak absolute positions to peak_locations:
    # peak_locations and peak_sequence (template_seq) are both post-exclusion
    # and share the same order, so this yields the correct index into the
    # sequence. The saved key_peak_seq_indices are NOT used here — they are
    # computed against the raw (pre-exclusion) peak list for template reload
    # and would land on the wrong peaks in the post-exclusion sequence.
    template_key_seq_indices = []
    template_key_polarities = []
    _kpp = key_peak_properties if key_peak_properties else []

    def _polarity_for(j, si):
        if j < len(_kpp) and _kpp[j] in ("min", "max", "regular"):
            return _kpp[j]
        if template_seq is not None and 0 <= si < len(template_seq):
            return "max" if template_seq[si] == 1 else "min"
        return "regular"

    # key_peaks are APEX positions. Match them against the apex peak list
    # (identical order/count to peak_locations) so FWHM-refined edge_method
    # doesn't shift peak_locations out from under the tolerance. Fall back to
    # peak_locations for settings saved before peak_locations_apex existed.
    _pl_len = len(peak_locations) if peak_locations is not None else 0
    match_locs = (
        peak_locations_apex
        if peak_locations_apex is not None and len(peak_locations_apex) == _pl_len
        else peak_locations
    )
    if (
        key_peaks is not None
        and len(key_peaks) > 0
        and match_locs is not None
        and len(match_locs) > 0
    ):
        # Match each key-peak position to its index in the post-exclusion list
        pl = np.asarray(match_locs)
        for j, kp in enumerate(key_peaks):
            dists = np.abs(pl - kp)
            closest = int(np.argmin(dists))
            if dists[closest] <= max(5, len(pl) * 0.01):
                template_key_seq_indices.append(closest)
                template_key_polarities.append(_polarity_for(j, closest))
    elif key_peak_seq_indices_in and len(key_peak_seq_indices_in) > 0:
        # Fallback only when positions are unavailable.
        template_key_seq_indices = [int(si) for si in key_peak_seq_indices_in]
        for j, si in enumerate(template_key_seq_indices):
            template_key_polarities.append(_polarity_for(j, si))

    # No proportional key peak positions needed — detection is purely
    # sequence-based (find min/max by amplitude, then count peaks to find others).

    logger.debug("=== Extracting Single Pulse Features ===")
    logger.debug(f"Template target peaks: +{target_pos} / -{target_neg}")
    logger.debug(f"Padding: {pad_length} samples ({pad_method} method)")
    logger.debug(
        f"Raw inputs: key_peaks={key_peaks}, key_peak_properties={key_peak_properties}"
    )
    if peak_locations is not None:
        logger.debug(
            f"Peak locations ({len(peak_locations)}): {np.asarray(peak_locations).tolist()}"
        )
    if exclusion_zones:
        logger.debug(f"Exclusion zones (raw): {exclusion_zones}")
    if template_key_seq_indices:
        logger.debug(f"Key peak seq indices: {template_key_seq_indices}")
        logger.debug(f"Key peak polarities: {template_key_polarities}")
        if template_seq is not None:
            logger.debug(
                f"Template sequence: {template_seq.tolist() if hasattr(template_seq, 'tolist') else template_seq}"
            )
    else:
        logger.debug(
            "No key peak seq indices resolved — check key_peaks vs peak_locations matching"
        )
    logger.debug(f"Processing {n_pulses} pulses...")

    # --- per-pulse processing ------------------------------------------------
    pulse_peak_locs = [None] * n_pulses
    rect_pulses = [None] * n_pulses
    rect_pulses_mean = [None] * n_pulses
    filtered_pulses = [None] * n_pulses
    pulse_diffs = [None] * n_pulses
    thresholds_used = [None] * n_pulses
    counts_used = [None] * n_pulses
    pulse_excl_ranges_list = [[] for _ in range(n_pulses)]
    pulse_key_peaks_list = [
        [] for _ in range(n_pulses)
    ]  # per-pulse key peak local indices
    pulse_key_peak_pols_list = [
        [] for _ in range(n_pulses)
    ]  # per-pulse key peak polarities
    pulse_key_peak_seqs_list = [
        [] for _ in range(n_pulses)
    ]  # per-pulse key peak template seq indices (for debug overlay)
    sequence_matches = np.ones(n_pulses, dtype=int)  # 0=reject, 1=accept, 2=warning
    pulse_start_indices = pulse_indices[:, 0].copy()

    for i in range(n_pulses):
        s_idx = max(0, int(pulse_indices[i, 0]))
        e_idx = min(len(data) - 1, int(pulse_indices[i, 1]))
        segment = data[s_idx : e_idx + 1].copy()
        pulse_duration = len(segment)

        if pulse_duration < 3:
            pulse_peak_locs[i] = np.array([], dtype=int)
            filtered_pulses[i] = segment
            pulse_diffs[i] = np.array([])
            rect_pulses[i] = segment
            thresholds_used[i] = (base_pos_thresh, base_neg_thresh)
            counts_used[i] = (0, 0)
            continue

        # Step 1: Pad signal
        actual_pad = min(pad_length, pulse_duration // 4)
        seg_padded = _pad_signal(segment, actual_pad, pad_method)

        # Step 2: Apply filters to padded signal
        seg_filt_padded = _apply_filter_stack(seg_padded, fs, filter_config)

        # Step 3: Remove padding → filtered pulse
        if actual_pad > 0:
            seg_filtered = seg_filt_padded[actual_pad : actual_pad + pulse_duration]
        else:
            seg_filtered = seg_filt_padded
        filtered_pulses[i] = seg_filtered

        # Step 4: Derivative on padded filtered, then re-filter derivative
        dseg_padded = np.diff(seg_filt_padded)
        dseg_filt_padded = _apply_filter_stack(dseg_padded, fs, filter_config)

        # Remove derivative padding
        if actual_pad > 0:
            dseg_filtered = dseg_filt_padded[
                actual_pad : actual_pad + pulse_duration - 1
            ]
        else:
            dseg_filtered = dseg_filt_padded
        pulse_diffs[i] = dseg_filtered

        # --- Step 9 (new): anchor + region peak matching --------------------
        # Phase 1: pick the *strongest* positive/negative peaks as the
        #          template-requested min/max anchors (no position
        #          constraint — they're whatever's strongest).
        # Phase 2: for each region bounded by anchors, match the template's
        #          per-region count + order.
        # Exclusion zones are no longer needed; the per-region check
        # subsumes their role.
        pulse_excl_ranges_list[i] = []

        # All candidate peaks of each polarity (no threshold — anchor
        # selection is amplitude-driven so we need the full set).
        kp_pos_locs, _ = find_peaks(dseg_filtered, height=None)
        kp_neg_locs, _ = find_peaks(-dseg_filtered, height=None)

        n_tmpl_min = sum(1 for p in template_key_polarities if p == "min")
        n_tmpl_max = sum(1 for p in template_key_polarities if p == "max")

        min_tmpl_seq = None
        max_tmpl_seq = None
        for _ki, _pol in enumerate(template_key_polarities):
            if _ki >= len(template_key_seq_indices):
                continue
            if _pol == "min":
                min_tmpl_seq = template_key_seq_indices[_ki]
            elif _pol == "max":
                max_tmpl_seq = template_key_seq_indices[_ki]

        # Phase 1: anchor selection.
        # Default: pick the globally strongest peak of each requested
        # polarity (worked well across the dataset).
        # Refinement: when both anchors are required and the global picks
        # land *degenerately close* (a noise blip near the real anchor),
        # keep the stronger of the two and re-pick the weaker one from
        # candidates farther away — using the template's own min↔max
        # spacing as the reference for "far enough".
        min_anchor = None
        max_anchor = None
        order_ok = True

        if n_tmpl_max == 1 and len(kp_pos_locs) > 0:
            max_anchor = int(kp_pos_locs[int(np.argmax(dseg_filtered[kp_pos_locs]))])
        if n_tmpl_min == 1 and len(kp_neg_locs) > 0:
            min_anchor = int(kp_neg_locs[int(np.argmin(dseg_filtered[kp_neg_locs]))])

        # Compute the per-pulse "active width" — span between this pulse's
        # first and last significant peak. Significance is gated on the
        # template's own thresholds, so noise-only candidates outside the
        # real pulse don't stretch the width. Used to scale the template's
        # min↔max ratio into the expected per-pulse min↔max distance.
        sig_peaks = []
        for _p in kp_pos_locs:
            if dseg_filtered[int(_p)] >= base_pos_thresh:
                sig_peaks.append(int(_p))
        for _p in kp_neg_locs:
            if dseg_filtered[int(_p)] <= base_neg_thresh:
                sig_peaks.append(int(_p))
        if len(sig_peaks) >= 2:
            per_pulse_width = max(sig_peaks) - min(sig_peaks)
        else:
            per_pulse_width = pulse_duration

        # Expected min↔max sample distance: template ratio × per-pulse width.
        tpl_locs = (
            np.asarray(peak_locations, dtype=float)
            if peak_locations is not None and len(peak_locations) > 0
            else None
        )
        expected_dist = None
        if (
            n_tmpl_min == 1
            and n_tmpl_max == 1
            and min_tmpl_seq is not None
            and max_tmpl_seq is not None
            and tpl_locs is not None
            and min_tmpl_seq < len(tpl_locs)
            and max_tmpl_seq < len(tpl_locs)
        ):
            tpl_span = float(tpl_locs[-1]) - float(tpl_locs[0])
            if tpl_span > 0:
                tpl_dist = abs(
                    float(tpl_locs[max_tmpl_seq]) - float(tpl_locs[min_tmpl_seq])
                )
                expected_dist = tpl_dist / tpl_span * per_pulse_width

        # Degeneracy refinement: if the global picks land within 30% of
        # the expected distance, re-pick the weaker-magnitude anchor from
        # candidates that sit outside that near-zone.
        #
        # Only when the per-pulse width is RELIABLE (>= 2 peaks cleared the
        # template threshold). For a weak pulse where nothing clears the
        # threshold, per_pulse_width falls back to the full pulse duration,
        # which inflates expected_dist and would wrongly relocate the
        # legitimately-close +/- anchors of a single spike to a far noise
        # peak. In that case keep the global-strongest anchors as-is.
        if (
            min_anchor is not None
            and max_anchor is not None
            and expected_dist is not None
            and expected_dist > 0
            and len(sig_peaks) >= 2
        ):
            actual_dist = abs(max_anchor - min_anchor)
            min_gap = 0.3 * expected_dist
            if actual_dist < min_gap:
                neg_mag = abs(float(dseg_filtered[min_anchor]))
                pos_mag = float(dseg_filtered[max_anchor])
                if pos_mag <= neg_mag:
                    # min is real; re-pick max away from min.
                    cands = [
                        int(p)
                        for p in kp_pos_locs
                        if abs(int(p) - min_anchor) >= min_gap
                    ]
                    if cands:
                        max_anchor = max(cands, key=lambda p: dseg_filtered[p])
                else:
                    # max is real; re-pick min away from max.
                    cands = [
                        int(n)
                        for n in kp_neg_locs
                        if abs(int(n) - max_anchor) >= min_gap
                    ]
                    if cands:
                        min_anchor = min(cands, key=lambda n: dseg_filtered[n])

        # Anchor order check: when both anchors exist, their pulse-sample
        # order must match the template's seq-index order.
        if (
            n_tmpl_min == 1
            and n_tmpl_max == 1
            and min_anchor is not None
            and max_anchor is not None
            and min_tmpl_seq is not None
            and max_tmpl_seq is not None
        ):
            tmpl_min_first = min_tmpl_seq < max_tmpl_seq
            pulse_min_first = min_anchor < max_anchor
            if tmpl_min_first != pulse_min_first:
                order_ok = False

        anchor_records = []
        if min_anchor is not None and min_tmpl_seq is not None:
            anchor_records.append((int(min_tmpl_seq), int(min_anchor), "min"))
        if max_anchor is not None and max_tmpl_seq is not None:
            anchor_records.append((int(max_tmpl_seq), int(max_anchor), "max"))
        anchor_records.sort(key=lambda r: r[0])

        pulse_key_peak_indices = [r[1] for r in anchor_records]
        _pulse_kp_pols = [r[2] for r in anchor_records]
        _pulse_kp_seqs = [r[0] for r in anchor_records]

        # Phase 2: region-aware regular peak matching.
        # Template-side regions are bounded by anchor seq indices.
        tmpl_anchor_seqs = sorted([r[0] for r in anchor_records])
        template_region_polarity = []
        if template_seq is not None and len(template_seq) > 0:
            t_bounds = [-1] + tmpl_anchor_seqs + [len(template_seq)]
            for r_idx in range(len(t_bounds) - 1):
                rs = t_bounds[r_idx] + 1
                re_ = t_bounds[r_idx + 1]
                template_region_polarity.append(
                    ["+" if template_seq[ii] == 1 else "-" for ii in range(rs, re_)]
                )
        else:
            template_region_polarity = [[]]

        # Pulse-side regions are bounded by anchor sample positions.
        pulse_anchor_sorted = sorted([r[1] for r in anchor_records])
        pulse_bounds = [0] + pulse_anchor_sorted + [pulse_duration]

        selected_indices = list(pulse_key_peak_indices)
        partial_match = False
        region_order_mismatch = False

        for r_idx in range(len(pulse_bounds) - 1):
            if r_idx >= len(template_region_polarity):
                continue
            region_template = template_region_polarity[r_idx]
            if not region_template:
                continue
            n_pos_expected = sum(1 for c in region_template if c == "+")
            n_neg_expected = sum(1 for c in region_template if c == "-")
            if n_pos_expected == 0 and n_neg_expected == 0:
                continue

            r_start = pulse_bounds[r_idx]
            r_end = pulse_bounds[r_idx + 1]
            region_pos = [
                int(l)
                for l in kp_pos_locs
                if r_start < int(l) < r_end and int(l) not in selected_indices
            ]
            region_neg = [
                int(l)
                for l in kp_neg_locs
                if r_start < int(l) < r_end and int(l) not in selected_indices
            ]

            region_pos_top = sorted(
                region_pos, key=lambda l: dseg_filtered[l], reverse=True
            )[:n_pos_expected]
            region_neg_top = sorted(
                region_neg, key=lambda l: abs(dseg_filtered[l]), reverse=True
            )[:n_neg_expected]

            if (
                len(region_pos_top) < n_pos_expected
                or len(region_neg_top) < n_neg_expected
            ):
                partial_match = True

            picks = sorted(region_pos_top + region_neg_top)
            region_pulse_pol = ["+" if dseg_filtered[l] >= 0 else "-" for l in picks]
            cmp_len = min(len(region_pulse_pol), len(region_template))
            if region_pulse_pol[:cmp_len] != region_template[:cmp_len]:
                region_order_mismatch = True

            selected_indices.extend(picks)

        all_idx = np.sort(np.unique(np.array(selected_indices, dtype=int)))
        # Apex positions stay in pulse_peak_locs (derivative markers / signs /
        # sequence / manual edits work in apex space); the FWHM edge method is
        # applied only where segment boundaries are consumed — the rect below
        # and the exported pulse_peak_locations (see _refine_pulse_bounds).
        bound_idx = refine_interior(seg_filtered, all_idx, edge_method)
        global_idx = all_idx + s_idx
        pulse_peak_locs[i] = global_idx

        total_pos = sum(
            1
            for idx in all_idx
            if 0 <= idx < len(dseg_filtered) and dseg_filtered[idx] >= 0
        )
        total_neg = len(all_idx) - total_pos
        thresholds_used[i] = (base_pos_thresh, base_neg_thresh)
        counts_used[i] = (total_pos, total_neg)
        pulse_key_peaks_list[i] = list(pulse_key_peak_indices)
        pulse_key_peak_pols_list[i] = list(_pulse_kp_pols)
        pulse_key_peak_seqs_list[i] = list(_pulse_kp_seqs)

        # Classification:
        #   red    — anchor or region order doesn't match the template
        #   yellow — partial: counts short in some region
        #   green  — full count + order match
        if (not order_ok) or region_order_mismatch:
            sequence_matches[i] = 0
        elif partial_match:
            sequence_matches[i] = 2
        else:
            sequence_matches[i] = 1

        # Step 8: Rectangularize directly on the filtered signal (both methods)
        rect_pulses[i], rect_pulses_mean[i] = _rectangularize_filtered(
            seg_filtered, bound_idx
        )

        # (Sequence classification is now set in Step 9 alongside peak
        # matching — no separate Step 11 needed.)

        pt, nt = thresholds_used[i]
        logger.debug(
            f"[Pulse {i + 1:3d}] | "
            f"Peaks: +{total_pos} / -{total_neg} | "
            f"Thresholds: upper={pt:.3e}, lower={nt:.3e}"
        )

    # --- Post-processing: flag shape outliers by segment proportions ----------
    _flag_shape_outliers(sequence_matches, pulse_peak_locs, pulse_indices, template_seq)

    logger.info("Feature extraction complete.")

    return {
        "data": data,
        "filtered_pulses": filtered_pulses,
        "rect_pulses": rect_pulses,
        "rect_pulses_mean": rect_pulses_mean,
        "pulse_peak_locs": pulse_peak_locs,
        "pulse_diffs": pulse_diffs,
        "thresholds_used": thresholds_used,
        "counts_used": counts_used,
        "sequence_matches": sequence_matches,
        "template_seq": template_seq,
        "pulse_excl_ranges_list": pulse_excl_ranges_list,
        "pulse_key_peaks_list": pulse_key_peaks_list,
        "pulse_key_peak_pols_list": pulse_key_peak_pols_list,
        "pulse_key_peak_seqs_list": pulse_key_peak_seqs_list,
        "template_key_seq_indices": template_key_seq_indices,
        "template_key_polarities": template_key_polarities,
        "pad_length": pad_length,
        "pad_method": pad_method,
        "pulse_start_indices": pulse_start_indices,
        "edge_method": edge_method,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flag_shape_outliers(
    sequence_matches, pulse_peak_locs, pulse_indices, template_seq
):
    """Downgrade accepted pulses (``sequence_matches`` 1 -> 2 "warning") whose
    inner segment-width proportions deviate from the population median (robust
    MAD, 4x with a 0.08 floor); requires >= 5 comparable pulses. Mutates
    ``sequence_matches`` in place; only 1 -> 2 (never touches rejects/0).

    This is CROSS-PULSE: the classification depends on the full set of pulses
    passed in. Extracted from _compute_zone so the coincident incremental
    recompute can re-run it on the merged population (a subset call would use a
    different median and skip the < 5 case). See
    ``_CoincidentReviewDialog._rebuild_compute_incremental``.
    """
    from utils.app_logger import logger

    pulse_indices = np.atleast_2d(np.asarray(pulse_indices, dtype=int))
    n_pulses = len(sequence_matches)
    n_inner_segs = (
        (len(template_seq) - 1)
        if template_seq is not None and len(template_seq) > 1
        else 0
    )
    if n_inner_segs <= 0:
        return
    all_proportions = []  # list of (pulse_idx, proportions_array)
    for i in range(n_pulses):
        if sequence_matches[i] == 0:
            continue
        locs = pulse_peak_locs[i]
        if locs is None or len(locs) == 0:
            continue
        s_idx = int(pulse_indices[i, 0])
        local_peaks = np.sort(np.asarray(locs) - s_idx)
        valid = local_peaks[(local_peaks >= 0)]
        if len(valid) < 2:
            continue
        # Inner segment widths: between consecutive peaks
        inner_widths = np.diff(valid).astype(float)
        pulse_width = float(valid[-1] - valid[0])
        if pulse_width <= 0:
            continue
        props = inner_widths / pulse_width
        if len(props) == n_inner_segs:
            all_proportions.append((i, props))

    if len(all_proportions) >= 5:
        # Robust population statistics: median + MAD-based threshold.
        prop_matrix = np.array([p for _, p in all_proportions])
        median_props = np.median(prop_matrix, axis=0)
        mad = np.median(np.abs(prop_matrix - median_props), axis=0)
        threshold_dev = np.maximum(mad * 4.0, 0.08)

        for idx, props in all_proportions:
            deviation = np.abs(props - median_props)
            if np.any(deviation > threshold_dev):
                if sequence_matches[idx] == 1:  # only downgrade accepted
                    sequence_matches[idx] = 2  # warning
                    logger.info(
                        f"Pulse {idx + 1}: shape WARNING "
                        f"(segment proportions deviate: "
                        f"max dev={np.max(deviation):.3f}, "
                        f"threshold={threshold_dev[np.argmax(deviation)]:.3f})"
                    )


def _resolve_sequence(peak_sequence):
    """Resolve template peak sequence from various input formats."""
    if peak_sequence is None:
        return None
    if isinstance(peak_sequence, dict):
        seq = peak_sequence.get("sequence", peak_sequence.get("peak_sequence"))
        if seq is not None:
            return np.asarray(seq, dtype=int)
        return None
    return np.asarray(peak_sequence, dtype=int)


def _pad_signal(segment, pad_len, method):
    """Pad signal with replicate or mirror method."""
    if pad_len <= 0:
        return segment.copy()
    n = len(segment)
    if method == "replicate":
        pad_start = np.full(pad_len, segment[0])
        pad_end = np.full(pad_len, segment[-1])
    elif method == "mirror":
        pad_start = segment[: min(pad_len, n)][::-1]
        pad_end = segment[max(0, n - pad_len) :][::-1]
    else:
        return segment.copy()
    return np.concatenate([pad_start, segment, pad_end])


def _apply_filter_stack(sig, fs, filter_config):
    """Apply the complete filter stack from filter_config to a signal.

    Uses the same _apply_single_filter from filter_ui.py to ensure identical
    filtering behavior (SOS form, reflect-padding, proper notch handling).
    """
    if not filter_config:
        return sig.copy()

    # Prefer full filter_info list (contains all per-filter params)
    filter_info = filter_config.get("filter_info")
    if filter_info and len(filter_info) > 0:
        from processing.filter_ui import _apply_single_filter

        filtered = sig.copy()
        for fi in filter_info:
            try:
                filtered = _apply_single_filter(
                    filtered,
                    fs,
                    fi.get("type", "lowpass"),
                    fi.get("cutoff1", 0),
                    fi.get("cutoff2", 0),
                    fi.get("notch_mode", "freqBW"),
                    fi.get("notch_bw", 1.0),
                )
            except Exception:
                pass
        return filtered

    # Fallback: reconstruct from legacy split fields
    filter_types = filter_config.get("filter_types", [])
    if not filter_types:
        return sig.copy()

    from processing.filter_ui import _apply_single_filter

    filtered = sig.copy()
    cutoffs_list = filter_config.get("cutoff_frequencies", [])
    notch_modes = filter_config.get("notch_modes", [])
    notch_bws = filter_config.get("notch_bws", [])
    n_filters = filter_config.get("num_filters", len(filter_types))

    for f_idx in range(n_filters):
        if f_idx >= len(filter_types):
            break
        ftype = filter_types[f_idx]
        cutoffs = cutoffs_list[f_idx] if f_idx < len(cutoffs_list) else [0, 0]
        notch_mode = notch_modes[f_idx] if f_idx < len(notch_modes) else "freqBW"
        notch_bw = notch_bws[f_idx] if f_idx < len(notch_bws) else 1.0
        try:
            filtered = _apply_single_filter(
                filtered,
                fs,
                ftype,
                cutoffs[0] if len(cutoffs) > 0 else 0,
                cutoffs[1] if len(cutoffs) > 1 else 0,
                notch_mode,
                notch_bw,
            )
        except Exception:
            pass

    return filtered


def _apply_lowpass(sig, fs, cutoff):
    """Extra Butterworth low-pass on top of the template filter stack."""
    from processing.filter_ui import _apply_single_filter

    try:
        return _apply_single_filter(sig, fs, "lowpass", cutoff, 0, "freqBW", 1.0)
    except Exception:
        return sig


def _empty_result():
    """Return empty result dict."""
    return {
        "pulse_peak_locations": [],
        "rectangularized_pulses": [],
        "acceptance_id": np.array([], dtype=bool),
        "pulse_start_indices": np.array([], dtype=int),
    }


# ---------------------------------------------------------------------------
# Interactive Review Dialog
# ---------------------------------------------------------------------------

_TOOL_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #444; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #aaa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #555; color: white; border-color: #555; }"
)
_EXPAND_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #0066aa; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #0066aa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #0066aa; color: white; }"
)
_TRIM_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #cc4400; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #cc4400; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #cc4400; color: white; }"
)
_CANCEL_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #888; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #aaa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #888; color: white; border-color: #888; }"
)
_PEAK_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,220); color: #6600aa; "
    "font-weight: bold; font-size: 9px; border-radius: 3px; border: 1px solid #6600aa; "
    "padding: 1px 5px; }"
    "QPushButton:hover { background: #6600aa; color: white; }"
)


class _LineOverlay(QWidget):
    """Lightweight Qt overlay that draws a vertical dashed line."""

    def __init__(self, parent=None, color=QColor(220, 0, 0)):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._x = -1
        self._y0 = 0
        self._y1 = 0
        self._color = color
        self.hide()

    def set_line(self, x, y0, y1):
        self._x = x
        self._y0 = y0
        self._y1 = y1
        self.update()

    def paintEvent(self, event):
        if self._x < 0:
            return
        p = QPainter(self)
        pen = QPen(self._color, 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(self._x, self._y0, self._x, self._y1)
        p.end()


class _CircleOverlay(QWidget):
    """Lightweight Qt overlay that draws a circle following the cursor."""

    RADIUS = 25  # search radius in physical (matplotlib) pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._cx = -1
        self._cy = -1
        self._has_peak = False  # whether a peak is inside the circle
        self._draw_radius = self.RADIUS  # logical-pixel radius for drawing
        self.hide()

    def set_pos(self, cx, cy, has_peak=False, draw_radius=None):
        self._cx = cx
        self._cy = cy
        self._has_peak = has_peak
        if draw_radius is not None:
            self._draw_radius = draw_radius
        r = self._draw_radius + 4
        self.setGeometry(cx - r, cy - r, 2 * r, 2 * r)
        self.update()

    def paintEvent(self, event):
        if self._cx < 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._draw_radius
        center_x = self.width() // 2
        center_y = self.height() // 2
        if self._has_peak:
            color = QColor(255, 60, 60, 180)
        else:
            color = QColor(100, 0, 170, 140)
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center_x - r, center_y - r, 2 * r, 2 * r)
        p.end()


class _PulseFeatureReviewDialog(QDialog):
    """Paginated pulse feature review dialog matching MATLAB UI.

    Shows 3 pulses per page: left plot (signal + rectangularized),
    right plot (derivative + thresholds + peaks), classification table.
    """

    def __init__(
        self,
        data,
        pulse_indices,
        filtered_pulses,
        rect_pulses,
        pulse_peak_locs,
        pulse_diffs,
        thresholds_used,
        counts_used,
        sequence_matches,
        template_seq,
        single_indices,
        fs,
        pulse_excl_ranges=None,
        pulse_key_peaks=None,
        pulse_key_peak_pols=None,
        pulse_key_peak_seqs=None,
        template_key_seq_indices=None,
        template_key_polarities=None,
        trendline=None,
        filter_config=None,
        pad_length=0,
        pad_method="replicate",
        rect_pulses_mean=None,
        parent=None,
        zones=None,
        active_zone=0,
        edge_method="peaks",
        ds_factor=None,
        file_names=None,
        file_boundaries=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Pulse Feature Review")
        self.resize(1500, 900)

        from utils.dialog_style import apply_dialog_style

        apply_dialog_style(self)

        self._edge_method = edge_method if edge_method in ("peaks", "fwhm") else "peaks"
        self._data = data
        self._pulse_indices = pulse_indices
        self._filtered = filtered_pulses
        self._rect = rect_pulses  # median
        self._rect_mean = rect_pulses_mean or [None] * len(rect_pulses)
        self._peak_locs = pulse_peak_locs
        self._diffs = pulse_diffs
        self._thresholds = thresholds_used
        self._counts = counts_used
        self._seq_matches = sequence_matches
        self._template_seq = template_seq
        self._template_key_seq_indices = set(template_key_seq_indices or [])
        # Build map: seq_index -> polarity for key peaks
        _tksi = template_key_seq_indices or []
        _tkp = template_key_polarities or []
        self._template_key_pol_map = {}
        for j, si in enumerate(_tksi):
            self._template_key_pol_map[si] = _tkp[j] if j < len(_tkp) else "regular"
        self._pulse_key_peaks = pulse_key_peaks or [
            [] for _ in range(len(pulse_peak_locs))
        ]
        self._pulse_key_peak_pols = pulse_key_peak_pols or [
            [] for _ in range(len(pulse_peak_locs))
        ]
        self._pulse_key_peak_seqs = pulse_key_peak_seqs or [
            [] for _ in range(len(pulse_peak_locs))
        ]
        self._single_indices = single_indices
        self._fs = fs
        self._pulse_excl_ranges = pulse_excl_ranges or [
            [] for _ in range(len(pulse_peak_locs))
        ]
        self._trendline = trendline
        self._ds_factor = ds_factor
        self._filter_config = filter_config
        self._pad_length = pad_length
        self._pad_method = pad_method
        # Per-file LP filter: source files of the (possibly concatenated)
        # signal, their sample offsets, and one independent LP setting each.
        self._file_names = [str(n) for n in file_names] if file_names else ["input"]
        if (
            file_boundaries is not None
            and len(file_boundaries) == len(self._file_names) + 1
        ):
            self._file_boundaries = np.asarray(file_boundaries, dtype=int)
        else:
            self._file_boundaries = np.array([0, len(data)], dtype=int)
        default_cut = round(max(float(fs) / 4.0, 1e-3), 3)
        self._file_filters = [
            {"enabled": False, "cutoff": default_cut} for _ in self._file_names
        ]
        self._lp_loading = False
        self._detrend = True
        self._rect_method = "mean"  # "median" or "mean"
        self._n_single = len(single_indices)
        self._page_size = 3
        self._total_pages = max(1, int(np.ceil(self._n_single / self._page_size)))
        self._page = 0

        # Acceptance: default from sequence match
        # 0 = reject, 1 = accept, 2 = warning (shape mismatch)
        self.acceptance = np.array(
            [sequence_matches[idx] for idx in single_indices], dtype=int
        )
        self.adjust_flags = np.zeros(self._n_single, dtype=bool)
        self.confirmed = False

        # Axes tracking for hover/click
        self._axes_left = []
        self._axes_right = []

        # Tool overlay state
        self._tool_overlay = None
        self._right_tool_overlay = None
        self._tool_sub_overlay = None
        self._hover_row = None
        self._active_tool = None  # None, 'trim', 'expand'
        self._active_tool_row = None
        self._trim_sample = None
        self._trim_line_qx = None
        self._peak_adjust_mode = False
        self._peak_adjust_row = None  # which row (0-2) peak adjust is active on
        self._mpl_cids = []

        # --- Multi-zone state ----------------------------------------------
        # Single zone (zones None or 1): degenerate — no selector, behavior
        # byte-for-byte identical to the legacy single-signal dialog. The
        # plain self._* attributes set above already point at zone 0.
        self._zones = self._build_zone_states(zones)
        self._active_zone = active_zone if 0 <= active_zone < len(self._zones) else 0
        # Overlay every zone's signal in the LEFT plots (right plots stay on the
        # active zone). Mirrors the Detection Review "All Zones" toggle; on by
        # default for multi-zone data.
        self._overlay_all_zones = len(self._zones) > 1

        self._build_ui()
        self._render_page()
        # Pristine snapshot for Restart — the default state.  Settings load
        # only from the settingsFile port (applied by the entry function);
        # an empty port leaves these defaults in place.
        self._initial_state = copy.deepcopy(self._get_state())

    # ---- Multi-zone state management ----------------------------------------

    # Per-zone attributes swapped on zone change. Includes the fixed compute
    # arrays AND the editable review state (acceptance, peaks, rect, indices).
    _ZONE_SWAP_ATTRS = (
        "_data",
        "_pulse_indices",
        "_filtered",
        "_rect",
        "_rect_mean",
        "_peak_locs",
        "_diffs",
        "_thresholds",
        "_counts",
        "_seq_matches",
        "_template_seq",
        "_template_key_seq_indices",
        "_template_key_pol_map",
        "_pulse_key_peaks",
        "_pulse_key_peak_pols",
        "_pulse_key_peak_seqs",
        "_single_indices",
        "_pulse_excl_ranges",
        "_trendline",
        "acceptance",
        "adjust_flags",
        "_n_single",
        "_total_pages",
        "_edge_method",
    )

    def _build_zone_states(self, zones):
        """Build the per-zone state list.

        ``zones`` is the entry's ``computes`` list (one ``_compute_zone`` dict
        per zone, each carrying ``"data"`` and a ``"trendline"`` per zone). The
        ACTIVE zone (zone 0 at construction) is represented by the live
        ``self._*`` attrs already set in ``__init__``; its snapshot is captured
        from those. Other zones are derived from their compute dict.

        For a single zone, returns a one-element list holding the current live
        state — no selector is built and switching never happens.
        """
        if not zones or len(zones) <= 1:
            return [self._snapshot_current_zone()]

        states = []
        for k, z in enumerate(zones):
            if k == self._init_active_zone_idx(zones):
                # The live attrs already reflect this zone (built from it).
                states.append(self._snapshot_current_zone())
            else:
                states.append(self._zone_state_from_compute(z))
        return states

    @staticmethod
    def _init_active_zone_idx(zones):
        # The dialog is always constructed from zone 0's arrays (the entry
        # passes computes[0] positionally), so the live attrs match zone 0.
        return 0

    def _snapshot_current_zone(self):
        """Capture the live self._* attrs into a per-zone dict (copies)."""
        snap = {}
        for a in self._ZONE_SWAP_ATTRS:
            v = getattr(self, a)
            snap[a] = v.copy() if isinstance(v, np.ndarray) else v
        return snap

    def _zone_state_from_compute(self, z):
        """Build a per-zone state dict from a ``_compute_zone`` result dict.

        The compute dict carries the per-pulse arrays + ``data``. The entry
        injects ``pulse_indices`` (shared Nx2) and ``trendline`` (per zone)
        into the dict before passing it here.
        """
        n = len(z["pulse_peak_locs"])
        single_indices = list(range(n))
        seq = z["sequence_matches"]
        tksi = z.get("template_key_seq_indices") or []
        tkp = z.get("template_key_polarities") or []
        pol_map = {}
        for j, si in enumerate(tksi):
            pol_map[si] = tkp[j] if j < len(tkp) else "regular"
        rect_mean = z.get("rect_pulses_mean") or [None] * len(z["rect_pulses"])
        n_single = len(single_indices)
        return {
            "_data": z["data"],
            "_pulse_indices": np.atleast_2d(
                np.asarray(z["pulse_indices"], dtype=int)
            ).copy(),
            "_filtered": list(z["filtered_pulses"]),
            "_rect": list(z["rect_pulses"]),
            "_rect_mean": list(rect_mean),
            "_peak_locs": list(z["pulse_peak_locs"]),
            "_diffs": list(z["pulse_diffs"]),
            "_thresholds": z["thresholds_used"],
            "_counts": z["counts_used"],
            "_seq_matches": seq,
            "_template_seq": z["template_seq"],
            "_template_key_seq_indices": set(tksi),
            "_template_key_pol_map": pol_map,
            "_pulse_key_peaks": z.get("pulse_key_peaks_list") or [[] for _ in range(n)],
            "_pulse_key_peak_pols": z.get("pulse_key_peak_pols_list")
            or [[] for _ in range(n)],
            "_pulse_key_peak_seqs": z.get("pulse_key_peak_seqs_list")
            or [[] for _ in range(n)],
            "_single_indices": single_indices,
            "_pulse_excl_ranges": z.get("pulse_excl_ranges_list")
            or [[] for _ in range(n)],
            "_trendline": z.get("trendline"),
            "acceptance": np.array([seq[idx] for idx in single_indices], dtype=int),
            "adjust_flags": np.zeros(n_single, dtype=bool),
            "_n_single": n_single,
            "_total_pages": max(1, int(np.ceil(n_single / self._page_size))),
            # per-zone edge method (each channel's template may differ)
            "_edge_method": z.get("edge_method", "peaks"),
        }

    def _save_active_zone(self):
        """Snapshot the live editable + fixed attrs into the active zone."""
        self._zones[self._active_zone] = self._snapshot_current_zone()

    def _load_active_zone(self, idx):
        """Repoint the live self._* attrs at zone *idx* and re-render."""
        self._active_zone = idx
        z = self._zones[idx]
        for a in self._ZONE_SWAP_ATTRS:
            setattr(self, a, z[a])
        # Refresh the Template-sequence indicator for this zone's template
        # (self._template_seq was just swapped in above).
        if getattr(self, "_lbl_seq", None) is not None:
            self._lbl_seq.setText(self._format_template_seq())
        # Reset transient tool / overlay state on zone switch.
        self._cancel_active_tool()
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._peak_adjust_mode = False
        self._peak_adjust_row = None
        self._hover_row = None
        self._page = max(0, min(self._page, self._total_pages - 1))
        self._render_page()

    def _on_zone_changed(self, idx):
        if idx == self._active_zone:
            return
        self._save_active_zone()
        self._load_active_zone(idx)

    def _on_overlay_toggled(self, checked):
        """Toggle the all-zones overlay on the left plots and re-render."""
        self._overlay_all_zones = bool(checked)
        self._render_page()

    def _overlay_other_zones_left(self, ax, pulse_idx, s, e, active_seg, ylim):
        """Overlay every non-active zone's signal AND rectangularized pulse for
        this pulse into the left axis *ax*, muted and baseline-stacked beneath
        the active zone.

        Each overlaid zone uses its own real values — detrended via the zone's
        own data when detrending is on (so all zones are detrended), or the
        zone's stored rect otherwise. The per-zone baseline re-centering and
        stack offset are applied for DISPLAY ONLY; nothing here feeds the
        block's output. Returns *ylim* widened to include the overlaid traces.
        """
        a = self._active_zone
        if active_seg is None or len(active_seg) == 0 or len(self._zones) <= 1:
            return ylim
        act_med = float(np.median(active_seg))
        arng = float(np.ptp(active_seg))
        if arng <= 0:
            arng = 1.0
        step = arng * 1.5
        use_mean = self._rect_method == "mean"
        lo, hi = ylim
        for zi in range(len(self._zones)):
            if zi == a:
                continue
            z = self._zones[zi]
            zdata = z.get("_data")
            if zdata is None:
                continue
            seg = np.asarray(zdata[s : min(e + 1, len(zdata))], dtype=float)
            if seg.size == 0:
                continue
            rect = None
            if self._detrend:
                res = self._detrend_pulse_segment(
                    zdata, z["_filtered"][pulse_idx], z["_peak_locs"][pulse_idx], s, e
                )
                if res is not None:
                    seg_dt, _, rect_med, rect_mean = res
                    seg = seg_dt
                    rect = rect_mean if use_mean else rect_med
            else:
                stored = (z["_rect_mean"] if use_mean else z["_rect"])[pulse_idx]
                if stored is not None and len(stored) == len(seg):
                    rect = np.asarray(stored, dtype=float)

            own_med = float(np.median(seg))
            # Smaller zone index sits higher (positive offset), matching
            # Detection Review's stacking order.
            offset = (a - zi) * step
            disp_shift = act_med - own_med + offset
            t = np.arange(len(seg)) / self._fs
            seg_shift = seg + disp_shift
            ax.plot(t, seg_shift, color="#999999", linewidth=0.5, alpha=0.35, zorder=0)
            lo = min(lo, float(np.min(seg_shift)))
            hi = max(hi, float(np.max(seg_shift)))
            if rect is not None and len(rect) == len(seg):
                rect_shift = rect + disp_shift
                ax.plot(
                    t, rect_shift, color="#555555", linewidth=1.0, alpha=0.55, zorder=0
                )
                lo = min(lo, float(np.min(rect_shift)))
                hi = max(hi, float(np.max(rect_shift)))
            ax.annotate(
                f"Z{zi + 1}",
                xy=(0, float(seg_shift[0])),
                xytext=(2, 0),
                textcoords="offset points",
                fontsize=6,
                color="#999999",
                va="center",
                zorder=0,
            )
        pad = (hi - lo) * 0.05
        return (lo - pad, hi + pad)

    def zone_results(self):
        """Per-zone output dicts, capturing the active zone's edits first.

        Each dict mirrors the legacy single-zone return shape. Only full
        matches are accepted (``acceptance_id = acceptance == 1``); rejected
        and warning pulses get their peak locs / rect zeroed.
        ``pulse_start_indices = pulse_indices[:, 0]``.

        ``_on_next`` already calls ``_finalize_detrended_rects`` on the active
        zone before ``accept()``; here we only snapshot the active zone's live
        edits into its slot before reading every zone.
        """
        self._save_active_zone()
        results = []
        for z in self._zones:
            n = z["_n_single"]
            single_indices = z["_single_indices"]
            acceptance = np.asarray(z["acceptance"])
            # Exported boundaries follow the template's edge method (FWHM
            # crossings of each pulse's own filtered signal), matching the
            # rect step positions so PulseSlicing cuts on the same samples.
            z_filtered = z["_filtered"]
            z_pidx = np.atleast_2d(np.asarray(z["_pulse_indices"], dtype=int))
            z_edge = z.get("_edge_method", "peaks")
            peak_locs = [
                _refine_pulse_bounds(
                    z["_peak_locs"][p],
                    z_filtered[p] if p < len(z_filtered) else None,
                    int(z_pidx[p, 0]),
                    z_edge,
                )
                for p in range(len(z["_peak_locs"]))
            ]
            # Output rects use each zone's real values: detrended per zone when
            # detrending is on (not just the active zone), undetrended otherwise.
            # The left-plot display shift never enters here.
            if self._detrend:
                rect_med, rect_mean = self._detrended_rects_for_zone(z)
            else:
                rect_med, rect_mean = z["_rect"], z["_rect_mean"]
            chosen_rect = list(rect_mean if self._rect_method == "mean" else rect_med)
            for si, idx in enumerate(single_indices):
                if int(acceptance[si]) != 1:
                    peak_locs[idx] = np.array([], dtype=int)
                    chosen_rect[idx] = np.array([])
            acceptance_id = np.zeros(n, dtype=bool)
            for si, idx in enumerate(single_indices):
                acceptance_id[idx] = int(acceptance[si]) == 1
            results.append(
                {
                    "pulse_peak_locations": peak_locs,
                    "rectangularized_pulses": chosen_rect,
                    "acceptance_id": acceptance_id,
                    "pulse_start_indices": np.asarray(z["_pulse_indices"])[:, 0].copy(),
                }
            )
        return results

    # ---- UI construction ----------------------------------------------------

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        # Left: matplotlib canvas
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(14, 9), tight_layout=False)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas)
        main.addWidget(left, stretch=5)

        # Trim line overlay
        self._trim_line = _LineOverlay(self.canvas)
        self._trim_line.resize(self.canvas.size())

        # Peak adjust cursor circle overlay
        self._cursor_circle = _CircleOverlay(self.canvas)

        # Connect matplotlib mouse events
        cid1 = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        cid2 = self.canvas.mpl_connect("button_press_event", self._on_click)
        cid3 = self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self._mpl_cids = [(self.canvas, cid1), (self.canvas, cid2), (self.canvas, cid3)]

        # Right: control panel
        panel = QWidget()
        panel.setFixedWidth(260)
        p_layout = QVBoxLayout(panel)
        p_layout.setContentsMargins(6, 6, 6, 6)
        p_layout.setSpacing(6)

        # --- Template sequence (top, with outline) ---
        seq_text = self._format_template_seq()
        self._lbl_seq = QLabel(seq_text)
        self._lbl_seq.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_seq.setWordWrap(True)
        self._lbl_seq.setStyleSheet(
            f"background: {theme.palette.panel_bg}; padding: 5px;"
            f" border: 1px solid {theme.palette.border}; "
            "border-radius: 3px; font-size: 11px;"
        )
        p_layout.addWidget(self._lbl_seq)

        # --- Information group ---
        from utils.sr_info import make_sr_info_group

        p_layout.addWidget(make_sr_info_group(self._fs, self._ds_factor))

        # --- Zone selector (multi-zone only) ---
        self._zone_btn_group = None
        self._chk_all_zones = None
        if len(self._zones) > 1:
            zone_grp = QGroupBox("Zone")
            zone_grp_layout = QVBoxLayout(zone_grp)
            zone_grp_layout.setContentsMargins(8, 4, 8, 6)
            zone_grp_layout.setSpacing(4)
            zone_row = QHBoxLayout()
            zone_row.setSpacing(4)
            self._zone_btn_group = QButtonGroup(self)
            for k in range(len(self._zones)):
                rb = QRadioButton(f"Zone {k + 1}")
                rb.setStyleSheet("font-size: 11px;")
                if k == self._active_zone:
                    rb.setChecked(True)
                self._zone_btn_group.addButton(rb, k)
                zone_row.addWidget(rb)
            zone_row.addStretch()
            self._zone_btn_group.idClicked.connect(self._on_zone_changed)
            zone_grp_layout.addLayout(zone_row)
            # Overlay all zones in the left plots (right plots stay active-zone).
            self._chk_all_zones = QCheckBox("All zones in left plots")
            self._chk_all_zones.setStyleSheet("font-size: 11px;")
            self._chk_all_zones.setChecked(self._overlay_all_zones)
            self._chk_all_zones.setToolTip(
                "Overlay every zone's signal in the left plots, stacked and "
                "muted beneath the active zone. The right (derivative) plots "
                "always show only the selected active zone."
            )
            self._chk_all_zones.toggled.connect(self._on_overlay_toggled)
            zone_grp_layout.addWidget(self._chk_all_zones)
            p_layout.addWidget(zone_grp)

        # --- Detection Summary group ---
        info_grp = QGroupBox("Detection Summary")
        info_layout = QVBoxLayout(info_grp)
        info_layout.setContentsMargins(8, 4, 8, 6)
        info_layout.setSpacing(2)

        self._lbl_page = QLabel("")
        self._lbl_page.setStyleSheet("font-weight: bold; font-size: 12px;")
        info_layout.addWidget(self._lbl_page)

        self._lbl_total = QLabel("")
        self._lbl_total.setStyleSheet("font-size: 11px;")
        info_layout.addWidget(self._lbl_total)
        self._lbl_accepted = QLabel("")
        self._lbl_accepted.setStyleSheet("font-size: 11px; color: #2ca02c;")
        info_layout.addWidget(self._lbl_accepted)
        self._lbl_warning = QLabel("")
        self._lbl_warning.setStyleSheet("font-size: 11px; color: #e67e22;")
        info_layout.addWidget(self._lbl_warning)
        self._lbl_rejected = QLabel("")
        self._lbl_rejected.setStyleSheet("font-size: 11px; color: #d62728;")
        info_layout.addWidget(self._lbl_rejected)
        p_layout.addWidget(info_grp)

        # --- Go To group ---
        goto_grp = QGroupBox("Go To")
        goto_layout = QHBoxLayout(goto_grp)
        goto_layout.setContentsMargins(6, 4, 6, 6)
        btn_jump_warn = QPushButton("Warning")
        btn_jump_warn.clicked.connect(self._on_jump_warning)
        goto_layout.addWidget(btn_jump_warn)
        btn_jump_rej = QPushButton("Rejected")
        btn_jump_rej.clicked.connect(self._on_jump_rejected)
        goto_layout.addWidget(btn_jump_rej)
        p_layout.addWidget(goto_grp)

        # --- Display Options group ---
        try:
            _has_trend = self._trendline is not None and np.size(self._trendline) > 0
        except Exception:
            _has_trend = False

        disp_grp = QGroupBox("Display Options")
        disp_layout = QVBoxLayout(disp_grp)
        disp_layout.setContentsMargins(8, 4, 8, 6)
        disp_layout.setSpacing(3)

        self._chk_show_trend = QCheckBox("Show Trendline")
        # Detrending shows the trendline by default (mirrors _on_detrend_toggled)
        self._chk_show_trend.setChecked(self._detrend and _has_trend)
        self._chk_show_trend.toggled.connect(lambda: self._render_page())
        disp_layout.addWidget(self._chk_show_trend)
        if not _has_trend:
            self._chk_show_trend.setVisible(False)

        # Diagnostics: label each key peak with its template seq index and
        # polarity so mismatches are visible at a glance.
        self._chk_debug_labels = QCheckBox("Debug Labels")
        self._chk_debug_labels.setChecked(False)
        self._chk_debug_labels.setToolTip(
            "Annotate each ★/▲/▼ key peak with its template sequence "
            "index and polarity. Use to diagnose peak-matching mismatches."
        )
        self._chk_debug_labels.toggled.connect(lambda: self._render_page())
        disp_layout.addWidget(self._chk_debug_labels)

        self._chk_detrend = QCheckBox("Enable Detrending")
        self._chk_detrend.setChecked(True)
        self._chk_detrend.toggled.connect(self._on_detrend_toggled)
        disp_layout.addWidget(self._chk_detrend)

        lbl_baseline = QLabel("Baseline value at:")
        lbl_baseline.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
        disp_layout.addWidget(lbl_baseline)

        self._rb_starting = QRadioButton("Start point (1st peak)")
        self._rb_starting.setChecked(True)
        self._rb_starting.setEnabled(self._detrend)
        self._rb_starting.setStyleSheet("font-size: 11px; margin-left: 16px;")

        # Samples input for start-point anchor
        from PySide6.QtWidgets import QSpinBox

        _samples_row = QHBoxLayout()
        _samples_row.setContentsMargins(24, 0, 0, 0)
        _lbl_samples = QLabel("Samples before peak:")
        _lbl_samples.setStyleSheet("font-size: 11px; color: #555;")
        self._spin_ref_samples = QSpinBox()
        self._spin_ref_samples.setRange(10, 10000)
        self._spin_ref_samples.setValue(1500)
        self._spin_ref_samples.setSingleStep(50)
        self._spin_ref_samples.setFixedWidth(70)
        self._spin_ref_samples.setStyleSheet("font-size: 11px;")
        self._spin_ref_samples.setEnabled(self._detrend)
        self._spin_ref_samples.editingFinished.connect(
            lambda: self._render_page() if self._detrend else None
        )
        _samples_row.addWidget(_lbl_samples)
        _samples_row.addWidget(self._spin_ref_samples)
        _samples_row.addStretch()

        self._rb_midpoint = QRadioButton("Mid point (key peaks)")
        self._rb_midpoint.setEnabled(self._detrend)
        self._rb_midpoint.setStyleSheet("font-size: 11px; margin-left: 16px;")
        self._detrend_offset_group = QButtonGroup(self)
        self._detrend_offset_group.addButton(self._rb_starting, 0)
        self._detrend_offset_group.addButton(self._rb_midpoint, 1)
        self._detrend_offset_group.buttonClicked.connect(
            lambda: self._render_page() if self._detrend else None
        )
        disp_layout.addWidget(self._rb_starting)
        disp_layout.addLayout(_samples_row)
        disp_layout.addWidget(self._rb_midpoint)

        self._lbl_ref_samples = _lbl_samples

        if not _has_trend:
            self._chk_detrend.setVisible(False)
            lbl_baseline.setVisible(False)
            self._rb_starting.setVisible(False)
            _lbl_samples.setVisible(False)
            self._spin_ref_samples.setVisible(False)
            self._rb_midpoint.setVisible(False)

        p_layout.addWidget(disp_grp)
        if not _has_trend:
            disp_grp.setVisible(False)

        # --- Processing group ---
        proc_grp = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_grp)
        proc_layout.setContentsMargins(8, 4, 8, 6)
        proc_layout.setSpacing(3)

        rect_row = QHBoxLayout()
        rect_row.addWidget(QLabel("Rect. method:"))
        self._rb_rect_median = QRadioButton("Median")
        self._rb_rect_mean = QRadioButton("Mean")
        self._rb_rect_mean.setChecked(True)
        self._rect_method_group = QButtonGroup(self)
        self._rect_method_group.addButton(self._rb_rect_median, 0)
        self._rect_method_group.addButton(self._rb_rect_mean, 1)
        self._rect_method_group.buttonClicked.connect(self._on_rect_method_changed)
        rect_row.addWidget(self._rb_rect_median)
        rect_row.addWidget(self._rb_rect_mean)
        rect_row.addStretch()
        proc_layout.addLayout(rect_row)

        # Edge indices: apex peaks vs FWHM crossings (mirrors the template
        # wizard's toggle; global across all zones/channels).
        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge:"))
        self._rb_edge_peaks = QRadioButton("Peaks")
        self._rb_edge_fwhm = QRadioButton("FWHM")
        self._edge_method_group = QButtonGroup(self)
        self._edge_method_group.addButton(self._rb_edge_peaks, 0)
        self._edge_method_group.addButton(self._rb_edge_fwhm, 1)
        (
            self._rb_edge_fwhm if self._edge_method == "fwhm" else self._rb_edge_peaks
        ).setChecked(True)
        self._edge_method_group.buttonClicked.connect(self._on_edge_method_changed)
        edge_row.addWidget(self._rb_edge_peaks)
        edge_row.addWidget(self._rb_edge_fwhm)
        edge_row.addStretch()
        proc_layout.addLayout(edge_row)
        p_layout.addWidget(proc_grp)

        # --- LP Filter group (per-file, on top of the template stack) ---
        from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

        lp_grp = QGroupBox("LP Filter")
        lp_layout = QVBoxLayout(lp_grp)
        lp_layout.setContentsMargins(8, 4, 8, 6)
        lp_layout.setSpacing(3)

        self._combo_lp_file = QComboBox()
        self._combo_lp_file.addItems(self._file_names)
        self._combo_lp_file.setStyleSheet("font-size: 11px;")
        self._combo_lp_file.setToolTip(
            "Settings below apply only to the selected file's pulses."
        )
        lp_layout.addWidget(self._combo_lp_file)

        self._chk_lp = QCheckBox("Enable low-pass")
        self._chk_lp.setStyleSheet("font-size: 11px;")
        lp_layout.addWidget(self._chk_lp)

        cut_row = QHBoxLayout()
        lbl_cut = QLabel("Cutoff (Hz):")
        lbl_cut.setStyleSheet("font-size: 11px; color: #555;")
        cut_row.addWidget(lbl_cut)
        self._spin_lp_cutoff = QDoubleSpinBox()
        nyq = max(float(self._fs) / 2.0, 2e-3)
        self._spin_lp_cutoff.setRange(1e-3, nyq * 0.99)
        self._spin_lp_cutoff.setDecimals(3)
        self._spin_lp_cutoff.setSingleStep(max(nyq / 50.0, 1e-3))
        self._spin_lp_cutoff.setStyleSheet("font-size: 11px;")
        cut_row.addWidget(self._spin_lp_cutoff)
        cut_row.addStretch()
        lp_layout.addLayout(cut_row)

        self._combo_lp_file.currentIndexChanged.connect(
            lambda _i: self._sync_lp_controls()
        )
        self._chk_lp.toggled.connect(lambda _c: self._on_lp_changed())
        self._spin_lp_cutoff.editingFinished.connect(self._on_lp_changed)
        self._sync_lp_controls()
        p_layout.addWidget(lp_grp)

        # --- Classification group ---
        class_grp = QGroupBox("Classification")
        class_layout = QVBoxLayout(class_grp)
        class_layout.setContentsMargins(4, 2, 4, 4)
        class_layout.setSpacing(2)

        self._table = QTableWidget(self._page_size, 2)
        self._table.setHorizontalHeaderLabels(["Accept", "Reject"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.setMaximumHeight(86)
        self._table.cellClicked.connect(self._on_table_click)
        class_layout.addWidget(self._table)
        p_layout.addWidget(class_grp)

        # --- Controls group ---
        ctrl_grp = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_grp)
        ctrl_layout.setContentsMargins(8, 4, 8, 6)
        ctrl_layout.setSpacing(3)

        from PySide6.QtWidgets import QSlider

        self._lbl_expand_val = QLabel("Expand: 25%")
        self._lbl_expand_val.setStyleSheet("font-size: 11px;")
        ctrl_layout.addWidget(self._lbl_expand_val)
        self._slider_expand = QSlider(Qt.Orientation.Horizontal)
        self._slider_expand.setRange(1, 10)  # 5% to 50% in steps of 5%
        self._slider_expand.setValue(5)  # 25%
        self._slider_expand.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_expand.setTickInterval(1)
        self._slider_expand.valueChanged.connect(
            lambda v: self._lbl_expand_val.setText(f"Expand: {v * 5}%")
        )
        ctrl_layout.addWidget(self._slider_expand)

        self._lbl_radius_val = QLabel("Cursor radius: 15 px")
        self._lbl_radius_val.setStyleSheet("font-size: 11px;")
        ctrl_layout.addWidget(self._lbl_radius_val)
        self._slider_cursor_radius = QSlider(Qt.Orientation.Horizontal)
        self._slider_cursor_radius.setRange(5, 30)
        self._slider_cursor_radius.setValue(15)
        self._slider_cursor_radius.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_cursor_radius.setTickInterval(5)
        self._slider_cursor_radius.valueChanged.connect(
            lambda v: self._lbl_radius_val.setText(f"Cursor radius: {v} px")
        )
        ctrl_layout.addWidget(self._slider_cursor_radius)

        p_layout.addWidget(ctrl_grp)

        p_layout.addStretch()

        # --- Export / Restart, then Save / Load on one line ---
        util_row = QHBoxLayout()
        from utils.export_figure_ui import make_export_button

        btn_export = make_export_button(lambda: self.fig, parent=self)
        util_row.addWidget(btn_export)
        self._btn_restart = QPushButton("Restart")
        self._btn_restart.setToolTip(
            "Reset all classifications and edits to the initial state"
        )
        self._btn_restart.clicked.connect(self._on_restart)
        util_row.addWidget(self._btn_restart)
        p_layout.addLayout(util_row)
        util_row2 = QHBoxLayout()
        btn_save_set = QPushButton("Save Settings")
        btn_save_set.clicked.connect(self._on_save_settings)
        util_row2.addWidget(btn_save_set)
        btn_load_set = QPushButton("Load Settings")
        btn_load_set.clicked.connect(self._on_load_settings)
        util_row2.addWidget(btn_load_set)
        p_layout.addLayout(util_row2)

        # --- Navigation ---
        nav = QHBoxLayout()
        self._btn_first = QPushButton("|<")
        self._btn_first.setFixedWidth(30)
        self._btn_first.clicked.connect(self._on_first)
        nav.addWidget(self._btn_first)
        self._btn_prev = QPushButton("< Prev")
        self._btn_prev.clicked.connect(self._on_prev)
        nav.addWidget(self._btn_prev)
        self._btn_next = QPushButton("Next >")
        self._btn_next.setObjectName("confirmBtn")
        self._btn_next.clicked.connect(self._on_next)
        nav.addWidget(self._btn_next)
        self._btn_last = QPushButton(">|")
        self._btn_last.setFixedWidth(30)
        self._btn_last.clicked.connect(self._on_last)
        nav.addWidget(self._btn_last)
        p_layout.addLayout(nav)

        main.addWidget(panel)

    def _format_template_seq(self):
        if self._template_seq is None or len(self._template_seq) == 0:
            return "Template: (unavailable)"
        parts = []
        for idx, v in enumerate(self._template_seq):
            color = "green" if v == 1 else "red"
            if idx in self._template_key_seq_indices:
                pol = self._template_key_pol_map.get(idx, "regular")
                if pol == "max":
                    sym = "\u25b2"  # ▲
                elif pol == "min":
                    sym = "\u25bc"  # ▼
                else:
                    sym = "\u2605"  # ★
                parts.append(
                    f'<span style="color:{color}; font-size:13px;">{sym}</span>'
                )
            else:
                parts.append(
                    f'<span style="color:{color}; font-size:13px;">\u25cf</span>'
                )
        return f"Template: {' '.join(parts)}"

    def _on_detrend_toggled(self, checked):
        self._detrend = checked
        self._rb_starting.setEnabled(checked)
        self._rb_midpoint.setEnabled(checked)
        self._spin_ref_samples.setEnabled(checked)
        # When detrending is enabled, also show the trendline
        if checked and not self._chk_show_trend.isChecked():
            self._chk_show_trend.setChecked(True)
        self._render_page()

    def _on_rect_method_changed(self):
        self._rect_method = "mean" if self._rb_rect_mean.isChecked() else "median"
        self._render_page()

    def _recompute_zone_rects(self, z):
        """Recompute a zone's stored median/mean rects from its apex peak locs
        and filtered pulses using the zone's edge method. Used when the Edge
        toggle moves boundaries (non-detrend path; detrend recomputes lazily
        in _render_page / zone_results)."""
        edge = z.get("_edge_method", "peaks")
        pidx = np.atleast_2d(np.asarray(z["_pulse_indices"], dtype=int))
        filt = z["_filtered"]
        peaks = z["_peak_locs"]
        rects = list(z["_rect"])
        rects_m = list(z["_rect_mean"])
        for i in range(len(pidx)):
            seg = filt[i] if i < len(filt) else None
            if seg is None or len(seg) == 0:
                continue
            seg = np.asarray(seg, dtype=float)
            s = int(pidx[i, 0])
            gp = peaks[i] if i < len(peaks) else None
            local = (
                (np.asarray(gp, dtype=int) - s)
                if gp is not None and len(gp) > 0
                else np.array([], dtype=int)
            )
            local = local[(local >= 0) & (local < len(seg))]
            bound = refine_interior(seg, local, edge)
            rects[i], rects_m[i] = _rectangularize_filtered(seg, bound)
        z["_rect"] = rects
        z["_rect_mean"] = rects_m

    def _on_edge_method_changed(self):
        new = "fwhm" if self._rb_edge_fwhm.isChecked() else "peaks"
        if new == getattr(self, "_edge_method", "peaks"):
            return
        # Global override across every zone/channel; fold live edits first so
        # the active zone's snapshot is current before we rewrite its rects.
        self._save_active_zone()
        for z in self._zones:
            z["_edge_method"] = new
            if not self._detrend:
                self._recompute_zone_rects(z)
        self._edge_method = new
        self._load_active_zone(self._active_zone)

    @property
    def rect_method(self):
        return self._rect_method

    def closeEvent(self, event):
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._hide_tool_sub_overlay()
        if self._trim_line is not None:
            self._trim_line.hide()
            self._trim_line = None
        for canvas, cid in self._mpl_cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids.clear()
        try:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        except Exception:
            pass
        super().closeEvent(event)

    # ---- page rendering -----------------------------------------------------

    def _render_page(self):
        self.fig.clear()
        self._axes_left = []
        self._axes_right = []
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._hide_tool_sub_overlay()
        self._active_tool = None
        self._active_tool_row = None
        self._hover_row = None
        self._clear_peak_hover()
        if self._trim_line is not None:
            self._trim_line.hide()

        p = self._page
        start = p * self._page_size
        end = min(start + self._page_size, self._n_single)
        n_show = end - start

        self._lbl_page.setText(f"Page {p + 1} / {self._total_pages}")
        self._update_summary()
        self._btn_first.setEnabled(p > 0)
        self._btn_prev.setEnabled(p > 0)
        is_last = p == self._total_pages - 1
        self._btn_next.setText("Finish" if is_last else "Next >")
        self._btn_last.setEnabled(not is_last)

        for i in range(n_show):
            si = start + i  # single-index
            pulse_idx = self._single_indices[si]  # global pulse index

            s = int(self._pulse_indices[pulse_idx, 0])
            e = int(self._pulse_indices[pulse_idx, 1])
            seg_raw = self._data[s : min(e + 1, len(self._data))]
            seg_filt = self._filtered[pulse_idx]
            seg_rect = self._rect[pulse_idx]
            dseg = self._diffs[pulse_idx]
            global_peaks = self._peak_locs[pulse_idx]

            # Extract trendline segment for this pulse (already smoothed from detection)
            _smooth_drift = None
            if self._trendline is not None:
                trend_seg = self._trendline[s : min(e + 1, len(self._trendline))]
                n = min(len(trend_seg), len(seg_raw))
                if n > 0:
                    _smooth_drift = trend_seg[:n].astype(float)

            local_peaks_raw = (
                global_peaks - s
                if global_peaks is not None and len(global_peaks) > 0
                else np.array([], dtype=int)
            )

            # Compute y-range
            _ymin = float(np.min(seg_raw))
            _ymax = float(np.max(seg_raw))
            _ypad = (_ymax - _ymin) * 0.08
            _ylim = (_ymin - _ypad, _ymax + _ypad)

            # Iterative baseline-fit detrending for all pulses.
            #
            # Anchor modes:
            #   Start point: anchor the first baseline segment to the
            #     mean of 100 samples before the first peak.
            #   Mid point: anchor the midpoint between first and last
            #     peak; the detrend pivot stays fixed there.
            #
            # Strategy:
            #   Pass 1 — fit a line through baseline regions (outside
            #     first..last peak), subtract it.
            #   Pass 2+ — measure residual slope from the first to last
            #     rect segment, subtract a linear correction anchored
            #     at the chosen pivot, repeat until both baseline
            #     segments converge.
            _do_detrend = self._detrend
            drift_total = None

            if _do_detrend:
                n = len(seg_raw)
                local_peaks_dt = (
                    global_peaks - s
                    if global_peaks is not None and len(global_peaks) > 0
                    else np.array([], dtype=int)
                )
                valid_lp = (
                    local_peaks_dt[(local_peaks_dt >= 0) & (local_peaks_dt < n)]
                    if len(local_peaks_dt) > 0
                    else np.array([], dtype=int)
                )
                # Follow the Edge method for the displayed detrend rect too
                # (the export path already does; keep them consistent).
                if seg_filt is not None and len(seg_filt) >= n:
                    valid_lp = refine_interior(
                        seg_filt[:n], valid_lp, self._edge_method
                    )
                bounds = np.sort(np.unique(np.concatenate([[0], valid_lp, [n]])))
                idx_arr = np.arange(n, dtype=float)

                # Baseline mask (regions outside first..last peak)
                bl_mask = np.ones(n, dtype=bool)
                if len(valid_lp) >= 2:
                    bl_mask[int(np.min(valid_lp)) : int(np.max(valid_lp)) + 1] = False

                # Determine anchor
                is_start_mode = self._rb_starting.isChecked()
                if is_start_mode:
                    first_pk = int(np.min(valid_lp)) if len(valid_lp) > 0 else 0
                    anchor_idx = max(first_pk - 1, 0)
                    # Reference region: N samples before 1st peak
                    ref_n = min(self._spin_ref_samples.value(), first_pk)
                    ref_n = max(ref_n, 1)
                    ref_start = first_pk - ref_n
                    ref_end = first_pk
                    anchor_ref = None  # computed after iteration
                    _ref_slice = (ref_start, ref_end)
                else:
                    if len(valid_lp) >= 2:
                        anchor_idx = (
                            int(np.min(valid_lp)) + int(np.max(valid_lp))
                        ) // 2
                    else:
                        anchor_idx = n // 2
                    anchor_ref = None

                def _rect_from(sig_in):
                    r = np.zeros(n, dtype=float)
                    if len(bounds) < 2:
                        r[:] = np.median(sig_in)
                    else:
                        for _k in range(len(bounds) - 1):
                            _bi, _be = int(bounds[_k]), int(bounds[_k + 1])
                            if _be <= _bi:
                                _be = _bi + 1
                            r[_bi:_be] = np.median(sig_in[_bi:_be])
                    return r

                _max_iter = 15
                _sig_range = float(np.ptp(seg_raw))
                _tol = max(1e-12, _sig_range * 5e-3)
                drift_total = np.zeros(n, dtype=float)

                first_seg_mid = (
                    (int(bounds[0]) + int(bounds[1])) / 2.0 if len(bounds) > 1 else 0.0
                )
                last_seg_mid = (
                    (int(bounds[-2]) + int(bounds[-1])) / 2.0
                    if len(bounds) > 1
                    else float(n)
                )

                for _it in range(_max_iter):
                    if _it == 0:
                        bl_idx = idx_arr[bl_mask]
                        if len(bl_idx) >= 2:
                            sig = (
                                seg_filt
                                if seg_filt is not None and len(seg_filt) >= n
                                else seg_raw
                            )
                            coeffs = np.polyfit(bl_idx, sig[:n][bl_mask], 1)
                            drift = (
                                np.polyval(coeffs, idx_arr)
                                - np.polyval(coeffs, [anchor_idx])[0]
                            )
                        else:
                            drift = np.linspace(seg_raw[0], seg_raw[-1], n) - seg_raw[0]
                    else:
                        residual = last_val - first_val
                        span = last_seg_mid - first_seg_mid
                        if span <= 0:
                            break
                        slope = residual / span
                        drift = slope * (idx_arr - float(anchor_idx))

                    seg_raw = seg_raw[:n] - drift
                    if seg_filt is not None and len(seg_filt) >= n:
                        seg_filt = seg_filt[:n] - drift
                    drift_total += drift

                    if seg_filt is not None and len(seg_filt) >= n:
                        seg_rect = _rect_from(seg_filt)
                        first_val = seg_rect[0]
                        last_val = seg_rect[-1]
                        if abs(first_val - last_val) <= _tol:
                            break
                    else:
                        seg_rect = _rect_from(seg_raw)
                        break

                # Anchor shift for start mode: set the first rect segment
                # to the mean of the reference region (after slope removal)
                if is_start_mode and "_ref_slice" in dir():
                    rs, re = _ref_slice
                    sig_ref = (
                        seg_filt
                        if seg_filt is not None and len(seg_filt) >= n
                        else seg_raw
                    )
                    anchor_ref = float(np.mean(sig_ref[rs:re]))
                    shift = seg_rect[0] - anchor_ref
                    seg_raw = seg_raw - shift
                    if seg_filt is not None and len(seg_filt) >= n:
                        seg_filt = seg_filt - shift
                    seg_rect = seg_rect - shift
                    drift_total += shift

            local_peaks = (
                global_peaks - s
                if global_peaks is not None and len(global_peaks) > 0
                else np.array([], dtype=int)
            )

            t_seg = np.arange(len(seg_raw)) / self._fs

            # --- left plot: signal + rectangularized ---
            ax_left = self.fig.add_axes([0.05, 0.97 - (i + 1) * 0.31, 0.42, 0.26])
            self._axes_left.append(ax_left)
            # Gray background: dataIn slice, detrended if enabled.
            seg_bg = self._data[s : min(e + 1, len(self._data))].copy()
            t_bg = np.arange(len(seg_bg)) / self._fs
            if _do_detrend and drift_total is not None:
                nb = min(len(drift_total), len(seg_bg))
                seg_bg[:nb] = seg_bg[:nb] - drift_total[:nb]
            ax_left.plot(t_bg, seg_bg, color="#bbbbbb", linewidth=0.8, label="Signal")

            # Overlay every other zone's signal for this pulse (left plots
            # only; the right derivative plot stays on the active zone). Widen
            # the y-limits so the stacked overlay traces stay visible.
            if self._overlay_all_zones and len(self._zones) > 1:
                _ylim = self._overlay_other_zones_left(
                    ax_left, pulse_idx, s, e, seg_bg, _ylim
                )

            # Show trendline overlay if enabled
            if self._chk_show_trend.isChecked() and _smooth_drift is not None:
                n_td = len(_smooth_drift)
                t_trend = np.arange(n_td) / self._fs
                # Show trendline dimmed when detrending is active
                alpha = 0.4 if _do_detrend else 0.7
                ax_left.plot(
                    t_trend,
                    _smooth_drift,
                    color="#e67e22",
                    linewidth=1.0,
                    linestyle="--" if _do_detrend else "-",
                    alpha=alpha,
                    label="Trendline",
                )
                # Keep the trendline inside the fixed y-view — it may sit at a
                # different absolute level than the (detrended) segment.
                _ylim = (
                    min(_ylim[0], float(np.min(_smooth_drift)) - _ypad),
                    max(_ylim[1], float(np.max(_smooth_drift)) + _ypad),
                )

            # Show anchor point and reference region when detrending
            if _do_detrend and drift_total is not None and "anchor_idx" in dir():
                a_t = anchor_idx / self._fs
                if "anchor_ref" in dir() and anchor_ref is not None:
                    a_val = anchor_ref
                else:
                    a_sig = (
                        seg_filt
                        if seg_filt is not None and len(seg_filt) > anchor_idx
                        else seg_raw
                    )
                    a_val = float(a_sig[min(anchor_idx, len(a_sig) - 1)])
                ax_left.plot(
                    a_t,
                    a_val,
                    "D",
                    color="#2196F3",
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=9,
                )
                # Highlight reference region (start mode only)
                if is_start_mode and "_ref_slice" in dir() and anchor_ref is not None:
                    rs, re = _ref_slice
                    t_rs = rs / self._fs
                    t_re = (re - 1) / self._fs
                    ax_left.axvspan(t_rs, t_re, color="#2196F3", alpha=0.12, zorder=0)
                    ax_left.axhline(
                        anchor_ref,
                        color="#2196F3",
                        linewidth=0.8,
                        linestyle=":",
                        alpha=0.5,
                        zorder=1,
                    )

            # Mean rect: re-rectangularize from detrended filtered if needed,
            # otherwise use pre-computed
            seg_rect_mean = self._rect_mean[pulse_idx]
            if _do_detrend and seg_filt is not None:
                n = len(seg_filt)
                local_peaks_dt = (
                    global_peaks - s
                    if global_peaks is not None and len(global_peaks) > 0
                    else np.array([], dtype=int)
                )
                valid_lp = (
                    local_peaks_dt[(local_peaks_dt >= 0) & (local_peaks_dt < n)]
                    if len(local_peaks_dt) > 0
                    else np.array([], dtype=int)
                )
                valid_lp = refine_interior(seg_filt, valid_lp, self._edge_method)
                bounds = np.sort(np.unique(np.concatenate([[0], valid_lp, [n]])))
                seg_rect_mean = np.zeros(n, dtype=float)
                if len(bounds) < 2:
                    seg_rect_mean[:] = np.mean(seg_filt)
                else:
                    for k in range(len(bounds) - 1):
                        bi, be = int(bounds[k]), int(bounds[k + 1])
                        if be <= bi:
                            be = bi + 1
                        seg_rect_mean[bi:be] = np.mean(seg_filt[bi:be])

            # Draw both rect lines: selected bold, other dimmed
            is_median = self._rect_method == "median"
            if seg_rect is not None and len(seg_rect) == len(t_seg):
                ax_left.plot(
                    t_seg,
                    seg_rect,
                    color="black",
                    linewidth=1.5 if is_median else 0.8,
                    alpha=1.0 if is_median else 0.3,
                    label="Median",
                )
            if seg_rect_mean is not None and len(seg_rect_mean) == len(t_seg):
                ax_left.plot(
                    t_seg,
                    seg_rect_mean,
                    color="#d62728",
                    linewidth=1.5 if not is_median else 0.8,
                    alpha=1.0 if not is_median else 0.3,
                    label="Mean",
                )

            # Color-coded peak markers on rect (key peaks as stars)
            # Use the selected rect method for peak marker positions
            active_rect = seg_rect if is_median else seg_rect_mean
            kp_list = self._pulse_key_peaks[pulse_idx]
            kp_pols = self._pulse_key_peak_pols[pulse_idx]
            kp_seqs = self._pulse_key_peak_seqs[pulse_idx]
            kp_set = set(kp_list)
            # Build polarity + template-seq lookups by local_peak_idx
            kp_pol_map = {}
            kp_seq_map = {}
            for kpi, kp_loc in enumerate(kp_list):
                kp_pol_map[int(kp_loc)] = (
                    kp_pols[kpi] if kpi < len(kp_pols) else "regular"
                )
                if kpi < len(kp_seqs):
                    kp_seq_map[int(kp_loc)] = int(kp_seqs[kpi])
            debug_labels = (
                hasattr(self, "_chk_debug_labels")
                and self._chk_debug_labels.isChecked()
            )
            if len(local_peaks) > 0 and dseg is not None and active_rect is not None:
                valid = (
                    (local_peaks >= 0)
                    & (local_peaks < len(dseg))
                    & (local_peaks < len(active_rect))
                )
                lp_valid = local_peaks[valid]
                if len(lp_valid) > 0:
                    pk_rect_vals = active_rect[lp_valid]
                    pk_diff_vals = dseg[lp_valid]
                    is_key = np.array([int(lp) in kp_set for lp in lp_valid])
                    pos_mask = pk_diff_vals >= 0
                    neg_mask = ~pos_mask
                    # Regular peaks (circles)
                    reg_pos = pos_mask & ~is_key
                    reg_neg = neg_mask & ~is_key
                    if np.any(reg_pos):
                        ax_left.plot(
                            t_seg[lp_valid[reg_pos]],
                            pk_rect_vals[reg_pos],
                            "o",
                            color="green",
                            markersize=5,
                            markerfacecolor="green",
                        )
                    if np.any(reg_neg):
                        ax_left.plot(
                            t_seg[lp_valid[reg_neg]],
                            pk_rect_vals[reg_neg],
                            "o",
                            color="red",
                            markersize=5,
                            markerfacecolor="red",
                        )
                    # Key peaks — different markers by polarity
                    for idx_j in range(len(lp_valid)):
                        if not is_key[idx_j]:
                            continue
                        lp = int(lp_valid[idx_j])
                        pol = kp_pol_map.get(lp, "regular")
                        color = "green" if pos_mask[idx_j] else "red"
                        if pol == "max":
                            marker, ms = "^", 8
                        elif pol == "min":
                            marker, ms = "v", 8
                        else:
                            marker, ms = "*", 10
                        ax_left.plot(
                            t_seg[lp],
                            pk_rect_vals[idx_j],
                            marker,
                            color=color,
                            markersize=ms,
                            markeredgecolor="none",
                            zorder=5,
                        )

            ax_left.set_ylabel("Amplitude", fontsize=8)
            ax_left.set_title(
                f"Pulse {pulse_idx + 1} - Signal & Rectangularized",
                fontsize=9,
                fontweight="bold",
            )
            ax_left.tick_params(labelsize=7)
            ax_left.grid(True, alpha=0.3)
            if len(t_seg) > 0:
                ax_left.set_xlim(0, t_seg[-1])
            ax_left.set_ylim(_ylim)

            # Outline colour based on acceptance
            self._apply_acceptance_outline(ax_left, si)

            # --- right plot: derivative + thresholds ---
            ax_right = self.fig.add_axes([0.54, 0.97 - (i + 1) * 0.31, 0.42, 0.26])
            self._axes_right.append(ax_right)
            if dseg is not None and len(dseg) > 0:
                t_diff = np.arange(len(dseg)) / self._fs
                ax_right.plot(t_diff, dseg, color="#888888", linewidth=1.0)

                # Threshold lines reflect the *effective* per-pulse cutoff:
                # detection keeps the strongest peaks per region, so the
                # de-facto threshold is the weakest selected peak of each
                # polarity. Fall back to the template thresholds if a polarity
                # has no selected peaks on this pulse.
                pos_t, neg_t = self._thresholds[pulse_idx]
                if len(local_peaks) > 0:
                    _sel = local_peaks[(local_peaks >= 0) & (local_peaks < len(dseg))]
                    if len(_sel) > 0:
                        _vals = dseg[_sel]
                        _pos = _vals[_vals >= 0]
                        _neg = _vals[_vals < 0]
                        if len(_pos) > 0:
                            pos_t = float(np.min(_pos))
                        if len(_neg) > 0:
                            neg_t = float(np.max(_neg))
                ax_right.axhline(pos_t, color="green", linestyle="--", linewidth=1.2)
                ax_right.axhline(neg_t, color="red", linestyle="--", linewidth=1.2)

                # Show exclusion zones as shaded regions with distinct colors
                _zone_colors = ["#FF8888", "#88AAFF", "#FFCC66", "#88DDAA", "#CC88FF"]
                for zi, (rs, re) in enumerate(self._pulse_excl_ranges[pulse_idx]):
                    ax_right.axvspan(
                        rs / self._fs,
                        re / self._fs,
                        color=_zone_colors[zi % len(_zone_colors)],
                        alpha=0.15,
                        zorder=0,
                    )

                if len(local_peaks) > 0:
                    valid = (local_peaks >= 0) & (local_peaks < len(dseg))
                    lp_valid = local_peaks[valid]
                    if len(lp_valid) > 0:
                        pk_vals = dseg[lp_valid]
                        is_key = np.array([int(lp) in kp_set for lp in lp_valid])
                        pos_mask = pk_vals >= 0
                        neg_mask = ~pos_mask
                        # Regular peaks (circles)
                        reg_pos = pos_mask & ~is_key
                        reg_neg = neg_mask & ~is_key
                        if np.any(reg_pos):
                            ax_right.plot(
                                t_diff[lp_valid[reg_pos]],
                                pk_vals[reg_pos],
                                "o",
                                color="green",
                                markersize=5,
                                markerfacecolor="green",
                            )
                        if np.any(reg_neg):
                            ax_right.plot(
                                t_diff[lp_valid[reg_neg]],
                                pk_vals[reg_neg],
                                "o",
                                color="red",
                                markersize=5,
                                markerfacecolor="red",
                            )
                        # Key peaks — different markers by polarity
                        for idx_j in range(len(lp_valid)):
                            if not is_key[idx_j]:
                                continue
                            lp = int(lp_valid[idx_j])
                            pol = kp_pol_map.get(lp, "regular")
                            color = "green" if pos_mask[idx_j] else "red"
                            if pol == "max":
                                marker, ms = "^", 10
                            elif pol == "min":
                                marker, ms = "v", 10
                            else:
                                marker, ms = "*", 12
                            ax_right.plot(
                                t_diff[lp],
                                pk_vals[idx_j],
                                marker,
                                color=color,
                                markersize=ms,
                                markeredgecolor="none",
                                zorder=5,
                            )
                            if debug_labels:
                                seq_idx = kp_seq_map.get(lp)
                                pol_short = {"min": "min", "max": "max"}.get(
                                    pol, "reg+" if pos_mask[idx_j] else "reg-"
                                )
                                label = (
                                    f"[{seq_idx}] {pol_short}"
                                    if seq_idx is not None
                                    else pol_short
                                )
                                ax_right.annotate(
                                    label,
                                    (t_diff[lp], pk_vals[idx_j]),
                                    xytext=(4, 6 if pos_mask[idx_j] else -10),
                                    textcoords="offset points",
                                    fontsize=7,
                                    color=color,
                                    zorder=6,
                                )
                ax_right.set_xlim(0, t_diff[-1])

            pc, nc = self._counts[pulse_idx] if self._counts[pulse_idx] else (0, 0)
            # Count actual plotted peaks
            if local_peaks is not None and len(local_peaks) > 0 and dseg is not None:
                vp = local_peaks[(local_peaks >= 0) & (local_peaks < len(dseg))]
                n_vis_pos = int(np.sum(dseg[vp] >= 0)) if len(vp) > 0 else 0
                n_vis_neg = len(vp) - n_vis_pos
            else:
                n_vis_pos, n_vis_neg = 0, 0
            count_ok = n_vis_pos == pc and n_vis_neg == nc
            title_str = f"Peaks: +{n_vis_pos} / -{n_vis_neg}"
            if not count_ok:
                title_str += f"  [target +{pc}/-{nc}]"
            ax_right.set_title(
                title_str,
                fontsize=9,
                fontweight="bold",
                color="black" if count_ok else "#c62828",
            )
            ax_right.set_ylabel("Derivative", fontsize=8)
            ax_right.tick_params(labelsize=7)
            ax_right.grid(False)
            if i == n_show - 1:
                ax_left.set_xlabel("Time (s)", fontsize=8)
                ax_right.set_xlabel("Time (s)", fontsize=8)

        # Update classification table
        self._table.setRowCount(n_show)
        for i in range(n_show):
            si = start + i
            pulse_idx = self._single_indices[si]
            self._table.setVerticalHeaderItem(
                i, QTableWidgetItem(f"Pulse {pulse_idx + 1}")
            )

            state = int(self.acceptance[si])
            acc = state == 1  # only full match accepted; warning defaults reject
            rej = state != 1

            for col, checked in enumerate([acc, rej]):
                item = QTableWidgetItem()
                item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                # Orange text for warning state
                if col == 0 and state == 2:
                    item.setForeground(QColor("#e67e22"))
                self._table.setItem(i, col, item)

        self.canvas.draw_idle()

    # ---- table interaction --------------------------------------------------

    def _on_table_click(self, row, col):
        p = self._page
        si = p * self._page_size + row
        if si >= self._n_single:
            return

        if col == 0:  # Accept
            self.acceptance[si] = 1
        elif col == 1:  # Reject
            self.acceptance[si] = 0

        # Update row checkboxes
        state = int(self.acceptance[si])
        acc = state == 1
        rej = state != 1
        for c, checked in enumerate([acc, rej]):
            item = self._table.item(row, c)
            if item:
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )

        # Update left plot outline colour
        if row < len(self._axes_left):
            self._apply_acceptance_outline(self._axes_left[row], si)
            self.canvas.draw_idle()

        self._update_summary()

    # ---- helpers ------------------------------------------------------------

    def _update_summary(self):
        """Update the acceptance summary labels."""
        n_total = self._n_single
        n_acc = int(np.sum(self.acceptance == 1))
        n_warn = int(np.sum(self.acceptance == 2))
        n_rej = int(np.sum(self.acceptance == 0))
        self._lbl_total.setText(f"Total: {n_total}")
        self._lbl_accepted.setText(f"Accepted: {n_acc}")
        self._lbl_warning.setText(f"Warning: {n_warn}")
        self._lbl_rejected.setText(f"Rejected: {n_rej}")

    def _apply_acceptance_outline(self, ax, single_idx):
        """Set spine colour of *ax*: green=accepted, orange=warning, red=rejected."""
        state = int(self.acceptance[single_idx])
        if state == 2:
            color = "#e67e22"  # orange — shape warning
        elif state >= 1:
            color = "#2ca02c"  # green — accepted
        else:
            color = "#d62728"  # red — rejected
        width = 2.5
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(width)

    def _check_sequence(self, pulse_idx):
        """Re-check peak sequence match for a pulse after peak adjustment.

        Compares the current peak sign sequence (from derivative values)
        against the template sequence. Only checks colour pattern match —
        key peak positions are not enforced.
        Updates self.acceptance for the corresponding single index.
        """
        dseg = self._diffs[pulse_idx]
        if dseg is None or len(dseg) == 0:
            return
        if self._template_seq is None or len(self._template_seq) == 0:
            return

        peaks = self._peak_locs[pulse_idx]
        s = int(self._pulse_indices[pulse_idx, 0])
        local_peaks = (
            peaks - s
            if peaks is not None and len(peaks) > 0
            else np.array([], dtype=int)
        )

        # Build pulse sequence: 1 for positive derivative, 0 for negative
        valid = local_peaks[(local_peaks >= 0) & (local_peaks < len(dseg))]
        pulse_seq = np.array(
            [1 if dseg[p] >= 0 else 0 for p in sorted(valid)], dtype=int
        )

        # Check: same length and same colour pattern
        seq_match = len(pulse_seq) == len(self._template_seq) and np.array_equal(
            pulse_seq, self._template_seq
        )

        # Find the single_index for this pulse_idx
        for si, idx in enumerate(self._single_indices):
            if idx == pulse_idx:
                # A still-mismatched pulse stays a WARNING (2, accepted), not a
                # reject (0) — matching the initial detection. Auto-rejecting on
                # edit would make build_result clear the pulse's data, silently
                # discarding the user's modification. Reject stays manual-only.
                self.acceptance[si] = 1 if seq_match else 2
                break

    def _get_left_row(self, event):
        """Return row index (0..2) if event is on a left-side axes."""
        if event.inaxes is None:
            return None
        for i, ax in enumerate(self._axes_left):
            if event.inaxes == ax:
                return i
        return None

    def _get_right_row(self, event):
        """Return row index (0..2) if event is on a right-side axes."""
        if event.inaxes is None:
            return None
        for i, ax in enumerate(self._axes_right):
            if event.inaxes == ax:
                return i
        return None

    def _axes_widget_rect(self, ax):
        """Return (x, y_top, w, h) in canvas widget coords for an axes."""
        bbox = ax.get_position()
        cw = self.canvas.width()
        ch = self.canvas.height()
        ax_x = int(bbox.x0 * cw)
        ax_y_top = int((1 - bbox.y1) * ch)
        ax_w = int(bbox.width * cw)
        ax_h = int(bbox.height * ch)
        return ax_x, ax_y_top, ax_w, ax_h

    def _show_line_on_axes(self, overlay, ax, event):
        """Position and show a _LineOverlay at event.xdata on the given axes."""
        pixel = ax.transData.transform((event.xdata, 0))
        bbox = ax.get_window_extent()
        px = int(pixel[0])
        canvas_h = self.canvas.height()
        dpr = self.canvas.devicePixelRatioF()
        qx = int(px / dpr)
        qy0 = int((canvas_h * dpr - bbox.y1) / dpr)
        qy1 = int((canvas_h * dpr - bbox.y0) / dpr)
        overlay.resize(self.canvas.size())
        overlay.set_line(qx, qy0, qy1)
        overlay.show()
        return qx

    # ---- tool overlays ------------------------------------------------------

    def _show_tool_overlay(self, row):
        """Show Expand/Trim buttons at the top of the left plot."""
        self._hide_tool_overlay()
        if row >= len(self._axes_left):
            return

        ax = self._axes_left[row]
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(ax)

        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(3)

        btn_expand = QPushButton("\u2194 Expand")
        btn_expand.setFixedHeight(20)
        btn_expand.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_expand.setToolTip("Expand pulse region")
        btn_expand.clicked.connect(lambda: self._show_expand_options(row))
        overlay_layout.addWidget(btn_expand)

        btn_trim = QPushButton("\u2702 Trim")
        btn_trim.setFixedHeight(20)
        btn_trim.setStyleSheet(_TRIM_BTN_STYLE)
        btn_trim.setToolTip("Trim pulse region")
        btn_trim.clicked.connect(lambda: self._activate_tool_mode("trim", row))
        overlay_layout.addWidget(btn_trim)

        overlay.adjustSize()
        ox = ax_x + (ax_w - overlay.width()) // 2
        overlay.move(ox, ax_y_top + 4)
        overlay.show()
        self._tool_overlay = overlay

    def _hide_tool_overlay(self):
        if self._tool_overlay is not None:
            self._tool_overlay.hide()
            self._tool_overlay.deleteLater()
            self._tool_overlay = None

    def _show_right_tool_overlay(self, row):
        """Show Peaks button at the top of the right (derivative) plot."""
        self._hide_right_tool_overlay()
        if row >= len(self._axes_right):
            return

        ax = self._axes_right[row]
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(ax)

        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(3)

        btn_peaks = QPushButton("\u2726 Peaks")
        btn_peaks.setFixedHeight(20)
        btn_peaks.setStyleSheet(_PEAK_BTN_STYLE)
        btn_peaks.setToolTip("Add/remove peaks on derivative plot")
        btn_peaks.clicked.connect(lambda: self._toggle_peak_adjust(row))
        overlay_layout.addWidget(btn_peaks)

        overlay.adjustSize()
        ox = ax_x + (ax_w - overlay.width()) // 2
        overlay.move(ox, ax_y_top + 4)
        overlay.show()
        self._right_tool_overlay = overlay

    def _hide_right_tool_overlay(self):
        if (
            hasattr(self, "_right_tool_overlay")
            and self._right_tool_overlay is not None
        ):
            self._right_tool_overlay.hide()
            self._right_tool_overlay.deleteLater()
            self._right_tool_overlay = None

    def _hide_tool_sub_overlay(self):
        if self._tool_sub_overlay is not None:
            self._tool_sub_overlay.hide()
            self._tool_sub_overlay.deleteLater()
            self._tool_sub_overlay = None

    def _cancel_active_tool(self):
        """Cancel any active tool mode and hide overlays."""
        self._active_tool = None
        self._active_tool_row = None
        self._trim_sample = None
        self._trim_line_qx = None
        self._peak_adjust_mode = False
        self._peak_adjust_row = None
        self._clear_peak_hover()
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._hide_tool_sub_overlay()
        if self._trim_line is not None:
            self._trim_line.hide()

    def _activate_tool_mode(self, mode, row):
        """Enter trim mode, locked to the given plot row."""
        self._active_tool = mode
        self._active_tool_row = row
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._hide_tool_sub_overlay()
        self._show_cancel_overlay(row)

    def _show_cancel_overlay(self, row):
        """Show a cancel button at the top of the target left plot."""
        self._hide_tool_sub_overlay()
        if row >= len(self._axes_left):
            return
        ax = self._axes_left[row]
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(ax)
        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(0)

        lbl = QLabel(f"Click on plot to {self._active_tool}")
        lbl.setStyleSheet("font-size: 9px; color: #666; background: transparent;")
        overlay_layout.addWidget(lbl)
        overlay_layout.addSpacing(6)

        btn_cancel = QPushButton("\u2715 Cancel")
        btn_cancel.setFixedHeight(20)
        btn_cancel.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_cancel.clicked.connect(self._cancel_active_tool)
        overlay_layout.addWidget(btn_cancel)

        overlay.adjustSize()
        ox = ax_x + (ax_w - overlay.width()) // 2
        overlay.move(ox, ax_y_top + 4)
        overlay.show()
        self._tool_sub_overlay = overlay

    # ---- mouse events -------------------------------------------------------

    def _on_mouse_move(self, event):
        left_row = self._get_left_row(event)
        right_row = self._get_right_row(event)
        hover_row = left_row if left_row is not None else right_row

        # Handle hover change for tool overlays
        if hover_row != self._hover_row:
            self._hover_row = hover_row
            no_active = (
                self._active_tool is None
                and not self._peak_adjust_mode
                and self._tool_sub_overlay is None
            )
            if no_active:
                if left_row is not None:
                    self._show_tool_overlay(left_row)
                    self._show_right_tool_overlay(left_row)
                elif right_row is not None:
                    self._show_tool_overlay(right_row)
                    self._show_right_tool_overlay(right_row)
                else:
                    self._hide_tool_overlay()
                    self._hide_right_tool_overlay()

        # Peak adjust mode: show cursor circle only on the active derivative plot
        if (
            self._peak_adjust_mode
            and right_row is not None
            and right_row == self._peak_adjust_row
            and event.xdata is not None
            and event.ydata is not None
        ):
            self._update_peak_hover(event, right_row)
        elif self._peak_adjust_mode:
            self._clear_peak_hover()
            # Cancel when cursor enters a different plot
            on_other = left_row is not None or (
                right_row is not None and right_row != self._peak_adjust_row
            )
            if on_other:
                self._cancel_active_tool()
        else:
            self._clear_peak_hover()

        # Show vertical trim line on the active left plot
        if self._active_tool == "trim":
            on_target = (
                left_row is not None
                and left_row == self._active_tool_row
                and event.xdata is not None
            )
            if on_target and self._trim_line is not None:
                self._show_line_on_axes(
                    self._trim_line, self._axes_left[left_row], event
                )
            elif self._trim_line is not None:
                self._trim_line.hide()

    def _on_axes_leave(self, event):
        self._hover_row = None
        no_active = (
            self._active_tool is None
            and not self._peak_adjust_mode
            and self._tool_sub_overlay is None
        )
        if no_active:
            self._hide_tool_overlay()
            self._hide_right_tool_overlay()
        self._clear_peak_hover()

    def _on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return

        left_row = self._get_left_row(event)

        # Trim mode: click on active left plot to select trim location
        if (
            self._active_tool == "trim"
            and left_row is not None
            and left_row == self._active_tool_row
            and event.xdata is not None
        ):
            p = self._page
            si = p * self._page_size + left_row
            if si >= self._n_single:
                return
            pulse_idx = self._single_indices[si]
            s = int(self._pulse_indices[pulse_idx, 0])
            e = int(self._pulse_indices[pulse_idx, 1])
            trim_sample = int(round(event.xdata * self._fs)) + s
            if trim_sample <= s or trim_sample >= e:
                return
            self._trim_sample = trim_sample
            ax = self._axes_left[left_row]
            self._trim_line_qx = self._show_line_on_axes(self._trim_line, ax, event)
            self._show_trim_options(left_row, pulse_idx)
            return

        # Peak adjust mode: click on the active right (derivative) plot only
        right_row = self._get_right_row(event)
        if (
            self._peak_adjust_mode
            and right_row is not None
            and right_row == self._peak_adjust_row
            and event.xdata is not None
            and event.ydata is not None
        ):
            p = self._page
            si = p * self._page_size + right_row
            if si >= self._n_single:
                return
            pulse_idx = self._single_indices[si]
            click_local = int(round(event.xdata * self._fs))
            self._handle_peak_click(
                pulse_idx, click_local, right_row, event.xdata, event.ydata
            )
            return

    # ---- expand tool --------------------------------------------------------

    def _show_expand_options(self, row):
        """Show Before/Both/After expand buttons."""
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._hide_tool_sub_overlay()
        if row >= len(self._axes_left):
            return

        ax = self._axes_left[row]
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(ax)

        overlay = QWidget(self.canvas)
        outer_layout = QVBoxLayout(overlay)
        outer_layout.setContentsMargins(4, 2, 4, 2)
        outer_layout.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.setSpacing(0)

        p = self._page
        si = p * self._page_size + row
        pulse_idx = self._single_indices[si] if si < self._n_single else 0

        btn_before = QPushButton("\u2190 Before")
        btn_before.setFixedHeight(20)
        btn_before.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_before.setToolTip("Expand 25% before pulse start")
        btn_before.clicked.connect(lambda: self._do_expand(pulse_idx, "before"))
        top_row.addWidget(btn_before, 0, Qt.AlignmentFlag.AlignLeft)
        top_row.addStretch()

        btn_both = QPushButton("\u2194 Both")
        btn_both.setFixedHeight(20)
        btn_both.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_both.setToolTip("Expand 25% on both ends")
        btn_both.clicked.connect(lambda: self._do_expand(pulse_idx, "both"))
        top_row.addWidget(btn_both)
        top_row.addStretch()

        btn_after = QPushButton("After \u2192")
        btn_after.setFixedHeight(20)
        btn_after.setStyleSheet(_EXPAND_BTN_STYLE)
        btn_after.setToolTip("Expand 25% after pulse end")
        btn_after.clicked.connect(lambda: self._do_expand(pulse_idx, "after"))
        top_row.addWidget(btn_after, 0, Qt.AlignmentFlag.AlignRight)
        outer_layout.addLayout(top_row)

        btn_cancel = QPushButton("\u2715 Cancel")
        btn_cancel.setFixedHeight(20)
        btn_cancel.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_cancel.clicked.connect(self._cancel_active_tool)
        outer_layout.addWidget(btn_cancel, 0, Qt.AlignmentFlag.AlignCenter)

        overlay.adjustSize()
        cx = ax_x + (ax_w - overlay.width()) // 2
        overlay.move(cx, ax_y_top + 4)
        overlay.show()
        self._tool_sub_overlay = overlay

    def _do_expand(self, pulse_idx, direction):
        """Expand pulse region by the configured percentage."""
        s = int(self._pulse_indices[pulse_idx, 0])
        e = int(self._pulse_indices[pulse_idx, 1])
        width = e - s
        pct = (self._slider_expand.value() * 5) / 100.0
        expand = max(1, int(width * pct))

        if direction == "before":
            self._pulse_indices[pulse_idx, 0] = max(0, s - expand)
        elif direction == "after":
            self._pulse_indices[pulse_idx, 1] = min(len(self._data) - 1, e + expand)
        else:  # both
            self._pulse_indices[pulse_idx, 0] = max(0, s - expand)
            self._pulse_indices[pulse_idx, 1] = min(len(self._data) - 1, e + expand)

        self._reprocess_pulse(pulse_idx)
        self._cancel_active_tool()
        self._render_page()

    # ---- trim tool ----------------------------------------------------------

    def _show_trim_options(self, row, pulse_idx):
        """Show Remove Before / Remove After / Cancel buttons."""
        self._hide_tool_sub_overlay()
        if row >= len(self._axes_left):
            return

        ax = self._axes_left[row]
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(ax)

        overlay = QWidget(self.canvas)
        overlay.setFixedWidth(ax_w)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(4, 2, 4, 2)
        overlay_layout.setSpacing(0)

        btn_before = QPushButton("Remove \u2190")
        btn_before.setFixedHeight(20)
        btn_before.setStyleSheet(_TRIM_BTN_STYLE)
        btn_before.setToolTip("Remove data before selected point")
        btn_before.clicked.connect(lambda: self._do_trim(pulse_idx, "before"))

        btn_after = QPushButton("\u2192 Remove")
        btn_after.setFixedHeight(20)
        btn_after.setStyleSheet(_TRIM_BTN_STYLE)
        btn_after.setToolTip("Remove data after selected point")
        btn_after.clicked.connect(lambda: self._do_trim(pulse_idx, "after"))

        btn_cancel = QPushButton("\u2715")
        btn_cancel.setFixedHeight(20)
        btn_cancel.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_cancel.clicked.connect(self._cancel_active_tool)

        overlay_layout.addWidget(btn_before, 0, Qt.AlignmentFlag.AlignLeft)
        overlay_layout.addStretch()
        overlay_layout.addWidget(btn_cancel)
        overlay_layout.addStretch()
        overlay_layout.addWidget(btn_after, 0, Qt.AlignmentFlag.AlignRight)

        overlay.adjustSize()
        overlay.move(ax_x, ax_y_top + 4)
        overlay.show()
        self._tool_sub_overlay = overlay

    def _do_trim(self, pulse_idx, direction):
        """Trim pulse at the previously selected sample location."""
        if self._trim_sample is None:
            self._cancel_active_tool()
            return

        s = int(self._pulse_indices[pulse_idx, 0])
        e = int(self._pulse_indices[pulse_idx, 1])

        if direction == "before":
            self._pulse_indices[pulse_idx, 0] = self._trim_sample
        else:
            self._pulse_indices[pulse_idx, 1] = self._trim_sample

        self._reprocess_pulse(pulse_idx)
        self._cancel_active_tool()
        self._render_page()

    # ---- peak adjustment ----------------------------------------------------

    def _toggle_peak_adjust(self, row):
        """Toggle peak adjustment mode."""
        self._peak_adjust_mode = not self._peak_adjust_mode
        self._hide_tool_overlay()
        self._hide_right_tool_overlay()
        self._hide_tool_sub_overlay()
        if self._peak_adjust_mode:
            self._peak_adjust_row = row
            self._show_peak_adjust_overlay(row)
        else:
            self._peak_adjust_row = None
            self._cancel_active_tool()

    def _show_peak_adjust_overlay(self, row):
        """Show peak adjust mode indicator with cancel button on derivative plot."""
        self._hide_tool_sub_overlay()
        if row >= len(self._axes_right):
            return
        ax = self._axes_right[row]
        ax_x, ax_y_top, ax_w, ax_h = self._axes_widget_rect(ax)

        overlay = QWidget(self.canvas)
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(2, 2, 2, 2)
        overlay_layout.setSpacing(4)

        lbl = QLabel("\u2726 Peak Adjust")
        lbl.setStyleSheet(
            "font-size: 9px; color: #6600aa; font-weight: bold; background: transparent;"
        )
        overlay_layout.addWidget(lbl)

        btn_done = QPushButton("\u2715 Done")
        btn_done.setFixedHeight(20)
        btn_done.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_done.clicked.connect(self._cancel_active_tool)
        overlay_layout.addWidget(btn_done)

        overlay.adjustSize()
        # Position at top-right corner inside the plot area
        ox = ax_x + ax_w - overlay.width() - 4
        oy = ax_y_top + 4
        overlay.move(ox, oy)
        overlay.show()
        self._tool_sub_overlay = overlay

    def _find_nearest_peak_display(self, ax, pulse_idx, data_x, data_y, radius_px=None):
        """Find the nearest peak to cursor using display (pixel) distance.

        Returns (nearest_idx, dist_px) or (None, inf) if no peak is close enough.
        """
        peaks = self._peak_locs[pulse_idx]
        s = int(self._pulse_indices[pulse_idx, 0])
        dseg = self._diffs[pulse_idx]
        if dseg is None or len(dseg) == 0:
            return None, float("inf")
        local_peaks = (
            peaks - s
            if peaks is not None and len(peaks) > 0
            else np.array([], dtype=int)
        )
        if len(local_peaks) == 0:
            return None, float("inf")

        if radius_px is None:
            radius_px = self._slider_cursor_radius.value()

        valid = (local_peaks >= 0) & (local_peaks < len(dseg))
        if not np.any(valid):
            return None, float("inf")

        t_diff = np.arange(len(dseg)) / self._fs
        # Convert cursor position to display coords
        cursor_disp = ax.transData.transform((data_x, data_y))
        # Convert each peak to display coords
        best_idx = None
        best_dist = float("inf")
        for i, (lp, v) in enumerate(zip(local_peaks, valid)):
            if not v:
                continue
            pk_x = t_diff[lp]
            pk_y = dseg[lp]
            pk_disp = ax.transData.transform((pk_x, pk_y))
            dist = np.sqrt(
                (cursor_disp[0] - pk_disp[0]) ** 2 + (cursor_disp[1] - pk_disp[1]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, best_dist

    def _update_peak_hover(self, event, right_row):
        """Move cursor circle on derivative plot during peak adjust mode."""
        p = self._page
        si = p * self._page_size + right_row
        if si >= self._n_single:
            self._clear_peak_hover()
            return
        pulse_idx = self._single_indices[si]
        ax = self._axes_right[right_row]

        # Convert cursor data coords to Qt widget coords.
        # matplotlib display coords are physical pixels with Y=0 at bottom;
        # Qt widget coords are logical pixels with Y=0 at top.
        cursor_disp = ax.transData.transform((event.xdata, event.ydata))
        dpr = self.canvas.devicePixelRatioF()
        canvas_h = self.canvas.height()
        cx = int(cursor_disp[0] / dpr)
        cy = int(canvas_h - cursor_disp[1] / dpr)

        # Check if any peak falls within the circle radius
        radius_px = self._slider_cursor_radius.value()
        nearest_idx, dist_px = self._find_nearest_peak_display(
            ax, pulse_idx, event.xdata, event.ydata, radius_px=radius_px
        )
        has_peak = nearest_idx is not None and dist_px <= radius_px

        self._cursor_circle.set_pos(cx, cy, has_peak, draw_radius=int(radius_px / dpr))
        self._cursor_circle.show()

    def _clear_peak_hover(self):
        """Hide the cursor circle overlay."""
        self._cursor_circle.hide()

    def _handle_peak_click(self, pulse_idx, click_local, row, data_x, data_y):
        """Handle click on derivative plot: add or remove a peak.

        If a marked peak is inside the cursor circle, remove it.
        Otherwise search for the strongest derivative peak within the circle
        and add it.
        """
        self._clear_peak_hover()
        peaks = self._peak_locs[pulse_idx]
        s = int(self._pulse_indices[pulse_idx, 0])
        dseg = self._diffs[pulse_idx]

        if dseg is None or len(dseg) == 0:
            return

        local_peaks = (
            peaks - s
            if peaks is not None and len(peaks) > 0
            else np.array([], dtype=int)
        )
        radius_px = self._slider_cursor_radius.value()
        ax = self._axes_right[row]

        # Check if any existing peak is inside the cursor circle -> remove it
        nearest_idx, dist_px = self._find_nearest_peak_display(
            ax, pulse_idx, data_x, data_y, radius_px=radius_px
        )
        if nearest_idx is not None and dist_px <= radius_px:
            # Remove this peak
            self._peak_locs[pulse_idx] = np.delete(peaks, nearest_idx)
            # Also remove from key peaks if present
            kp = self._pulse_key_peaks[pulse_idx]
            if isinstance(kp, (list, np.ndarray)) and int(local_peaks[nearest_idx]) in [
                int(x) for x in kp
            ]:
                kp_list = [int(x) for x in kp]
                kp_list.remove(int(local_peaks[nearest_idx]))
                self._pulse_key_peaks[pulse_idx] = kp_list
            self._recompute_rect(pulse_idx)
            self._check_sequence(pulse_idx)
            self._render_page()
            self._peak_adjust_mode = True
            self._peak_adjust_row = row
            self._show_peak_adjust_overlay(row)
            return

        # No existing peak in circle — search for new peak within the circle's
        # x-range (converted from display pixels to sample indices)
        cursor_disp = ax.transData.transform((data_x, data_y))
        left_data = ax.transData.inverted().transform(
            (cursor_disp[0] - radius_px, cursor_disp[1])
        )
        right_data = ax.transData.inverted().transform(
            (cursor_disp[0] + radius_px, cursor_disp[1])
        )
        lo_sample = max(0, int(round(left_data[0] * self._fs)))
        hi_sample = min(len(dseg), int(round(right_data[0] * self._fs)) + 1)

        if hi_sample <= lo_sample:
            return

        # Among candidates in the x-range, pick the one closest in display space
        t_diff = np.arange(len(dseg)) / self._fs
        candidates = np.arange(lo_sample, hi_sample)
        # Filter by circular display distance
        best_local = None
        best_abs = -1
        for ci in candidates:
            pk_disp = ax.transData.transform((t_diff[ci], dseg[ci]))
            d = np.sqrt(
                (cursor_disp[0] - pk_disp[0]) ** 2 + (cursor_disp[1] - pk_disp[1]) ** 2
            )
            if d <= radius_px and abs(dseg[ci]) > best_abs:
                best_abs = abs(dseg[ci])
                best_local = ci

        if best_local is None:
            return

        new_global = best_local + s
        # Don't add if already exists
        if len(peaks) > 0 and new_global in peaks:
            return

        self._peak_locs[pulse_idx] = np.sort(np.append(peaks, new_global))
        self._recompute_rect(pulse_idx)
        self._check_sequence(pulse_idx)
        self._render_page()
        self._peak_adjust_mode = True
        self._peak_adjust_row = row
        self._show_peak_adjust_overlay(row)

    # ---- per-file LP filter -------------------------------------------------

    def _file_of_start(self, s):
        """Index of the source file containing global sample *s*."""
        b = self._file_boundaries
        return int(
            np.clip(
                np.searchsorted(b, s, side="right") - 1, 0, len(self._file_names) - 1
            )
        )

    def _lp_cutoff_for_pulse(self, pulse_idx):
        """Cutoff (Hz) of the pulse's file LP filter, or None if disabled."""
        cfg = self._file_filters[
            self._file_of_start(int(self._pulse_indices[pulse_idx, 0]))
        ]
        if cfg.get("enabled") and cfg.get("cutoff", 0) > 0:
            return float(cfg["cutoff"])
        return None

    def _reprocess_files(self, file_ids):
        """Re-filter/re-detect every pulse of the given files, in all zones."""
        if not file_ids:
            return
        self._save_active_zone()
        active = self._active_zone
        for k in range(len(self._zones)):
            self._load_active_zone(k)
            for i in range(len(self._pulse_indices)):
                if self._file_of_start(int(self._pulse_indices[i, 0])) in file_ids:
                    self._reprocess_pulse(i)
            self._save_active_zone()
        self._load_active_zone(active)

    def _sync_lp_controls(self):
        """Reflect the combo-selected file's LP settings in the controls."""
        cfg = self._file_filters[self._combo_lp_file.currentIndex()]
        self._lp_loading = True
        try:
            self._chk_lp.setChecked(bool(cfg["enabled"]))
            self._spin_lp_cutoff.setValue(float(cfg["cutoff"]))
        finally:
            self._lp_loading = False

    def _on_lp_changed(self):
        """Apply the controls to the selected file only; reprocess its pulses."""
        if self._lp_loading:
            return
        fi = self._combo_lp_file.currentIndex()
        old = self._file_filters[fi]
        new = {
            "enabled": self._chk_lp.isChecked(),
            "cutoff": float(self._spin_lp_cutoff.value()),
        }
        if new == old:
            return
        self._file_filters[fi] = new
        old_eff = old["cutoff"] if old["enabled"] else None
        new_eff = new["cutoff"] if new["enabled"] else None
        if old_eff != new_eff:
            self._reprocess_files({fi})

    def _restore_file_filters(self, saved):
        """Restore per-file LP settings (matched by file name, else index) and
        reprocess the files whose effective filtering changed."""
        by_name = {str(d.get("file")): d for d in saved if isinstance(d, dict)}
        affected = set()
        for i, name in enumerate(self._file_names):
            d = by_name.get(name)
            if d is None and i < len(saved) and isinstance(saved[i], dict):
                d = saved[i]
            if d is None:
                continue
            old = self._file_filters[i]
            new = {
                "enabled": bool(d.get("enabled")),
                "cutoff": float(d.get("cutoff", old["cutoff"])),
            }
            old_eff = old["cutoff"] if old["enabled"] else None
            new_eff = new["cutoff"] if new["enabled"] else None
            if old_eff != new_eff:
                affected.add(i)
            self._file_filters[i] = new
        self._sync_lp_controls()
        self._reprocess_files(affected)

    # ---- reprocessing -------------------------------------------------------

    def _reprocess_pulse(self, pulse_idx):
        """Re-run filter/derivative/peak-detection for a pulse after region change."""
        s = max(0, int(self._pulse_indices[pulse_idx, 0]))
        e = min(len(self._data) - 1, int(self._pulse_indices[pulse_idx, 1]))
        segment = self._data[s : e + 1].copy()
        n = len(segment)

        if n < 3:
            self._filtered[pulse_idx] = segment
            self._diffs[pulse_idx] = np.array([])
            self._rect[pulse_idx] = segment
            self._peak_locs[pulse_idx] = np.array([], dtype=int)
            return

        # Pad + filter
        actual_pad = min(self._pad_length, n // 4)
        seg_padded = _pad_signal(segment, actual_pad, self._pad_method)
        seg_filt_padded = _apply_filter_stack(seg_padded, self._fs, self._filter_config)
        lp_cut = self._lp_cutoff_for_pulse(pulse_idx)
        if lp_cut:
            seg_filt_padded = _apply_lowpass(seg_filt_padded, self._fs, lp_cut)
        if actual_pad > 0:
            seg_filtered = seg_filt_padded[actual_pad : actual_pad + n]
        else:
            seg_filtered = seg_filt_padded[:n]
        self._filtered[pulse_idx] = seg_filtered

        # Derivative
        dseg_padded = np.diff(seg_filt_padded)
        dseg_filt_padded = _apply_filter_stack(
            dseg_padded, self._fs, self._filter_config
        )
        if lp_cut:
            dseg_filt_padded = _apply_lowpass(dseg_filt_padded, self._fs, lp_cut)
        if actual_pad > 0:
            dseg = dseg_filt_padded[actual_pad : actual_pad + n]
        else:
            dseg = dseg_filt_padded[:n]
        self._diffs[pulse_idx] = dseg

        # Re-detect peaks using existing thresholds
        pos_t, neg_t = (
            self._thresholds[pulse_idx] if self._thresholds[pulse_idx] else (0, 0)
        )
        pos_peaks, _ = find_peaks(dseg, height=pos_t)
        neg_peaks, _ = find_peaks(-dseg, height=-neg_t)
        all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
        self._peak_locs[pulse_idx] = all_peaks + s  # store as global indices

        # Rectangularize from unfiltered segment
        self._recompute_rect(pulse_idx)

    def _recompute_rect(self, pulse_idx):
        """Recompute rectangularized pulse (median and mean) from current peaks."""
        s = max(0, int(self._pulse_indices[pulse_idx, 0]))
        e = min(len(self._data) - 1, int(self._pulse_indices[pulse_idx, 1]))
        segment = self._data[s : e + 1]
        n = len(segment)

        peaks = self._peak_locs[pulse_idx]
        local_peaks = (
            peaks - s
            if peaks is not None and len(peaks) > 0
            else np.array([], dtype=int)
        )
        valid_lp = (
            local_peaks[(local_peaks >= 0) & (local_peaks < n)]
            if len(local_peaks) > 0
            else np.array([], dtype=int)
        )
        # Follow the template's edge method: FWHM crossings of this pulse's own
        # filtered signal (or the apex peaks). Count/order preserved.
        valid_lp = refine_interior(
            self._filtered[pulse_idx], valid_lp, self._edge_method
        )

        bounds = np.sort(np.unique(np.concatenate([[0], valid_lp, [n]])))
        seg_rect = np.zeros(n, dtype=float)
        seg_rect_mean = np.zeros(n, dtype=float)
        if len(bounds) < 2:
            seg_rect[:] = np.median(segment)
            seg_rect_mean[:] = np.mean(segment)
        else:
            for k in range(len(bounds) - 1):
                bi = int(bounds[k])
                be = int(bounds[k + 1])
                if be <= bi:
                    be = bi + 1
                seg_rect[bi:be] = np.median(segment[bi:be])
                seg_rect_mean[bi:be] = np.mean(segment[bi:be])
        self._rect[pulse_idx] = seg_rect
        self._rect_mean[pulse_idx] = seg_rect_mean

    # ---- navigation ---------------------------------------------------------

    def _on_first(self):
        if self._page > 0:
            self._page = 0
            self._render_page()

    def _on_last(self):
        if self._page < self._total_pages - 1:
            self._page = self._total_pages - 1
            self._render_page()

    def _on_jump_warning(self):
        """Jump to next page containing a warning pulse (from current page)."""
        self._jump_to_state(2, "warning")

    def _on_jump_rejected(self):
        """Jump to next page containing a rejected pulse (from current page)."""
        self._jump_to_state(0, "rejected")

    def _jump_to_state(self, target_state, label):
        """Cycle to next pulse with the given acceptance state, wrapping around."""
        matches = [
            si
            for si in range(self._n_single)
            if int(self.acceptance[si]) == target_state
        ]
        if not matches:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(self, "Go To", f"No {label} pulses found.")
            return

        # Track last jumped position per state to enable cycling
        key = f"_last_jump_{target_state}"
        last_si = getattr(self, key, -1)

        # Find next match after last jumped position, wrapping around
        next_match = None
        for si in matches:
            if si > last_si:
                next_match = si
                break
        if next_match is None:
            next_match = matches[0]  # wrap to first

        setattr(self, key, next_match)
        target_page = next_match // self._page_size
        self._page = target_page
        self._render_page()

    def _on_prev(self):
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def _on_next(self):
        if self._page < self._total_pages - 1:
            self._page += 1
            self._render_page()
        else:
            self.confirmed = True
            self._finalize_detrended_rects()
            self._autosave()
            self.accept()

    def _detrend_pulse_segment(self, data, seg_filt_orig, global_peaks, s, e):
        """Iterative baseline-fit detrend for ONE pulse segment of one zone.

        Thin Qt wrapper over processing.pulse_detrend.detrend_pulse_segment.
        """
        from processing.pulse_detrend import detrend_pulse_segment

        mode = "starting" if self._rb_starting.isChecked() else "mean"
        bounds = _refine_pulse_bounds(global_peaks, seg_filt_orig, s, self._edge_method)
        return detrend_pulse_segment(
            data,
            seg_filt_orig,
            bounds,
            s,
            e,
            mode=mode,
            ref_samples=self._spin_ref_samples.value(),
        )

    def _detrended_rects_for_zone(self, z):
        """Return ``(rect_list, rect_mean_list)`` for zone-state *z* with the
        current detrend settings applied to every pulse, using the zone's own
        pristine data. Pulses too short to detrend keep their stored rect.
        """
        data = z["_data"]
        filtered = z["_filtered"]
        peak_locs = z["_peak_locs"]
        pidx = np.atleast_2d(np.asarray(z["_pulse_indices"], dtype=int))
        rect_list = list(z["_rect"])
        rect_mean_list = list(z["_rect_mean"])
        for i in range(len(pidx)):
            res = self._detrend_pulse_segment(
                data,
                filtered[i] if i < len(filtered) else None,
                peak_locs[i] if i < len(peak_locs) else None,
                int(pidx[i, 0]),
                int(pidx[i, 1]),
            )
            if res is None:
                continue
            _, _, rect_med, rect_mean = res
            rect_list[i] = rect_med
            rect_mean_list[i] = rect_mean
        return rect_list, rect_mean_list

    def _finalize_detrended_rects(self):
        """When detrending is on, persist detrended rect arrays for every
        pulse of the active zone so downstream blocks (PulseSlicing) see what
        the user previewed. _render_page detrends only locally for display.
        """
        if not self._detrend:
            return
        for pulse_idx in range(len(self._pulse_indices)):
            res = self._detrend_pulse_segment(
                self._data,
                self._filtered[pulse_idx],
                self._peak_locs[pulse_idx],
                int(self._pulse_indices[pulse_idx, 0]),
                int(self._pulse_indices[pulse_idx, 1]),
            )
            if res is None:
                continue
            _, _, seg_rect, seg_rect_mean = res
            self._rect[pulse_idx] = seg_rect
            self._rect_mean[pulse_idx] = seg_rect_mean

    def _autosave(self):
        """Auto-save current settings to SavedTemplates/Processing/temp.json."""
        import json
        import os

        folder = self._default_folder()
        target = os.path.join(folder, "temp.json")
        tmp = target + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._get_state(), f, indent=2)
            os.replace(tmp, target)
        except Exception as e:
            print(f"[BatchProcessing] Autosave failed: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    # ---- save / load settings -----------------------------------------------

    @staticmethod
    def _to_int_list(seq):
        """Convert a sequence (ndarray or list) to a plain Python int list."""
        if isinstance(seq, np.ndarray):
            return seq.tolist()
        return [int(x) for x in seq]

    def _get_state(self):
        """Capture all user-adjustable state as a serialisable dict.

        The per-pulse edits (acceptance, peaks, exclusions) are captured
        PER ZONE under ``state["zones"]`` so each zone restores its own work.
        The flat top-level copies mirror the active zone for single-zone /
        legacy consumers.
        """
        self._save_active_zone()  # fold the active zone's live edits into _zones
        state = {
            "acceptance": self.acceptance.tolist(),
            "rect_method": self._rect_method,
            "edge_method": self._edge_method,
            "detrend": self._detrend,
            "show_trendline": self._chk_show_trend.isChecked(),
            "detrend_mode": (
                "starting" if self._rb_starting.isChecked() else "midpoint"
            ),
            "page": self._page,
            "pulse_indices": self._pulse_indices.tolist(),
            "peak_locs": [self._to_int_list(p) for p in self._peak_locs],
            "pulse_key_peaks": [self._to_int_list(p) for p in self._pulse_key_peaks],
            "pulse_key_peak_pols": [list(p) for p in self._pulse_key_peak_pols],
            "pulse_excl_ranges": [
                [[int(a), int(b)] for a, b in r] for r in self._pulse_excl_ranges
            ],
            "expand_pct": self._slider_expand.value(),
            "cursor_radius": self._slider_cursor_radius.value(),
            "ref_samples": self._spin_ref_samples.value(),
            "file_filters": [
                {"file": self._file_names[i], **self._file_filters[i]}
                for i in range(len(self._file_names))
            ],
        }
        state["zones"] = [self._zone_edit_state(z) for z in self._zones]
        return state

    def _zone_edit_state(self, z):
        """Serialise one zone's per-pulse user edits from its snapshot dict."""
        return {
            "peak_locs": [self._to_int_list(p) for p in z["_peak_locs"]],
            "pulse_key_peaks": [self._to_int_list(p) for p in z["_pulse_key_peaks"]],
            "pulse_key_peak_pols": [list(p) for p in z["_pulse_key_peak_pols"]],
            "pulse_excl_ranges": [
                [[int(a), int(b)] for a, b in r] for r in z["_pulse_excl_ranges"]
            ],
            "acceptance": np.asarray(z["acceptance"]).astype(int).tolist(),
        }

    def _apply_zone_edit_state(self, z, zs):
        """Write one zone's saved per-pulse edits into its snapshot dict *z*."""
        if "peak_locs" in zs:
            for i in range(min(len(zs["peak_locs"]), len(z["_peak_locs"]))):
                z["_peak_locs"][i] = np.array(zs["peak_locs"][i], dtype=int)
        if "pulse_key_peaks" in zs:
            for i in range(min(len(zs["pulse_key_peaks"]), len(z["_pulse_key_peaks"]))):
                z["_pulse_key_peaks"][i] = list(zs["pulse_key_peaks"][i])
        if "pulse_key_peak_pols" in zs:
            for i in range(
                min(len(zs["pulse_key_peak_pols"]), len(z["_pulse_key_peak_pols"]))
            ):
                z["_pulse_key_peak_pols"][i] = list(zs["pulse_key_peak_pols"][i])
        if "pulse_excl_ranges" in zs:
            for i in range(
                min(len(zs["pulse_excl_ranges"]), len(z["_pulse_excl_ranges"]))
            ):
                z["_pulse_excl_ranges"][i] = list(zs["pulse_excl_ranges"][i])
        if "acceptance" in zs:
            a = np.asarray(zs["acceptance"], dtype=int)
            za = np.asarray(z["acceptance"], dtype=int)
            m = min(len(a), len(za))
            za[:m] = a[:m]
            z["acceptance"] = za

    def _apply_state(self, state):
        """Apply a previously saved state dict to the dialog.

        Per-pulse edits restore PER ZONE from ``state["zones"]`` when present
        (so a zone's peaks never land on another zone); otherwise the flat
        top-level fields restore the active zone (legacy single-zone format).
        """
        _zones_state = state.get("zones")
        _has_zones = isinstance(_zones_state, list) and len(_zones_state) > 0

        if "acceptance" in state and not _has_zones:
            saved = state["acceptance"]
            n = min(len(saved), len(self.acceptance))
            self.acceptance[:n] = np.array(saved[:n], dtype=int)

        if "rect_method" in state:
            self._rect_method = state["rect_method"]
            if self._rect_method == "mean":
                self._rb_rect_mean.setChecked(True)
            else:
                self._rb_rect_median.setChecked(True)

        # Restore the edge method before the peak/rect recompute loops below so
        # they rebuild rects at the saved boundaries. Propagate to every zone
        # (global setting); the recompute loops read it via self._edge_method.
        if "edge_method" in state and state["edge_method"] in ("peaks", "fwhm"):
            self._edge_method = state["edge_method"]
            (
                self._rb_edge_fwhm
                if self._edge_method == "fwhm"
                else self._rb_edge_peaks
            ).setChecked(True)
            for z in self._zones:
                z["_edge_method"] = self._edge_method

        if "show_trendline" in state:
            self._chk_show_trend.setChecked(state["show_trendline"])

        if "detrend" in state:
            self._detrend = state["detrend"]
            self._chk_detrend.setChecked(self._detrend)

        if "detrend_mode" in state:
            if state["detrend_mode"] in ("mean", "midpoint"):
                self._rb_midpoint.setChecked(True)
            else:
                self._rb_starting.setChecked(True)

        if "pulse_indices" in state:
            saved = np.array(state["pulse_indices"])
            if saved.shape == self._pulse_indices.shape:
                self._pulse_indices[:] = saved

        # Restore per-file LP filters (and reprocess affected files) BEFORE the
        # saved peak/rect edits below so those edits land on the re-filtered
        # pulses instead of being clobbered.
        if "file_filters" in state:
            self._restore_file_filters(state["file_filters"])

        if "peak_locs" in state and not _has_zones:
            saved = state["peak_locs"]
            n = min(len(saved), len(self._peak_locs))
            for i in range(n):
                self._peak_locs[i] = np.array(saved[i], dtype=int)
            # Recompute rectangularised signals from updated peaks
            for i in range(n):
                self._recompute_rect(i)

        if "pulse_key_peaks" in state and not _has_zones:
            saved = state["pulse_key_peaks"]
            n = min(len(saved), len(self._pulse_key_peaks))
            for i in range(n):
                self._pulse_key_peaks[i] = list(saved[i])

        if "pulse_key_peak_pols" in state and not _has_zones:
            saved = state["pulse_key_peak_pols"]
            n = min(len(saved), len(self._pulse_key_peak_pols))
            for i in range(n):
                self._pulse_key_peak_pols[i] = list(saved[i])

        if "pulse_excl_ranges" in state and not _has_zones:
            saved = state["pulse_excl_ranges"]
            n = min(len(saved), len(self._pulse_excl_ranges))
            for i in range(n):
                self._pulse_excl_ranges[i] = list(saved[i])

        if "expand_pct" in state:
            self._slider_expand.setValue(int(state["expand_pct"]))

        if "cursor_radius" in state:
            self._slider_cursor_radius.setValue(int(state["cursor_radius"]))

        if "ref_samples" in state:
            self._spin_ref_samples.setValue(int(state["ref_samples"]))

        if _has_zones:
            # Restore each zone's own edits into its snapshot, recompute that
            # zone's rect from the restored peaks, then display the active zone.
            active = self._active_zone if self._active_zone < len(self._zones) else 0
            for k, zs in enumerate(_zones_state):
                if k >= len(self._zones):
                    break
                self._apply_zone_edit_state(self._zones[k], zs)
            for k in range(len(self._zones)):
                self._load_active_zone(k)
                for i in range(len(self._peak_locs)):
                    self._recompute_rect(i)
                self._save_active_zone()
            if "page" in state:
                self._page = max(0, min(state["page"], self._total_pages - 1))
            self._load_active_zone(active)
            return

        if "page" in state:
            self._page = max(0, min(state["page"], self._total_pages - 1))

        self._render_page()

    def _default_folder(self):
        from utils.paths import saved_templates_dir

        return saved_templates_dir("Processing")

    def _on_restart(self):
        """Reset all classifications and edits back to the initial state."""
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Restart Review",
            "Reset all classifications and edits back to the initial state?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._apply_state(copy.deepcopy(self._initial_state))

    def _on_save_settings(self):
        """Save current review state to a JSON file."""
        import json
        import os
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Review Settings",
            self._default_folder(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._get_state(), f, indent=2)
        os.replace(tmp, path)

    def _on_load_settings(self):
        """Load review state from a JSON file."""
        import json
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Review Settings",
            self._default_folder(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        with open(path, "r") as f:
            state = json.load(f)
        self._apply_state(state)

    def load_state_from_file(self, path):
        """Load state from a file path or name.

        Accepts:
        - Absolute path to a JSON file
        - Relative path from project root (e.g. SavedTemplates/Processing/foo.json)
        - Just a name (e.g. "Sorter Demo") → resolved to
          SavedTemplates/Processing/<name>.json
        """
        import json
        import os
        from utils.paths import project_root

        if not path:
            return
        path = str(path).strip()
        root = project_root()

        if os.path.isabs(path):
            resolved = path
        else:
            resolved = os.path.join(root, path)

        # If the direct path doesn't exist, try as a name in the default folder
        if not os.path.isfile(resolved):
            name = path
            if not name.lower().endswith(".json"):
                name += ".json"
            resolved = os.path.join(self._default_folder(), name)

        if not os.path.isfile(resolved):
            print(f"[BatchProcessing] Settings file not found: {path}")
            return
        try:
            with open(resolved, "r") as f:
                state = json.load(f)
            self._apply_state(state)
        except Exception as e:
            print(f"[BatchProcessing] Failed to load settings from {resolved}: {e}")
