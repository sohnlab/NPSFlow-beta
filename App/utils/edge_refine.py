"""FWHM boundary refinement for segment slicing (pure numpy, no Qt).

Shared by the Template Processing wizard (template segments) and Batch /
Coincident pulse processing (per-pulse segments) so the same "Edge Indices"
choice — apex peaks vs FWHM crossings — drives both.

The core moves boundary POSITIONS only; count and order are preserved.
"""

import numpy as np


def _nearest_crossing(filtered, level, ws, we, apex):
    """Index of the signal's crossing of *level* nearest *apex* within
    [ws, we]; returns *apex* when the window never crosses the level."""
    ws = max(int(ws), 0)
    we = min(int(we), len(filtered) - 1)
    if we <= ws:
        return int(apex)
    d = filtered[ws:we + 1] - level
    cross = np.nonzero(d[:-1] * d[1:] <= 0)[0]
    cross = cross[(d[cross] != 0) | (d[cross + 1] != 0)]
    if len(cross) == 0:
        return int(apex)
    f0, f1 = d[cross], d[cross + 1]
    t = np.where(f1 == f0, 0.0, f0 / np.where(f1 == f0, 1.0, f0 - f1))
    pos = ws + cross + t
    return int(round(pos[np.argmin(np.abs(pos - apex))]))


def fwhm_refine_boundaries(filtered, all_idx):
    """Move apex-based segment boundaries to FWHM (50%-crossing) positions.

    *all_idx* is the full boundary list INCLUDING the ``[0, n-1]`` sentinels;
    only interior boundaries move. Per segment, half level = midpoint of its
    plateau mean and its baseline (the higher flanking plateau mean). Each
    interior boundary becomes the crossing of the LOWER adjacent segment's
    half level — both boundaries of a pulse use the pulse's own half-depth
    level, so its width is its FWHM. When that level lies outside the
    transition the midpoint of the two plateau levels is used; when no
    crossing exists the apex is kept. Boundary count and order are preserved.
    """
    idx = np.asarray(all_idx, dtype=int)
    n = len(filtered)
    if len(idx) < 3 or n == 0:
        return idx.copy()

    refined = idx.copy()
    for _ in range(2):  # initial pass + one refinement with updated means
        nseg = len(refined) - 1
        levels = np.empty(nseg)
        for k in range(nseg):
            s = int(refined[k])
            e = int(refined[k + 1]) if k < nseg - 1 else n
            levels[k] = (float(np.mean(filtered[s:e])) if e > s
                         else float(filtered[min(s, n - 1)]))

        base = np.empty(nseg)
        for k in range(nseg):
            flanks = ([levels[k - 1]] if k > 0 else []) + \
                     ([levels[k + 1]] if k < nseg - 1 else [])
            base[k] = max(flanks) if flanks else levels[k]
        half = 0.5 * (base + levels)

        out = refined.copy()
        for i in range(1, len(idx) - 1):
            la, lb = levels[i - 1], levels[i]
            lo, hi = (la, lb) if la <= lb else (lb, la)
            level = half[i - 1] if la <= lb else half[i]
            if not (lo < level < hi):
                level = 0.5 * (la + lb)
            # Search only within this boundary's own apex-cell (midpoints to
            # the neighboring apexes) so a corrupted level can't drag the
            # boundary onto an adjacent transition; out-of-cell -> keep apex.
            ws = (int(idx[i - 1]) + int(idx[i])) // 2
            we = (int(idx[i]) + int(idx[i + 1]) + 1) // 2
            out[i] = _nearest_crossing(filtered, level, ws, we, idx[i])
        for i in range(1, len(out) - 1):
            out[i] = max(out[i], out[i - 1] + 1)
            out[i] = min(out[i], int(idx[-1]) - (len(out) - 1 - i))
        refined = out
    return refined


def refine_interior(filtered_seg, interior_local, edge_method):
    """Refine interior boundary positions inside one segment.

    *interior_local* are boundary indices strictly inside *filtered_seg*
    (apex positions, no sentinels). For ``edge_method == 'fwhm'`` they are
    moved to FWHM crossings via :func:`fwhm_refine_boundaries` (sentinels
    ``0`` and ``len-1`` added internally); otherwise they are returned
    unchanged. Count and order are preserved. Returns an int ndarray.
    """
    interior = np.asarray(interior_local, dtype=int).ravel()
    if edge_method != 'fwhm' or len(interior) == 0 or len(filtered_seg) < 3:
        return interior.copy()
    n = len(filtered_seg)
    sent = np.concatenate([[0], interior, [n - 1]])
    refined = fwhm_refine_boundaries(np.asarray(filtered_seg, dtype=float),
                                     sent)
    return refined[1:-1].astype(int)
