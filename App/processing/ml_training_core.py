"""Core (non-Qt) machinery for the ML Training block.

Trains a 3-class event classifier — baseline (0) / single (1) / coincident (2)
— from labeled files produced by the Data Labeling block, and provides a
causal streaming classifier for real-time inference. Noise and background
share label 0 ("baseline"); uncertain (3) is dropped.

Adapted from scripts/train_event_classifier.py (onset-triggered early
classifier, see docs/superpowers/specs/2026-07-02-event-early-classifier-
design.md). Files are split train/validation/test at the *file* level so
windows from one recording never leak across splits.
"""

import json
import os

import numpy as np

from utils.paths import project_root

DATA_KEYS =("labeledData.data", "data")
LABEL_KEYS = ("labeledData.labels", "labels")

CLASS_NAMES = {0: "baseline", 1: "single", 2: "coincident"}
DROP_LABEL = 3  # uncertain — excluded from training and noise sampling

DEFAULT_HORIZONS_MS = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]  # + "full" appended
DEFAULT_FS = 50000.0
BASELINE_MS = 5.0     # pre-onset window used to zero/normalize each zone
ACTIVE_K = 4.0        # a zone sample is "active" when |z-score| exceeds this
SUBPULSE_MIN_SEP_MS = 0.5  # min separation for multi-pulse peak picking
NOISE_GUARD_MS = 5.0  # keep sampled noise windows this far from any event

DEFAULT_TRIGGER = {
    "trigger_k": 4.0,      # smoothed max-zone |z| above this arms the trigger
    "release_k": 2.0,      # smoothed max-zone |z| below this counts as quiet
    "smooth_ms": 0.6,      # causal moving-average window for the |z| statistic
    "min_active_ms": 0.3,  # sustained-active time to confirm an onset
    "end_quiet_ms": 2.0,   # sustained-quiet time to close an event
    "max_event_ms": 500.0,
    "baseline_ms": BASELINE_MS,
}

DEFAULT_CONFIG = {
    "folder": "",
    "fs": DEFAULT_FS,
    "noise_ids": [],      # raw label ids remapped to baseline (0)
    "drop_ids": [3, 4],   # raw label ids dropped as "uncertain"
    "horizons_ms": list(DEFAULT_HORIZONS_MS),
    "model": "rf",              # rf | gb
    "strategy": "two_stage",    # two_stage | flat
    "oversample": "smote",      # smote | random | none
    "oversample_ratio": 1.0,
    "smote_k": 5,
    "noise_ratio": 1.0,
    "hard_negative_ratio": 1.0,
    "seed": 0,
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "assignments": {},          # file name -> auto|train|val|test
    "final_refit": "train+val", # train | train+val | all
    "stream_eval": True,
    "trigger": dict(DEFAULT_TRIGGER),
    "modelPath": os.path.join("Output", "EventClassifier",
                              "event_classifier.joblib"),
}


def parse_ids(text):
    """Ints from a comma/space-separated string, skipping bad tokens."""
    out = []
    for tok in str(text).replace(",", " ").split():
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


def parse_floats(text):
    """Floats from a comma/space-separated string, skipping bad tokens."""
    out = []
    for tok in str(text).replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


# ----------------------------------------------------------------------------
# Data loading & event parsing (pure)
# ----------------------------------------------------------------------------

def _pick(source, candidates):
    for name in candidates:
        if name in source:
            return name
    return None


def find_labeled_files(folder):
    """Recursively list labeled .npz / .csv files under *folder*, sorted."""
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".npz", ".csv")):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_labeled_npz(path):
    """Return (data (N,Z) float64, labels (N,) int32) from a labeled .npz."""
    with np.load(path, allow_pickle=True) as npz:
        dk = _pick(npz.files, DATA_KEYS)
        lk = _pick(npz.files, LABEL_KEYS)
        if dk is None or lk is None:
            raise ValueError(f"{path}: missing data/labels arrays")
        data = np.asarray(npz[dk], dtype=np.float64)
        labels = np.asarray(npz[lk]).astype(np.int32)
    if data.ndim == 1:
        data = data[:, None]
    if len(data) != len(labels):
        raise ValueError(f"{path}: data/labels length mismatch")
    return data, labels


def load_labeled_csv(path):
    """Return (data, labels) from a Data Labeling CSV export.

    Expects a ``label_id`` column and one or more ``value*`` columns
    (``value`` or ``value_zone1..N``); falls back to last column = labels,
    all other numeric columns = data.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = list(df.columns)
    label_col = _pick(cols, ("label_id", "label", "labels"))
    value_cols = [c for c in cols if str(c).lower().startswith("value")]
    if label_col is None:
        label_col = cols[-1]
    if not value_cols:
        value_cols = [c for c in cols
                      if c != label_col and str(c).lower() != "sample_index"]
    if not value_cols:
        raise ValueError(f"{path}: no value columns found")
    data = df[value_cols].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy().astype(np.int32)
    return data, labels


def load_labeled_file(path):
    if path.lower().endswith(".npz"):
        return load_labeled_npz(path)
    return load_labeled_csv(path)


def load_labels_only(path):
    """Labels array without loading the (much larger) data — for fast scans."""
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as npz:
            lk = _pick(npz.files, LABEL_KEYS)
            if lk is None:
                raise ValueError(f"{path}: missing labels array")
            return np.asarray(npz[lk]).astype(np.int32)
    import pandas as pd
    head = pd.read_csv(path, nrows=0)
    label_col = _pick(list(head.columns), ("label_id", "label", "labels")) \
        or list(head.columns)[-1]
    return pd.read_csv(path, usecols=[label_col])[label_col] \
        .to_numpy().astype(np.int32)


def canonicalize_labels(labels, noise_ids=(), drop_ids=(3, 4)):
    """Map raw label ids onto the fixed 0/1/2/uncertain convention.

    1/2 stay single/coincident, *drop_ids* become uncertain (excluded from
    training, ignored in scoring), *noise_ids* and everything else collapse
    to baseline (0) — noise is baseline by definition for this classifier.
    """
    lab = np.asarray(labels).astype(np.int32).ravel()
    out = np.zeros_like(lab)
    out[lab == 1] = 1
    out[lab == 2] = 2
    for i in drop_ids:
        if int(i) not in (1, 2):
            out[lab == int(i)] = DROP_LABEL
    for i in noise_ids:
        out[lab == int(i)] = 0
    return out


def parse_events(labels):
    """Split a per-sample label array into maximal contiguous runs.

    Returns a list of (start, end, label) with end exclusive, covering every
    run including background (label 0). Callers filter by label.
    """
    labels = np.asarray(labels)
    n = len(labels)
    events = []
    i = 0
    while i < n:
        v = int(labels[i])
        j = i + 1
        while j < n and labels[j] == v:
            j += 1
        events.append((i, j, v))
        i = j
    return events


def sample_noise_windows(events, fs, n_windows, length_pool, rng,
                         guard_ms=NOISE_GUARD_MS, baseline_ms=BASELINE_MS):
    """Sample baseline "events" (start, end, 0) from label-0 runs.

    Each window is placed inside a background run with room for a pre-onset
    baseline and a guard margin from the run edges (which abut real events).
    ``length_pool`` supplies synthetic event lengths drawn to match the real
    event-length distribution.
    """
    guard = int(round(guard_ms / 1000.0 * fs))
    baseline = int(round(baseline_ms / 1000.0 * fs))
    length_pool = np.asarray(length_pool, dtype=int)
    if length_pool.size == 0:
        length_pool = np.array([int(0.02 * fs)])  # 20 ms fallback

    bg = [(s, e) for (s, e, v) in events if v == 0]
    min_room = baseline + guard + int(length_pool.min()) + guard
    usable = [(s, e) for (s, e) in bg if (e - s) >= min_room]
    if not usable:
        return []

    weights = np.array([e - s for (s, e) in usable], dtype=float)
    weights /= weights.sum()

    out = []
    attempts = 0
    while len(out) < n_windows and attempts < n_windows * 20:
        attempts += 1
        run = usable[rng.choice(len(usable), p=weights)]
        s, e = run
        length = int(length_pool[rng.integers(len(length_pool))])
        lo = s + baseline + guard
        hi = e - guard - length
        if hi <= lo:
            length = min(length, (e - guard) - (s + baseline + guard))
            hi = e - guard - length
            if hi <= lo or length <= 0:
                continue
        onset = int(rng.integers(lo, hi + 1))
        out.append((onset, onset + length, 0))
    return out


# ----------------------------------------------------------------------------
# Feature extraction (pure)
# ----------------------------------------------------------------------------

def smooth_causal(x, n):
    """Causal moving average over the trailing *n* samples (ramp-up at t<n)."""
    if n <= 1:
        return np.asarray(x, dtype=float)
    c = np.cumsum(np.concatenate([[0.0], np.asarray(x, dtype=float)]))
    lo = np.maximum(0, np.arange(len(x)) + 1 - n)
    return (c[1:] - c[lo]) / (np.arange(len(x)) + 1 - lo)


_PER_ZONE_STATS = [
    "mean", "std", "rms", "min", "max", "range", "area",
    "maxabs", "argmax_frac", "slope", "n_cross", "n_extrema",
]


def feature_names(n_zones):
    names = []
    for z in range(n_zones):
        names += [f"z{z}_{s}" for s in _PER_ZONE_STATS]
    for a in range(n_zones):
        for b in range(a + 1, n_zones):
            names.append(f"corr_{a}{b}")
    names += ["amp_max", "amp_min", "amp_ratio", "amp_std", "n_zones_active"]
    names += ["n_subpulses", "n_soft_subpulses", "subpeak_spacing_frac",
              "active_frac"]
    names += ["n_rising_edges", "late_activity_ratio", "tail_slope",
              "tail_head_amp_ratio"]
    names += ["horizon_samples"]
    return names


def _linfit_slope(y):
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(((x - xm) * (y - y.mean())).sum() / denom)


def extract_features(data, onset, h_eff, fs,
                     baseline_ms=BASELINE_MS, active_k=ACTIVE_K,
                     subpulse_min_sep_ms=SUBPULSE_MIN_SEP_MS):
    """Causal feature vector for the window ``data[onset:onset+h_eff]``.

    Uses only samples up to ``onset + h_eff`` plus a *pre-onset* baseline
    window (past samples, causally available). Returns a 1-D float array whose
    order matches ``feature_names(data.shape[1])``.
    """
    from scipy.signal import find_peaks

    n_total, n_zones = data.shape
    h_eff = int(max(1, min(h_eff, n_total - onset)))
    win = data[onset:onset + h_eff]  # (h_eff, Z)

    baseline = int(round(baseline_ms / 1000.0 * fs))
    b0 = max(0, onset - baseline)
    pre = data[b0:onset]
    if len(pre) >= 2:
        base_mean = pre.mean(axis=0)
        base_std = pre.std(axis=0)
    else:
        # No usable pre-window (event at very start): use the window itself.
        base_mean = win.mean(axis=0)
        base_std = win.std(axis=0)
    eps = 1e-9
    base_std = base_std + eps

    dev = win - base_mean            # (h_eff, Z) baseline-corrected
    zsc = dev / base_std             # per-zone z-score
    maxabs = np.max(np.abs(dev), axis=0)  # (Z,)

    feats = []
    for z in range(n_zones):
        d = dev[:, z]
        feats += [
            float(d.mean()),
            float(d.std()),
            float(np.sqrt(np.mean(d * d))),
            float(d.min()),
            float(d.max()),
            float(d.max() - d.min()),
            float(d.sum() / fs),                       # signed area (unit·s)
            float(maxabs[z]),
            float(np.argmax(np.abs(d)) / max(1, len(d) - 1)),
            _linfit_slope(d),
            float(np.sum(np.diff((np.abs(zsc[:, z]) > active_k).astype(int)) == 1)),
            _count_extrema(np.abs(d), find_peaks),
        ]

    for a in range(n_zones):
        for b in range(a + 1, n_zones):
            feats.append(_safe_corr(dev[:, a], dev[:, b]))

    amp = maxabs
    feats += [
        float(amp.max()),
        float(amp.min()),
        float(amp.max() / (amp.min() + eps)),
        float(amp.std()),
        float(np.sum(np.max(np.abs(zsc), axis=0) > active_k)),
    ]

    # shape / multi-pulse on combined magnitude across zones.
    # prominence requires a real dip between peaks, so noise ripple on a single
    # broad pulse is not miscounted as multiple sub-pulses (single vs coincident
    # hinges on this: coincident = >=2 prominent, separated sub-pulses).
    mag = np.max(np.abs(zsc), axis=1)  # (h_eff,)
    min_sep = max(2, int(round(subpulse_min_sep_ms / 1000.0 * fs)))
    peaks, _ = find_peaks(mag, height=active_k, distance=min_sep,
                          prominence=active_k)
    n_sub = int(len(peaks))
    # "soft" counter with a shallower dip requirement catches a partially
    # overlapping second pulse earlier than the strict counter.
    soft_peaks, _ = find_peaks(mag, height=active_k, distance=min_sep,
                               prominence=active_k * 0.5)
    n_soft = int(len(soft_peaks))
    if n_sub >= 2:
        spacing = float((peaks[-1] - peaks[0]) / (n_sub - 1) / max(1, len(mag)))
    else:
        spacing = 0.0
    feats += [
        float(n_sub),
        float(n_soft),
        spacing,
        float(np.mean(mag > active_k)),
    ]

    # Coincident-targeted early-arrival cues: a second pulse shows up as a
    # fresh rising edge / growing activity in the *tail* of the window, often
    # before the strict sub-pulse counter can confirm a separated peak.
    rising = _count_bursts(mag > active_k, min_gap=min_sep,
                           min_run=max(2, min_sep // 5))
    q = max(1, len(mag) // 3)
    head_max = float(np.max(mag[:q]))
    tail_max = float(np.max(mag[-q:]))
    late_ratio = tail_max / (head_max + eps)
    tail_slope = _linfit_slope(mag[-q:]) if q >= 2 else 0.0
    head_amp = float(np.max(np.abs(dev[:q]))) if q >= 1 else 0.0
    tail_amp = float(np.max(np.abs(dev[-q:]))) if q >= 1 else 0.0
    tail_head_amp = tail_amp / (head_amp + eps)
    feats += [
        float(rising),
        float(late_ratio),
        float(tail_slope),
        float(tail_head_amp),
    ]

    feats.append(float(h_eff))
    return np.asarray(feats, dtype=np.float64)


def _count_bursts(active, min_gap, min_run):
    """Count sustained 'active' runs, debounced against threshold noise.

    Runs of True separated by a False gap shorter than ``min_gap`` are merged
    (a brief dip inside one pulse), and runs shorter than ``min_run`` are
    dropped (a 1-2 sample noise spike).
    """
    idx = np.flatnonzero(np.asarray(active, dtype=bool))
    if idx.size == 0:
        return 0
    start = prev = idx[0]
    runs = []
    for k in idx[1:]:
        if k - prev <= min_gap:
            prev = k
        else:
            runs.append((start, prev))
            start = prev = k
    runs.append((start, prev))
    return sum(1 for (s, e) in runs if (e - s + 1) >= min_run)


def _count_extrema(mag, find_peaks):
    if len(mag) < 3:
        return 0.0
    thr = np.mean(mag) + np.std(mag)
    peaks, _ = find_peaks(mag, height=thr)
    return float(len(peaks))


def _safe_corr(a, b):
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


# ----------------------------------------------------------------------------
# Dataset assembly & splitting
# ----------------------------------------------------------------------------

def horizon_label(h_ms):
    """Stable string key for a horizon (used for per-horizon metric slicing)."""
    return "full" if h_ms is None else f"{h_ms:g}ms"


def horizon_sort_key(hz):
    if hz == "full":
        return float("inf")
    return float(hz.replace("ms", ""))


def refine_event_extent(data, s, e, fs, trigger=None, fallback=True):
    """Trigger-aligned (onset, end) inside a labeled span [s, e).

    Labeled regions carry leading/trailing margin; the streaming detector
    anchors at the sustained-threshold crossing instead. Re-deriving the same
    extent here makes training windows match deployment windows. When no
    sustained crossing exists, returns the span unchanged (*fallback*) or
    None (fallback=False, used for hard-negative mining).
    """
    cfg = dict(DEFAULT_TRIGGER)
    cfg.update(trigger or {})
    baseline = int(round(cfg["baseline_ms"] / 1000.0 * fs))
    pre = data[max(0, s - baseline):s]
    if len(pre) < 2:
        return s, e
    mean = pre.mean(axis=0)
    std = pre.std(axis=0) + 1e-9
    smooth_n = max(1, int(round(cfg["smooth_ms"] / 1000.0 * fs)))
    mag = smooth_causal(
        np.abs((data[s:e] - mean) / std).max(axis=1), smooth_n)
    act = mag > cfg["trigger_k"]
    min_active = max(1, int(round(cfg["min_active_ms"] / 1000.0 * fs)))
    onset = None
    run = 0
    for i, a in enumerate(act):
        run = run + 1 if a else 0
        if run >= min_active:
            onset = s + i - (min_active - 1)
            break
    if onset is None:
        return (s, e) if fallback else None
    loud = mag > cfg["release_k"]
    last = int(np.flatnonzero(loud).max()) if loud.any() else (e - s - 1)
    return onset, max(s + last + 1, onset + 1)


def sample_hard_negatives(data, labels, fs, trigger, n_max,
                          step_ms=100.0, win_ms=100.0):
    """Background stretches where the stream trigger would actually fire.

    Harvested as label-0 examples so the classifier learns to reject the
    false triggers it will face live — random baseline windows alone are
    too easy a negative class.
    """
    if n_max <= 0:
        return []
    baseline = int(round((dict(DEFAULT_TRIGGER, **(trigger or {})))
                         ["baseline_ms"] / 1000.0 * fs))
    step = max(1, int(round(step_ms / 1000.0 * fs)))
    win = max(baseline + 2, int(round(win_ms / 1000.0 * fs)))
    labels = np.asarray(labels)
    out = []
    for s in range(baseline, len(labels) - win, step):
        if labels[s - baseline:s + win].any():
            continue
        ext = refine_event_extent(data, s, s + win, fs, trigger,
                                  fallback=False)
        if ext is None:
            continue
        out.append((ext[0], ext[1], 0))
        if len(out) >= n_max:
            break
    return out


def build_file_dataset(data, labels, fs, horizons_ms, rng, noise_ratio=1.0,
                       trigger=None, hard_negative_ratio=1.0):
    """Build feature rows for one file.

    Returns dict with X (rows,F), y (rows,), horizon (rows,) str,
    example_id (rows,) int. Each event (real + sampled baseline windows +
    mined hard negatives) emits one row per horizon (nominal ms values plus
    "full"). Real events are trigger-aligned via refine_event_extent.
    """
    events_all = parse_events(labels)
    real = [(s, e, v) for (s, e, v) in events_all
            if v in (1, 2)]  # single / coincident only (drop uncertain)
    real = [refine_event_extent(data, s, e, fs, trigger) + (v,)
            for (s, e, v) in real]
    length_pool = [e - s for (s, e, v) in real] or \
                  [e - s for (s, e, v) in events_all if v != 0]

    n_single = sum(1 for _, _, v in real if v == 1)
    n_noise = int(round(noise_ratio * max(1, n_single)))
    noise = sample_noise_windows(events_all, fs, n_noise, length_pool, rng)
    hard = sample_hard_negatives(
        data, labels, fs, trigger,
        int(round(hard_negative_ratio * max(1, n_single))))

    examples = real + noise + hard
    horizons = list(horizons_ms) + [None]  # None -> full event

    X, y, horiz, ex_id = [], [], [], []
    for eid, (s, e, v) in enumerate(examples):
        ev_len = e - s
        for h_ms in horizons:
            if h_ms is None:
                h_eff = ev_len
            else:
                h_eff = min(int(round(h_ms / 1000.0 * fs)), ev_len)
            X.append(extract_features(data, s, h_eff, fs))
            y.append(int(v))
            horiz.append(horizon_label(h_ms))
            ex_id.append(eid)
    # keep the feature dimension even with no examples (e.g. a segment whose
    # only event is uncertain) so datasets stack cleanly
    X_arr = np.asarray(X, dtype=np.float64) if X else \
        np.zeros((0, len(feature_names(data.shape[1]))))
    return {
        "X": X_arr,
        "y": np.asarray(y, dtype=np.int32),
        "horizon": np.asarray(horiz),
        "example_id": np.asarray(ex_id, dtype=np.int32),
    }


def event_counts(labels):
    """{'single': n, 'coincident': n, 'uncertain': n} event counts."""
    events = parse_events(labels)
    return {
        "single": sum(1 for _, _, v in events if v == 1),
        "coincident": sum(1 for _, _, v in events if v == 2),
        "uncertain": sum(1 for _, _, v in events if v == DROP_LABEL),
    }


def auto_split(counts_by_file, fractions, seed, assignments=None):
    """Assign files to train/val/test at the file level.

    *counts_by_file* maps name -> event_counts() dict. Files with explicit
    (non-"auto") entries in *assignments* keep them; the rest are assigned
    greedily — richest files first (coincident, then single) to the split
    with the largest remaining event-weight deficit — so rare coincident
    events spread across all splits. Every split with a nonzero fraction is
    guaranteed at least one file when enough files exist.
    """
    assignments = dict(assignments or {})
    fixed = {n: a for n, a in assignments.items()
             if a in ("train", "val", "test") and n in counts_by_file}
    free = [n for n in counts_by_file if n not in fixed]

    fracs = {k: max(0.0, float(fractions.get(k, 0.0)))
             for k in ("train", "val", "test")}
    total_frac = sum(fracs.values()) or 1.0
    fracs = {k: v / total_frac for k, v in fracs.items()}

    def weight(name):
        c = counts_by_file[name]
        return c["single"] + c["coincident"] + 1  # +1 so empty files count

    total_w = sum(weight(n) for n in counts_by_file)
    got = {"train": 0.0, "val": 0.0, "test": 0.0}
    for n, a in fixed.items():
        got[a] += weight(n)

    rng = np.random.default_rng(seed)
    order = [free[i] for i in rng.permutation(len(free))]
    order.sort(key=lambda n: (counts_by_file[n]["coincident"],
                              counts_by_file[n]["single"]), reverse=True)

    out = dict(fixed)
    for n in order:
        deficits = {k: fracs[k] * total_w - got[k]
                    for k in ("train", "val", "test") if fracs[k] > 0}
        best = max(deficits, key=lambda k: deficits[k])
        out[n] = best
        got[best] += weight(n)

    # Guarantee nonempty splits (steal from the largest split).
    for k in ("test", "val", "train"):
        if fracs[k] <= 0 or any(a == k for a in out.values()):
            continue
        by_size = sorted(
            ((sum(1 for a in out.values() if a == s), s)
             for s in ("train", "val", "test")), reverse=True)
        donor = by_size[0][1]
        movable = [n for n in order if out.get(n) == donor]
        if movable:
            out[movable[-1]] = k
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def make_model(kind, seed):
    if kind == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=seed)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        min_samples_leaf=2, n_jobs=-1, random_state=seed,
    )


def _smote(Xm, n_synth, k, rng):
    """Interpolate ``n_synth`` synthetic minority rows from ``Xm`` (SMOTE).

    Neighbours are found in standardized space; interpolation is in raw
    feature space. Falls back to replication when there are too few seeds.
    """
    n = len(Xm)
    if n < 2:
        return Xm[rng.integers(0, max(1, n), size=n_synth)] if n else \
            np.empty((0, Xm.shape[1]))
    from sklearn.neighbors import NearestNeighbors
    kk = min(k, n - 1)
    std = Xm.std(axis=0)
    std[std == 0] = 1.0
    Xs = (Xm - Xm.mean(axis=0)) / std
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xs)
    _, nbrs = nn.kneighbors(Xs)
    out = np.empty((n_synth, Xm.shape[1]), dtype=float)
    for t in range(n_synth):
        i = int(rng.integers(n))
        j = int(nbrs[i, rng.integers(1, nbrs.shape[1])])  # skip self at col 0
        g = rng.random()
        out[t] = Xm[i] + g * (Xm[j] - Xm[i])
    return out


def oversample_train(X, y, horizon, rng, kind, ratio, k,
                     minority=2, majority_ref=1):
    """Grow the minority class per horizon so it isn't drowned out.

    Oversampling is done *within each horizon group* so a synthesized row
    never mixes a 1 ms window with a 40 ms one. ``kind`` is
    'none' | 'smote' | 'random'.
    """
    if kind == "none" or ratio <= 0:
        return X, y
    Xa, ya = [X], [y]
    for hz in np.unique(horizon):
        m = horizon == hz
        Xh, yh = X[m], y[m]
        n_min = int(np.sum(yh == minority))
        n_maj = int(np.sum(yh == majority_ref))
        target = int(round(ratio * n_maj))
        n_synth = target - n_min
        if n_min < 1 or n_synth <= 0:
            continue
        Xmin = Xh[yh == minority]
        if kind == "smote":
            syn = _smote(Xmin, n_synth, k, rng)
        else:
            syn = Xmin[rng.integers(0, len(Xmin), size=n_synth)]
        Xa.append(syn)
        ya.append(np.full(len(syn), minority, dtype=y.dtype))
    return np.vstack(Xa), np.concatenate(ya)


def train_classifier(X, y, horizon, strategy, model_kind, seed, rng,
                     oversample="smote", ratio=1.0, k=5):
    """Fit the classifier and return a predictor bundle (dict).

    strategy 'flat'      -> single 3-way model.
    strategy 'two_stage' -> stage A event-vs-baseline, stage B single-vs-
                            coincident trained on event rows only.
    """
    if strategy == "flat":
        Xo, yo = oversample_train(X, y, horizon, rng, oversample, ratio, k)
        model = make_model(model_kind, seed)
        model.fit(Xo, yo)
        return {"strategy": "flat", "flat": model}

    yA = (y != 0).astype(np.int32)          # 1 = event, 0 = baseline
    stageA = make_model(model_kind, seed)
    stageA.fit(X, yA)

    ev = y != 0
    XB, yB, hB = X[ev], y[ev], horizon[ev]
    XoB, yoB = oversample_train(XB, yB, hB, rng, oversample, ratio, k)
    stageB = make_model(model_kind, seed)
    stageB.fit(XoB, yoB)
    return {"strategy": "two_stage", "stageA": stageA, "stageB": stageB}


def predict_classifier(bundle, X):
    """3-way prediction (0/1/2) from a bundle produced by train_classifier."""
    if bundle["strategy"] == "flat":
        return bundle["flat"].predict(X).astype(np.int32)
    pred = np.zeros(len(X), dtype=np.int32)
    is_event = bundle["stageA"].predict(X) == 1
    if is_event.any():
        pred[is_event] = bundle["stageB"].predict(X[is_event]).astype(np.int32)
    return pred


def predict_proba_classifier(bundle, X):
    """(n,3) class probabilities [baseline, single, coincident]."""
    out = np.zeros((len(X), 3), dtype=float)
    if bundle["strategy"] == "flat":
        model = bundle["flat"]
        proba = model.predict_proba(X)
        for i, cls in enumerate(model.classes_):
            out[:, int(cls)] = proba[:, i]
        return out
    pA = bundle["stageA"].predict_proba(X)
    a_classes = list(bundle["stageA"].classes_)
    p_event = pA[:, a_classes.index(1)] if 1 in a_classes else np.zeros(len(X))
    out[:, 0] = 1.0 - p_event
    pB = bundle["stageB"].predict_proba(X)
    for i, cls in enumerate(bundle["stageB"].classes_):
        out[:, int(cls)] += p_event * pB[:, i]
    return out


def feature_importance_ranking(bundle, names, top=15):
    """[(name, importance)] sorted desc, averaged across stages."""
    models = ([bundle["flat"]] if bundle["strategy"] == "flat"
              else [bundle["stageA"], bundle["stageB"]])
    imps = [m.feature_importances_ for m in models
            if hasattr(m, "feature_importances_")]
    if not imps:
        return []
    mean_imp = np.mean(imps, axis=0)
    order = np.argsort(mean_imp)[::-1][:top]
    return [(names[i], float(mean_imp[i])) for i in order]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def per_horizon_metrics(y_true, y_pred, horizons, present_labels):
    """Return {horizon: {accuracy, macro_f1, per_class, confusion, n}}."""
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix)
    out = {}
    label_order = [0, 1, 2]
    for hz in sorted(set(horizons), key=horizon_sort_key):
        m = horizons == hz
        yt, yp = y_true[m], y_pred[m]
        if len(yt) == 0:
            continue
        prec, rec, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=label_order, zero_division=0)
        per_class = {}
        for i, lab in enumerate(label_order):
            present = lab in present_labels
            per_class[CLASS_NAMES[lab]] = {
                "precision": float(prec[i]) if present else None,
                "recall": float(rec[i]) if present else None,
                "f1": float(f1[i]) if present else None,
                "support": int(sup[i]),
            }
        macro_labels = [l for l in label_order if l in present_labels]
        out[hz] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, labels=macro_labels,
                                       average="macro", zero_division=0)),
            "per_class": per_class,
            "confusion": confusion_matrix(
                yt, yp, labels=label_order).tolist(),
            "n": int(len(yt)),
        }
    return out


def evaluate_bundle(bundle, datasets, names):
    """Pool the named datasets and score per horizon."""
    names = [n for n in names if len(datasets[n]["y"])]
    if not names:
        return {}
    X = np.vstack([datasets[n]["X"] for n in names])
    y = np.concatenate([datasets[n]["y"] for n in names])
    hz = np.concatenate([datasets[n]["horizon"] for n in names])
    pred = predict_classifier(bundle, X)
    present = set(int(v) for v in np.unique(y))
    return per_horizon_metrics(y, pred, hz, present)


# ----------------------------------------------------------------------------
# Streaming classifier (real-time inference)
# ----------------------------------------------------------------------------

class StreamEventClassifier:
    """Causal chunk-by-chunk classifier for a live data stream.

    Feed raw samples via :meth:`process` (any chunk size, (n,) or (n, Z));
    finalized events come back as dicts. Unlabeled stretches are baseline by
    definition; a trigger that the model rejects is returned with label 0.

    Detection statistic: per-sample max-zone |z| against a rolling baseline,
    smoothed by a short causal moving average so isolated noisy samples
    neither trigger nor block release. The baseline only adapts while the
    smoothed statistic is deeply quiet (hysteresis), freezes during events,
    and re-seeds from scratch after each event so post-pulse level shifts
    self-heal. Sustained smoothed activity above trigger_k opens an event;
    the model classifies at each latency horizon as the window grows
    (optional ``on_update`` callback for early decisions); sustained quiet
    below release_k — or max_event_ms — closes it, and the final label comes
    from the full-event window.
    """

    def __init__(self, saved, on_update=None, **trigger_overrides):
        self.bundle = saved["bundle"]
        self.fs = float(saved["fs"])
        self.n_zones = int(saved.get("n_zones", 0)) or None
        self.horizons_ms = sorted(float(h) for h in saved.get(
            "horizons_ms", DEFAULT_HORIZONS_MS))
        cfg = dict(DEFAULT_TRIGGER)
        cfg.update(saved.get("trigger") or {})
        cfg.update(trigger_overrides)
        self.cfg = cfg
        self.on_update = on_update

        fs = self.fs
        self._baseline_n = max(2, int(round(cfg["baseline_ms"] / 1000 * fs)))
        self._smooth_n = max(1, int(round(cfg["smooth_ms"] / 1000 * fs)))
        self._min_active = max(1, int(round(cfg["min_active_ms"] / 1000 * fs)))
        self._end_quiet = max(1, int(round(cfg["end_quiet_ms"] / 1000 * fs)))
        self._max_event = max(self._end_quiet + 1,
                              int(round(cfg["max_event_ms"] / 1000 * fs)))
        self._horizon_samp = [max(1, int(round(h / 1000 * fs)))
                              for h in self.horizons_ms]

        self._buf = None          # (n, Z) rolling buffer
        self._buf_g0 = 0          # global index of buf[0]
        self._n_seen = 0
        self._cursor_g = 0        # next sample to scan
        self._seed_from_g = 0     # where the next baseline seed starts
        self._mean = None
        self._std = None
        self._mag_tail = np.empty(0)
        self._state = "quiet"
        self._onset_g = None
        self._active_run = 0
        self._quiet_run = 0
        self._updates = []
        self._next_h = 0

    # -- public API --------------------------------------------------------

    def process(self, chunk):
        """Consume a chunk of samples; return list of finalized events."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.size == 0:
            return []
        if self.n_zones is None:
            self.n_zones = chunk.shape[1]
        elif chunk.shape[1] != self.n_zones:
            raise ValueError(
                f"chunk has {chunk.shape[1]} zones, model expects "
                f"{self.n_zones}")

        if self._buf is None:
            self._buf = chunk.copy()
        else:
            self._buf = np.vstack([self._buf, chunk])
        self._n_seen += len(chunk)

        events = []
        while True:
            i = self._cursor_g - self._buf_g0
            end = len(self._buf)
            if i >= end:
                break
            if self._mean is None:
                lo = self._seed_from_g - self._buf_g0
                if end - lo < self._baseline_n:
                    break  # not enough post-seed history yet
                seed = self._buf[lo:lo + self._baseline_n]
                self._mean = seed.mean(axis=0)
                self._std = seed.std(axis=0) + 1e-9
                self._mag_tail = np.empty(0)
                self._cursor_g = max(self._cursor_g,
                                     self._seed_from_g + self._baseline_n)
                continue
            if self._state == "quiet":
                consumed, done = self._run_quiet(i, end)
            else:
                consumed, done = self._run_event(i, end)
            events.extend(done)
            if consumed <= i:
                break
            self._cursor_g = self._buf_g0 + consumed
        self._trim()
        return events

    def finish(self):
        """Flush an open event at end-of-stream; returns finalized events."""
        if self._state != "event":
            return []
        ev = self._finalize(self._n_seen)
        self._reset_after_event(self._n_seen)
        return [ev]

    # -- internals ---------------------------------------------------------

    def _mag(self, lo, hi):
        """Raw + causally smoothed max-zone |z| for buf[lo:hi]."""
        raw = np.abs((self._buf[lo:hi] - self._mean) / self._std).max(axis=1)
        ext = np.concatenate([self._mag_tail, raw]) \
            if self._mag_tail.size else raw
        sm = smooth_causal(ext, self._smooth_n)
        return sm[len(ext) - len(raw):], ext

    def _keep_tail(self, ext, tail_len, consumed_count):
        upto = tail_len + consumed_count
        if self._smooth_n > 1:
            self._mag_tail = ext[:upto][-(self._smooth_n - 1):]
        else:
            self._mag_tail = np.empty(0)

    def _first_sustained(self, flags, carry, need):
        """(run_start, confirm) indices in *flags* coords for the first run of
        *need* Trues (counting *carry* carried-in Trues), or None."""
        a2 = np.concatenate([np.ones(carry, dtype=bool),
                             np.asarray(flags, dtype=bool)])
        if len(a2) < need:
            return None
        c = np.convolve(a2.astype(np.int32),
                        np.ones(need, dtype=np.int32), "valid")
        hits = np.flatnonzero(c >= need)
        if not hits.size:
            return None
        confirm = int(hits[0]) + need - 1
        falses = np.flatnonzero(~a2[:confirm + 1])
        run_start = int(falses.max()) + 1 if falses.size else 0
        return run_start - carry, confirm - carry

    def _trailing_run(self, flags, carry):
        rev = np.flatnonzero(~np.asarray(flags, dtype=bool))
        if rev.size:
            return len(flags) - (int(rev.max()) + 1)
        return carry + len(flags)

    def _update_baseline(self, lo, sm_slice):
        """Fold deeply-quiet samples of buf[lo:lo+len] into the rolling stats."""
        if len(sm_slice) == 0:
            return
        m = sm_slice < self.cfg["release_k"]
        if not m.any():
            return
        block = self._buf[lo:lo + len(sm_slice)][m]
        w = min(1.0, len(block) / self._baseline_n)
        self._mean = (1 - w) * self._mean + w * block.mean(axis=0)
        self._std = (1 - w) * self._std + w * (block.std(axis=0) + 1e-9)

    def _run_quiet(self, i, end):
        sm, ext = self._mag(i, end)
        tail_len = len(ext) - len(sm)
        act = sm > self.cfg["trigger_k"]
        hit = self._first_sustained(act, self._active_run, self._min_active)
        if hit is None:
            self._active_run = min(self._trailing_run(act, self._active_run),
                                   self._min_active - 1)
            n_quiet = len(sm) - min(self._active_run, len(sm))
            self._update_baseline(i, sm[:n_quiet])
            self._keep_tail(ext, tail_len, len(sm))
            return end, []
        run_start, confirm = hit
        self._update_baseline(i, sm[:max(0, run_start)])
        self._onset_g = self._buf_g0 + i + run_start
        self._state = "event"
        self._updates = []
        self._next_h = 0
        self._quiet_run = 0
        self._active_run = 0
        consumed = i + confirm + 1
        self._keep_tail(ext, tail_len, consumed - i)
        return consumed, []

    def _run_event(self, i, end):
        sm, ext = self._mag(i, end)
        tail_len = len(ext) - len(sm)
        onset_local = self._onset_g - self._buf_g0
        quiet = sm < self.cfg["release_k"]
        hit = self._first_sustained(quiet, self._quiet_run, self._end_quiet)
        exp_local = onset_local + self._max_event - 1
        if hit is not None:
            run_start, confirm = hit
            close_local = i + confirm       # sample where quiet confirmed
            quiet_start = i + run_start     # event actually ended here
        else:
            close_local = quiet_start = None
        if exp_local < end and (close_local is None
                                or exp_local < close_local):
            stop_local = max(i, exp_local)
            end_g = self._buf_g0 + stop_local + 1
        elif close_local is not None:
            stop_local = close_local
            end_g = self._buf_g0 + max(quiet_start, onset_local + 1)
        else:
            self._fire_updates(end - onset_local)
            self._quiet_run = min(self._trailing_run(quiet, self._quiet_run),
                                  self._end_quiet - 1)
            self._keep_tail(ext, tail_len, len(sm))
            return end, []
        self._fire_updates(stop_local + 1 - onset_local)
        ev = self._finalize(end_g)
        consumed = stop_local + 1
        self._keep_tail(ext, tail_len, consumed - i)
        self._reset_after_event(end_g)
        return consumed, [ev]

    def _fire_updates(self, elapsed):
        while (self._next_h < len(self._horizon_samp)
               and elapsed >= self._horizon_samp[self._next_h]):
            h = self._horizon_samp[self._next_h]
            label, proba = self._classify(h)
            upd = {"h_ms": self.horizons_ms[self._next_h],
                   "label": label, "name": CLASS_NAMES[label],
                   "proba": proba}
            self._updates.append(upd)
            self._next_h += 1
            if self.on_update:
                self.on_update(self._onset_g, upd)

    def _classify(self, h_eff):
        onset_local = self._onset_g - self._buf_g0
        feats = extract_features(
            self._buf, onset_local, h_eff, self.fs,
            baseline_ms=self.cfg["baseline_ms"])[None, :]
        label = int(predict_classifier(self.bundle, feats)[0])
        try:
            proba = predict_proba_classifier(self.bundle, feats)[0].tolist()
        except Exception:
            proba = None
        return label, proba

    def _finalize(self, end_g):
        h_eff = max(1, end_g - self._onset_g)
        label, proba = self._classify(h_eff)
        return {
            "onset": int(self._onset_g),
            "end": int(end_g),
            "duration_ms": 1000.0 * h_eff / self.fs,
            "label": label,
            "name": CLASS_NAMES[label],
            "proba": proba,
            "updates": list(self._updates),
        }

    def _reset_after_event(self, end_g):
        self._state = "quiet"
        self._onset_g = None
        self._active_run = 0
        self._quiet_run = 0
        self._updates = []
        self._next_h = 0
        self._mean = None
        self._std = None
        self._mag_tail = np.empty(0)
        self._seed_from_g = end_g

    def _trim(self):
        if self._buf is None:
            return
        if self._state == "event":
            keep_g = self._onset_g - self._baseline_n - self._smooth_n
        elif self._mean is None:
            keep_g = min(self._seed_from_g, self._cursor_g)
        else:
            keep_g = self._cursor_g - (self._baseline_n + self._smooth_n
                                       + self._min_active + self._end_quiet
                                       + 8)
        cut = max(0, keep_g - self._buf_g0)
        if cut > 0:
            self._buf = self._buf[cut:]
            self._buf_g0 += cut


def simulate_stream(saved, data, chunk_size=65536, progress=None):
    """Replay *data* through a StreamEventClassifier; return event list."""
    clf = StreamEventClassifier(saved)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    events = []
    n = len(data)
    for s in range(0, n, chunk_size):
        events.extend(clf.process(data[s:s + chunk_size]))
        if progress:
            progress(min(1.0, (s + chunk_size) / n))
    events.extend(clf.finish())
    return events


def events_to_labels(events, n):
    """Per-sample predicted label array (int32, length n) from stream events."""
    out = np.zeros(int(n), dtype=np.int32)
    for ev in events:
        if ev.get("label", 0) > 0:
            out[max(0, int(ev["onset"])):max(0, int(ev["end"]))] = ev["label"]
    return out


def score_stream(events, labels, fs):
    """Event-level scoring of streamed predictions against per-sample labels.

    Ground truth = contiguous label-1/2 runs (uncertain runs are ignored, not
    penalized). A predicted event with label>0 matches the ground-truth event
    it overlaps most; unmatched positives count as false positives.
    """
    gt = [(s, e, v) for (s, e, v) in parse_events(labels) if v in (1, 2)]
    unc = [(s, e) for (s, e, v) in parse_events(labels) if v == DROP_LABEL]
    pos = [ev for ev in events if ev["label"] > 0]

    def overlap(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0))

    matched = {}
    used = set()
    for gi, (s, e, _v) in enumerate(gt):
        best, best_ov = None, 0
        for pi, ev in enumerate(pos):
            if pi in used:
                continue
            ov = overlap(s, e, ev["onset"], ev["end"])
            if ov > best_ov:
                best, best_ov = pi, ov
        if best is not None:
            matched[gi] = best
            used.add(best)

    conf = np.zeros((2, 3), dtype=int)  # rows: true 1,2; cols: missed,1,2
    for gi, (s, e, v) in enumerate(gt):
        r = v - 1
        if gi not in matched:
            conf[r, 0] += 1
        else:
            conf[r, pos[matched[gi]]["label"]] += 1

    fp = 0
    for pi, ev in enumerate(pos):
        if pi in used:
            continue
        if any(overlap(s, e, ev["onset"], ev["end"]) > 0 for s, e in unc):
            continue
        fp += 1

    minutes = len(labels) / fs / 60.0
    n_gt = len(gt)
    n_det = len(matched)
    n_correct = sum(1 for gi, pi in matched.items()
                    if pos[pi]["label"] == gt[gi][2])
    false_triggers = sum(1 for ev in events if ev["label"] == 0)
    return {
        "n_events": n_gt,
        "n_detected": n_det,
        "n_correct": n_correct,
        "detection_recall": n_det / n_gt if n_gt else None,
        "classification_accuracy": n_correct / n_det if n_det else None,
        "confusion": conf.tolist(),  # rows true[1,2] x cols [missed,pred1,pred2]
        "false_positives": int(fp),
        "fp_per_min": fp / minutes if minutes > 0 else None,
        "rejected_triggers": int(false_triggers),
        "minutes": minutes,
    }


# ----------------------------------------------------------------------------
# Orchestration & persistence
# ----------------------------------------------------------------------------

def resolve_path(p):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(project_root(), p)


def save_bundle(path, payload):
    from joblib import dump
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dump(payload, path)
    return path


def load_bundle(path):
    from joblib import load
    return load(path)


def run_training(config, progress=None, cancel=None):
    """Full pipeline: load folder → split → train → evaluate → save model.

    *progress* is an optional callable(str). *cancel* is an optional
    callable() -> bool checked between steps (raises InterruptedError).
    Returns a dict with split assignments, metrics (val/test per horizon,
    stream eval per test file), feature importances, and the model path.
    """
    def say(msg):
        if progress:
            progress(msg)

    def check():
        if cancel and cancel():
            raise InterruptedError("Training cancelled.")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in (config or {}).items() if v is not None})
    fs = float(cfg["fs"])
    rng = np.random.default_rng(int(cfg["seed"]))

    folder = resolve_path(cfg["folder"])
    if not folder or not os.path.isdir(folder):
        raise ValueError(f"Labeled-data folder not found: {folder!r}")
    paths = find_labeled_files(folder)
    if len(paths) < 3:
        raise ValueError(
            f"Need at least 3 labeled files for train/val/test, found "
            f"{len(paths)} in {folder}")

    raw, counts, datasets = {}, {}, {}
    n_zones = None
    for k, path in enumerate(paths):
        check()
        name = os.path.basename(path)
        say(f"Loading {name} ({k + 1}/{len(paths)})...")
        data, labels = load_labeled_file(path)
        labels = canonicalize_labels(labels, cfg.get("noise_ids") or (),
                                     cfg.get("drop_ids") or ())
        if n_zones is None:
            n_zones = data.shape[1]
        elif data.shape[1] != n_zones:
            raise ValueError(
                f"{name}: {data.shape[1]} zones, expected {n_zones} "
                f"(all files must share one zone count)")
        raw[name] = (data, labels)
        counts[name] = event_counts(labels)
        say(f"Building features for {name}...")
        datasets[name] = build_file_dataset(
            data, labels, fs, cfg["horizons_ms"], rng,
            noise_ratio=float(cfg["noise_ratio"]),
            trigger=cfg.get("trigger"),
            hard_negative_ratio=float(cfg.get("hard_negative_ratio", 1.0)))

    split = auto_split(counts, cfg["split"], int(cfg["seed"]),
                       cfg.get("assignments"))
    groups = {k: sorted(n for n, a in split.items() if a == k)
              for k in ("train", "val", "test")}
    say("Split: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    def stack(names):
        X = np.vstack([datasets[n]["X"] for n in names])
        y = np.concatenate([datasets[n]["y"] for n in names])
        hz = np.concatenate([datasets[n]["horizon"] for n in names])
        return X, y, hz

    check()
    say(f"Training {cfg['model']} ({cfg['strategy']}) on "
        f"{len(groups['train'])} files...")
    Xtr, ytr, htr = stack(groups["train"])
    bundle = train_classifier(
        Xtr, ytr, htr, cfg["strategy"], cfg["model"], int(cfg["seed"]), rng,
        oversample=cfg["oversample"], ratio=float(cfg["oversample_ratio"]),
        k=int(cfg["smote_k"]))

    check()
    say("Evaluating on validation / test files...")
    metrics = {
        "val": evaluate_bundle(bundle, datasets, groups["val"]),
        "test": evaluate_bundle(bundle, datasets, groups["test"]),
    }

    names = feature_names(n_zones)
    importances = feature_importance_ranking(bundle, names)

    # Refit the shipped model on more data (test stays untouched unless the
    # user explicitly opts into 'all').
    refit = cfg.get("final_refit", "train+val")
    refit_names = {
        "train": groups["train"],
        "train+val": groups["train"] + groups["val"],
        "all": groups["train"] + groups["val"] + groups["test"],
    }.get(refit, groups["train"] + groups["val"])
    final_bundle = bundle
    if refit_names != groups["train"]:
        check()
        say(f"Refitting final model on {refit} ({len(refit_names)} files)...")
        Xf, yf, hf = stack(refit_names)
        final_bundle = train_classifier(
            Xf, yf, hf, cfg["strategy"], cfg["model"], int(cfg["seed"]), rng,
            oversample=cfg["oversample"],
            ratio=float(cfg["oversample_ratio"]), k=int(cfg["smote_k"]))

    saved = {
        "bundle": final_bundle,
        "feature_names": names,
        "class_names": dict(CLASS_NAMES),
        "fs": fs,
        "n_zones": n_zones,
        "horizons_ms": list(cfg["horizons_ms"]),
        "trigger": dict(cfg.get("trigger") or DEFAULT_TRIGGER),
        "final_refit": refit,
    }

    if cfg.get("stream_eval") and groups["test"]:
        stream = {}
        # Score streaming with the evaluation model so test stays honest.
        eval_saved = dict(saved, bundle=bundle)
        for name in groups["test"]:
            check()
            say(f"Simulating real-time stream on {name}...")
            data, labels = raw[name]
            events = simulate_stream(eval_saved, data)
            stream[name] = score_stream(events, labels, fs)
        metrics["stream_test"] = stream

    check()
    model_path = resolve_path(cfg["modelPath"])
    say(f"Saving model to {model_path}...")
    saved["metrics"] = metrics
    saved["config"] = {k: v for k, v in cfg.items() if k != "assignments"}
    save_bundle(model_path, saved)
    try:
        with open(os.path.splitext(model_path)[0] + "_metrics.json",
                  "w") as f:
            json.dump({"split": split, "counts": counts,
                       "metrics": metrics,
                       "importances": importances}, f, indent=2)
    except Exception:
        pass

    say("Done.")
    return {
        "modelPath": model_path,
        "split": split,
        "counts": counts,
        "metrics": metrics,
        "importances": importances,
        "n_zones": n_zones,
        "class_names": dict(CLASS_NAMES),
    }


# ----------------------------------------------------------------------------
# Entry-based API (modular ML Split / ML Train / ML Evaluate blocks)
# ----------------------------------------------------------------------------

def extract_entries(data_in):
    """Normalize a Load File struct / entry list / single entry to a list."""
    if isinstance(data_in, dict):
        if isinstance(data_in.get("files"), (list, tuple)):
            data_in = data_in["files"]
        else:
            data_in = [data_in]
    if not isinstance(data_in, (list, tuple)):
        raise ValueError("Expected a Load File struct or list of entries.")
    return [e for e in data_in if isinstance(e, dict)]


def entry_data_labels(entry):
    """(data, labels) from a file entry, honoring flat and dot-prefixed keys.

    Either may be None when the entry lacks that field.
    """
    return (entry.get("data", entry.get("labeledData.data")),
            entry.get("labels", entry.get("labeledData.labels")))


def segment_entries(entries):
    """Cut each labeled entry into per-event segments (data views).

    Cuts land at the midpoints of the background gaps between consecutive
    labeled events, so every segment carries one event with generous
    background margins; leading/trailing background attaches to the first/
    last segment. Entries without events pass through whole. Segment names
    are '<fileName>#<k>'. Used by ML Split's "by event" mode — segments
    from all files can then be pooled and mixed across train/val/test
    (note: splits then share each recording's noise character, so scores
    read optimistic vs the by-file split).
    """
    out = []
    for i, e in enumerate(entries):
        name = str(e.get("fileName") or f"file{i}")
        data, labels = entry_data_labels(e)
        if data is None or labels is None:
            out.append(e)
            continue
        data = np.asarray(data)
        labels = np.asarray(labels).ravel()
        runs = [(s, t, v) for (s, t, v) in parse_events(labels) if v != 0]
        if not runs:
            out.append(e)
            continue
        cuts = [0]
        for (a, b) in zip(runs[:-1], runs[1:]):
            cuts.append((a[1] + b[0]) // 2)
        cuts.append(len(labels))
        for k in range(len(runs)):
            s, t = cuts[k], cuts[k + 1]
            out.append({"fileName": f"{name}#{k}",
                        "data": data[s:t], "labels": labels[s:t]})
    return out


def entries_to_datasets(entries, cfg, rng, progress=None):
    """Per-file feature datasets from in-memory Load File entries.

    Returns (datasets {name: rows}, raw {name: (data, canonical labels)},
    n_zones). Entries must carry per-sample labels.
    """
    fs = float(cfg["fs"])
    datasets, raw = {}, {}
    n_zones = None
    for k, e in enumerate(entries):
        name = str(e.get("fileName") or f"file{k}")
        data, labels = entry_data_labels(e)
        if data is None or labels is None:
            raise ValueError(f"{name}: entry needs 'data' and 'labels' "
                             f"(a labeled recording)")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        labels = canonicalize_labels(labels, cfg.get("noise_ids") or (),
                                     cfg.get("drop_ids") or (3, 4))
        if len(labels) != len(data):
            raise ValueError(f"{name}: data/labels length mismatch")
        if n_zones is None:
            n_zones = data.shape[1]
        elif data.shape[1] != n_zones:
            raise ValueError(f"{name}: {data.shape[1]} zones, expected "
                             f"{n_zones}")
        if progress:
            progress(f"Building features for {name} "
                     f"({k + 1}/{len(entries)})...")
        datasets[name] = build_file_dataset(
            data, labels, fs, cfg["horizons_ms"], rng,
            noise_ratio=float(cfg["noise_ratio"]),
            trigger=cfg.get("trigger"),
            hard_negative_ratio=float(cfg.get("hard_negative_ratio", 1.0)))
        raw[name] = (data, labels)
    if not datasets:
        raise ValueError("No labeled file entries provided.")
    return datasets, raw, n_zones


def _stack_datasets(datasets, names):
    names = [n for n in names if len(datasets[n]["y"])]
    if not names:
        raise ValueError("No feature rows in the provided entries.")
    X = np.vstack([datasets[n]["X"] for n in names])
    y = np.concatenate([datasets[n]["y"] for n in names])
    hz = np.concatenate([datasets[n]["horizon"] for n in names])
    return X, y, hz


def train_from_entries(entries, config=None, progress=None):
    """Train on the given entries only; saves the bundle to cfg['modelPath'].

    The counterpart of run_training for the modular pipeline — splitting
    and evaluation live in their own blocks.
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in (config or {}).items() if v is not None})
    rng = np.random.default_rng(int(cfg["seed"]))
    datasets, raw, n_zones = entries_to_datasets(entries, cfg, rng, progress)
    names = sorted(datasets)
    if progress:
        progress(f"Training {cfg['model']} ({cfg['strategy']}) on "
                 f"{len(names)} file(s)...")
    X, y, hz = _stack_datasets(datasets, names)
    bundle = train_classifier(
        X, y, hz, cfg["strategy"], cfg["model"], int(cfg["seed"]), rng,
        oversample=cfg["oversample"], ratio=float(cfg["oversample_ratio"]),
        k=int(cfg["smote_k"]))
    fnames = feature_names(n_zones)
    saved = {
        "bundle": bundle,
        "feature_names": fnames,
        "class_names": dict(CLASS_NAMES),
        "fs": float(cfg["fs"]),
        "n_zones": n_zones,
        "horizons_ms": list(cfg["horizons_ms"]),
        "trigger": dict(cfg.get("trigger") or DEFAULT_TRIGGER),
        "config": {k: v for k, v in cfg.items() if k != "assignments"},
    }
    model_path = resolve_path(cfg["modelPath"])
    save_bundle(model_path, saved)
    return {
        "modelPath": model_path,
        "files": names,
        "counts": {n: event_counts(raw[n][1]) for n in names},
        "rows": int(len(y)),
        "n_zones": n_zones,
        "importances": feature_importance_ranking(bundle, fnames),
    }


def evaluate_on_entries(saved, entries, stream_eval=False, progress=None):
    """Score a trained bundle on labeled entries.

    Windows are rebuilt with the training config stored in the bundle so the
    numbers are comparable across runs. Returns {'metrics': per-horizon,
    'stream': {name: score} | None}.
    """
    if isinstance(saved, str):
        saved = load_bundle(resolve_path(saved))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(saved.get("config") or {})
    cfg["fs"] = float(saved.get("fs", cfg["fs"]))
    cfg["horizons_ms"] = list(saved.get("horizons_ms", cfg["horizons_ms"]))
    cfg["trigger"] = dict(saved.get("trigger") or cfg["trigger"])
    rng = np.random.default_rng(int(cfg["seed"]))
    datasets, raw, n_zones = entries_to_datasets(entries, cfg, rng, progress)
    if saved.get("n_zones") and n_zones != saved["n_zones"]:
        raise ValueError(f"data has {n_zones} zones, model expects "
                         f"{saved['n_zones']}")
    names = sorted(datasets)
    metrics = evaluate_bundle(saved["bundle"], datasets, names)
    stream = None
    if stream_eval:
        stream = {}
        for name in names:
            if progress:
                progress(f"Simulating real-time stream on {name}...")
            data, labels = raw[name]
            stream[name] = score_stream(simulate_stream(saved, data),
                                        labels, cfg["fs"])
    return {"metrics": metrics, "stream": stream}
