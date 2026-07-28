"""CWT Evaluator: compare multiple wavelet/scoring configurations on the
same signal.

Data model
----------
The user maintains a list of "detector configurations" (configs); each
config picks one mother-wavelet shape (Template, Ricker, Morlet, DoG,
Haar) and its own scoring settings (threshold k, scoring method,
scale range). Each enabled config is run through the standard CWT
pipeline (`build_kernels` → `compute_score` → `mad_threshold` →
`detect_regions`) and its regions are post-processed (merge / min-
event / padding) before scoring against optional ground truth.

This lets the user compare not only different wavelet shapes but also
different scoring settings for the *same* wavelet (e.g. Template with
`max` vs `rss`, or with different scale ranges) by adding multiple
Template configs.

Pure helpers + an interactive QDialog with a tabbed config panel,
shared common settings, a stacked-score plot, and a metrics table.
"""

import os
import json

import numpy as np

from utils.paths import saved_templates_dir


_SHAPES = ("Template", "Ricker", "Morlet", "DoG", "Haar")


def _new_config(shape="Ricker", **overrides):
    """Build a fresh config dict with default values for `shape`."""
    cfg = {
        "name":           shape,
        "shape":          shape,
        "enabled":        True,
        "thresholdK":     5.0,
        "scoringMethod":  "max",       # "max" or "rss"
        "scaleMin":       0.5,
        "scaleMax":       2.0,
        "scaleCount":     3,
        "baseLengthMs":   20.0,        # used by non-template wavelets
        "invertPolarity": False,       # template only
        "morletOmega":    5.0,         # morlet only
    }
    cfg.update(overrides)
    return cfg


_DEFAULT_PARAMS = {
    # Default config list — one per shape so the user has an immediate
    # baseline. Template is included even if no template is connected;
    # the runner / dialog disable it in that case.
    "configs": [
        _new_config("Template"),
        _new_config("Ricker"),
        _new_config("Morlet"),
        _new_config("DoG"),
        _new_config("Haar"),
    ],
    # Common post-processing applied to every config's regions.
    "mergeGapMs":         0.0,
    "minEventMs":         0.0,
    "padBeforeFactor":    0.0,
    "padAfterFactor":     0.0,
    # Region-pipeline knobs forwarded to detect_regions().
    "minSeparationFrac":  1.0,
    "splitCoincidents":   True,
    "splitReturnSigma":   3.0,
    "splitMergeGapFrac":  0.1,
    # Ground-truth matching.
    "iouThreshold":       0.30,
    "winner":             "auto",
}


# --- Wavelet builders ------------------------------------------------------

def _ricker(width_samples):
    a = max(1.0, float(width_samples))
    half = max(1, int(round(4.0 * a)))
    t = np.arange(-half, half + 1, dtype=float)
    x = (t / a) ** 2
    return (1.0 - x) * np.exp(-0.5 * x)


def _morlet_real(width_samples, omega=5.0):
    a = max(1.0, float(width_samples))
    half = max(1, int(round(4.0 * a)))
    t = np.arange(-half, half + 1, dtype=float)
    return np.cos(omega * t / a) * np.exp(-0.5 * (t / a) ** 2)


def _dog(width_samples):
    a = max(1.0, float(width_samples))
    half = max(1, int(round(4.0 * a)))
    t = np.arange(-half, half + 1, dtype=float)
    return -(t / a) * np.exp(-0.5 * (t / a) ** 2)


def _haar_step(width_samples):
    w = max(2, int(round(width_samples)))
    k = np.empty(2 * w, dtype=float)
    k[:w] = -1.0
    k[w:] = 1.0
    return k


def build_wavelet_from_config(config, sample_rate, template):
    """Build the mother-wavelet array for one config."""
    shape = config.get("shape", "Ricker")
    if shape == "Template":
        if template is None or len(template) < 2:
            raise ValueError("Template wavelet not provided on input port")
        w = np.asarray(template, dtype=float).ravel()
        if config.get("invertPolarity", False):
            w = -w
        return w
    base_len = max(2, int(round(float(config.get("baseLengthMs", 20.0))
                                * float(sample_rate) / 1000.0)))
    if shape == "Ricker":
        return _ricker(base_len)
    if shape == "Morlet":
        return _morlet_real(base_len,
                            omega=float(config.get("morletOmega", 5.0)))
    if shape == "DoG":
        return _dog(base_len)
    if shape == "Haar":
        return _haar_step(base_len)
    raise ValueError(f"Unknown shape: {shape!r}")


# --- Region post-processing ------------------------------------------------

def post_process_regions(regions, n_samples, sample_rate, merge_gap_ms,
                         min_event_ms, pad_before_factor, pad_after_factor):
    """Merge close regions, drop short ones, then pad. Mirrors the
    Combined block's mergeGapMs / minEventMs / padBeforeFactor /
    padAfterFactor pipeline so post-processed counts are comparable
    across both blocks.
    """
    if regions is None or len(regions) == 0:
        return np.zeros((0, 2), dtype=int)
    regs = np.asarray(regions, dtype=int).reshape(-1, 2)
    regs = regs[regs[:, 0].argsort()]
    sr = max(float(sample_rate), 1.0)
    gap = max(0, int(round(float(merge_gap_ms) * sr / 1000.0)))
    merged = []
    for a, b in regs:
        a, b = int(a), int(b)
        if merged and a - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    min_len = max(0, int(round(float(min_event_ms) * sr / 1000.0)))
    if min_len > 0:
        merged = [(a, b) for a, b in merged if (b - a + 1) >= min_len]
    out = []
    pb = float(pad_before_factor)
    pa = float(pad_after_factor)
    for a, b in merged:
        w = b - a + 1
        a2 = a - int(round(pb * w))
        b2 = b + int(round(pa * w))
        out.append((max(0, a2), min(int(n_samples) - 1, b2)))
    return np.asarray(out, dtype=int).reshape(-1, 2)


# --- Metrics ---------------------------------------------------------------

def _iou(a, b):
    s = max(int(a[0]), int(b[0]))
    e = min(int(a[1]), int(b[1]))
    inter = max(0, e - s + 1)
    if inter == 0:
        return 0.0
    union = (int(a[1]) - int(a[0]) + 1) + (int(b[1]) - int(b[0]) + 1) - inter
    return inter / union if union > 0 else 0.0


def match_predictions(pred, truth, iou_threshold):
    pred = [tuple(map(int, r)) for r in (pred if pred is not None else [])]
    truth = [tuple(map(int, r)) for r in (truth if truth is not None else [])]
    if not truth:
        return 0, len(pred), 0, 0.0
    if not pred:
        return 0, 0, len(truth), 0.0
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(truth):
            iou = _iou(p, g)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, tp_ious = set(), set(), []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        tp_ious.append(iou)
    tp = len(tp_ious)
    fp = len(pred) - tp
    fn = len(truth) - tp
    mean_iou = float(np.mean(tp_ious)) if tp_ious else 0.0
    return tp, fp, fn, mean_iou


# --- Per-config evaluation -------------------------------------------------

def evaluate_one(signal, wavelet, config, region_pipeline_params,
                 post_params, sample_rate):
    """Run the full CWT detection pipeline for one wavelet, then apply
    region post-processing. Returns dict of arrays + scalar metrics.
    """
    from processing.realtime_detection_cwt import (
        normalize_wavelet, build_kernels, compute_score, mad_threshold,
        detect_regions, autocorr_min_separation,
    )
    w, _ = normalize_wavelet(np.asarray(wavelet, dtype=float).ravel(),
                             auto=True)
    smin = float(config.get("scaleMin", 0.5))
    smax = float(config.get("scaleMax", 2.0))
    sct = max(1, int(config.get("scaleCount", 3)))
    if sct == 1:
        scales = [smin]
    else:
        scales = list(np.geomspace(max(smin, 1e-6),
                                   max(smax, smin + 1e-6), sct))
    kernels = build_kernels(w, scales)
    score = compute_score(signal, kernels,
                          method=config.get("scoringMethod", "max"))
    thr, sigma = mad_threshold(score, k=float(config.get("thresholdK", 5.0)))
    autocorr_floor = autocorr_min_separation(w, threshold_frac=0.1)
    sig = np.asarray(signal, dtype=float).ravel()
    sig_baseline = float(np.median(sig)) if sig.size else 0.0
    sig_sigma = (1.4826 * float(np.median(np.abs(sig - sig_baseline)))
                 if sig.size else 0.0)
    raw_regions, _ = detect_regions(
        score, thr, template_len=len(w),
        min_separation_frac=region_pipeline_params.get("minSeparationFrac", 1.0),
        split_coincidents=region_pipeline_params.get("splitCoincidents", True),
        signal=sig, signal_baseline=sig_baseline, signal_sigma=sig_sigma,
        split_return_sigma=region_pipeline_params.get("splitReturnSigma", 3.0),
        split_merge_gap_frac=region_pipeline_params.get("splitMergeGapFrac", 0.1),
        autocorr_floor_samples=autocorr_floor,
    )
    regions = post_process_regions(
        raw_regions, n_samples=len(score), sample_rate=sample_rate,
        merge_gap_ms=post_params.get("mergeGapMs", 0.0),
        min_event_ms=post_params.get("minEventMs", 0.0),
        pad_before_factor=post_params.get("padBeforeFactor", 0.0),
        pad_after_factor=post_params.get("padAfterFactor", 0.0),
    )
    peak_snr = []
    if sigma > 0:
        for a, b in regions:
            a, b = int(a), int(b)
            if 0 <= a <= b < len(score):
                peak_snr.append(float(np.max(score[a:b + 1])) / sigma)
    return {
        "regions":     regions,
        "rawRegions":  np.asarray(raw_regions, dtype=int).reshape(-1, 2),
        "score":       score,
        "threshold":   float(thr),
        "sigma":       float(sigma),
        "peakSnrMean": float(np.mean(peak_snr)) if peak_snr else 0.0,
        "peakSnrMax":  float(np.max(peak_snr)) if peak_snr else 0.0,
        "wavelet":     w,
    }


def evaluate_configs(signal, sample_rate, template, params=None,
                     ground_truth=None):
    """Run every enabled config and return {name: metrics_dict}."""
    p = dict(_DEFAULT_PARAMS)
    if params:
        for k, v in params.items():
            if k == "configs" and v is not None:
                p["configs"] = list(v)
            else:
                p[k] = v
    region_pipeline = {
        "minSeparationFrac": p.get("minSeparationFrac", 1.0),
        "splitCoincidents":  p.get("splitCoincidents", True),
        "splitReturnSigma":  p.get("splitReturnSigma", 3.0),
        "splitMergeGapFrac": p.get("splitMergeGapFrac", 0.1),
    }
    post_params = {
        "mergeGapMs":      p.get("mergeGapMs", 0.0),
        "minEventMs":      p.get("minEventMs", 0.0),
        "padBeforeFactor": p.get("padBeforeFactor", 0.0),
        "padAfterFactor":  p.get("padAfterFactor", 0.0),
    }
    iou_thr = float(p.get("iouThreshold", 0.30))
    out = {}
    seen = {}
    for cfg in p["configs"]:
        if not cfg.get("enabled", True):
            continue
        # Resolve a unique display name for this config. Prefer the
        # user-supplied "name"; fall back to "Shape" / "Shape-N" if a
        # collision occurs (which can happen when the user clones tabs).
        base = cfg.get("name") or cfg.get("shape", "Cfg")
        n = seen.get(base, 0) + 1
        seen[base] = n
        name = base if n == 1 else f"{base}-{n}"
        try:
            wavelet = build_wavelet_from_config(cfg, sample_rate, template)
            res = evaluate_one(signal, wavelet, cfg, region_pipeline,
                               post_params, sample_rate)
        except Exception as e:
            out[name] = {"error": str(e), "config": dict(cfg)}
            continue
        regions_list = [tuple(r) for r in res["regions"].tolist()]
        tp, fp, fn, mean_iou = match_predictions(
            regions_list, ground_truth, iou_thr)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        res.update({
            "config":    dict(cfg),
            "shape":     cfg.get("shape"),
            "count":     int(len(regions_list)),
            "tp":        int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall),
            "f1":        float(f1), "meanIoU": float(mean_iou),
        })
        out[name] = res
    return out


def pick_winner(metrics, prefer="auto", has_ground_truth=False):
    valid = [n for n, m in metrics.items() if "error" not in m]
    if not valid:
        return None
    if prefer != "auto" and prefer in valid:
        return prefer
    if has_ground_truth:
        return max(valid, key=lambda n: (metrics[n]["f1"],
                                         metrics[n]["peakSnrMean"]))
    template_rows = [n for n in valid
                     if metrics[n].get("shape") == "Template"]
    if template_rows:
        return template_rows[0]
    return max(valid, key=lambda n: (metrics[n]["count"],
                                     metrics[n]["peakSnrMean"]))


# --- Settings persistence -------------------------------------------------

def save_state(params, filepath):
    payload = {"version": 2}
    for k in _DEFAULT_PARAMS:
        if k in params:
            payload[k] = params[k]
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, filepath)


def load_state(filepath):
    """Load params, migrating v1 (`enabled` dict + flat thresholdK) to
    the v2 configs-list shape if needed."""
    with open(filepath, "r") as f:
        loaded = json.load(f)
    out = dict(_DEFAULT_PARAMS)
    out["configs"] = [dict(c) for c in _DEFAULT_PARAMS["configs"]]
    if loaded.get("version", 1) == 1:
        # v1 migration: build configs from the old global params.
        en = loaded.get("enabled", {})
        new_configs = []
        for shape in _SHAPES:
            if not en.get(shape, True):
                continue
            new_configs.append(_new_config(
                shape,
                thresholdK=loaded.get("thresholdK", 5.0),
                scoringMethod=loaded.get("scoringMethod", "max"),
                scaleMin=loaded.get("scaleMin", 0.5),
                scaleMax=loaded.get("scaleMax", 2.0),
                scaleCount=loaded.get("scaleCount", 3),
                baseLengthMs=loaded.get("baseLengthMs", 20.0),
            ))
        if new_configs:
            out["configs"] = new_configs
    else:
        if "configs" in loaded:
            out["configs"] = [{**_new_config(c.get("shape", "Ricker")), **c}
                              for c in loaded["configs"]]
    for k in _DEFAULT_PARAMS:
        if k != "configs" and k in loaded:
            out[k] = loaded[k]
    return out


# --- Dialog ---------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QGroupBox, QFileDialog, QMessageBox, QWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QInputDialog,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure


_SHAPE_COLORS = {
    "Template": "#3aa55a",
    "Ricker":   "#9b59b6",
    "Morlet":   "#e08533",
    "DoG":      "#3b6fb6",
    "Haar":     "#d23f3f",
}


class _ConfigTab(QWidget):
    """Single tab — one detector configuration."""

    def __init__(self, config, template_available, on_shape_changed):
        super().__init__()
        self._cfg = dict(config)
        self._template_available = template_available
        self._on_shape_changed_cb = on_shape_changed
        self._build()
        self._sync_from_cfg()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.enabled_chk = QCheckBox("Enabled")
        form.addRow(self.enabled_chk)

        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)

        self.shape_combo = QComboBox()
        for s in _SHAPES:
            self.shape_combo.addItem(s)
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        form.addRow("Wavelet:", self.shape_combo)

        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(0.1, 50.0); self.k_spin.setSingleStep(0.5)
        self.k_spin.setDecimals(2)
        form.addRow("Threshold k:", self.k_spin)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["max", "rss"])
        self.method_combo.setToolTip(
            "max: keep only positive correlations across scales (direction-"
            "aware). rss: sqrt-sum-of-squares — sign-agnostic.")
        form.addRow("Scoring:", self.method_combo)

        self.smin_spin = QDoubleSpinBox()
        self.smin_spin.setRange(0.05, 10.0); self.smin_spin.setSingleStep(0.1)
        self.smin_spin.setDecimals(2)
        form.addRow("Scale min:", self.smin_spin)
        self.smax_spin = QDoubleSpinBox()
        self.smax_spin.setRange(0.05, 10.0); self.smax_spin.setSingleStep(0.1)
        self.smax_spin.setDecimals(2)
        form.addRow("Scale max:", self.smax_spin)
        self.sct_spin = QSpinBox()
        self.sct_spin.setRange(1, 10)
        form.addRow("Scale count:", self.sct_spin)

        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(0.1, 1000.0); self.len_spin.setSingleStep(1.0)
        self.len_spin.setDecimals(2)
        self.len_spin.setToolTip(
            "Pre-resampling base length for non-template wavelets.")
        form.addRow("Base length (ms):", self.len_spin)

        self.invert_chk = QCheckBox("Invert template polarity")
        self.invert_chk.setToolTip(
            "Multiply the template by −1 before normalization. Use when "
            "the template was extracted upside-down vs the actual pulses.")
        form.addRow(self.invert_chk)

        self.omega_spin = QDoubleSpinBox()
        self.omega_spin.setRange(1.0, 30.0); self.omega_spin.setSingleStep(0.5)
        self.omega_spin.setDecimals(2)
        self.omega_spin.setToolTip(
            "Morlet carrier frequency ω. Higher = more oscillations under "
            "the Gaussian envelope.")
        form.addRow("Morlet ω:", self.omega_spin)

        layout.addLayout(form)
        layout.addStretch(1)

    def _sync_from_cfg(self):
        c = self._cfg
        self.enabled_chk.setChecked(bool(c.get("enabled", True)))
        self.name_edit.setText(str(c.get("name", c.get("shape", "Cfg"))))
        idx = self.shape_combo.findText(c.get("shape", "Ricker"))
        if idx >= 0:
            self.shape_combo.setCurrentIndex(idx)
        self.k_spin.setValue(float(c.get("thresholdK", 5.0)))
        i = self.method_combo.findText(c.get("scoringMethod", "max"))
        if i >= 0:
            self.method_combo.setCurrentIndex(i)
        self.smin_spin.setValue(float(c.get("scaleMin", 0.5)))
        self.smax_spin.setValue(float(c.get("scaleMax", 2.0)))
        self.sct_spin.setValue(int(c.get("scaleCount", 3)))
        self.len_spin.setValue(float(c.get("baseLengthMs", 20.0)))
        self.invert_chk.setChecked(bool(c.get("invertPolarity", False)))
        self.omega_spin.setValue(float(c.get("morletOmega", 5.0)))
        self._on_shape_changed(self.shape_combo.currentText())

    def _on_shape_changed(self, shape):
        is_template = shape == "Template"
        is_morlet = shape == "Morlet"
        # Template needs no base length; standard wavelets need one.
        self.len_spin.setEnabled(not is_template)
        self.invert_chk.setEnabled(is_template)
        self.omega_spin.setEnabled(is_morlet)
        if is_template and not self._template_available:
            self.enabled_chk.setChecked(False)
            self.enabled_chk.setEnabled(False)
            self.enabled_chk.setToolTip(
                "No template wavelet connected to input port.")
        else:
            self.enabled_chk.setEnabled(True)
            self.enabled_chk.setToolTip("")
        if self._on_shape_changed_cb is not None:
            self._on_shape_changed_cb(self)

    def read_config(self):
        return {
            "name":           self.name_edit.text().strip() or "Cfg",
            "shape":          self.shape_combo.currentText(),
            "enabled":        self.enabled_chk.isChecked(),
            "thresholdK":     float(self.k_spin.value()),
            "scoringMethod":  self.method_combo.currentText(),
            "scaleMin":       float(self.smin_spin.value()),
            "scaleMax":       float(self.smax_spin.value()),
            "scaleCount":     int(self.sct_spin.value()),
            "baseLengthMs":   float(self.len_spin.value()),
            "invertPolarity": bool(self.invert_chk.isChecked()),
            "morletOmega":    float(self.omega_spin.value()),
        }


class CwtEvaluatorDialog(QDialog):
    """Tabbed wavelet-comparison dialog. Each tab is one config; common
    settings (post-processing, IoU) live in a separate group below the
    tab widget."""

    def __init__(self, signal, sample_rate, template, params,
                 ground_truth=None, current_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CWT Evaluator")
        self.resize(1600, 880)

        try:
            from utils.dialog_style import apply_dialog_style
            apply_dialog_style(self)
        except ImportError:
            pass

        self._signal = np.asarray(signal, dtype=float).ravel()
        self._sample_rate = float(sample_rate)
        self._template = (np.asarray(template, dtype=float).ravel()
                          if template is not None else None)
        self._template_available = (self._template is not None
                                    and self._template.size >= 2)
        self._ground_truth = (np.asarray(ground_truth, dtype=int).reshape(-1, 2)
                              if ground_truth is not None else None)
        self._has_gt = self._ground_truth is not None

        self._params = dict(_DEFAULT_PARAMS)
        self._params["configs"] = [dict(c) for c in _DEFAULT_PARAMS["configs"]]
        if params:
            self._params.update(params)
            if "configs" in params and params["configs"]:
                self._params["configs"] = [dict(c) for c in params["configs"]]

        self._current_path = current_path
        if current_path:
            self.setWindowTitle(
                f"CWT Evaluator — {os.path.basename(current_path)}")

        self._metrics = {}
        self._build_ui()
        self._populate_tabs_from_params()
        self._sync_common_from_params()
        self._run_evaluation()

    # --- UI ---------------------------------------------------------------

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        # Plot column.
        plot_col = QWidget()
        pl = QVBoxLayout(plot_col)
        pl.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(8, 6))
        style_mpl_figure(self._fig)
        self._canvas = FigureCanvas(self._fig)
        self._nav = NavigationToolbar(self._canvas, plot_col)
        pl.addWidget(self._nav)
        pl.addWidget(self._canvas, 1)
        main.addWidget(plot_col, 3)

        # Right column.
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        # Tabs (one per detector config).
        cfg_box = QGroupBox("Detector configurations")
        cfg_layout = QVBoxLayout(cfg_box)
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_close_tab)
        cfg_layout.addWidget(self._tabs)

        tab_btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add tab")
        add_btn.clicked.connect(self._on_add_tab)
        clone_btn = QPushButton("Clone current")
        clone_btn.clicked.connect(self._on_clone_tab)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._on_rename_tab)
        tab_btn_row.addWidget(add_btn)
        tab_btn_row.addWidget(clone_btn)
        tab_btn_row.addWidget(rename_btn)
        tab_btn_row.addStretch(1)
        cfg_layout.addLayout(tab_btn_row)
        rl.addWidget(cfg_box)

        # Common settings group.
        com_box = QGroupBox("Common settings")
        com_grid = QGridLayout(com_box)
        row = 0
        com_grid.addWidget(QLabel("Merge gap (ms):"), row, 0)
        self._merge_spin = QDoubleSpinBox()
        self._merge_spin.setRange(0.0, 10000.0); self._merge_spin.setSingleStep(5.0)
        self._merge_spin.setDecimals(1)
        self._merge_spin.setToolTip(
            "Two regions whose gap is < this are merged into one. 0 = off.")
        com_grid.addWidget(self._merge_spin, row, 1); row += 1
        com_grid.addWidget(QLabel("Min event (ms):"), row, 0)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(0.0, 10000.0); self._min_spin.setSingleStep(1.0)
        self._min_spin.setDecimals(1)
        self._min_spin.setToolTip(
            "Drop regions shorter than this after merging. 0 = off.")
        com_grid.addWidget(self._min_spin, row, 1); row += 1
        com_grid.addWidget(QLabel("Pad before (×):"), row, 0)
        self._padb_spin = QDoubleSpinBox()
        self._padb_spin.setRange(0.0, 5.0); self._padb_spin.setSingleStep(0.05)
        self._padb_spin.setDecimals(2)
        self._padb_spin.setToolTip(
            "Extend each region's start by this fraction of its width.")
        com_grid.addWidget(self._padb_spin, row, 1); row += 1
        com_grid.addWidget(QLabel("Pad after (×):"), row, 0)
        self._pada_spin = QDoubleSpinBox()
        self._pada_spin.setRange(0.0, 5.0); self._pada_spin.setSingleStep(0.05)
        self._pada_spin.setDecimals(2)
        com_grid.addWidget(self._pada_spin, row, 1); row += 1
        com_grid.addWidget(QLabel("IoU threshold:"), row, 0)
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.0, 1.0); self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setDecimals(2)
        self._iou_spin.setToolTip(
            "Minimum IoU for a predicted region to count as a TP "
            "match against ground truth.")
        com_grid.addWidget(self._iou_spin, row, 1); row += 1
        rl.addWidget(com_box)

        run_btn = QPushButton("Run evaluation")
        run_btn.clicked.connect(self._run_evaluation)
        rl.addWidget(run_btn)

        # Metrics table.
        cols = ["Name", "Shape", "Method", "Count", "SNR̄", "Thr"]
        if self._has_gt:
            cols += ["TP", "FP", "FN", "P", "R", "F1", "IoU̅"]
        self._metric_cols = cols
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        rl.addWidget(self._table, 1)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Winner:"))
        self._winner_combo = QComboBox()
        win_row.addWidget(self._winner_combo, 1)
        rl.addLayout(win_row)

        # Save / Load / OK / Cancel.
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        save_as_btn = QPushButton("Save as…")
        save_as_btn.clicked.connect(self._on_save_as)
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(save_as_btn)
        btn_row.addWidget(load_btn)
        rl.addLayout(btn_row)

        ok_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_row.addStretch(1)
        ok_row.addWidget(ok_btn)
        ok_row.addWidget(cancel_btn)
        rl.addLayout(ok_row)

        main.addWidget(right, 2)

    # --- Tab management ---------------------------------------------------

    def _populate_tabs_from_params(self):
        # Clear, then add one tab per config.
        while self._tabs.count():
            self._tabs.removeTab(0)
        for cfg in self._params.get("configs", []):
            self._add_tab_widget(cfg)
        if self._tabs.count() == 0:
            self._add_tab_widget(_new_config("Ricker"))
        self._update_close_buttons()

    def _add_tab_widget(self, config):
        tab = _ConfigTab(config, self._template_available, None)
        title = self._tab_title(tab)
        self._tabs.addTab(tab, title)
        # Refresh title when the user edits name/shape.
        tab.name_edit.editingFinished.connect(
            lambda t=tab: self._refresh_tab_title(t))
        tab.shape_combo.currentTextChanged.connect(
            lambda _, t=tab: self._refresh_tab_title(t))
        return tab

    def _tab_title(self, tab):
        c = tab.read_config()
        return f"{c['name']} ({c['shape']})"

    def _refresh_tab_title(self, tab):
        idx = self._tabs.indexOf(tab)
        if idx >= 0:
            self._tabs.setTabText(idx, self._tab_title(tab))

    def _update_close_buttons(self):
        # Minimum 1 tab — disable close on the last remaining tab.
        bar = self._tabs.tabBar()
        if bar is None:
            return
        only_one = self._tabs.count() <= 1
        for i in range(self._tabs.count()):
            for side in (bar.ButtonPosition.RightSide,
                         bar.ButtonPosition.LeftSide):
                btn = bar.tabButton(i, side)
                if btn is not None:
                    btn.setEnabled(not only_one)

    def _on_add_tab(self):
        # Default new tab to Ricker (always available).
        tab = self._add_tab_widget(_new_config("Ricker"))
        self._tabs.setCurrentWidget(tab)
        self._update_close_buttons()

    def _on_clone_tab(self):
        cur = self._tabs.currentWidget()
        if cur is None:
            return
        cfg = cur.read_config()
        cfg["name"] = cfg["name"] + "*"
        tab = self._add_tab_widget(cfg)
        self._tabs.setCurrentWidget(tab)
        self._update_close_buttons()

    def _on_rename_tab(self):
        cur = self._tabs.currentWidget()
        if cur is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename tab", "New name:", text=cur.name_edit.text())
        if ok and new_name.strip():
            cur.name_edit.setText(new_name.strip())
            self._refresh_tab_title(cur)

    def _on_close_tab(self, index):
        if self._tabs.count() <= 1:
            return
        self._tabs.removeTab(index)
        self._update_close_buttons()

    # --- Param read/write -------------------------------------------------

    def _read_configs_from_tabs(self):
        out = []
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            out.append(tab.read_config())
        return out

    def _read_params(self):
        return {
            "configs":           self._read_configs_from_tabs(),
            "mergeGapMs":        float(self._merge_spin.value()),
            "minEventMs":        float(self._min_spin.value()),
            "padBeforeFactor":   float(self._padb_spin.value()),
            "padAfterFactor":    float(self._pada_spin.value()),
            "iouThreshold":      float(self._iou_spin.value()),
            "minSeparationFrac": self._params.get("minSeparationFrac", 1.0),
            "splitCoincidents":  self._params.get("splitCoincidents", True),
            "splitReturnSigma":  self._params.get("splitReturnSigma", 3.0),
            "splitMergeGapFrac": self._params.get("splitMergeGapFrac", 0.1),
            "winner":            self._params.get("winner", "auto"),
        }

    def _sync_common_from_params(self):
        p = self._params
        self._merge_spin.setValue(float(p.get("mergeGapMs", 0.0)))
        self._min_spin.setValue(float(p.get("minEventMs", 0.0)))
        self._padb_spin.setValue(float(p.get("padBeforeFactor", 0.0)))
        self._pada_spin.setValue(float(p.get("padAfterFactor", 0.0)))
        self._iou_spin.setValue(float(p.get("iouThreshold", 0.30)))

    # --- Run + render -----------------------------------------------------

    def _run_evaluation(self):
        self._params = self._read_params()
        try:
            self._metrics = evaluate_configs(
                self._signal, self._sample_rate, self._template,
                self._params,
                ground_truth=(self._ground_truth.tolist()
                              if self._ground_truth is not None else None),
            )
        except Exception as e:
            QMessageBox.critical(self, "Evaluation failed", str(e))
            return
        self._populate_table()
        self._populate_winner_combo()
        self._redraw()

    def _populate_table(self):
        names = list(self._metrics.keys())
        self._table.setRowCount(len(names))
        for r, name in enumerate(names):
            m = self._metrics[name]
            shape = m.get("shape", m.get("config", {}).get("shape", ""))
            method = m.get("config", {}).get("scoringMethod", "")
            cells = [name, shape, method]
            if "error" in m:
                cells += [f"err: {m['error'][:40]}"]
                cells += [""] * (len(self._metric_cols) - len(cells))
            else:
                cells += [str(m["count"]),
                          f"{m['peakSnrMean']:.2f}",
                          f"{m['threshold']:.3g}"]
                if self._has_gt:
                    cells += [str(m["tp"]), str(m["fp"]), str(m["fn"]),
                              f"{m['precision']:.2f}",
                              f"{m['recall']:.2f}",
                              f"{m['f1']:.2f}",
                              f"{m['meanIoU']:.2f}"]
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                if c == 1:  # shape column gets the color
                    from PySide6.QtGui import QColor
                    item.setForeground(QColor(_SHAPE_COLORS.get(shape, "#444")))
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()

    def _populate_winner_combo(self):
        cur_pref = self._params.get("winner", "auto")
        self._winner_combo.blockSignals(True)
        self._winner_combo.clear()
        self._winner_combo.addItem("auto")
        for name, m in self._metrics.items():
            if "error" not in m:
                self._winner_combo.addItem(name)
        idx = self._winner_combo.findText(cur_pref)
        self._winner_combo.setCurrentIndex(max(0, idx))
        self._winner_combo.blockSignals(False)

    def _redraw(self):
        # Snapshot current view limits so re-running evaluation doesn't
        # reset the user's zoom/pan. Keyed by detector name (first line of
        # the ylabel) so axes survive reordering when detectors change.
        prev_xlim = None
        prev_ylims = {}
        if self._fig.axes:
            prev_xlim = self._fig.axes[0].get_xlim()
            for _ax in self._fig.axes:
                _key = (_ax.get_ylabel() or "").split("\n", 1)[0]
                if _key:
                    prev_ylims[_key] = _ax.get_ylim()

        self._fig.clear()
        active = [(n, m) for n, m in self._metrics.items() if "error" not in m]
        if not active:
            self._canvas.draw_idle()
            return
        sr = max(self._sample_rate, 1.0)
        n = len(self._signal)
        t = np.arange(n) / sr
        n_axes = 1 + len(active)
        axes = self._fig.subplots(n_axes, 1, sharex=True)
        if n_axes == 1:
            axes = [axes]
        ax_sig = axes[0]
        ax_sig.plot(t, self._signal, color="#3b6fb6", lw=0.7)
        ax_sig.set_ylabel("signal")
        ax_sig.set_title("CWT Evaluator")
        if self._ground_truth is not None:
            for a, b in self._ground_truth:
                ax_sig.axvspan(int(a) / sr, int(b) / sr,
                               color="#888888", alpha=0.30)
        for i, (name, m) in enumerate(active):
            ax = axes[1 + i]
            shape = m.get("shape", "")
            color = _SHAPE_COLORS.get(shape, "#444444")
            ax.plot(t, m["score"], color=color, lw=0.7)
            ax.axhline(m["threshold"], color=color, ls="--", lw=1.0,
                       alpha=0.7)
            ax.axhline(0, color="#888", ls=":", lw=0.5)
            ax.set_ylabel(f"{name}\n(σ={m['sigma']:.2g})")
            for a, b in m["regions"]:
                ax.axvspan(int(a) / sr, int(b) / sr, color=color, alpha=0.20)
                ax_sig.axvspan(int(a) / sr, int(b) / sr,
                               color=color, alpha=0.10)
        axes[-1].set_xlabel("time (s)")
        self._fig.tight_layout()

        # Restore previous view limits (sharex propagates xlim across axes).
        if prev_xlim is not None:
            axes[0].set_xlim(prev_xlim)
        for _ax in self._fig.axes:
            _key = (_ax.get_ylabel() or "").split("\n", 1)[0]
            if _key in prev_ylims:
                _ax.set_ylim(prev_ylims[_key])

        self._canvas.draw_idle()

    # --- Save / Load ------------------------------------------------------

    def _settings_folder(self):
        return saved_templates_dir(os.path.join("Detection", "CwtEvaluator"))

    def _on_save(self):
        if not self._current_path:
            return self._on_save_as()
        try:
            save_state(self._read_params(), self._current_path)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _on_save_as(self):
        folder = self._settings_folder()
        os.makedirs(folder, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CWT Evaluator settings", folder,
            "JSON files (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            save_state(self._read_params(), path)
            self._current_path = path
            self.setWindowTitle(
                f"CWT Evaluator — {os.path.basename(path)}")
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _on_load(self):
        folder = self._settings_folder()
        if not os.path.isdir(folder):
            folder = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CWT Evaluator settings", folder,
            "JSON files (*.json)")
        if not path:
            return
        try:
            self._params = load_state(path)
            self._populate_tabs_from_params()
            self._sync_common_from_params()
            self._current_path = path
            self.setWindowTitle(
                f"CWT Evaluator — {os.path.basename(path)}")
            self._run_evaluation()
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Load failed", str(e))

    # --- Result accessors -------------------------------------------------

    def result_state(self):
        params = self._read_params()
        params["winner"] = self._winner_combo.currentText()
        winner_name = pick_winner(
            self._metrics, prefer=params["winner"],
            has_ground_truth=self._has_gt)
        winner = (self._metrics.get(winner_name)
                  if winner_name else None)
        return params, self._metrics, winner_name, winner
