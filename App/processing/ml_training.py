"""Interactive UI for the ML Training block.

Scans a folder of labeled files (Data Labeling output), splits them into
train/validation/test at the file level, trains the baseline/single/
coincident event classifier, and reports window-level metrics plus a
simulated real-time stream evaluation on the held-out test files. The
trained model bundle (.joblib) is saved for use on live data streams via
ml_training_core.StreamEventClassifier.
"""

import json
import os
import warnings

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QPlainTextEdit, QTabWidget,
    QFileDialog, QWidget, QAbstractItemView, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.dialog_style import apply_dialog_style, style_mpl_figure
from utils.paths import project_root, saved_templates_dir
from processing import ml_training_core as core
from processing.ml_training_core import (
    parse_floats as _parse_floats, parse_ids as _parse_ids)

_SPLITS = ["auto", "train", "val", "test"]
_MODELS = [("Random Forest", "rf"), ("Gradient Boosting", "gb")]
_STRATEGIES = [("Two-stage", "two_stage"), ("Flat 3-way", "flat")]
_OVERSAMPLE = [("SMOTE", "smote"), ("Random", "random"), ("None", "none")]
_REFIT = [("train", "train"), ("train + val", "train+val"), ("all", "all")]


class _ScanWorker(QThread):
    file_scanned = Signal(str, dict)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, folder, noise_ids, drop_ids, parent=None):
        super().__init__(parent)
        self._folder = folder
        self._noise_ids = noise_ids
        self._drop_ids = drop_ids
        self.stop = False

    def run(self):
        try:
            paths = core.find_labeled_files(self._folder)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        n = 0
        for path in paths:
            if self.stop:
                return
            name = os.path.basename(path)
            try:
                labels = core.canonicalize_labels(
                    core.load_labels_only(path),
                    self._noise_ids, self._drop_ids)
                counts = core.event_counts(labels)
            except Exception as exc:
                counts = {"error": str(exc)}
            self.file_scanned.emit(name, counts)
            n += 1
        self.done.emit(n)


class _TrainWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self.cancel = False

    def run(self):
        try:
            res = core.run_training(self._cfg,
                                    progress=self.progress.emit,
                                    cancel=lambda: self.cancel)
            self.finished_ok.emit(res)
        except InterruptedError:
            self.failed.emit("Training cancelled.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _MLTrainingDialog(QDialog):
    def __init__(self, folder="", sample_rate=None, init_settings=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("ML Training — Event Classifier")
        self.resize(1320, 880)
        apply_dialog_style(self)

        self.confirmed = False
        self.result = None
        self._counts = {}
        self._scan_worker = None
        self._train_worker = None
        self._fs_from_port = sample_rate is not None and sample_rate > 1

        self._build_ui()

        cfg = dict(core.DEFAULT_CONFIG)
        if init_settings:
            cfg.update({k: v for k, v in init_settings.items()
                        if v is not None})
        if folder:
            cfg["folder"] = folder
        self._apply_settings(cfg)
        if self._fs_from_port:
            self._spin_fs.setValue(float(np.ravel(sample_rate)[0]))
            self._spin_fs.setEnabled(False)
            self._spin_fs.setToolTip("Sample rate wired from the "
                                     "sampleRate input port")
        if self._edit_folder.text():
            self._rescan()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        # Folder row
        row = QHBoxLayout()
        row.addWidget(QLabel("Labeled data folder:"))
        self._edit_folder = QLineEdit()
        self._edit_folder.editingFinished.connect(self._rescan)
        row.addWidget(self._edit_folder, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_folder)
        row.addWidget(btn)
        self._lbl_scan = QLabel("")
        row.addWidget(self._lbl_scan)
        main.addLayout(row)

        body = QHBoxLayout()
        main.addLayout(body, 1)

        # Left: file table over results tabs
        left = QVBoxLayout()
        body.addLayout(left, 1)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["File", "Single", "Coincident", "Uncertain", "Split"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 5):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        left.addWidget(self._table, 2)

        self._tabs = QTabWidget()
        self._tab_metrics = QTableWidget(0, 8)
        self._tab_metrics.setHorizontalHeaderLabels(
            ["Set", "Horizon", "n", "Accuracy", "Macro-F1",
             "R baseline", "R single", "R coincident"])
        self._tab_metrics.verticalHeader().setVisible(False)
        self._tab_metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tab_metrics.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._tabs.addTab(self._tab_metrics, "Metrics")

        self.fig = Figure(figsize=(9, 3.2))
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self._tabs.addTab(self.canvas, "Plots")

        self._tab_stream = QTableWidget(0, 7)
        self._tab_stream.setHorizontalHeaderLabels(
            ["Test file", "Events", "Detected", "Correct", "False pos",
             "FP / min", "Rejected triggers"])
        self._tab_stream.verticalHeader().setVisible(False)
        self._tab_stream.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tab_stream.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._tabs.addTab(self._tab_stream, "Stream Test")

        self._tab_feat = QTableWidget(0, 2)
        self._tab_feat.setHorizontalHeaderLabels(["Feature", "Importance"])
        self._tab_feat.verticalHeader().setVisible(False)
        self._tab_feat.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tab_feat.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._tabs.addTab(self._tab_feat, "Features")
        left.addWidget(self._tabs, 3)

        # Right: settings column
        right_host = QScrollArea()
        right_host.setWidgetResizable(True)
        right_host.setFixedWidth(340)
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(4, 0, 4, 0)
        right_host.setWidget(right_widget)
        body.addWidget(right_host)

        g = QGroupBox("Training")
        f = QFormLayout(g)
        self._spin_fs = QDoubleSpinBox()
        self._spin_fs.setRange(1, 1e8)
        self._spin_fs.setDecimals(0)
        self._spin_fs.setSuffix(" Hz")
        f.addRow("Sample rate", self._spin_fs)
        self._cmb_model = QComboBox()
        for label, _v in _MODELS:
            self._cmb_model.addItem(label)
        f.addRow("Model", self._cmb_model)
        self._cmb_strategy = QComboBox()
        for label, _v in _STRATEGIES:
            self._cmb_strategy.addItem(label)
        f.addRow("Strategy", self._cmb_strategy)
        self._cmb_oversample = QComboBox()
        for label, _v in _OVERSAMPLE:
            self._cmb_oversample.addItem(label)
        f.addRow("Oversample", self._cmb_oversample)
        self._spin_ratio = QDoubleSpinBox()
        self._spin_ratio.setRange(0.0, 5.0)
        self._spin_ratio.setSingleStep(0.1)
        f.addRow("Oversample ratio", self._spin_ratio)
        self._spin_noise = QDoubleSpinBox()
        self._spin_noise.setRange(0.0, 10.0)
        self._spin_noise.setSingleStep(0.1)
        f.addRow("Baseline ratio", self._spin_noise)
        self._spin_hard = QDoubleSpinBox()
        self._spin_hard.setRange(0.0, 10.0)
        self._spin_hard.setSingleStep(0.1)
        self._spin_hard.setToolTip(
            "Trigger-style false positives mined from background as extra "
            "baseline examples, relative to the single count")
        f.addRow("Hard negatives", self._spin_hard)
        self._edit_horizons = QLineEdit()
        self._edit_horizons.setToolTip(
            "Decision latencies (ms after onset); a 'full' horizon is "
            "always added")
        f.addRow("Horizons (ms)", self._edit_horizons)
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 99999)
        f.addRow("Seed", self._spin_seed)
        right.addWidget(g)

        g = QGroupBox("Split (by file)")
        f = QFormLayout(g)
        self._spin_split = {}
        for key in ("train", "val", "test"):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.05)
            self._spin_split[key] = sp
            f.addRow(key, sp)
        self._cmb_refit = QComboBox()
        for label, _v in _REFIT:
            self._cmb_refit.addItem(label)
        self._cmb_refit.setToolTip(
            "Which splits the saved model is refit on after evaluation")
        f.addRow("Final refit on", self._cmb_refit)
        self._chk_stream = QCheckBox("Simulate real-time stream on test")
        f.addRow(self._chk_stream)
        right.addWidget(g)

        g = QGroupBox("Label mapping")
        f = QFormLayout(g)
        self._edit_noise_ids = QLineEdit()
        self._edit_noise_ids.setToolTip(
            "Raw label ids treated as baseline (noise = baseline)")
        f.addRow("Noise ids", self._edit_noise_ids)
        self._edit_drop_ids = QLineEdit()
        self._edit_drop_ids.setToolTip(
            "Raw label ids dropped as uncertain")
        f.addRow("Uncertain ids", self._edit_drop_ids)
        for e in (self._edit_noise_ids, self._edit_drop_ids):
            e.editingFinished.connect(self._rescan)
        right.addWidget(g)

        g = QGroupBox("Stream trigger")
        f = QFormLayout(g)
        self._trig = {}
        for key, label, lo, hi, step in (
                ("trigger_k", "Trigger (σ)", 0.5, 50.0, 0.5),
                ("release_k", "Release (σ)", 0.1, 20.0, 0.5),
                ("smooth_ms", "Smoothing (ms)", 0.0, 20.0, 0.1),
                ("min_active_ms", "Min active (ms)", 0.0, 50.0, 0.1),
                ("end_quiet_ms", "End quiet (ms)", 0.1, 100.0, 0.5),
                ("max_event_ms", "Max event (ms)", 10.0, 10000.0, 50.0),
                ("baseline_ms", "Baseline (ms)", 0.5, 100.0, 0.5)):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            self._trig[key] = sp
            f.addRow(label, sp)
        right.addWidget(g)

        g = QGroupBox("Model output")
        v = QVBoxLayout(g)
        rowm = QHBoxLayout()
        self._edit_model_path = QLineEdit()
        rowm.addWidget(self._edit_model_path, 1)
        btn = QPushButton("…")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._browse_model)
        rowm.addWidget(btn)
        v.addLayout(rowm)
        right.addWidget(g)

        self._btn_train = QPushButton("Train")
        self._btn_train.clicked.connect(self._toggle_train)
        right.addWidget(self._btn_train)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setMinimumHeight(120)
        right.addWidget(self._log, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        bottom.addWidget(self._btn_confirm)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        main.addLayout(bottom)

    # ── settings <-> widgets ─────────────────────────────────────────────

    def _apply_settings(self, cfg):
        def pick(pairs, value, combo):
            for i, (_label, v) in enumerate(pairs):
                if v == value:
                    combo.setCurrentIndex(i)
                    return

        self._edit_folder.setText(str(cfg.get("folder") or ""))
        self._spin_fs.setValue(float(cfg.get("fs") or core.DEFAULT_FS))
        pick(_MODELS, cfg.get("model", "rf"), self._cmb_model)
        pick(_STRATEGIES, cfg.get("strategy", "two_stage"),
             self._cmb_strategy)
        pick(_OVERSAMPLE, cfg.get("oversample", "smote"),
             self._cmb_oversample)
        self._spin_ratio.setValue(float(cfg.get("oversample_ratio", 1.0)))
        self._spin_noise.setValue(float(cfg.get("noise_ratio", 1.0)))
        self._spin_hard.setValue(float(cfg.get("hard_negative_ratio", 1.0)))
        self._edit_horizons.setText(", ".join(
            f"{h:g}" for h in cfg.get("horizons_ms",
                                      core.DEFAULT_HORIZONS_MS)))
        self._spin_seed.setValue(int(cfg.get("seed", 0)))
        split = cfg.get("split") or {}
        for key, sp in self._spin_split.items():
            sp.setValue(float(split.get(key,
                                        core.DEFAULT_CONFIG["split"][key])))
        pick(_REFIT, cfg.get("final_refit", "train+val"), self._cmb_refit)
        self._chk_stream.setChecked(bool(cfg.get("stream_eval", True)))
        self._edit_noise_ids.setText(", ".join(
            str(i) for i in cfg.get("noise_ids") or []))
        self._edit_drop_ids.setText(", ".join(
            str(i) for i in cfg.get("drop_ids") or [3, 4]))
        trig = dict(core.DEFAULT_TRIGGER)
        trig.update(cfg.get("trigger") or {})
        for key, sp in self._trig.items():
            sp.setValue(float(trig.get(key, core.DEFAULT_TRIGGER[key])))
        self._edit_model_path.setText(
            str(cfg.get("modelPath") or core.DEFAULT_CONFIG["modelPath"]))
        self._assignments = dict(cfg.get("assignments") or {})

    def current_config(self):
        assignments = {}
        for r in range(self._table.rowCount()):
            name = self._table.item(r, 0).text()
            combo = self._table.cellWidget(r, 4)
            if combo and combo.currentText() != "auto":
                assignments[name] = combo.currentText()
        horizons = _parse_floats(self._edit_horizons.text()) \
            or list(core.DEFAULT_HORIZONS_MS)
        return {
            "folder": self._edit_folder.text().strip(),
            "fs": float(self._spin_fs.value()),
            "model": _MODELS[self._cmb_model.currentIndex()][1],
            "strategy": _STRATEGIES[self._cmb_strategy.currentIndex()][1],
            "oversample":
                _OVERSAMPLE[self._cmb_oversample.currentIndex()][1],
            "oversample_ratio": float(self._spin_ratio.value()),
            "smote_k": 5,
            "noise_ratio": float(self._spin_noise.value()),
            "hard_negative_ratio": float(self._spin_hard.value()),
            "horizons_ms": horizons,
            "seed": int(self._spin_seed.value()),
            "split": {k: float(sp.value())
                      for k, sp in self._spin_split.items()},
            "assignments": assignments,
            "final_refit": _REFIT[self._cmb_refit.currentIndex()][1],
            "stream_eval": self._chk_stream.isChecked(),
            "noise_ids": _parse_ids(self._edit_noise_ids.text()),
            "drop_ids": _parse_ids(self._edit_drop_ids.text()),
            "trigger": {k: float(sp.value())
                        for k, sp in self._trig.items()},
            "modelPath": self._edit_model_path.text().strip(),
        }

    # ── folder scan ──────────────────────────────────────────────────────

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Labeled data folder",
            self._edit_folder.text() or project_root())
        if path:
            self._edit_folder.setText(path)
            self._rescan()

    def _browse_model(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Model file",
            core.resolve_path(self._edit_model_path.text()),
            "Model bundle (*.joblib)")
        if path:
            self._edit_model_path.setText(path)

    def _rescan(self):
        folder = core.resolve_path(self._edit_folder.text().strip())
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.stop = True
            self._scan_worker.wait(1000)
        self._table.setRowCount(0)
        self._counts = {}
        if not folder or not os.path.isdir(folder):
            self._lbl_scan.setText("folder not found" if folder else "")
            return
        self._lbl_scan.setText("scanning…")
        self._scan_worker = _ScanWorker(
            folder, _parse_ids(self._edit_noise_ids.text()),
            _parse_ids(self._edit_drop_ids.text()), parent=self)
        self._scan_worker.file_scanned.connect(self._on_file_scanned)
        self._scan_worker.done.connect(
            lambda n: self._lbl_scan.setText(f"{n} file(s)"))
        self._scan_worker.failed.connect(
            lambda msg: self._lbl_scan.setText(msg))
        self._scan_worker.start()

    def _on_file_scanned(self, name, counts):
        self._counts[name] = counts
        r = self._table.rowCount()
        self._table.insertRow(r)
        if "error" in counts:
            vals = [name, "—", "—", counts["error"]]
        else:
            vals = [name, str(counts["single"]), str(counts["coincident"]),
                    str(counts["uncertain"])]
        for c, text in enumerate(vals):
            item = QTableWidgetItem(text)
            if c:
                item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, c, item)
        combo = QComboBox()
        combo.addItems(_SPLITS)
        combo.setCurrentText(self._assignments.get(name, "auto"))
        self._table.setCellWidget(r, 4, combo)

    # ── training ─────────────────────────────────────────────────────────

    def _toggle_train(self):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.cancel = True
            self._btn_train.setText("Stopping…")
            self._btn_train.setEnabled(False)
            return
        cfg = self.current_config()
        if not cfg["folder"]:
            self._log.appendPlainText("Set a labeled-data folder first.")
            return
        self._log.appendPlainText("— training started —")
        self._btn_train.setText("Stop")
        self._btn_confirm.setEnabled(False)
        self._train_worker = _TrainWorker(cfg, parent=self)
        self._train_worker.progress.connect(self._log.appendPlainText)
        self._train_worker.finished_ok.connect(self._on_trained)
        self._train_worker.failed.connect(self._on_train_failed)
        self._train_worker.start()

    def _on_train_failed(self, msg):
        self._log.appendPlainText(f"Error: {msg}")
        self._btn_train.setText("Train")
        self._btn_train.setEnabled(True)

    def _on_trained(self, res):
        self.result = res
        self._btn_train.setText("Train")
        self._btn_train.setEnabled(True)
        self._btn_confirm.setEnabled(True)
        for r in range(self._table.rowCount()):
            name = self._table.item(r, 0).text()
            combo = self._table.cellWidget(r, 4)
            if combo and combo.currentText() == "auto":
                combo.setToolTip(f"auto → {res['split'].get(name, '?')}")
        self._render_metrics(res)
        self._render_plots(res)
        self._render_stream(res)
        self._render_features(res)

    # ── results rendering ────────────────────────────────────────────────

    @staticmethod
    def _fmt(v, digits=3):
        return "n/a" if v is None else f"{v:.{digits}f}"

    def _render_metrics(self, res):
        t = self._tab_metrics
        t.setRowCount(0)
        for part in ("val", "test"):
            m = res["metrics"].get(part) or {}
            for hz in sorted(m, key=core.horizon_sort_key):
                d = m[hz]
                pc = d["per_class"]
                row = [part, hz, str(d["n"]), self._fmt(d["accuracy"]),
                       self._fmt(d["macro_f1"]),
                       self._fmt(pc["baseline"]["recall"], 2),
                       self._fmt(pc["single"]["recall"], 2),
                       self._fmt(pc["coincident"]["recall"], 2)]
                r = t.rowCount()
                t.insertRow(r)
                for c, text in enumerate(row):
                    item = QTableWidgetItem(text)
                    if c > 1:
                        item.setTextAlignment(Qt.AlignCenter)
                    t.setItem(r, c, item)
        self._tabs.setCurrentIndex(0)

    def _render_plots(self, res):
        self.fig.clear()
        style_mpl_figure(self.fig)
        ax1 = self.fig.add_subplot(1, 3, 1)
        for part, ls in (("val", "--"), ("test", "-")):
            m = res["metrics"].get(part) or {}
            hzs = sorted(m, key=core.horizon_sort_key)
            if not hzs:
                continue
            xs = range(len(hzs))
            ax1.plot(xs, [m[h]["accuracy"] for h in hzs], ls, marker="o",
                     color="#1f77b4", label=f"{part} acc", linewidth=1.2)
            ax1.plot(xs, [m[h]["macro_f1"] for h in hzs], ls, marker="s",
                     color="#ff7f0e", label=f"{part} mF1", linewidth=1.2)
            ax1.set_xticks(list(xs))
            ax1.set_xticklabels(hzs, fontsize=7)
        ax1.set_ylim(0, 1.02)
        ax1.set_title("score vs decision latency", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7)

        names = [core.CLASS_NAMES[i] for i in (0, 1, 2)]
        for k, part in enumerate(("val", "test")):
            m = res["metrics"].get(part) or {}
            if "full" not in m:
                continue
            cm = np.asarray(m["full"]["confusion"], dtype=float)
            ax = self.fig.add_subplot(1, 3, 2 + k)
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3))
            ax.set_xticklabels(names, fontsize=7)
            ax.set_yticks(range(3))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("predicted", fontsize=8)
            ax.set_ylabel("true", fontsize=8)
            ax.set_title(f"{part} @ full", fontsize=9)
            total = cm.sum(axis=1, keepdims=True)
            total[total == 0] = 1
            for i in range(3):
                for j in range(3):
                    frac = cm[i, j] / total[i, 0]
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center",
                            va="center", fontsize=8,
                            color="white" if frac > 0.5 else "black")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*tight_layout.*")
            self.fig.tight_layout()
        self.canvas.draw_idle()

    def _render_stream(self, res):
        t = self._tab_stream
        t.setRowCount(0)
        stream = res["metrics"].get("stream_test") or {}
        for name, s in stream.items():
            row = [name, str(s["n_events"]), str(s["n_detected"]),
                   str(s["n_correct"]), str(s["false_positives"]),
                   self._fmt(s["fp_per_min"], 2),
                   str(s["rejected_triggers"])]
            r = t.rowCount()
            t.insertRow(r)
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignCenter)
                t.setItem(r, c, item)

    def _render_features(self, res):
        t = self._tab_feat
        t.setRowCount(0)
        for name, imp in res.get("importances") or []:
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(f"{imp:.4f}")
            item.setTextAlignment(Qt.AlignCenter)
            t.setItem(r, 1, item)

    # ── confirm / close ──────────────────────────────────────────────────

    def _on_confirm(self):
        if self.result is None:
            return
        self._save_settings()
        self.confirmed = True
        self.accept()

    def _save_settings(self):
        path = os.path.join(saved_templates_dir("ML"), "temp.json")
        try:
            with open(path, "w") as f:
                json.dump(self.current_config(), f, indent=2)
        except Exception as exc:
            from utils.app_logger import logger
            logger.warning(f"[MLTraining] settings autosave failed: {exc}")

    def closeEvent(self, event):
        for w in (self._scan_worker, self._train_worker):
            if w and w.isRunning():
                if hasattr(w, "cancel"):
                    w.cancel = True
                if hasattr(w, "stop"):
                    w.stop = True
                w.wait(3000)
        super().closeEvent(event)


def ml_training_ui(folder="", sample_rate=None, init_settings=None):
    """Entry point called by the runner.

    Returns dict with 'modelFile', 'metrics', and the dialog settings.
    Raises if the dialog is closed without confirming a trained model.
    """
    dlg = _MLTrainingDialog(folder=folder, sample_rate=sample_rate,
                            init_settings=init_settings)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()

    if not dlg.confirmed or dlg.result is None:
        raise ValueError("User closed ML Training without confirming "
                         "a trained model.")
    res = dlg.result
    return {
        "modelFile": res["modelPath"],
        "metrics": {
            "split": res["split"],
            "counts": res["counts"],
            "metrics": res["metrics"],
            "importances": res["importances"],
            "class_names": res["class_names"],
        },
        "settings": dlg.current_config(),
    }
