"""Interactive UI for the ML Classify block.

Streams a recording through a trained event-classifier bundle
(StreamEventClassifier) and plots the predictions against the true labels
when they are available: true regions are shaded under the signal, predicted
events are drawn as a colored strip above it (false positives outlined in
red). Confirm passes events / per-sample predicted labels / score summary
downstream.
"""

import warnings

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton,
)
from PySide6.QtCore import QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from utils.dialog_style import apply_dialog_style, style_mpl_figure
from utils.decimate import minmax_decimate
from processing import ml_training_core as core

_CLASS_COLORS = {1: "#009900", 2: "#dd7700"}  # single, coincident
_UNCERTAIN_COLOR = "#8e44ad"


class _ClassifyWorker(QThread):
    progress = Signal(float)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, saved, data, parent=None):
        super().__init__(parent)
        self._saved = saved
        self._data = data

    def run(self):
        try:
            events = core.simulate_stream(
                self._saved, self._data, progress=self.progress.emit)
            self.finished_ok.emit(events)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _MLClassifyDialog(QDialog):
    def __init__(self, saved, data, labels=None, file_name="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("ML Classify — Stream Review")
        self.resize(1400, 800)
        apply_dialog_style(self)

        self._saved = saved
        self._data = np.asarray(data, dtype=np.float64)
        if self._data.ndim == 1:
            self._data = self._data[:, None]
        self._labels = None if labels is None else \
            np.asarray(labels).astype(np.int32).ravel()
        self._fs = float(saved["fs"])
        self._file_name = file_name
        self.events = None
        self.summary = None
        self.confirmed = False

        self._build_ui()
        self._worker = _ClassifyWorker(saved, self._data, parent=self)
        self._worker.progress.connect(
            lambda f: self._pbar.setValue(int(f * 100)))
        self._worker.finished_ok.connect(self._on_classified)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        head = QHBoxLayout()
        name = f"  —  {self._file_name}" if self._file_name else ""
        self._lbl_head = QLabel(
            f"Classifying{name}  ({len(self._data):,} samples @ "
            f"{self._fs:g} Hz, {self._data.shape[1]} zone(s))")
        head.addWidget(self._lbl_head)
        head.addStretch()
        main.addLayout(head)

        self._pbar = QProgressBar()
        self._pbar.setRange(0, 100)
        main.addWidget(self._pbar)

        self.fig = Figure(figsize=(13, 6))
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self._toolbar = NavigationToolbar2QT(self.canvas, self)
        main.addWidget(self._toolbar)
        main.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        self._lbl_score = QLabel("")
        bottom.addWidget(self._lbl_score)
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

    # ── results ──────────────────────────────────────────────────────────

    def _on_failed(self, msg):
        self._lbl_head.setText(f"Error: {msg}")
        self._pbar.setVisible(False)

    def _on_classified(self, events):
        self.events = events
        self._pbar.setVisible(False)
        pos = [e for e in events if e["label"] > 0]
        n1 = sum(1 for e in pos if e["label"] == 1)
        n2 = sum(1 for e in pos if e["label"] == 2)
        head = (f"{self._file_name + '  —  ' if self._file_name else ''}"
                f"{len(pos)} events predicted "
                f"({n1} single, {n2} coincident, "
                f"{len(events) - len(pos)} rejected triggers)")
        self._lbl_head.setText(head)

        if self._labels is not None and len(self._labels) == len(self._data):
            self.summary = core.score_stream(events, self._labels, self._fs)
            s = self.summary
            self._lbl_score.setText(
                f"true events: {s['n_events']}   detected: "
                f"{s['n_detected']}   correct: {s['n_correct']}   "
                f"false pos: {s['false_positives']} "
                f"({0 if not s['fp_per_min'] else s['fp_per_min']:.1f}/min)")
        self._render()
        self._btn_confirm.setEnabled(True)

    def _render(self):
        self.fig.clear()
        style_mpl_figure(self.fig)
        ax = self.fig.add_subplot(111)
        n, n_zones = self._data.shape
        t = 1.0 / self._fs

        # stacked min/max envelopes per zone
        step = 2.4
        for z in range(n_zones):
            y = self._data[:, z]
            med = np.median(y)
            scale = max(np.percentile(np.abs(y - med), 99.5), 1e-9)
            xi, yi = minmax_decimate(np.arange(n), y, 8000)
            offset = (n_zones - 1 - z) * step
            ax.plot(xi * t, (yi - med) / scale + offset,
                    linewidth=0.5, zorder=3)
        y_top = (n_zones - 1) * step + 1.6

        # true regions: shaded background
        if self._labels is not None and len(self._labels) == n:
            for (s, e, v) in core.parse_events(self._labels):
                if v == 0:
                    continue
                color = _CLASS_COLORS.get(v, _UNCERTAIN_COLOR)
                ax.axvspan(s * t, e * t, color=color,
                           alpha=0.18 if v in _CLASS_COLORS else 0.10,
                           zorder=0, linewidth=0)

        # predicted events: strip above the signal
        gt = [] if self._labels is None else \
            [(s, e) for (s, e, v) in core.parse_events(self._labels)
             if v in (1, 2)]
        for ev in self.events:
            if ev["label"] == 0:
                continue
            x0, x1 = ev["onset"] * t, ev["end"] * t
            is_fp = self._labels is not None and not any(
                min(x1 / t, e) > max(x0 / t, s) for s, e in gt)
            rect = Rectangle((x0, y_top), x1 - x0, 0.6,
                             facecolor=_CLASS_COLORS[ev["label"]],
                             edgecolor="#cc0000" if is_fp else "none",
                             linewidth=1.2 if is_fp else 0,
                             alpha=0.9, zorder=4)
            ax.add_patch(rect)

        ax.set_ylim(-1.6, y_top + 1.0)
        ax.set_xlim(0, n * t)
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.2)
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=_CLASS_COLORS[1], alpha=0.9),
            Rectangle((0, 0), 1, 1, facecolor=_CLASS_COLORS[2], alpha=0.9),
        ]
        names = ["pred single", "pred coincident"]
        if self._labels is not None:
            handles += [
                Rectangle((0, 0), 1, 1, facecolor=_CLASS_COLORS[1],
                          alpha=0.18),
                Rectangle((0, 0), 1, 1, facecolor=_CLASS_COLORS[2],
                          alpha=0.18),
                Rectangle((0, 0), 1, 1, facecolor="none",
                          edgecolor="#cc0000", linewidth=1.2),
            ]
            names += ["true single", "true coincident", "false positive"]
        ax.legend(handles, names, loc="upper right", fontsize=8, ncols=2)
        ax.set_title("predicted events (strip) vs true labels (shading)",
                     fontsize=10)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*tight_layout.*")
            self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_confirm(self):
        if self.events is None:
            return
        self.confirmed = True
        self.accept()

    def closeEvent(self, event):
        if self._worker.isRunning():
            self._worker.wait(3000)
        super().closeEvent(event)


def ml_classify_ui(model_path, data, labels=None, file_name=""):
    """Entry point called by the runner.

    Returns {'events', 'predLabels', 'summary'}; raises if the dialog is
    closed without confirming.
    """
    saved = core.load_bundle(model_path)
    dlg = _MLClassifyDialog(saved, data, labels=labels, file_name=file_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()
    if not dlg.confirmed or dlg.events is None:
        raise ValueError("User closed ML Classify without confirming.")
    return {
        "events": dlg.events,
        "predLabels": core.events_to_labels(dlg.events, len(dlg._data)),
        "summary": dlg.summary,
    }
