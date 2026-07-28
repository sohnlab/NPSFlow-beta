"""Threshold Detection UI — setup, live sim, per-file results, summary."""

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QStackedWidget, QWidget, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from processing.threshold_detection_algo import (
    detect_causal, compute_metrics, find_regions,
)
from utils.decimate import minmax_decimate
from utils.dialog_style import apply_dialog_style, style_mpl_figure


# ---------- plotting helpers ----------

def _downsample_for_plot(y, max_points=20000):
    """Min/max downsample so the plot still shows extremes."""
    return minmax_decimate(np.arange(len(y)), y, max_points)


def _shade_regions(ax, regions, color, alpha=0.25, label=None):
    for i, (s, e) in enumerate(regions):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label if i == 0 else None)


# ---------- Page 1: setup ----------

class _SetupPage(QWidget):
    def __init__(self, parent_dialog, first_item, sample_rate, settings):
        super().__init__()
        self._dlg = parent_dialog
        self._data = np.asarray(first_item["data"], dtype=np.float64)
        self._labels = np.asarray(first_item["labels"])
        self._fs = sample_rate
        self._diff_full = np.diff(self._data) if len(self._data) > 1 else np.zeros(0)

        self._t_pos = float(settings.get("tPos", _auto_threshold(self._diff_full, 1)))
        self._t_neg = float(settings.get("tNeg", _auto_threshold(self._diff_full, -1)))
        self._filter_on = bool(settings.get("filterOn", False))
        self._filter_alpha = float(settings.get("filterAlpha", 0.5))
        self._samples_per_frame = int(settings.get("samplesPerFrame", 2000))

        self._build_ui()
        self._redraw()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self._fig = Figure(figsize=(9, 6))
        self._canvas = FigureCanvas(self._fig)
        self._ax_data = self._fig.add_subplot(2, 1, 1)
        self._ax_diff = self._fig.add_subplot(2, 1, 2, sharex=self._ax_data)
        style_mpl_figure(self._fig)
        layout.addWidget(self._canvas, 1)

        # Mouse drag state for threshold lines
        self._drag_line = None
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

        controls = QHBoxLayout()

        self._t_pos_spin = QDoubleSpinBox()
        self._t_pos_spin.setDecimals(4)
        self._t_pos_spin.setRange(-1e12, 1e12)
        self._t_pos_spin.setSingleStep(
            max(abs(self._t_pos) * 0.05, 1e-6))
        self._t_pos_spin.setValue(self._t_pos)
        self._t_pos_spin.valueChanged.connect(self._on_tpos_spin)
        controls.addWidget(QLabel("T+"))
        controls.addWidget(self._t_pos_spin)

        self._t_neg_spin = QDoubleSpinBox()
        self._t_neg_spin.setDecimals(4)
        self._t_neg_spin.setRange(-1e12, 1e12)
        self._t_neg_spin.setSingleStep(
            max(abs(self._t_neg) * 0.05, 1e-6))
        self._t_neg_spin.setValue(self._t_neg)
        self._t_neg_spin.valueChanged.connect(self._on_tneg_spin)
        controls.addWidget(QLabel("T−"))
        controls.addWidget(self._t_neg_spin)

        self._filter_chk = QCheckBox("Filter diff")
        self._filter_chk.setChecked(self._filter_on)
        self._filter_chk.toggled.connect(self._on_filter_toggle)
        controls.addWidget(self._filter_chk)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setRange(0.01, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setValue(self._filter_alpha)
        self._alpha_spin.setEnabled(self._filter_on)
        self._alpha_spin.valueChanged.connect(self._on_alpha_change)
        controls.addWidget(QLabel("α"))
        controls.addWidget(self._alpha_spin)

        controls.addStretch()

        controls.addWidget(QLabel("Samples/frame"))
        self._spf_spin = QSpinBox()
        self._spf_spin.setRange(10, 200000)
        self._spf_spin.setValue(self._samples_per_frame)
        self._spf_spin.valueChanged.connect(self._on_spf_change)
        controls.addWidget(self._spf_spin)

        layout.addLayout(controls)

        btns = QHBoxLayout()
        btns.addStretch()
        btn_sim = QPushButton("Run Live Sim")
        btn_sim.clicked.connect(lambda: self._dlg.go_to_sim())
        btn_skip = QPushButton("Skip to Results")
        btn_skip.clicked.connect(lambda: self._dlg.go_to_results())
        btns.addWidget(btn_sim)
        btns.addWidget(btn_skip)
        layout.addLayout(btns)

    def settings(self):
        return {
            "tPos": float(self._t_pos),
            "tNeg": float(self._t_neg),
            "filterOn": bool(self._filter_on),
            "filterAlpha": float(self._filter_alpha) if self._filter_on else None,
            "samplesPerFrame": int(self._samples_per_frame),
        }

    def _on_tpos_spin(self, v):
        self._t_pos = float(v)
        self._redraw()

    def _on_tneg_spin(self, v):
        self._t_neg = float(v)
        self._redraw()

    def _on_filter_toggle(self, checked):
        self._filter_on = bool(checked)
        self._alpha_spin.setEnabled(self._filter_on)
        self._redraw()

    def _on_alpha_change(self, v):
        self._filter_alpha = float(v)
        if self._filter_on:
            self._redraw()

    def _on_spf_change(self, v):
        self._samples_per_frame = int(v)

    def _filter_alpha_for_algo(self):
        return self._filter_alpha if self._filter_on else None

    def _redraw(self):
        data = self._data
        diff = self._diff_full
        if self._filter_on and len(diff) > 0:
            # Same IIR for plotting as the algo uses
            from processing.threshold_detection_algo import _apply_iir
            diff_plot = _apply_iir(diff, self._filter_alpha)
        else:
            diff_plot = diff

        self._ax_data.clear()
        self._ax_diff.clear()

        x_d, y_d = _downsample_for_plot(data)
        self._ax_data.plot(x_d, y_d, color="#333", linewidth=0.8)
        self._ax_data.set_ylabel("data")
        self._ax_data.set_title(
            f"Preview: {self._dlg._items[0].get('fileName', 'file 0')}")

        # Preview detection overlay
        detected = detect_causal(data, self._t_pos, self._t_neg,
                                 self._filter_alpha_for_algo())
        _shade_regions(self._ax_data, find_regions(detected > 0),
                       color="#ff7a3a", alpha=0.25, label="detected")
        _shade_regions(self._ax_data, find_regions(self._labels > 0),
                       color="#3a7aff", alpha=0.18, label="true")
        self._ax_data.legend(loc="upper right", fontsize=8)

        x_f, y_f = _downsample_for_plot(diff_plot)
        self._ax_diff.plot(x_f, y_f, color="#666", linewidth=0.6)
        self._ax_diff.axhline(self._t_pos, color="#d14", linestyle="--", linewidth=1.0)
        self._ax_diff.axhline(self._t_neg, color="#14d", linestyle="--", linewidth=1.0)
        self._ax_diff.set_ylabel("diff")
        self._ax_diff.set_xlabel("sample index")

        self._fig.tight_layout()
        self._canvas.draw_idle()

    # --- threshold line dragging ---

    def _on_press(self, event):
        if event.inaxes is not self._ax_diff or event.ydata is None:
            return
        yrange = self._diff_full.max() - self._diff_full.min() if len(self._diff_full) else 1.0
        tol = abs(yrange) * 0.05
        if abs(event.ydata - self._t_pos) < tol:
            self._drag_line = "pos"
        elif abs(event.ydata - self._t_neg) < tol:
            self._drag_line = "neg"

    def _on_motion(self, event):
        if self._drag_line is None or event.inaxes is not self._ax_diff or event.ydata is None:
            return
        if self._drag_line == "pos":
            self._t_pos = float(event.ydata)
            self._t_pos_spin.blockSignals(True)
            self._t_pos_spin.setValue(self._t_pos)
            self._t_pos_spin.blockSignals(False)
        else:
            self._t_neg = float(event.ydata)
            self._t_neg_spin.blockSignals(True)
            self._t_neg_spin.setValue(self._t_neg)
            self._t_neg_spin.blockSignals(False)
        self._redraw()

    def _on_release(self, event):
        self._drag_line = None


def _auto_threshold(diff, sign):
    """Pick an initial threshold based on data std."""
    if len(diff) == 0:
        return sign * 1.0
    return sign * float(np.std(diff) * 4.0)


# ---------- Page 2: live sim ----------

class _SimPage(QWidget):
    def __init__(self, parent_dialog):
        super().__init__()
        self._dlg = parent_dialog
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._cursor = 0
        self._paused = False

        layout = QVBoxLayout(self)
        self._fig = Figure(figsize=(9, 5))
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(1, 1, 1)
        style_mpl_figure(self._fig)
        layout.addWidget(self._canvas, 1)

        controls = QHBoxLayout()
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        controls.addWidget(self._pause_btn)

        controls.addWidget(QLabel("Frame interval (ms)"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(10, 1000)
        self._interval_spin.setValue(33)
        self._interval_spin.valueChanged.connect(
            lambda v: self._timer.setInterval(int(v)))
        controls.addWidget(self._interval_spin)

        controls.addStretch()
        btn_exit = QPushButton("Exit Sim → Results")
        btn_exit.clicked.connect(lambda: self._dlg.go_to_results())
        controls.addWidget(btn_exit)
        layout.addLayout(controls)

    def start(self):
        item = self._dlg._items[0]
        self._data = np.asarray(item["data"], dtype=np.float64)
        settings = self._dlg.current_settings()
        self._detected_full = self._dlg._detections[0]
        self._spf = int(settings["samplesPerFrame"])
        self._t_pos = settings["tPos"]
        self._t_neg = settings["tNeg"]
        self._cursor = 0
        self._paused = False
        self._pause_btn.setText("Pause")
        self._window = int(self._dlg._sample_rate * 2)  # 2-second window
        self._timer.start(int(self._interval_spin.value()))
        self._redraw()

    def stop(self):
        self._timer.stop()

    def _toggle_pause(self):
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")
        if self._paused:
            self._timer.stop()
        else:
            self._timer.start(int(self._interval_spin.value()))

    def _tick(self):
        if self._paused:
            return
        self._cursor = min(self._cursor + self._spf, len(self._data))
        self._redraw()
        if self._cursor >= len(self._data):
            self._timer.stop()

    def _redraw(self):
        end = self._cursor
        start = max(0, end - self._window)
        self._ax.clear()
        seg = self._data[start:end]
        if len(seg) == 0:
            self._canvas.draw_idle()
            return
        xs = np.arange(start, end)
        self._ax.plot(xs, seg, color="#333", linewidth=0.8)
        det_seg = self._detected_full[start:end] > 0
        regions = find_regions(det_seg)
        for s, e in regions:
            self._ax.axvspan(start + s, start + e, color="#ff7a3a", alpha=0.3)
        self._ax.set_xlim(start, max(end, start + self._window))
        self._ax.set_title(
            f"Sim: {self._dlg._items[0].get('fileName', 'file 0')} — "
            f"{end}/{len(self._data)} samples")
        self._ax.set_xlabel("sample index")
        self._fig.tight_layout()
        self._canvas.draw_idle()


# ---------- Page 3: results ----------

class _ResultsPage(QWidget):
    def __init__(self, parent_dialog):
        super().__init__()
        self._dlg = parent_dialog
        self._index = 0

        layout = QVBoxLayout(self)
        self._fig = Figure(figsize=(9, 6))
        self._canvas = FigureCanvas(self._fig)
        self._ax_data = self._fig.add_subplot(2, 1, 1)
        self._ax_diff = self._fig.add_subplot(2, 1, 2, sharex=self._ax_data)
        style_mpl_figure(self._fig)
        layout.addWidget(self._canvas, 1)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("< Prev")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn = QPushButton("Next >")
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._file_combo = QComboBox()
        self._file_combo.currentIndexChanged.connect(self._on_combo_changed)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._file_combo, 1)
        nav.addWidget(self._next_btn)

        btn_summary = QPushButton("Summary →")
        btn_summary.clicked.connect(lambda: self._dlg.go_to_summary())
        nav.addWidget(btn_summary)
        layout.addLayout(nav)

    def load_files(self):
        self._file_combo.blockSignals(True)
        self._file_combo.clear()
        for i, item in enumerate(self._dlg._items):
            name = item.get("fileName", f"file {i}")
            self._file_combo.addItem(f"[{i+1}/{len(self._dlg._items)}] {name}")
        self._file_combo.blockSignals(False)
        self._index = 0
        self._file_combo.setCurrentIndex(0)
        self._redraw()

    def _on_combo_changed(self, idx):
        if idx < 0:
            return
        self._index = idx
        self._redraw()

    def _step(self, delta):
        new_idx = max(0, min(len(self._dlg._items) - 1, self._index + delta))
        if new_idx != self._index:
            self._file_combo.setCurrentIndex(new_idx)

    def _redraw(self):
        item = self._dlg._items[self._index]
        data = np.asarray(item["data"], dtype=np.float64)
        labels = np.asarray(item["labels"])
        detected = self._dlg._detections[self._index]
        diff = np.diff(data) if len(data) > 1 else np.zeros(0)
        if self._dlg._settings.get("filterOn"):
            from processing.threshold_detection_algo import _apply_iir
            diff = _apply_iir(diff, self._dlg._settings.get("filterAlpha"))

        self._ax_data.clear()
        self._ax_diff.clear()

        x_d, y_d = _downsample_for_plot(data)
        self._ax_data.plot(x_d, y_d, color="#333", linewidth=0.8)
        _shade_regions(self._ax_data, find_regions(detected > 0),
                       color="#ff7a3a", alpha=0.3, label="detected")
        _shade_regions(self._ax_data, find_regions(labels > 0),
                       color="#3a7aff", alpha=0.18, label="true")
        self._ax_data.legend(loc="upper right", fontsize=8)
        self._ax_data.set_ylabel("data")
        self._ax_data.set_title(item.get("fileName", f"file {self._index}"))

        x_f, y_f = _downsample_for_plot(diff)
        self._ax_diff.plot(x_f, y_f, color="#666", linewidth=0.6)
        self._ax_diff.axhline(self._dlg._settings["tPos"],
                              color="#d14", linestyle="--", linewidth=1.0)
        self._ax_diff.axhline(self._dlg._settings["tNeg"],
                              color="#14d", linestyle="--", linewidth=1.0)
        self._ax_diff.set_ylabel("diff")
        self._ax_diff.set_xlabel("sample index")

        self._fig.tight_layout()
        self._canvas.draw_idle()

        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < len(self._dlg._items) - 1)


# ---------- Page 4: summary ----------

class _SummaryPage(QWidget):
    def __init__(self, parent_dialog):
        super().__init__()
        self._dlg = parent_dialog

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Per-file performance (sample-level + event-level, IoU ≥ 0.5):"))

        self._table = QTableWidget()
        cols = ["File", "TP", "FP", "FN", "Precision", "Recall", "F1",
                "Det events", "GT events", "Matched", "Missed", "False alarms"]
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)

        btns = QHBoxLayout()
        btn_back = QPushButton("< Back to Results")
        btn_back.clicked.connect(lambda: self._dlg.go_to_results())
        btns.addWidget(btn_back)
        btns.addStretch()
        btn_finish = QPushButton("Finish & Save")
        btn_finish.clicked.connect(self._dlg.finish)
        btns.addWidget(btn_finish)
        layout.addLayout(btns)

    def load_summary(self):
        items = self._dlg._items
        detections = self._dlg._detections
        rows = []
        agg = {"tp": 0, "fp": 0, "fn": 0,
               "det_events": 0, "gt_events": 0,
               "matched": 0, "missed": 0, "false_alarms": 0}
        for item, det in zip(items, detections):
            m = compute_metrics(det, np.asarray(item["labels"]))
            rows.append((item.get("fileName", ""), m))
            for k in agg:
                agg[k] += m[k]
        prec = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0.0
        rec = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        self._table.setRowCount(len(rows) + 1)
        for r, (name, m) in enumerate(rows):
            self._fill_row(r, name, m)
        self._fill_row(len(rows), "— TOTAL —", {
            **agg, "precision": prec, "recall": rec, "f1": f1,
        }, bold=True)

    def _fill_row(self, r, name, m, bold=False):
        def cell(v):
            it = QTableWidgetItem(v)
            if bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            return it
        vals = [
            name,
            str(m["tp"]), str(m["fp"]), str(m["fn"]),
            f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
            str(m["det_events"]), str(m["gt_events"]),
            str(m["matched"]), str(m["missed"]), str(m["false_alarms"]),
        ]
        for c, v in enumerate(vals):
            self._table.setItem(r, c, cell(v))


# ---------- main dialog ----------

class _ThresholdDetectionDialog(QDialog):
    def __init__(self, items, sample_rate, settings):
        super().__init__()
        self.setWindowTitle("Threshold Detection")
        self.resize(1100, 750)
        apply_dialog_style(self)
        self._items = items
        self._sample_rate = sample_rate
        self._settings = settings  # populated from setup page
        self._detections = []  # list of binary int32 arrays, one per item
        self._result = None

        self._setup_page = _SetupPage(self, items[0], sample_rate, settings)
        self._sim_page = _SimPage(self)
        self._results_page = _ResultsPage(self)
        self._summary_page = _SummaryPage(self)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._setup_page)
        self._stack.addWidget(self._sim_page)
        self._stack.addWidget(self._results_page)
        self._stack.addWidget(self._summary_page)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._stack)

    def current_settings(self):
        return self._setup_page.settings()

    def _compute_all(self):
        """Run the causal detector on every item using current settings."""
        self._settings = self.current_settings()
        alpha = self._settings.get("filterAlpha") if self._settings.get("filterOn") else None
        self._detections = []
        for item in self._items:
            det = detect_causal(
                np.asarray(item["data"], dtype=np.float64),
                self._settings["tPos"],
                self._settings["tNeg"],
                alpha,
            )
            self._detections.append(det)

    def go_to_sim(self):
        self._compute_all()
        self._sim_page.start()
        self._stack.setCurrentWidget(self._sim_page)

    def go_to_results(self):
        self._sim_page.stop()
        if not self._detections:
            self._compute_all()
        self._results_page.load_files()
        self._stack.setCurrentWidget(self._results_page)

    def go_to_summary(self):
        self._summary_page.load_summary()
        self._stack.setCurrentWidget(self._summary_page)

    def finish(self):
        # Attach detected arrays to a shallow-copied list of dicts
        out_items = []
        for item, det in zip(self._items, self._detections):
            new_item = dict(item)
            new_item["detected"] = det
            out_items.append(new_item)
        self._result = {
            "labeledDataList": out_items,
            "settings": self._settings,
        }
        self.accept()

    def reject(self):
        self._sim_page.stop()
        self._result = None
        super().reject()


# ---------- public entry point ----------

def threshold_detection_ui(items, sample_rate=10000.0, init_settings=None):
    """Launch the UI, return {'labeledDataList': [...], 'settings': {...}} or None."""
    settings = dict(init_settings or {})
    dlg = _ThresholdDetectionDialog(items, sample_rate, settings)
    dlg.exec()
    return dlg._result
