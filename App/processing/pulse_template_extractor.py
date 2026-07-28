"""Interactive pulse template extraction via SpanSelector range selection.

Drag a region on the signal to mark the pulse template range. The shaded
band is reshapable from either edge and persists across window-scroll;
Start / End spinboxes below the plot stay in sync for precise input.
"""

import math

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QDoubleSpinBox, QLabel, QComboBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from utils.dialog_style import style_mpl_figure


class _PulseTemplateExtractorDialog(QDialog):
    """Interactive dialog for selecting a pulse template region."""

    def __init__(self, data, sample_rate, parent=None, settings_file=None,
                 files=None):
        super().__init__(parent)
        self.setWindowTitle("Pulse Template Extraction")
        self.resize(1050, 490)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        # (name, 2-D array) per selectable file; bare input → one unnamed file.
        self._files = files if files else [("", data)]
        self._file_idx = 0
        self._per_file_state = {}
        self._sample_rate = sample_rate
        self._data = self._files[0][1]
        self._n_samples, self._n_channels = self._data.shape

        self._window_size = self._default_window()
        self._step_size = max(1, self._window_size // 3)
        self._current_start = 0

        # Default selection at 25%/75% of the first window
        ws0 = min(self._window_size, self._n_samples)
        self._start_idx = int(0.25 * ws0)
        self._end_idx = int(0.75 * ws0)
        self.confirmed = False
        self._span = None
        # X-axis display unit: "index" (sample number) or "time" (seconds).
        self._x_unit = "index"

        self._compute_scale()
        self._build_ui()
        # Settings load only from the loadFile port (empty → defaults).
        if settings_file:
            self._load_state_from(settings_file)
        self._sync_unit_combo()
        self._configure_spins()
        self._plot_window()

    def _default_window(self):
        """2 s of samples, clamped so a bad sample rate can't collapse the
        view to a handful of points."""
        return max(16, min(int(round(2 * self._sample_rate)),
                           self._n_samples))

    def _compute_scale(self):
        """Display scale for the index spinboxes on very large records."""
        if self._n_samples >= 10000:
            k = int(math.floor(math.log10(self._n_samples)))
            self._scale = 10 ** k
            self._scale_exp = k
        else:
            self._scale = 1
            self._scale_exp = 0

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if len(self._files) > 1:
            file_row = QHBoxLayout()
            file_row.setSpacing(6)
            file_row.addWidget(QLabel("File:"))
            self._file_combo = QComboBox()
            self._file_combo.addItems(
                [n or f"file {i + 1}" for i, (n, _) in enumerate(self._files)])
            self._file_combo.setMinimumWidth(280)
            self._file_combo.setToolTip("Signal file to extract the template from")
            self._file_combo.currentIndexChanged.connect(self._on_file_changed)
            file_row.addWidget(self._file_combo)
            file_row.addStretch()
            layout.addLayout(file_row)

        # Matplotlib canvas — 21:9, matching Trim Data
        self.fig = Figure(figsize=(10.5, 4.5), tight_layout=True)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        layout.addWidget(self.canvas, 1)

        # Single bottom row: Start | End | navigation | view | export | actions
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        btn_layout.addWidget(QLabel("X-axis:"))
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["Index", "Time (s)"])
        self._unit_combo.setFixedWidth(96)
        self._unit_combo.setToolTip("Units for the x-axis and Start/End fields")
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        btn_layout.addWidget(self._unit_combo)
        btn_layout.addSpacing(12)

        btn_layout.addWidget(QLabel("Start:"))
        self._start_spin = self._make_spin()
        self._start_spin.valueChanged.connect(self._on_spin_changed)
        btn_layout.addWidget(self._start_spin)
        btn_layout.addSpacing(6)
        btn_layout.addWidget(QLabel("End:"))
        self._end_spin = self._make_spin()
        self._end_spin.valueChanged.connect(self._on_spin_changed)
        btn_layout.addWidget(self._end_spin)

        btn_layout.addStretch()

        nav_btn_width = 100
        self._btn_prev = QPushButton("<< Previous")
        self._btn_prev.setFixedWidth(nav_btn_width)
        self._btn_prev.clicked.connect(self._on_previous)
        btn_layout.addWidget(self._btn_prev)
        self._btn_next = QPushButton("Next >>")
        self._btn_next.setFixedWidth(nav_btn_width)
        self._btn_next.clicked.connect(self._on_next)
        btn_layout.addWidget(self._btn_next)
        btn_layout.addSpacing(12)

        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedWidth(32)
        self._btn_zoom_in.setToolTip("Zoom in (x-axis)")
        self._btn_zoom_in.clicked.connect(self._on_zoom_in)
        btn_layout.addWidget(self._btn_zoom_in)
        self._btn_zoom_out = QPushButton("−")
        self._btn_zoom_out.setFixedWidth(32)
        self._btn_zoom_out.setToolTip("Zoom out (x-axis)")
        self._btn_zoom_out.clicked.connect(self._on_zoom_out)
        btn_layout.addWidget(self._btn_zoom_out)

        btn_layout.addSpacing(24)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self._btn_reset)

        from utils.export_figure_ui import make_export_button
        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        btn_layout.addWidget(self._btn_export)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self._btn_confirm)

        layout.addLayout(btn_layout)

    # ------------------------------------------------- unit / scale helpers
    def _make_spin(self):
        sb = QDoubleSpinBox()
        sb.setFixedWidth(120)
        sb.setKeyboardTracking(False)
        sb.wheelEvent = lambda ev: ev.ignore()
        return sb

    def _configure_spins(self):
        """Set spinbox range / decimals / suffix for the current x-axis unit."""
        for sb in (self._start_spin, self._end_spin):
            sb.blockSignals(True)
        if self._x_unit == "time":
            hi = (self._n_samples - 1) / self._sample_rate
            for sb in (self._start_spin, self._end_spin):
                sb.setDecimals(6)
                sb.setRange(0.0, hi)
                sb.setSingleStep(1.0 / self._sample_rate)
                sb.setSuffix(" s")
        else:
            hi = (self._n_samples - 1) / self._scale
            suffix = "" if self._scale == 1 else f" ×1e{self._scale_exp}"
            for sb in (self._start_spin, self._end_spin):
                sb.setDecimals(0 if self._scale == 1 else 4)
                sb.setRange(0.0, hi)
                sb.setSingleStep(1.0 if self._scale == 1 else 0.01)
                sb.setSuffix(suffix)
        for sb in (self._start_spin, self._end_spin):
            sb.blockSignals(False)
        self._sync_spinboxes()

    def _sync_unit_combo(self):
        self._unit_combo.blockSignals(True)
        self._unit_combo.setCurrentIndex(1 if self._x_unit == "time" else 0)
        self._unit_combo.blockSignals(False)

    # Sample index <-> displayed spinbox value (scaled index, or seconds).
    def _spin_of_idx(self, idx):
        if self._x_unit == "time":
            return idx / self._sample_rate
        return idx / self._scale

    def _idx_of_spin(self, v):
        if self._x_unit == "time":
            idx = int(round(float(v) * self._sample_rate))
        else:
            idx = int(round(float(v) * self._scale))
        return max(0, min(self._n_samples - 1, idx))

    # Sample index <-> x-axis coordinate (raw index, or seconds).
    def _axis_x(self, idx):
        return idx / self._sample_rate if self._x_unit == "time" else float(idx)

    def _idx_of_axis(self, x):
        idx = x * self._sample_rate if self._x_unit == "time" else x
        return max(0, min(self._n_samples - 1, int(round(idx))))

    def _sync_spinboxes(self):
        self._start_spin.blockSignals(True)
        self._end_spin.blockSignals(True)
        self._start_spin.setValue(self._spin_of_idx(self._start_idx))
        self._end_spin.setValue(self._spin_of_idx(self._end_idx))
        self._start_spin.blockSignals(False)
        self._end_spin.blockSignals(False)

    def _on_unit_changed(self, _index):
        self._x_unit = "time" if self._unit_combo.currentIndex() == 1 else "index"
        self._configure_spins()
        self._plot_window()

    def _on_file_changed(self, idx):
        if idx == self._file_idx or not (0 <= idx < len(self._files)):
            return
        self._per_file_state[self._file_idx] = (
            self._start_idx, self._end_idx,
            self._window_size, self._current_start)
        self._apply_file(idx)
        self._configure_spins()
        self._plot_window()

    def _apply_file(self, idx):
        """Switch the active file; restore its saved view or the defaults."""
        self._file_idx = idx
        self._data = self._files[idx][1]
        self._n_samples, self._n_channels = self._data.shape
        state = self._per_file_state.get(idx)
        if state:
            self._start_idx, self._end_idx, ws, cs = state
            self._window_size = min(ws, self._n_samples)
        else:
            self._window_size = self._default_window()
            cs = 0
            ws0 = min(self._window_size, self._n_samples)
            self._start_idx = int(0.25 * ws0)
            self._end_idx = int(0.75 * ws0)
        self._step_size = max(1, self._window_size // 3)
        self._current_start = max(0, min(cs, self._n_samples - 1))
        self._compute_scale()

    def _plot_window(self):
        self.ax.clear()
        cs = self._current_start
        ws = self._window_size
        end_window = min(cs + ws, self._n_samples)
        idx = np.arange(cs, end_window)
        xvals = idx / self._sample_rate if self._x_unit == "time" else idx

        if self._n_channels == 1:
            self.ax.plot(xvals, self._data[idx, 0], linewidth=1.0,
                         label="_nolegend_")
            self.ax.set_ylabel("Signal Amplitude")
        else:
            zero_mean = self._data[idx] - np.mean(self._data[idx], axis=0)
            spacing = 0
            for i in range(self._n_channels - 1):
                req = 2 * (np.max(zero_mean[:, i + 1]) - np.min(zero_mean[:, i]))
                spacing = max(spacing, req)
            if spacing <= 0:
                spacing = 0.1 * np.mean(np.ptp(zero_mean, axis=0))
            for i in range(self._n_channels):
                offset = (self._n_channels - i) * spacing / 2
                self.ax.plot(xvals, zero_mean[:, i] + offset, linewidth=1.0,
                            label=f"Zone {i + 1}")
            self.ax.set_ylabel("Resistance (offset)")
            self.ax.legend(loc="upper right")

        self.ax.set_xlabel("Time (s)" if self._x_unit == "time" else "Sample Index")
        self.ax.set_title("Select the pulse template range")
        self.ax.grid(True)
        self.ax.set_xlim(self._axis_x(cs), self._axis_x(end_window))

        # Re-attach SpanSelector to the fresh axes.
        self._span = SpanSelector(
            self.ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="#4a86e8"),
            interactive=True,
            drag_from_anywhere=True,
            ignore_event_outside=True,
        )
        try:
            self._span.extents = (self._axis_x(self._start_idx),
                                  self._axis_x(self._end_idx))
        except (AttributeError, ValueError):
            pass

        self._sync_spinboxes()
        self.canvas.draw_idle()

    # ----------------------------------------------------------- callbacks
    def _on_span_select(self, xmin, xmax):
        si = self._idx_of_axis(xmin)
        ei = self._idx_of_axis(xmax)
        if si >= ei:
            return
        self._start_idx, self._end_idx = si, ei
        self._sync_spinboxes()

    def _on_spin_changed(self, _value):
        si = self._idx_of_spin(self._start_spin.value())
        ei = self._idx_of_spin(self._end_spin.value())
        if si >= ei:
            return
        self._start_idx, self._end_idx = si, ei
        if self._span is not None:
            try:
                self._span.extents = (self._axis_x(si), self._axis_x(ei))
            except (AttributeError, ValueError):
                pass
        self.canvas.draw_idle()

    # Navigation scrolls the view AND shifts the selection band so it keeps
    # the same relative position inside the window (its absolute position
    # in the data shifts with the window).
    def _shift_selection_with_window(self, old_start, new_start):
        """Translate start/end by the same delta as the window scroll."""
        delta = new_start - old_start
        si = int(np.clip(self._start_idx + delta, 0, self._n_samples - 1))
        ei = int(np.clip(self._end_idx + delta, 0, self._n_samples - 1))
        if si < ei:
            self._start_idx, self._end_idx = si, ei

    def _on_previous(self):
        old = self._current_start
        new = max(0, old - self._step_size)
        if new == old:
            return
        self._current_start = new
        self._shift_selection_with_window(old, new)
        self._plot_window()

    def _on_next(self):
        old = self._current_start
        new = old + self._step_size
        if new + self._window_size > self._n_samples:
            return
        self._current_start = new
        self._shift_selection_with_window(old, new)
        self._plot_window()

    def _on_reset(self):
        self._window_size = self._default_window()
        self._step_size = max(1, self._window_size // 3)
        self._current_start = 0
        ws0 = min(self._window_size, self._n_samples)
        self._start_idx = int(0.25 * ws0)
        self._end_idx = int(0.75 * ws0)
        self._plot_window()

    def _on_zoom_in(self):
        # Recenter on the selection midpoint if the band is inside the data;
        # otherwise keep the current center.
        if 0 <= self._start_idx <= self._end_idx < self._n_samples:
            center = (self._start_idx + self._end_idx) // 2
        else:
            center = self._current_start + self._window_size // 2
        self._window_size = max(int(0.1 * self._sample_rate),
                                self._window_size // 2)
        self._step_size = max(1, self._window_size // 3)
        self._current_start = max(
            0, min(self._n_samples - self._window_size,
                   center - self._window_size // 2))
        self._plot_window()

    def _on_zoom_out(self):
        if 0 <= self._start_idx <= self._end_idx < self._n_samples:
            center = (self._start_idx + self._end_idx) // 2
        else:
            center = self._current_start + self._window_size // 2
        self._window_size = min(self._n_samples, self._window_size * 2)
        self._step_size = max(1, self._window_size // 3)
        self._current_start = max(
            0, min(self._n_samples - self._window_size,
                   center - self._window_size // 2))
        self._plot_window()

    def _on_confirm(self):
        if self._start_idx is None or self._end_idx is None:
            return
        if self._start_idx > self._end_idx:
            self._start_idx, self._end_idx = self._end_idx, self._start_idx
        self.confirmed = True
        self._autosave()
        self.accept()

    def _autosave_path(self):
        import os
        from utils.paths import saved_templates_dir
        return os.path.join(saved_templates_dir("PulseShape"),
                            "extractor_temp.json")

    def _autosave(self):
        import json
        try:
            settings = {
                "start_idx": int(self._start_idx),
                "end_idx": int(self._end_idx),
                "window_size": int(self._window_size),
                "current_start": int(self._current_start),
                "n_samples": int(self._n_samples),
                "x_unit": self._x_unit,
                "file_index": int(self._file_idx),
                "file_name": self._files[self._file_idx][0],
            }
            with open(self._autosave_path(), "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _load_state_from(self, path):
        import json, os
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                s = json.load(f)
        except Exception:
            return
        # Restore the active file first — the bounds below check against it.
        if len(self._files) > 1:
            idx = next((i for i, (name, _) in enumerate(self._files)
                        if name and name == s.get("file_name")), None)
            if idx is None:
                fi = s.get("file_index")
                if isinstance(fi, int) and 0 <= fi < len(self._files):
                    idx = fi
            if idx is not None and idx != self._file_idx:
                self._apply_file(idx)
                self._file_combo.blockSignals(True)
                self._file_combo.setCurrentIndex(idx)
                self._file_combo.blockSignals(False)
        # Saved indices are absolute sample positions; only restore if
        # they fit within the current data bounds.
        n = self._n_samples
        si = int(s.get("start_idx", -1))
        ei = int(s.get("end_idx", -1))
        if not (0 <= si < n and 0 <= ei < n and si <= ei):
            return
        ws = int(s.get("window_size", self._window_size))
        ws = max(int(0.1 * self._sample_rate), min(ws, n))
        cs = int(s.get("current_start", max(0, si - ws // 4)))
        cs = max(0, min(cs, n - ws))
        # Ensure both lines are visible inside the restored window.
        if si < cs or ei > cs + ws:
            cs = max(0, min(n - ws, si - max(1, ws // 8)))
        self._window_size = ws
        self._step_size = max(1, ws // 3)
        self._current_start = cs
        self._start_idx = si
        self._end_idx = ei
        xu = s.get("x_unit")
        if xu in ("index", "time"):
            self._x_unit = xu

    def get_result(self):
        if self.confirmed and self._start_idx is not None and self._end_idx is not None:
            s, e = self._start_idx, self._end_idx
            template = self._data[s:e + 1].copy()
            if self._n_channels == 1:
                template = template.ravel()
            return {
                "template": template,
                "start_idx": s,
                "end_idx": e,
                "file_index": self._file_idx,
                "file_name": self._files[self._file_idx][0],
                "accepted_pulses": [template],
                "accepted_indices": [(s, e)],
            }
        return {
            "template": None,
            "start_idx": None,
            "end_idx": None,
            "accepted_pulses": [],
            "accepted_indices": [],
        }


def _as_2d(arr):
    a = np.atleast_1d(np.asarray(arr))
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _entry_signal(entry):
    """Best numeric array from a Load File entry (dot-prefixes tolerated)."""
    e = {k.split(".", 1)[-1]: v for k, v in entry.items()}
    if "data" in e:
        return np.asarray(e["data"])
    best = None
    for k, v in e.items():
        if k in ("labels", "fileName"):
            continue
        arr = np.asarray(v)
        if arr.dtype.kind in "iuf" and arr.ndim >= 1 and arr.size:
            if best is None or arr.size > best.size:
                best = arr
    return best


def _normalize_files(data):
    """Flexible input → [(name, 2-D array)]. Accepts a bare array, a Load
    File struct ({files: [...]}), a single file entry, or a list of entries."""
    entries = None
    if isinstance(data, dict):
        entries = data.get("files", [data])
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
        entries = data
    if entries is None:
        return [("", _as_2d(data))]
    files = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        sig = _entry_signal(entry)
        if sig is None:
            continue
        files.append((str(entry.get("fileName") or f"file {i + 1}"),
                      _as_2d(sig)))
    if not files:
        raise ValueError("No numeric signal found in the Load File struct.")
    return files


def pulse_template_extractor(data, sample_rate, settings_file=None):
    """Interactive UI for selecting and extracting a pulse template region.

    Parameters
    ----------
    data : np.ndarray | dict | list
        1-D or 2-D array (time x channels), a Load File struct
        ({fileNames, files}), a single file entry, or a list of entries.
        Multiple files get a File selector in the dialog.
    sample_rate : float
        Sampling rate in Hz.
    settings_file : str | None
        Path to saved extractor settings to pre-load (from the loadFile port).
        ``None`` (empty port) starts fresh with default selection.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'template': extracted data segment (ndarray or None)
        - 'start_idx': start sample index
        - 'end_idx': end sample index
        - 'file_index' / 'file_name': which loaded file the template came from
        - 'accepted_pulses': list containing the single extracted segment
        - 'accepted_indices': list of accepted pulse index tuples
    """
    files = _normalize_files(data)
    dlg = _PulseTemplateExtractorDialog(files[0][1], sample_rate,
                                        settings_file=settings_file,
                                        files=files)
    dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Template Extractor without confirming.")

    return dlg.get_result()
