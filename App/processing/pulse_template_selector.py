"""Interactive pulse template start/end selection.

Translates MATLAB PulseTemplateSelector.m: interactive UI with draggable
start and end vertical lines for selecting the pulse template boundaries
within a filtered signal.
Uses a QDialog with embedded FigureCanvas and side panel.
"""

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGroupBox, QWidget,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure


class _TemplateSelectorDialog(QDialog):
    """Interactive template boundary selection dialog."""

    def __init__(self, data, sample_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pulse Template Selection")
        self.resize(1400, 800)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._data = data.copy()
        if self._data.ndim == 1:
            self._data = self._data.reshape(-1, 1)
        self._n_samples, self._n_channels = self._data.shape
        self._sample_rate = sample_rate

        window_size = int(round(2 * sample_rate))
        if window_size > self._n_samples:
            window_size = self._n_samples
        self._window_size = window_size
        self._step_size = window_size // 3
        self._current_start = 0

        self._start_idx = int(0.25 * window_size)
        self._end_idx = int(0.75 * window_size)
        self.confirmed = False

        self._dragging = False
        self._active_line = None

        self._build_ui()
        self._plot()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left: canvas
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(13, 7), tight_layout=False)
        style_mpl_figure(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(left_widget, stretch=4)

        # Mouse interaction
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        # Right: side panel
        panel = QWidget()
        panel.setFixedWidth(200)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)

        info_group = QGroupBox("Selection")
        info_layout = QVBoxLayout(info_group)
        self._lbl_start = QLabel(f"Start: {self._start_idx}")
        self._lbl_end = QLabel(f"End: {self._end_idx}")
        self._lbl_length = QLabel(f"Length: {self._end_idx - self._start_idx}")
        info_layout.addWidget(self._lbl_start)
        info_layout.addWidget(self._lbl_end)
        info_layout.addWidget(self._lbl_length)
        info_layout.addWidget(QLabel("(Drag green/red lines)"))
        panel_layout.addWidget(info_group)

        panel_layout.addSpacing(10)

        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)

        self._btn_home = QPushButton("Home")
        self._btn_home.clicked.connect(self._on_home)
        nav_layout.addWidget(self._btn_home)

        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("← Prev")
        self._btn_prev.clicked.connect(self._on_prev)
        nav_row.addWidget(self._btn_prev)
        self._btn_next_nav = QPushButton("Next →")
        self._btn_next_nav.clicked.connect(self._on_next)
        nav_row.addWidget(self._btn_next_nav)
        nav_layout.addLayout(nav_row)

        zoom_row = QHBoxLayout()
        self._btn_zoom_in = QPushButton("Zoom In")
        self._btn_zoom_in.clicked.connect(self._on_zoom_in)
        zoom_row.addWidget(self._btn_zoom_in)
        self._btn_zoom_out = QPushButton("Zoom Out")
        self._btn_zoom_out.clicked.connect(self._on_zoom_out)
        zoom_row.addWidget(self._btn_zoom_out)
        nav_layout.addLayout(zoom_row)

        panel_layout.addWidget(nav_group)

        from utils.export_figure_ui import make_export_button
        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        panel_layout.addWidget(self._btn_export)

        panel_layout.addSpacing(10)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        panel_layout.addWidget(self._btn_reset)

        panel_layout.addStretch()

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        bottom_layout.addWidget(self._btn_cancel)
        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.clicked.connect(self._on_confirm)
        bottom_layout.addWidget(self._btn_confirm)
        panel_layout.addLayout(bottom_layout)

        main_layout.addWidget(panel)

    def _plot(self):
        self.ax.clear()
        cs = self._current_start
        ws = self._window_size
        end_window = min(cs + ws, self._n_samples)
        idx_range = np.arange(cs, end_window)

        if self._n_channels == 1:
            self.ax.plot(idx_range, self._data[idx_range, 0], linewidth=1.2)
            self.ax.set_ylabel('Signal Amplitude')
        else:
            zm = self._data[idx_range] - np.mean(self._data[idx_range], axis=0)
            spacing = 0
            for i in range(self._n_channels - 1):
                req = 2 * (np.max(zm[:, i + 1]) - np.min(zm[:, i]))
                spacing = max(spacing, req)
            if spacing <= 0:
                spacing = 0.1 * np.mean(np.ptp(zm, axis=0))
            for i in range(self._n_channels):
                offset = (self._n_channels - i) * spacing / 2
                self.ax.plot(idx_range, zm[:, i] + offset, linewidth=1.0,
                             label=f'Zone {i + 1}')
            self.ax.legend(loc='upper right')
            self.ax.set_ylabel('Resistance (offset)')

        self.ax.set_xlabel('Sample Index')
        self.ax.set_title('Drag the vertical lines to set START and END, then press Confirm')
        self.ax.grid(True)
        self.ax.set_xlim(cs, end_window)

        self._start_idx = np.clip(self._start_idx, 0, self._n_samples - 1)
        self._end_idx = np.clip(self._end_idx, 0, self._n_samples - 1)

        self._vline_start = self.ax.axvline(self._start_idx, color='green',
                                             linestyle='--', linewidth=2)
        self._vline_end = self.ax.axvline(self._end_idx, color='red',
                                           linestyle='--', linewidth=2)

        self._update_labels()
        self.canvas.draw_idle()

    def _update_labels(self):
        self._lbl_start.setText(f"Start: {self._start_idx}")
        self._lbl_end.setText(f"End: {self._end_idx}")
        self._lbl_length.setText(f"Length: {self._end_idx - self._start_idx}")

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x = event.xdata
        tol = 0.02 * self._window_size
        d_s = abs(x - self._start_idx)
        d_e = abs(x - self._end_idx)
        if d_s > tol and d_e > tol:
            return
        self._dragging = True
        self._active_line = 'start' if d_s <= d_e else 'end'

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        x = int(round(np.clip(event.xdata, 0, self._n_samples - 1)))
        if self._active_line == 'start':
            self._start_idx = x
        else:
            self._end_idx = x
        if self._start_idx > self._end_idx:
            self._start_idx, self._end_idx = self._end_idx, self._start_idx
            self._active_line = 'end' if self._active_line == 'start' else 'start'
        self._vline_start.set_xdata([self._start_idx])
        self._vline_end.set_xdata([self._end_idx])
        self._update_labels()
        self.canvas.draw_idle()

    def _on_release(self, event):
        self._dragging = False
        self._active_line = None

    def _on_home(self):
        self._current_start = 0
        self.ax.set_xlim(0, self._n_samples)
        self.ax.autoscale(axis='y')
        self.canvas.draw_idle()

    def _on_prev(self):
        self._current_start = max(0, self._current_start - self._step_size)
        self._plot()

    def _on_next(self):
        new = self._current_start + self._step_size
        if new + self._window_size <= self._n_samples:
            self._current_start = new
        self._plot()

    def _on_zoom_in(self):
        mid = (self._start_idx + self._end_idx) // 2
        self._window_size = max(int(0.1 * self._sample_rate),
                                 self._window_size // 2)
        self._current_start = max(0, mid - self._window_size // 2)
        self._plot()

    def _on_zoom_out(self):
        mid = (self._start_idx + self._end_idx) // 2
        self._window_size = min(self._n_samples, self._window_size * 2)
        self._current_start = max(0, mid - self._window_size // 2)
        self._plot()

    def _on_reset(self):
        ws = self._window_size
        self._start_idx = self._current_start + int(0.25 * ws)
        self._end_idx = self._current_start + int(0.75 * ws)
        self._plot()

    def _on_confirm(self):
        if self._start_idx > self._end_idx:
            self._start_idx, self._end_idx = self._end_idx, self._start_idx
        self.confirmed = True
        self.accept()


def pulse_template_selector(detrended_template):
    """Interactive UI for selecting pulse template start and end points.

    Parameters
    ----------
    detrended_template : np.ndarray or dict
        If ndarray: 1-D or 2-D (time x channels) detrended signal.
        If dict: expects key 'detrended_template' containing the array
        and optionally 'sample_rate'.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'start_idx': selected start index
        - 'end_idx': selected end index
        - 'selected_data': extracted segment
        - 'selected_pulses': list of selected pulse arrays
    """
    sample_rate = 1.0
    if isinstance(detrended_template, dict):
        data = np.atleast_1d(detrended_template.get('detrended_template',
                                                     detrended_template.get('template', np.array([]))))
        sample_rate = detrended_template.get('sample_rate', 1.0)
    else:
        data = np.atleast_1d(detrended_template)

    data = data.copy().astype(float)

    dlg = _TemplateSelectorDialog(data, sample_rate)
    dlg.exec()

    if dlg.confirmed:
        s, e = dlg._start_idx, dlg._end_idx
        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        selected = data_2d[s:e + 1]
        if data_2d.shape[1] == 1:
            selected = selected.ravel()
        return {
            'start_idx': s,
            'end_idx': e,
            'selected_data': selected,
            'selected_pulses': [selected],
        }
    else:
        return {
            'start_idx': None,
            'end_idx': None,
            'selected_data': None,
            'selected_pulses': [],
        }
