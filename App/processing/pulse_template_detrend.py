"""Interactive pulse template detrending via two-point baseline selection.

Translates MATLAB PulseTemplateDetrend.m: interactive UI with draggable
baseline points, detrend preview, and per-channel processing.
Uses a QDialog with embedded FigureCanvas.
"""

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure


class _DetrendChannelDialog(QDialog):
    """Interactive dialog for detrending a single channel."""

    def __init__(self, channel_data, ch_idx, total_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Pulse Baseline Selection - Zone {ch_idx}")
        self.resize(975, 562)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._n = len(channel_data)
        self._idx = np.arange(self._n)
        self._zero_mean = channel_data - np.mean(channel_data)
        self._ch_idx = ch_idx

        self._point1 = [min(10, self._n - 1),
                        float(self._zero_mean[min(10, self._n - 1)])]
        self._point2 = [max(self._n - 10, 0),
                        float(self._zero_mean[max(self._n - 10, 0)])]
        self._step = 1  # 1 = baseline selection, 2 = detrended view
        self._detrended = None
        self._slope = None
        self._intercept = None
        self._dragging = False
        self._active_point = None
        self.confirmed = False

        # Persistent artist handles for fast drag updates
        self._signal_line = None
        self._point1_marker = None
        self._point2_marker = None
        self._baseline_line = None
        self._bg_cache = None

        self._build_ui()
        self._setup_dragging()
        self._plot()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.fig = Figure(figsize=(9.75, 5.25), tight_layout=True)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        layout.addWidget(self.canvas)

        # Button row: Restart | Export Fig | ... | Detrend | Confirm
        btn_layout = QHBoxLayout()

        self._btn_restart = QPushButton("Restart")
        self._btn_restart.clicked.connect(self._on_restart)
        btn_layout.addWidget(self._btn_restart)

        from utils.export_figure_ui import make_export_button
        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        btn_layout.addWidget(self._btn_export)

        btn_layout.addStretch()

        self._btn_detrend = QPushButton("Detrend")
        self._btn_detrend.setObjectName("confirmBtn")
        self._btn_detrend.clicked.connect(self._on_detrend)
        btn_layout.addWidget(self._btn_detrend)

        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self._btn_confirm)

        layout.addLayout(btn_layout)

    def _plot(self):
        """Full redraw — used for initial render and step changes."""
        self.ax.clear()
        n = self._n
        idx = self._idx
        zero_mean = self._zero_mean

        # Reset artist handles
        self._signal_line = None
        self._point1_marker = None
        self._point2_marker = None
        self._baseline_line = None
        self._bg_cache = None

        if self._step == 1:
            self._signal_line, = self.ax.plot(idx, zero_mean, linewidth=1.2)
            y_range = np.ptp(zero_mean)
            if y_range == 0:
                y_range = 1e-6
            margin = 0.1 * y_range
            self.ax.set_ylim(np.min(zero_mean) - margin,
                             np.max(zero_mean) + margin)
            self.ax.set_ylabel("Signal Amplitude")
            self.ax.set_xlabel("Sample Index")
            self.ax.set_title(
                f"Zone {self._ch_idx}: Drag the red dots to set baseline",
                fontsize=10,
            )
            self.ax.grid(True)
            self.ax.set_xlim(0, n)

            p1 = self._point1
            p2 = self._point2
            self._point1_marker, = self.ax.plot(
                p1[0], p1[1], 'o', color='red', markersize=10, animated=True)
            self._point2_marker, = self.ax.plot(
                p2[0], p2[1], 'o', color='red', markersize=10, animated=True)

            # Baseline line
            bl_vals = self._calc_baseline()
            self._baseline_line, = self.ax.plot(
                idx, bl_vals, 'r--', linewidth=2, animated=True)
        else:
            # Detrended view
            self.ax.plot(idx, zero_mean, linewidth=1.0, color="grey",
                         alpha=0.7)
            self.ax.plot(idx, self._detrended, linewidth=1.2)
            y_min = min(np.min(zero_mean), np.min(self._detrended))
            y_max = max(np.max(zero_mean), np.max(self._detrended))
            y_range = y_max - y_min
            if y_range <= 0:
                y_range = max(1e-6, abs(y_max))
            margin = 0.1 * y_range
            self.ax.set_ylim(y_min - margin, y_max + margin)
            self.ax.set_ylabel("Signal Amplitude")
            self.ax.set_xlabel("Sample Index")
            self.ax.set_title(
                f"Zone {self._ch_idx}: Detrended Template", fontsize=10,
            )
            self.ax.grid(True)
            self.ax.set_xlim(0, n)

        self.canvas.draw()
        # _grab_bg is called automatically via the 'draw_event' handler

    def _grab_bg(self):
        """Capture background and draw animated artists on top.

        Called on every 'draw_event' (initial paint, resize, expose, etc.)
        so the blit cache stays valid.
        """
        if self._step != 1 or self._point1_marker is None:
            return
        if getattr(self, '_in_grab_bg', False):
            return
        self._in_grab_bg = True
        try:
            self._bg_cache = self.canvas.copy_from_bbox(self.ax.bbox)
            self.ax.draw_artist(self._point1_marker)
            self.ax.draw_artist(self._point2_marker)
            self.ax.draw_artist(self._baseline_line)
            self.canvas.blit(self.ax.bbox)
        finally:
            self._in_grab_bg = False

    def _calc_baseline(self):
        """Compute baseline y-values from the two control points."""
        p1 = self._point1
        p2 = self._point2
        if p2[0] != p1[0]:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        else:
            slope = 0
        intercept = p1[1] - slope * p1[0]
        return slope * self._idx + intercept

    def _update_artists(self):
        """Fast blit update — only redraws the draggable artists."""
        if self._bg_cache is None:
            # Fallback: full redraw if no cache
            self._plot()
            return

        self.canvas.restore_region(self._bg_cache)

        # Update marker positions
        self._point1_marker.set_data([self._point1[0]], [self._point1[1]])
        self._point2_marker.set_data([self._point2[0]], [self._point2[1]])

        # Update baseline
        self._baseline_line.set_ydata(self._calc_baseline())

        self.ax.draw_artist(self._point1_marker)
        self.ax.draw_artist(self._point2_marker)
        self.ax.draw_artist(self._baseline_line)
        self.canvas.blit(self.ax.bbox)

    def _setup_dragging(self):
        self.canvas.mpl_connect("draw_event", lambda evt: self._grab_bg())
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        if self._step != 1 or event.inaxes != self.ax:
            return
        x_click, y_click = event.xdata, event.ydata
        if x_click is None:
            return
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        x_range = xlims[1] - xlims[0]
        y_range = ylims[1] - ylims[0]
        dist_threshold = 0.03 * max(x_range, y_range)

        d1 = np.sqrt((x_click - self._point1[0]) ** 2 +
                      ((y_click - self._point1[1]) / y_range * x_range) ** 2)
        d2 = np.sqrt((x_click - self._point2[0]) ** 2 +
                      ((y_click - self._point2[1]) / y_range * x_range) ** 2)

        if d1 < dist_threshold:
            self._dragging = True
            self._active_point = 'point1'
        elif d2 < dist_threshold:
            self._dragging = True
            self._active_point = 'point2'

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax:
            return
        x_click, y_click = event.xdata, event.ydata
        if x_click is None:
            return
        x_click = int(round(np.clip(x_click, 0, self._n - 1)))
        if self._active_point == 'point1':
            self._point1 = [x_click, y_click]
        else:
            self._point2 = [x_click, y_click]
        self._update_artists()

    def _on_release(self, event):
        self._dragging = False
        self._active_point = None

    def _on_detrend(self):
        p1 = self._point1
        p2 = self._point2
        if p2[0] != p1[0]:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        else:
            slope = 0
        intercept = p1[1] - slope * p1[0]
        bl = slope * self._idx + intercept
        detrended = self._zero_mean - bl
        # Align first point
        offset = self._zero_mean[0] - detrended[0]
        detrended = detrended + offset
        self._detrended = detrended
        self._slope = slope
        self._intercept = intercept
        self._step = 2
        # Swap highlight: unhighlight Detrend, highlight Confirm
        self._btn_detrend.setObjectName("")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.setEnabled(True)
        self.setStyleSheet(self.styleSheet())  # force style refresh
        self._plot()

    def _on_restart(self):
        self._step = 1
        self._detrended = None
        self._point1 = [min(10, self._n - 1),
                        float(self._zero_mean[min(10, self._n - 1)])]
        self._point2 = [max(self._n - 10, 0),
                        float(self._zero_mean[max(self._n - 10, 0)])]
        # Swap highlight back: highlight Detrend, unhighlight Confirm
        self._btn_detrend.setObjectName("confirmBtn")
        self._btn_confirm.setObjectName("")
        self._btn_confirm.setEnabled(False)
        self.setStyleSheet(self.styleSheet())  # force style refresh
        self._plot()

    def _on_confirm(self):
        if self._detrended is None:
            return
        self.confirmed = True
        self.accept()

    def get_result(self):
        if self.confirmed and self._detrended is not None:
            return {
                'detrended': self._detrended,
                'slope': self._slope,
                'intercept': self._intercept,
            }
        return None


def pulse_template_detrend(processed_template):
    """Interactive detrending of pulse template data.

    The user drags two points to define a linear baseline, then presses
    Detrend to subtract it.  A before/after overlay is shown for
    confirmation.

    Parameters
    ----------
    processed_template : np.ndarray or dict
        If ndarray: 1-D or 2-D (time x channels) pulse template data.
        If dict: expects key 'pulse_template_rec' containing the array.

    Returns
    -------
    detrended_template : np.ndarray
        Detrended data (same shape as input).
        Returns empty array if user cancels.
    """
    if isinstance(processed_template, dict):
        data = np.atleast_1d(processed_template.get('pulse_template_rec',
                                                     processed_template.get('template', np.array([]))))
    else:
        data = np.atleast_1d(processed_template)

    data = data.copy().astype(float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    detrended_all = np.empty_like(data)

    for ch in range(n_channels):
        channel_data = data[:, ch]
        dlg = _DetrendChannelDialog(channel_data, ch + 1, n_channels)
        dlg.exec()
        result_ch = dlg.get_result()
        if result_ch is None:
            raise ValueError("User closed Template Detrend without confirming.")
        detrended_all[:, ch] = result_ch['detrended']

    if n_channels == 1:
        detrended_all = detrended_all.ravel()

    return detrended_all
