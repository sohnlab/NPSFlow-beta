"""Interactive data range selector.

Click-and-drag a region on the signal plot to select the range to keep.
The shaded band shows the kept range and is reshapable from either edge or
draggable as a whole; Start / End spinboxes in the right-side control panel
stay in sync for precise input.
"""

import numpy as np

import math

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QLabel,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from utils.decimate import minmax_decimate


MAX_PLOT_POINTS = 5000


class _DataRangeDialog(QDialog):
    """Dialog with a drag-to-select range band, synced numeric inputs."""

    def __init__(self, signal_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trim Data")
        self.resize(1200, 500)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._signal_data = signal_data
        self._is_multi_zone = signal_data.ndim == 2 and signal_data.shape[1] > 1
        self._N = signal_data.shape[0]
        self._start_default = max(1, int(np.ceil(0.25 * self._N)))
        self._end_default = min(self._N, int(np.floor(0.75 * self._N)))
        self._start = self._start_default
        self._end = self._end_default
        self._result_start = None
        self._result_end = None
        self.confirmed = False

        # Display scale for the Start/End inputs. Matches the x-axis tick
        # offset (e.g. 1e7 when N≈15.9M) so the user can type 0.2 → 0.2 × 1e7.
        if self._N >= 10000:
            k = int(math.floor(math.log10(self._N)))
            self._scale = 10 ** k
            self._scale_suffix = f"× 1e{k}"
        else:
            self._scale = 1
            self._scale_suffix = ""

        self._build_ui()
        self._plot_signal()
        self._init_span_selector()
        # Apply the initial selection so band + spinboxes agree.
        self._update_range(self._start_default, self._end_default,
                           from_span=False)

    # ------------------------------------------------------------- build UI
    def _build_ui(self):
        # Canvas on the left, grouped control column on the right
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Matplotlib canvas
        if self._is_multi_zone:
            fig_height = min(4 + self._signal_data.shape[1], 12)
            self.fig = Figure(figsize=(12, fig_height), tight_layout=True)
        else:
            self.fig = Figure(figsize=(10.5, 4.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        outer.addWidget(self.canvas, 1)

        side = QVBoxLayout()
        side.setSpacing(10)

        # --- Range group: Start / End inputs, Trim to View, Reset ---
        range_group = QGroupBox("Range")
        rg = QVBoxLayout(range_group)
        rg.setContentsMargins(8, 6, 8, 6)
        rg.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel("Start:"), 0, 0)
        self._start_spin = self._make_index_spin()
        self._start_spin.valueChanged.connect(self._on_spin_changed)
        grid.addWidget(self._start_spin, 0, 1)
        grid.addWidget(QLabel("End:"), 1, 0)
        self._end_spin = self._make_index_spin()
        self._end_spin.valueChanged.connect(self._on_spin_changed)
        grid.addWidget(self._end_spin, 1, 1)
        if self._scale_suffix:
            grid.addWidget(QLabel(self._scale_suffix), 0, 2)
            grid.addWidget(QLabel(self._scale_suffix), 1, 2)
        rg.addLayout(grid)

        self._btn_view = QPushButton("Trim to View")
        self._btn_view.setToolTip("Use the current x-axis view as the range")
        self._btn_view.clicked.connect(self._on_trim_to_view)
        rg.addWidget(self._btn_view)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip("Reset the selected range")
        self._btn_reset.clicked.connect(self._on_reset)
        rg.addWidget(self._btn_reset)
        side.addWidget(range_group)

        # --- View group: x-axis zoom and reset view ---
        view_group = QGroupBox("View")
        vg = QVBoxLayout(view_group)
        vg.setContentsMargins(8, 6, 8, 6)
        vg.setSpacing(6)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)
        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setToolTip("Zoom in (x-axis)")
        self._btn_zoom_in.clicked.connect(lambda: self._zoom_x(1 / 1.5))
        zoom_row.addWidget(self._btn_zoom_in)
        self._btn_zoom_out = QPushButton("−")
        self._btn_zoom_out.setToolTip("Zoom out (x-axis)")
        self._btn_zoom_out.clicked.connect(lambda: self._zoom_x(1.5))
        zoom_row.addWidget(self._btn_zoom_out)
        vg.addLayout(zoom_row)

        self._btn_home = QPushButton("Reset View")
        self._btn_home.clicked.connect(self._on_home)
        vg.addWidget(self._btn_home)
        side.addWidget(view_group)

        side.addStretch()

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        side.addWidget(self._btn_cancel)
        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.clicked.connect(self._on_confirm)
        side.addWidget(self._btn_confirm)

        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(200)
        outer.addWidget(side_w)

    # ----------------------------------------------------- scale helpers
    def _make_index_spin(self):
        """Return a spinbox covering [1, N], scaled to self._scale for display."""
        if self._scale == 1:
            sb = QSpinBox()
            sb.setRange(1, self._N)
        else:
            sb = QDoubleSpinBox()
            sb.setRange(1.0 / self._scale, self._N / self._scale)
            sb.setDecimals(4)
            sb.setSingleStep(0.01)
        sb.setFixedWidth(84)
        sb.wheelEvent = lambda ev: ev.ignore()
        return sb

    def _spin_value_to_index(self, v) -> int:
        return max(1, min(self._N, int(round(float(v) * self._scale))))

    def _index_to_spin_value(self, idx):
        return idx / self._scale if self._scale != 1 else int(idx)

    # ---------------------------------------------------------------- plot
    def _plot_signal(self):
        N = self._N
        index_vector = np.arange(1, N + 1)

        if self._is_multi_zone:
            n_zones = self._signal_data.shape[1]
            zm = self._signal_data - self._signal_data.mean(axis=0)
            min_spacing = 0
            for i in range(n_zones - 1):
                req = 2 * (np.max(zm[:, i + 1]) - np.min(zm[:, i]))
                min_spacing = max(min_spacing, req)
            if min_spacing <= 0:
                min_spacing = 0.1 * np.mean(
                    np.max(zm, axis=0) - np.min(zm, axis=0))
            for i in range(n_zones):
                offset = (n_zones - i) * min_spacing / 2
                dx, dy = minmax_decimate(index_vector, zm[:, i] + offset,
                                         MAX_PLOT_POINTS)
                self.ax.plot(dx, dy, label=f"Zone {i+1}")
            self.ax.set_ylabel("Resistance (offset)")
            self.ax.legend(loc="best")
        else:
            dx, dy = minmax_decimate(index_vector, self._signal_data[:, 0],
                                     MAX_PLOT_POINTS)
            self.ax.plot(dx, dy, color="#1f77b4", linewidth=0.5)
            self.ax.set_ylabel("Signal Amplitude")

        self.ax.set_title("Select the data range to keep")
        self.ax.set_xlabel("Sample Index")
        self.ax.set_xlim(1, N)
        # Snapshot the initial view so "Reset View" returns to exactly here.
        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()
        self.canvas.draw()

    def _init_span_selector(self):
        """Interactive SpanSelector — drag a region, then refine its edges."""
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

    # ----------------------------------------------------------- callbacks
    def _on_span_select(self, xmin, xmax):
        si = max(1, int(round(xmin)))
        ei = min(self._N, int(round(xmax)))
        if si >= ei:
            return
        self._update_range(si, ei, from_span=True)

    def _on_spin_changed(self, _value):
        si = self._spin_value_to_index(self._start_spin.value())
        ei = self._spin_value_to_index(self._end_spin.value())
        if si >= ei:
            return  # ignored; user will fix it
        self._update_range(si, ei, from_span=False)

    def _on_trim_to_view(self):
        x0, x1 = self.ax.get_xlim()
        si = max(1, int(round(x0)))
        ei = min(self._N, int(round(x1)))
        if si >= ei:
            return
        self._update_range(si, ei, from_span=False)

    def _on_reset(self):
        self._update_range(self._start_default, self._end_default,
                           from_span=False)

    def _on_home(self):
        self.ax.set_xlim(*self._home_xlim)
        self.ax.set_ylim(*self._home_ylim)
        self.canvas.draw_idle()

    def _zoom_x(self, factor):
        """Scale the x-axis around the selection center by *factor*.

        Zoom-in is capped at the selection range with 10% padding on each side;
        zoom-out is capped at the full data range.
        """
        sel_w = max(1.0, float(self._end - self._start))
        sel_center = 0.5 * (self._start + self._end)
        min_half = 0.5 * sel_w * 1.2  # 10% padding on each side

        x0, x1 = self.ax.get_xlim()
        cur_half = 0.5 * (x1 - x0)
        new_half = max(cur_half * factor, min_half)

        new_x0 = sel_center - new_half
        new_x1 = sel_center + new_half
        # Slide back into [1, N] if the window overruns either edge.
        if new_x0 < 1:
            new_x1 += (1 - new_x0)
            new_x0 = 1
        if new_x1 > self._N:
            new_x0 -= (new_x1 - self._N)
            new_x1 = self._N
        new_x0 = max(1, new_x0)
        new_x1 = min(self._N, new_x1)
        if new_x1 <= new_x0:
            return
        self.ax.set_xlim(new_x0, new_x1)
        self.canvas.draw_idle()

    def _on_confirm(self):
        if self._start >= self._end:
            return
        self._result_start = self._start
        self._result_end = self._end
        self.confirmed = True
        self.accept()

    # --------------------------------------------------------------- sync
    def _update_range(self, si, ei, *, from_span):
        self._start, self._end = si, ei
        if not from_span:
            try:
                self._span.extents = (si, ei)
            except AttributeError:
                pass  # SpanSelector not yet built
        self._start_spin.blockSignals(True)
        self._end_spin.blockSignals(True)
        self._start_spin.setValue(self._index_to_spin_value(si))
        self._end_spin.setValue(self._index_to_spin_value(ei))
        self._start_spin.blockSignals(False)
        self._end_spin.blockSignals(False)
        self.canvas.draw_idle()

    # --------------------------------------------------------------- result
    def get_result(self, signal_data):
        """Return the selected slice of signal_data."""
        if self._result_start is None:
            return None
        si = self._result_start - 1  # 1-based → 0-based
        ei = self._result_end         # exclusive end for slicing
        if signal_data.shape[1] > 1:
            return signal_data[si:ei, :]
        return signal_data[si:ei, 0]
