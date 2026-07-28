"""Interactive segment selection UI with template visualization and checkboxes.

Shows the pulse template with labeled segments/regions overlaid. The user
selects which segments to include via checkboxes. Returns selected segments
and their corresponding slice definitions. Uses a QDialog with embedded
FigureCanvas.
"""

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox,
    QGroupBox, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class _RegionSelectionDialog(QDialog):
    """Dialog for selecting pulse template regions via checkboxes."""

    def __init__(self, pulse_features, processed_template=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Region Selection")
        self.resize(1200, 650)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self.confirmed = False

        pf = pulse_features
        if isinstance(pf, list):
            pf = pf[0] if len(pf) > 0 else {}

        self._bounds = np.asarray(pf.get("bounds", []))
        if self._bounds.size == 0:
            raise ValueError("No segment bounds defined in pulse_features.")
        if self._bounds.ndim == 1:
            self._bounds = self._bounds.reshape(1, -1)

        self._num_segments = self._bounds.shape[0]
        self._labels = pf.get("labels",
                               [f"Segment {i+1}" for i in range(self._num_segments)])

        # Template signal
        template_signal = pf.get("signal", None)
        if processed_template and isinstance(processed_template, dict):
            if "pulseTemplate_rec" in processed_template:
                template_signal = processed_template["pulseTemplate_rec"]

        if template_signal is not None:
            self._template = np.asarray(template_signal, dtype=float).ravel()
        else:
            max_bound = int(np.max(self._bounds))
            self._template = np.zeros(max_bound)

        self._N = len(self._template)
        self._x = np.arange(self._N)

        # Colors
        cmap = plt.get_cmap("tab10", max(self._num_segments, 3))
        self._colors = [cmap(i) for i in range(self._num_segments)]

        # Selection state
        self._selected = [True] * self._num_segments

        self._build_ui()
        self._draw()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Left: plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(plot_widget, stretch=3)

        # Right: checkboxes + buttons
        right_widget = QWidget()
        right_widget.setFixedWidth(200)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Checkbox group
        check_group = QGroupBox("Regions")
        check_layout = QVBoxLayout(check_group)

        self._checkboxes = []
        for i in range(self._num_segments):
            cb = QCheckBox(str(self._labels[i]))
            cb.setChecked(True)
            # Store color as hex for stylesheet
            r, g, b = int(self._colors[i][0]*255), int(self._colors[i][1]*255), int(self._colors[i][2]*255)
            cb.setStyleSheet(f"color: rgb({r},{g},{b}); font-size: 10px;")
            cb.stateChanged.connect(self._on_check_changed)
            check_layout.addWidget(cb)
            self._checkboxes.append(cb)

        right_layout.addWidget(check_group)

        # All / None buttons
        sel_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(self._on_all)
        sel_row.addWidget(btn_all)

        btn_none = QPushButton("None")
        btn_none.clicked.connect(self._on_none)
        sel_row.addWidget(btn_none)
        right_layout.addLayout(sel_row)

        right_layout.addStretch()

        # Confirm button
        btn_confirm = QPushButton("Confirm")
        btn_confirm.setObjectName("confirmBtn")
        btn_confirm.clicked.connect(self._on_confirm)
        right_layout.addWidget(btn_confirm)

        layout.addWidget(right_widget, stretch=0)

    def _draw(self):
        self.ax.clear()
        self.ax.plot(self._x, self._template, "k-", linewidth=1.5, label="Template")

        y_min = np.min(self._template)
        y_max = np.max(self._template)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1
        y_pad = 0.1 * y_range

        for k in range(self._num_segments):
            s = max(0, int(self._bounds[k, 0]) - 1)
            e = min(self._N, int(self._bounds[k, 1]))
            alpha = 0.3 if self._selected[k] else 0.08
            self.ax.axvspan(s, e, alpha=alpha, color=self._colors[k],
                           label=str(self._labels[k]))
            mid = (s + e) / 2
            self.ax.text(mid, y_max + 0.05 * y_range, str(self._labels[k]),
                        ha="center", va="bottom", fontsize=8,
                        color=self._colors[k], fontweight="bold")

        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Select Regions (check to include)")
        self.ax.set_xlim(0, self._N)
        self.ax.set_ylim(y_min - y_pad, y_max + y_pad + 0.15 * y_range)
        self.ax.grid(True)
        self.canvas.draw_idle()

    def _on_check_changed(self):
        for i, cb in enumerate(self._checkboxes):
            self._selected[i] = cb.isChecked()
        self._draw()

    def _on_all(self):
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _on_none(self):
        for cb in self._checkboxes:
            cb.setChecked(False)

    def _on_confirm(self):
        self.confirmed = True
        self.accept()

    def get_results(self):
        selected_segments = [i for i in range(self._num_segments) if self._selected[i]]
        segment_slices = []
        for i in selected_segments:
            segment_slices.append({
                "segmentId": i + 1,
                "label": str(self._labels[i]),
                "bounds": (int(self._bounds[i, 0]), int(self._bounds[i, 1])),
            })
        return selected_segments, segment_slices


def region_selection_ui(pulse_features, processed_template=None):
    """Show interactive region selection and return user choices.

    Parameters
    ----------
    pulse_features : dict
        Must contain 'bounds' (Nx2 array), 'labels' (list of str), and
        optionally 'signal' (template waveform) and 'peakLocations'.
    processed_template : dict, optional
        Additional template processing data (e.g., rectangularized pulse).

    Returns
    -------
    selected_segments : list[int]
        0-based indices of the selected segments.
    segment_slices : list[dict]
        Slice definitions for each selected segment, each containing
        'segmentId', 'label', 'bounds'.
    """
    dlg = _RegionSelectionDialog(pulse_features, processed_template)
    dlg.exec()

    return dlg.get_results()
