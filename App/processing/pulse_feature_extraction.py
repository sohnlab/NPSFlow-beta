"""Interactive segment-based pulse feature extraction.

Translates MATLAB PulseFeatureExtraction.m: interactive UI with draggable
segment boundaries on the pulse template, editable segment labels,
split/combine operations, and segment metadata generation.
Uses a QDialog with embedded FigureCanvas and side panel.
"""

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGroupBox, QWidget, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
import sys as _sys

_SANS = "Helvetica Neue" if _sys.platform == "darwin" else "Segoe UI"


class _SegmentDialog(QDialog):
    """Interactive segment definition dialog matching MATLAB version.

    Features:
    - Editable table with Segment #, Label, Start, End columns
    - Draggable boundary lines on the plot
    - Multi-select rows for Combine
    - Split / Combine / Reset operations
    """

    def __init__(self, signal_rect=None, peak_locations=None,
                 template_original=None, methods=None, parent=None,
                 zones=None, active_zone=0):
        super().__init__(parent)
        self.setWindowTitle("Template Slicing")
        self.resize(1400, 550)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        # Build per-zone inputs. Old single-template constructor maps to one
        # zone; the multi-zone path passes a list of input dicts via `zones`.
        if zones is None:
            zones = [{
                'signal_rect': signal_rect,
                'peak_locations': peak_locations,
                'template_original': template_original,
                'methods': methods,
            }]
        self._zones = [self._make_zone_state(z) for z in zones]
        self._active_zone = int(active_zone) if 0 <= active_zone < len(self._zones) else 0

        self._selected_segments = set()
        self._max_segments = 20
        self.confirmed = False

        # Drag state
        self._dragging_idx = None
        self._drag_line = None

        # Point the plain attrs the rest of the dialog uses at the active zone.
        self._bind_active_zone(self._active_zone)

        self._build_ui()
        self._plot()
        self._update_table()

    # ------------------------------------------------------------------
    # Per-zone state
    # ------------------------------------------------------------------

    def _make_zone_state(self, z):
        """Build an editable per-zone state dict from an input dict."""
        signal_rect = np.atleast_1d(z.get('signal_rect')).ravel().astype(float)
        peak_locations = z.get('peak_locations')
        if peak_locations is None:
            peak_locations = np.array([], dtype=int)
        peak_locations = np.atleast_1d(peak_locations).ravel().astype(int)
        template_original = z.get('template_original')
        methods = z.get('methods')
        n = len(signal_rect)

        boundaries = sorted(set([0] + list(int(p) for p in peak_locations)
                                + [n - 1]))
        n_seg = len(boundaries) - 1
        labels = [f'{i + 1}' for i in range(n_seg)]
        if methods and len(methods) >= n_seg:
            zmethods = list(methods[:n_seg])
        else:
            zmethods = ['mean'] * n_seg

        return {
            'signal': signal_rect.copy(),
            'n': n,
            'peak_locations': set(int(p) for p in peak_locations),
            'peak_locations_arr': peak_locations.copy(),
            'template_original': template_original,
            'show_original': (template_original is not None
                              and len(template_original) == n),
            'boundaries': boundaries,
            'initial_boundaries': list(boundaries),
            'labels': labels,
            'methods': zmethods,
            'initial_methods': list(zmethods),
            'accepted': False,  # set when the zone's OK button is pressed
        }

    def _bind_active_zone(self, idx):
        """Repoint the plain attrs the rest of the dialog uses at zone *idx*."""
        z = self._zones[idx]
        self._signal = z['signal']
        self._n = z['n']
        self._peak_locations = z['peak_locations']
        self._peak_locations_arr = z['peak_locations_arr']
        self._template_original = z['template_original']
        self._show_original = z['show_original']
        self._boundaries = z['boundaries']
        self._initial_boundaries = z['initial_boundaries']
        self._labels = z['labels']
        self._methods = z['methods']
        self._initial_methods = z['initial_methods']

    def _save_active_zone(self):
        """Copy the editable plain attrs back into the active zone slot."""
        z = self._zones[self._active_zone]
        z['boundaries'] = list(self._boundaries)
        z['labels'] = list(self._labels)
        z['methods'] = list(self._methods)

    def _load_active_zone(self, idx):
        """Make zone *idx* active and refresh the view."""
        self._active_zone = idx
        self._bind_active_zone(idx)
        # Reset transient drag/selection state.
        self._dragging_idx = None
        self._drag_line = None
        self._selected_segments = set()
        self._plot()
        self._update_table()
        self._refresh_finish_button()

    def _on_zone_changed(self, idx):
        if idx == self._active_zone:
            return
        self._save_active_zone()
        self._load_active_zone(idx)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --- Left: canvas (65% width) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(13, 5.5), tight_layout=False)
        style_mpl_figure(self.fig)
        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.10)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(left_widget, stretch=65)

        # Connect mouse events for dragging boundaries
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        # --- Right: Feature Definition panel (capped width) ---
        panel = QWidget()
        panel.setMaximumWidth(400)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(6)

        # Title
        title = QLabel("Feature Definition")
        title.setFont(QFont(_SANS, 13, QFont.Weight.Bold))
        panel_layout.addWidget(title)

        # Zone selector (only shown for multi-zone templates)
        self._zone_btn_group = None
        if len(self._zones) > 1:
            from PySide6.QtWidgets import QButtonGroup, QRadioButton
            zone_row = QHBoxLayout()
            zone_row.setSpacing(8)
            zone_row.addWidget(QLabel("Zone:"))
            self._zone_btn_group = QButtonGroup(self)
            for i in range(len(self._zones)):
                rb = QRadioButton(f"Zone {i + 1}")
                if i == self._active_zone:
                    rb.setChecked(True)
                self._zone_btn_group.addButton(rb, i)
                zone_row.addWidget(rb)
            zone_row.addStretch(1)
            self._zone_btn_group.idClicked.connect(self._on_zone_changed)
            panel_layout.addLayout(zone_row)

        # Segment table: #, Label, Method, Start, End
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["#", "Segment Label", "Method", "Start", "End"])
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 28)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 55)
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 50)
        self._table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(4, 50)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QTableWidget::item:selected { background: #d0e0f0; color: black; }")
        self._table.cellChanged.connect(self._on_table_edit)
        panel_layout.addWidget(self._table, stretch=1)

        # Instruction text
        hint = QLabel(
            "Select segment(s) in table. Split/Combine\n"
            "boundaries; drag lines to adjust;\n"
            "double-click label to rename.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10px;")
        panel_layout.addWidget(hint)

        # Operations row
        ops_layout = QHBoxLayout()
        ops_layout.setSpacing(6)
        self._btn_split = QPushButton("Split")
        self._btn_split.clicked.connect(self._on_split)
        ops_layout.addWidget(self._btn_split)
        self._btn_combine = QPushButton("Combine")
        self._btn_combine.clicked.connect(self._on_combine)
        ops_layout.addWidget(self._btn_combine)
        panel_layout.addLayout(ops_layout)

        # Bottom buttons: Export, Reset, then the confirm button (OK/Finish).
        from utils.export_figure_ui import make_export_button
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(6)
        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        bottom_layout.addWidget(self._btn_export)
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        bottom_layout.addWidget(self._btn_reset)
        # Confirm button: "OK" on non-last zones (accept + advance), "Finish"
        # on the last zone (enabled only once every other zone is accepted).
        self._btn_finish = QPushButton("Finish")
        self._btn_finish.setObjectName("confirmBtn")
        self._btn_finish.clicked.connect(self._on_finish)
        bottom_layout.addWidget(self._btn_finish)
        panel_layout.addLayout(bottom_layout)
        self._refresh_finish_button()

        main_layout.addWidget(panel, stretch=35)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _plot(self):
        self.ax.clear()
        n = self._n

        # Signals
        if self._show_original:
            self.ax.plot(np.arange(n), self._template_original, '--',
                         color=[0.7, 0.7, 0.7], linewidth=1.0,
                         label='Original')
        self.ax.plot(np.arange(n), self._signal, '-',
                     color=[0, 0.4470, 0.7410], linewidth=1.4,
                     label='Rectangularized')

        self.ax.set_xlim(-0.5, n + 0.5)
        y_min = float(np.min(self._signal))
        y_max = float(np.max(self._signal))
        if self._show_original:
            y_min = min(y_min, float(np.min(self._template_original)))
            y_max = max(y_max, float(np.max(self._template_original)))
        y_range = y_max - y_min if y_max != y_min else 1.0
        self.ax.set_ylim(y_min - 0.10 * y_range, y_max + 0.25 * y_range)
        self.ax.set_xlabel('Sample Index')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)

        bounds = self._boundaries
        labs = self._labels
        ylims = self.ax.get_ylim()
        y_range_plot = ylims[1] - ylims[0]

        # Draw internal boundary lines
        self._boundary_lines = []
        for i in range(1, len(bounds) - 1):
            is_peak = int(bounds[i]) in self._peak_locations
            color = [0.85, 0.33, 0.10] if is_peak else [0.95, 0.60, 0.30]
            line = self.ax.axvline(bounds[i], color=color, linewidth=1.5)
            self._boundary_lines.append((i, line))

        # Draw segment labels
        for i in range(len(bounds) - 1):
            x_center = (bounds[i] + bounds[i + 1]) / 2
            y_pos = ylims[1] - (0.04 if i % 2 == 0 else 0.11) * y_range_plot
            self.ax.text(x_center, y_pos, labs[i], fontsize=8,
                         ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7))

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _update_table(self):
        """Rebuild the table from current boundaries and labels."""
        self._table.blockSignals(True)
        bounds = self._boundaries
        n_seg = len(bounds) - 1

        # Ensure methods list matches segment count
        while len(self._methods) < n_seg:
            self._methods.append('mean')
        self._methods = self._methods[:n_seg]

        self._table.setRowCount(n_seg)
        for i in range(n_seg):
            # Column 0: segment number (read-only)
            num_item = QTableWidgetItem(str(i + 1))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 0, num_item)

            # Column 1: label (editable)
            label_item = QTableWidgetItem(self._labels[i])
            self._table.setItem(i, 1, label_item)

            # Column 2: method (read-only, from input)
            method_item = QTableWidgetItem(self._methods[i])
            method_item.setFlags(method_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            method_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 2, method_item)

            # Column 3: start (read-only)
            start_item = QTableWidgetItem(str(int(bounds[i])))
            start_item.setFlags(start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            start_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 3, start_item)

            # Column 4: end (read-only)
            end_item = QTableWidgetItem(str(int(bounds[i + 1])))
            end_item.setFlags(end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            end_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 4, end_item)

        self._table.blockSignals(False)

    def _on_table_edit(self, row, col):
        """Handle label edits in the table."""
        if col == 1 and row < len(self._labels):
            new_text = self._table.item(row, col).text().strip()
            if new_text:
                self._labels[row] = new_text
                self._plot()

    # ------------------------------------------------------------------
    # Draggable boundaries
    # ------------------------------------------------------------------

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x = event.xdata
        bounds = self._boundaries

        # Check if near an internal boundary line
        xlim = self.ax.get_xlim()
        threshold = (xlim[1] - xlim[0]) * 0.01  # 1% of visible range
        for i in range(1, len(bounds) - 1):
            if abs(x - bounds[i]) < threshold:
                self._dragging_idx = i
                return

        # Otherwise, select the segment under the click
        for i in range(len(bounds) - 1):
            if bounds[i] <= x <= bounds[i + 1]:
                self._table.selectRow(i)
                break

    def _on_motion(self, event):
        if self._dragging_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        idx = self._dragging_idx
        bounds = self._boundaries

        # Clamp to stay between neighbors
        left = bounds[idx - 1] + 1
        right = bounds[idx + 1] - 1
        new_x = int(round(event.xdata))
        new_x = max(left, min(right, new_x))
        bounds[idx] = new_x

        # Update the plot and table in real time
        self._plot()
        self._update_table()

    def _on_release(self, event):
        self._dragging_idx = None

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _on_split(self):
        bounds = self._boundaries
        n_seg = len(bounds) - 1
        if n_seg >= self._max_segments:
            return

        # Use first selected row in table
        selected = self._table.selectedItems()
        if selected:
            k = selected[0].row()
        else:
            k = 0
        if k < 0 or k >= n_seg:
            k = n_seg - 1

        s = bounds[k]
        e = bounds[k + 1]
        if e - s < 2:
            return
        mid = (s + e) // 2
        self._boundaries = bounds[:k + 1] + [mid] + bounds[k + 1:]
        n_new = len(self._boundaries) - 1
        self._labels = [f'{i + 1}' for i in range(n_new)]
        self._methods = ['mean'] * n_new
        self._plot()
        self._update_table()

    def _on_combine(self):
        """Combine all selected segments into one."""
        bounds = self._boundaries
        n_seg = len(bounds) - 1
        if n_seg < 2:
            return

        # Get selected row indices
        selected_rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()))
        if len(selected_rows) < 2:
            # Need at least 2 segments selected
            return

        # Find contiguous range
        k1 = selected_rows[0]
        k2 = selected_rows[-1]
        if k1 < 0:
            k1 = 0
        if k2 >= n_seg:
            k2 = n_seg - 1

        # Remove boundaries between k1 and k2
        keep_label = self._labels[k1]
        keep_method = self._methods[k1] if k1 < len(self._methods) else 'mean'
        self._boundaries = bounds[:k1 + 1] + bounds[k2 + 1:]
        n_new = len(self._boundaries) - 1
        self._labels = [f'{i + 1}' for i in range(n_new)]
        self._labels[min(k1, n_new - 1)] = keep_label
        self._methods = ['mean'] * n_new
        self._methods[min(k1, n_new - 1)] = keep_method
        self._plot()
        self._update_table()

    def _on_reset(self):
        self._boundaries = list(self._initial_boundaries)
        n_seg = len(self._boundaries) - 1
        self._labels = [f'{i + 1}' for i in range(n_seg)]
        self._methods = list(self._initial_methods)
        self._plot()
        self._update_table()

    def _refresh_finish_button(self):
        """Label/enable the confirm button per the active zone.

        Last (or only) zone → "Finish", enabled only once every OTHER zone is
        accepted. Other zones → "OK" (accept + advance), always enabled.
        Single zone: it is the last zone with no others, so "Finish" is enabled.
        """
        if not hasattr(self, '_btn_finish'):
            return
        last = len(self._zones) - 1
        if self._active_zone == last:
            self._btn_finish.setText("Finish")
            others_ok = all(self._zones[i].get('accepted')
                            for i in range(len(self._zones)) if i != last)
            self._btn_finish.setEnabled(others_ok)
            self._btn_finish.setToolTip(
                "Finish" if others_ok
                else "Accept every other zone (OK) before finishing")
        else:
            self._btn_finish.setText("OK")
            self._btn_finish.setEnabled(True)
            self._btn_finish.setToolTip("Accept this zone and go to the next")

    def _on_finish(self):
        self._save_active_zone()
        last = len(self._zones) - 1
        self._zones[self._active_zone]['accepted'] = True
        if self._active_zone == last:
            # Finish: confirm the whole dialog (only reachable when enabled).
            self.confirmed = True
            self.accept()
            return
        # OK: accept this zone, advance to the next one.
        nxt = self._active_zone + 1
        if self._zone_btn_group is not None:
            btn = self._zone_btn_group.button(nxt)
            if btn is not None:
                btn.setChecked(True)  # does not emit idClicked
        self._load_active_zone(nxt)


def _segment_result(boundaries, labels, methods, signal_rect, peak_locations):
    """Build the segment result dict for one zone from its boundaries.

    Mirrors the legacy single-zone result-building logic exactly.
    """
    bounds = list(boundaries)
    labs = list(labels)
    n_seg = len(bounds) - 1

    bounds_arr = np.array([[bounds[i], bounds[i + 1]] for i in range(n_seg)],
                          dtype=int)
    lines_x = bounds[1:-1]

    # Segment metadata
    segment_start_peaks = np.zeros(n_seg, dtype=int)
    for i in range(n_seg):
        if i == 0:
            segment_start_peaks[i] = 0
            continue
        seg_start = bounds[i]
        matches = np.where(peak_locations == seg_start)[0]
        if len(matches) > 0:
            segment_start_peaks[i] = int(matches[0]) + 1
        else:
            after = np.where(peak_locations >= seg_start)[0]
            segment_start_peaks[i] = int(after[0]) + 1 if len(after) > 0 else -1

    return {
        'bounds': bounds_arr,
        'labels': labs[:n_seg],
        'methods': list(methods[:n_seg]),
        'lines_x': lines_x,
        'signal': signal_rect.copy(),
        'peak_locations': peak_locations.copy(),
        'segment_start_peaks': segment_start_peaks,
        'segment_numbers': np.arange(1, n_seg + 1),
        'segment_names': labs[:n_seg],
    }


def pulse_feature_extraction(tp_settings, sample_rate=None):
    """Interactive UI to define labeled segments within pulse template(s).

    Processes every zone of a (possibly multi-zone) TPsettings struct: each
    zone gets its own rectangularized template, peak locations, original
    overlay and methods, selectable via the dialog's Zone selector.

    Parameters
    ----------
    tp_settings : dict
        Zone-grouped TPsettings struct (or legacy flat dict for one zone).
        Each zone supplies 'pulse_template_rec', 'peak_locations',
        'template_original' and 'methods'.
    sample_rate : float, optional
        Sampling rate in Hz.

    Returns
    -------
    result : dict
        For a single zone, a bare segment dict (identical to the legacy
        output). For multiple zones, a zone-grouped struct
        ``{num_zones, zones:[segment dict, ...]}``.
    """
    from utils.zones import n_zones, wrap_zones
    from utils.tpsettings_io import get_zone

    nz = n_zones(tp_settings)

    zone_inputs = []
    for k in range(nz):
        tpz = get_zone(tp_settings, k)
        template_rect = tpz.get('pulse_template_rec')
        if template_rect is None:
            raise ValueError(
                "No rectangularized template found in TPsettings.")
        peak_locations = tpz.get('peak_locations')
        if peak_locations is None:
            peak_locations = np.array([], dtype=int)
        peak_locations = np.atleast_1d(peak_locations).ravel().astype(int)
        zone_inputs.append({
            'signal_rect': np.atleast_1d(template_rect).ravel().astype(float),
            'peak_locations': peak_locations,
            'template_original': tpz.get('template_original'),
            'methods': tpz.get('methods'),
        })

    dlg = _SegmentDialog(zones=zone_inputs, active_zone=0)
    dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Feature Extraction without confirming.")

    results = []
    for k, z in enumerate(dlg._zones):
        results.append(_segment_result(
            z['boundaries'], z['labels'], z['methods'],
            zone_inputs[k]['signal_rect'], zone_inputs[k]['peak_locations']))

    return wrap_zones(results)
