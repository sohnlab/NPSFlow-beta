"""Runner for TrimData block - interactive data trimming with QDialog."""

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QCheckBox, QGridLayout, QGroupBox, QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from utils.decimate import minmax_decimate


MAX_PLOT_POINTS = 5000


def _get_time_axis(num_samples, sample_rate):
    """Compute time vector and label with appropriate units."""
    time_s = np.arange(num_samples) / sample_rate
    duration = time_s[-1] if len(time_s) > 0 else 0
    if duration >= 1:
        return time_s, "Time [s]"
    elif duration >= 1e-3:
        return time_s * 1e3, "Time [ms]"
    else:
        return time_s * 1e6, "Time [\u00b5s]"


class _TrimDataDialog(QDialog):
    """Interactive trim data dialog with embedded matplotlib canvas."""

    def __init__(self, data, sample_rate, is_mz, filename=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualize Data")
        self.resize(1180, 480)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._original_data = data.copy()
        self._sample_rate = sample_rate
        self._is_mz = is_mz
        self._filename = filename
        self.confirmed = False
        self._n_zones = data.shape[1]
        # Derived-state model — _data_current is recomputed from these:
        #   trim range (rows into original), selected zones, inverted zones.
        self._trim_start = 0
        self._trim_end = data.shape[0]
        self._selected_zones = list(range(self._n_zones))  # original indices
        self._invert_zones = set()                          # original indices
        self._recompute()

        self._build_ui()
        self._plot_data(self._data_current)

    def _recompute(self):
        """Rebuild _data_current from trim range, zone selection and inverts."""
        cols = self._selected_zones or list(range(self._n_zones))
        d = self._original_data[self._trim_start:self._trim_end, cols].copy()
        for j, z in enumerate(cols):
            if z in self._invert_zones:
                d[:, j] = -d[:, j]
        self._data_current = d

    def _build_ui(self):
        # Canvas on the left, grouped control column on the right
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # Matplotlib canvas
        self.fig = Figure(figsize=(10.5, 4.5), tight_layout=True)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        outer.addWidget(self.canvas, stretch=1)

        # Right-side control column: [Zones] [Invert] [Edit] [File] <stretch> [Continue]
        side = QVBoxLayout()
        side.setSpacing(10)

        # Zones group — inline zone selection (mzNPS only, 6 zones in 2 columns)
        if self._is_mz and self._n_zones > 1:
            self._zone_group = QGroupBox("Zones")
            z_l = QGridLayout(self._zone_group)
            z_l.setContentsMargins(8, 6, 8, 6)
            z_l.setHorizontalSpacing(10)
            z_l.setVerticalSpacing(4)
            self._zone_layout = z_l
            self._zone_checks = {}
            self._refresh_zone_group()
            side.addWidget(self._zone_group)

        # Invert group — per-zone (mzNPS, 6 zones in 2 columns) or whole-signal
        self._invert_group = QGroupBox("Invert")
        inv_l = QGridLayout(self._invert_group)
        inv_l.setContentsMargins(8, 6, 8, 6)
        inv_l.setHorizontalSpacing(10)
        inv_l.setVerticalSpacing(4)
        self._invert_layout = inv_l
        self._invert_checks = []
        self._refresh_invert_group()
        side.addWidget(self._invert_group)

        # Edit group — Reset / Select Zones / Trim
        edit_group = QGroupBox("Edit")
        edit_l = QVBoxLayout(edit_group)
        edit_l.setContentsMargins(8, 6, 8, 6)
        edit_l.setSpacing(4)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        edit_l.addWidget(self._btn_reset)

        self._btn_trim = QPushButton("Trim")
        self._btn_trim.setObjectName("primaryBtn")
        self._btn_trim.clicked.connect(self._on_trim)
        edit_l.addWidget(self._btn_trim)
        side.addWidget(edit_group)

        # File group — Save / Load / Export
        file_group = QGroupBox("File")
        file_l = QVBoxLayout(file_group)
        file_l.setContentsMargins(8, 6, 8, 6)
        file_l.setSpacing(4)

        self._btn_save_settings = QPushButton("Save")
        self._btn_save_settings.clicked.connect(self._on_save_settings)
        file_l.addWidget(self._btn_save_settings)

        self._btn_load_settings = QPushButton("Load")
        self._btn_load_settings.clicked.connect(self._on_load_settings)
        file_l.addWidget(self._btn_load_settings)

        from utils.export_figure_ui import make_export_button
        self._btn_save = make_export_button(
            lambda: self.fig, parent=self,
            get_info=lambda: {"filename": self._filename} if self._filename else None,
        )
        self._btn_save.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Fixed)
        file_l.addWidget(self._btn_save)
        side.addWidget(file_group)

        side.addStretch()

        self._btn_continue = QPushButton("Continue")
        self._btn_continue.setObjectName("confirmBtn")
        self._btn_continue.clicked.connect(self._on_continue)
        side.addWidget(self._btn_continue)

        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(170)
        outer.addWidget(side_w)

    def _plot_data(self, d):
        self.ax.clear()
        n_zones = d.shape[1]
        time_vec, time_label = _get_time_axis(d.shape[0], self._sample_rate)

        if self._is_mz and n_zones > 1:
            zmd = (d - d.mean(axis=0)) / 1e6
            min_spacing = 0
            for zi in range(n_zones - 1):
                max_next = np.max(zmd[:, zi + 1])
                min_curr = np.min(zmd[:, zi])
                req = 2 * (max_next - min_curr)
                min_spacing = max(min_spacing, req)
            if min_spacing <= 0:
                ranges = np.max(zmd, axis=0) - np.min(zmd, axis=0)
                min_spacing = 0.1 * np.mean(ranges)

            for zi in range(n_zones):
                offset = (n_zones - zi) * min_spacing / 2
                tx, ty = minmax_decimate(time_vec, zmd[:, zi] + offset,
                                         MAX_PLOT_POINTS)
                self.ax.plot(tx, ty, label=f"Zone {zi + 1}")
            self.ax.set_ylabel("Resistance [M\u03a9] (offset)")
            self.ax.legend(loc="best")
        else:
            tx, ty = minmax_decimate(time_vec, d[:, 0], MAX_PLOT_POINTS)
            self.ax.plot(tx, ty, color="#1f77b4", linewidth=0.5)
            self.ax.set_ylabel("Signal Amplitude")

        self.ax.set_xlabel(time_label)
        self.ax.set_xlim(time_vec[0], time_vec[-1])
        self.ax.grid(True)

        if self._filename:
            self.ax.set_title(self._filename)

        self.canvas.draw()

    @staticmethod
    def _clear_grid(lay):
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _refresh_zone_group(self):
        """Rebuild the zone-selection checkboxes — a fixed 6-zone grid
        (2 columns); a zone that doesn't exist in the data is disabled."""
        self._clear_grid(self._zone_layout)
        self._zone_checks = {}
        n_show = max(6, self._n_zones)
        n_rows = (n_show + 1) // 2
        for zi in range(n_show):
            cb = QCheckBox(f"Zone {zi + 1}")
            if zi < self._n_zones:
                cb.blockSignals(True)
                cb.setChecked(zi in self._selected_zones)
                cb.blockSignals(False)
                cb.toggled.connect(
                    lambda checked, z=zi: self._on_zone_selected(z, checked))
            else:
                cb.setEnabled(False)
            self._zone_layout.addWidget(cb, zi % n_rows, zi // n_rows)
            self._zone_checks[zi] = cb

    def _refresh_invert_group(self):
        """Rebuild the invert checkboxes.

        mzNPS: a fixed 6-zone grid (2 columns) labelled ``Zone N``; a zone
        not currently selected is shown disabled. Single-signal data gets a
        lone ``Invert`` toggle.
        """
        self._clear_grid(self._invert_layout)
        self._invert_checks = []

        if not (self._is_mz and self._n_zones > 1):
            cb = QCheckBox("Invert")
            cb.blockSignals(True)
            cb.setChecked(0 in self._invert_zones)
            cb.blockSignals(False)
            cb.toggled.connect(lambda checked: self._on_invert_toggled(0, checked))
            self._invert_layout.addWidget(cb, 0, 0)
            self._invert_checks.append(cb)
            return

        n_show = max(6, self._n_zones)
        n_rows = (n_show + 1) // 2
        for zi in range(n_show):
            cb = QCheckBox(f"Zone {zi + 1}")
            if zi in self._selected_zones:
                cb.blockSignals(True)
                cb.setChecked(zi in self._invert_zones)
                cb.blockSignals(False)
                cb.toggled.connect(
                    lambda checked, z=zi: self._on_invert_toggled(z, checked))
            else:
                cb.setEnabled(False)
            self._invert_layout.addWidget(cb, zi % n_rows, zi // n_rows)
            self._invert_checks.append(cb)

    def _on_zone_selected(self, zone, checked):
        """Keep or drop a zone. At least one zone must remain selected."""
        if checked:
            if zone not in self._selected_zones:
                self._selected_zones = sorted(self._selected_zones + [zone])
        else:
            if len(self._selected_zones) <= 1:
                cb = self._zone_checks.get(zone)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(True)
                    cb.blockSignals(False)
                return
            self._selected_zones = [z for z in self._selected_zones if z != zone]
            self._invert_zones.discard(zone)
        self._recompute()
        self._refresh_invert_group()
        self._plot_data(self._data_current)

    def _on_invert_toggled(self, zone, checked):
        """Toggle negation of a zone (by original index) and redraw."""
        if checked:
            self._invert_zones.add(zone)
        else:
            self._invert_zones.discard(zone)
        self._recompute()
        self._plot_data(self._data_current)

    def _on_trim(self):
        from utils.select_data_range import _DataRangeDialog
        dlg = _DataRangeDialog(self._data_current)
        dlg.exec()
        if not dlg.confirmed or dlg._result_start is None:
            return
        # Convert 1-based selection into original-row indices (current rows
        # map to original rows [_trim_start:_trim_end]).
        si = dlg._result_start - 1
        ei = dlg._result_end
        new_start = self._trim_start + si
        self._trim_end = self._trim_start + ei
        self._trim_start = new_start
        self._recompute()
        self._plot_data(self._data_current)

    def _on_reset(self):
        self._trim_start = 0
        self._trim_end = self._original_data.shape[0]
        self._selected_zones = list(range(self._n_zones))
        self._invert_zones = set()
        self._recompute()
        if hasattr(self, "_zone_layout"):
            self._refresh_zone_group()
        self._refresh_invert_group()
        self._plot_data(self._data_current)

    def _on_continue(self):
        self.confirmed = True
        self._autosave()
        self.accept()

    def _autosave(self):
        """Auto-save current settings to SavedTemplates/Trim/temp.json."""
        import json, os
        from utils.paths import project_root
        root = project_root()
        folder = os.path.join(root, "SavedTemplates", "Trim")
        os.makedirs(folder, exist_ok=True)
        settings = {
            "trim_start": int(self._trim_start),
            "trim_end": int(self._trim_end),
            "kept_zones": list(self._selected_zones),
            "invert_flags": [z in self._invert_zones for z in self._selected_zones],
        }
        try:
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _default_folder(self):
        import os
        from utils.paths import project_root
        root = project_root()
        folder = os.path.join(root, "SavedTemplates", "Trim")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _on_save_settings(self):
        import json
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Trim Settings", self._default_folder(),
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        settings = {
            "trim_start": int(self._trim_start),
            "trim_end": int(self._trim_end),
            "kept_zones": list(self._selected_zones),
            "invert_flags": [z in self._invert_zones for z in self._selected_zones],
        }
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)

    def _on_load_settings(self):
        import json
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Trim Settings", self._default_folder(),
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        with open(path, "r") as f:
            settings = json.load(f)
        self._apply_settings(settings)

    def _apply_settings(self, settings):
        """Apply loaded trim/zone/invert settings on top of the original data."""
        n_rows = self._original_data.shape[0]
        ts = max(0, min(settings.get("trim_start", 0), n_rows))
        te = max(ts, min(settings.get("trim_end", n_rows), n_rows))
        self._trim_start = ts
        self._trim_end = te

        zones = settings.get("kept_zones")
        if zones is not None:
            valid = sorted(z for z in zones if 0 <= z < self._n_zones)
            self._selected_zones = valid or list(range(self._n_zones))
        else:
            self._selected_zones = list(range(self._n_zones))

        # invert_flags is aligned to the kept columns (see _autosave)
        inv = settings.get("invert_flags") or []
        self._invert_zones = {
            z for j, z in enumerate(self._selected_zones)
            if j < len(inv) and inv[j]
        }

        self._recompute()
        if hasattr(self, "_zone_layout"):
            self._refresh_zone_group()
        self._refresh_invert_group()
        self._plot_data(self._data_current)

    def load_settings_from_file(self, path):
        """Load settings from file path (called from runner for input port)."""
        import json, os
        from utils.paths import project_root
        if not path:
            return
        path = str(path).strip()
        if not os.path.isabs(path):
            root = project_root()
            path = os.path.join(root, path)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                settings = json.load(f)
            self._apply_settings(settings)
        except Exception as e:
            print(f"[TrimData] Failed to load settings from {path}: {e}")

    def get_data(self):
        return self._data_current

    def get_trim_range(self):
        """Return [start, end] sample indices of the kept window in the
        original data (0-based, end exclusive)."""
        return [int(self._trim_start), int(self._trim_end)]


def run(inputs, params, block):
    data = inputs.get("data")
    info = inputs.get("info") or {}
    zones = inputs.get("zones", None)
    sample_rate = inputs.get("sampleRate", params.get("sampleRate", 1))

    if data is None:
        raise ValueError("No data provided to TrimData.")

    filename = info.get("filename") if isinstance(info, dict) else None
    platform = info.get("platform") if isinstance(info, dict) else None

    data = np.asarray(data, dtype=float)

    # Auto-detect platform if not carried in the bundle
    if platform is None:
        platform = "mzNPS" if data.ndim == 2 and data.shape[1] > 1 else "mechanoNPS"

    is_mz = platform.lower() == "mznps"
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Apply pre-set zone selection (1-based indices) before opening the dialog
    if is_mz and zones is not None and data.shape[1] > 1:
        col_indices = [z - 1 for z in zones if 1 <= z <= data.shape[1]]
        if col_indices:
            data = data[:, col_indices]

    dlg = _TrimDataDialog(data, sample_rate, is_mz, filename=filename)

    # Load saved settings if a path was wired in. `/lastSettings` resolves
    # to the block's autosave (SavedTemplates/Trim/temp.json) — and if the
    # block already carries per-instance lastSettings from the workflow
    # JSON, that takes priority. Empty input starts fresh.
    import os
    from utils.paths import project_root
    from utils.settings_input import resolve_settings_input
    from utils.lastsettings_capture import capture
    from utils.paths import saved_templates_dir
    autosave_abs = os.path.join(saved_templates_dir("Trim"), "temp.json")
    settings_path, _ = resolve_settings_input(
        inputs.get("settingsFile"), autosave_abs, block=block)
    if settings_path:
        dlg.load_settings_from_file(settings_path)
    else:
        # No settings port wired: still reopen with this block instance's own
        # last-used trim/zone/invert selection (carried in the workflow JSON),
        # so closing and reopening the block doesn't drop the configuration.
        params = getattr(block, "parameters", None)
        last = params.get("lastSettings") if isinstance(params, dict) else None
        if isinstance(last, dict):
            dlg._apply_settings(last)

    dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Trim Data without confirming.")

    # Mirror the just-written autosave into the block so the settings
    # travel with the workflow JSON.
    capture(block, autosave_abs)

    result = dlg.get_data()
    if result.ndim == 2 and result.shape[1] == 1:
        result = result.ravel()

    return {"data": result, "trimIndex": dlg.get_trim_range()}
