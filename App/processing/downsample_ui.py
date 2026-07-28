"""Interactive UI for the Downsample block — the JOVE preprocessing chain.

Single preview plot (raw signal before Apply; the detrended output after)
plus per-stage controls with enable checkboxes. Signal maths live in
jove_preprocess.py; this module is UI only.
"""

import json
from utils.paths import project_root
import os

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
    QWidget, QProgressDialog, QApplication, QSizePolicy,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.dialog_style import style_mpl_figure, apply_dialog_style
from processing import jove_preprocess as jp
from processing.filter_ui import MAX_PLOT_POINTS
from utils.decimate import minmax_decimate


class _Cancelled(Exception):
    """Raised from the progress callback when the user cancels processing."""


def _to_float(text, fallback):
    try:
        return float(text)
    except (TypeError, ValueError):
        return fallback


def _fmt(v):
    return f"{v:g}"


class _DownsampleDialog(QDialog):
    """Interactive JOVE-preprocessing dialog for one Downsample block."""

    def __init__(self, data, sample_rate, init_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preprocess")
        self.resize(1120, 600)
        apply_dialog_style(self)

        self._fs = float(sample_rate) if sample_rate else 0.0
        from utils.zones import zone_columns
        self._zones = zone_columns(data)
        self._n_zones = len(self._zones)
        self._active = 0

        self._params = dict(jp.DEFAULT_PARAMS)
        if init_params:
            self._params.update({k: v for k, v in init_params.items()
                                 if k in jp.DEFAULT_PARAMS})
        self.confirmed = False

        # Navigation state: single plot; y auto-fits the visible data with a
        # 10% margin. Zoom Box / Pan are manual (drag).
        self._x_full = None
        self._zb_mode = False
        self._pan_mode = False
        self._zb_start = None
        self._zb_span = None
        self._zb_ax = None
        self._zb_bg = None  # cached background for blitting the rubber-band
        self._pan_start = None
        self._applied = False  # processing runs only once Apply is clicked
        self._busy = False     # guards against re-entrant Apply during progress
        # Full (un-decimated) series currently plotted, as (t, y, line) tuples;
        # re-decimated per view so zooming in reveals real detail instead of the
        # coarse full-range line.
        self._plot_series = []

        self._build_ui()
        self._show_raw()

    # ── UI ──

    def _build_ui(self):
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)
        p = self._params

        # plot (row 0, col 0)
        self.fig = Figure(figsize=(9, 2.8))
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.14)
        self.canvas.mpl_connect("button_press_event", self._mpl_press)
        self.canvas.mpl_connect("motion_notify_event", self._mpl_motion)
        self.canvas.mpl_connect("button_release_event", self._mpl_release)
        grid.addWidget(self.canvas, 0, 0)

        # navigation (row 0, col 1) — top-aligned so it stays compact instead
        # of stretching into a tall empty box beside the plot.
        grid.addWidget(self._build_nav(), 0, 1, Qt.AlignmentFlag.AlignTop)

        # stage groups (row 1, col 0)
        stages = QWidget()
        srow = QHBoxLayout(stages)
        srow.setContentsMargins(0, 0, 0, 0)
        for g in (self._build_smoothing(p), self._build_downsample(p),
                  self._build_lowpass(p), self._build_asls(p)):
            g.setSizePolicy(QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Expanding)  # equal height
            srow.addWidget(g, 1)
        grid.addWidget(stages, 1, 0)

        # Uniform width for every value control so the spinboxes and dropdowns
        # line up across the stage groups.
        for w in (self.sp_width, self.cmb_type, self.sp_factor, self.sp_cutoff,
                  self.cmb_preset, self.le_lambda, self.le_p, self.le_margin,
                  self.sp_iter):
            w.setFixedWidth(120)

        # buttons (row 1, col 1) — top-aligned to sit level with the stage groups.
        grid.addWidget(self._build_buttons(), 1, 1, Qt.AlignmentFlag.AlignTop)

        # plot width == stage-groups width: col 0 stretches, col 1 natural.
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)

        self._apply_local_style()

    def _build_smoothing(self, p):
        g = QGroupBox("Smoothing")
        gl = QVBoxLayout(g)
        self.cb_smooth = QCheckBox("Enable")
        self.cb_smooth.setChecked(p["smooth_enable"])
        gl.addWidget(self.cb_smooth)
        row = QHBoxLayout()
        row.addWidget(QLabel("Width"))
        self.sp_width = QSpinBox()
        self.sp_width.setRange(2, 1000000)
        self.sp_width.setValue(int(p["smooth_width"]))
        row.addWidget(self.sp_width, 1)
        gl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Type"))
        self.cmb_type = QComboBox()
        for t, label in jp.SMOOTH_TYPES.items():
            self.cmb_type.addItem(label, t)
        idx = self.cmb_type.findData(int(p["smooth_type"]))
        self.cmb_type.setCurrentIndex(max(0, idx))
        row.addWidget(self.cmb_type, 1)
        gl.addLayout(row)
        gl.addStretch()
        return g

    def _build_downsample(self, p):
        g = QGroupBox("Downsample")
        gl = QVBoxLayout(g)
        self.cb_ds = QCheckBox("Enable")
        self.cb_ds.setChecked(p["ds_enable"])
        gl.addWidget(self.cb_ds)
        row = QHBoxLayout()
        row.addWidget(QLabel("Factor N"))
        self.sp_factor = QSpinBox()
        self.sp_factor.setRange(1, 100000)
        self.sp_factor.setValue(int(p["ds_factor"]))
        row.addWidget(self.sp_factor, 1)
        gl.addLayout(row)
        gl.addStretch()
        return g

    def _build_lowpass(self, p):
        g = QGroupBox("Low-pass filter")
        gl = QVBoxLayout(g)
        self.cb_lp = QCheckBox("Enable")
        self.cb_lp.setChecked(p["lp_enable"])
        gl.addWidget(self.cb_lp)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cutoff (Hz)"))
        self.sp_cutoff = QDoubleSpinBox()
        self.sp_cutoff.setDecimals(1)
        self.sp_cutoff.setRange(0.1, max(self._fs / 2.0, 1e6))
        self.sp_cutoff.setValue(float(p["lp_cutoff"]))
        row.addWidget(self.sp_cutoff, 1)
        gl.addLayout(row)
        self.cb_pad = QCheckBox("Padding")
        self.cb_pad.setChecked(p["lp_pad"])
        gl.addWidget(self.cb_pad)
        gl.addStretch()
        return g

    def _build_asls(self, p):
        g = QGroupBox("ASLS detrend")
        gl = QVBoxLayout(g)
        self.cb_asls = QCheckBox("Enable")
        self.cb_asls.setChecked(p["asls_enable"])
        gl.addWidget(self.cb_asls)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItem("— preset —", None)
        for name in jp.ASLS_PRESETS:
            self.cmb_preset.addItem(name, name)
        self.cmb_preset.currentIndexChanged.connect(
            lambda _i: self._on_preset(self.cmb_preset.currentData()))
        row.addWidget(self.cmb_preset, 1)
        gl.addLayout(row)
        grid = QGridLayout()
        grid.addWidget(QLabel("lambda"), 0, 0)
        self.le_lambda = QLineEdit(_fmt(p["asls_lambda"]))
        grid.addWidget(self.le_lambda, 0, 1)
        grid.addWidget(QLabel("p"), 1, 0)
        self.le_p = QLineEdit(_fmt(p["asls_p"]))
        grid.addWidget(self.le_p, 1, 1)
        grid.addWidget(QLabel("noise margin"), 2, 0)
        self.le_margin = QLineEdit(_fmt(p["asls_margin"]))
        grid.addWidget(self.le_margin, 2, 1)
        grid.addWidget(QLabel("max iter"), 3, 0)
        self.sp_iter = QSpinBox()
        self.sp_iter.setRange(1, 500)
        self.sp_iter.setValue(int(p["asls_iter"]))
        grid.addWidget(self.sp_iter, 3, 1)
        gl.addLayout(grid)
        gl.addStretch()
        return g

    def _build_nav(self):
        nav = QGroupBox("Navigation")
        nl = QVBoxLayout(nav)
        if self._n_zones > 1:
            zr = QHBoxLayout()
            zr.addWidget(QLabel("Zone"))
            self.cmb_zone = QComboBox()
            for zi in range(self._n_zones):
                self.cmb_zone.addItem(f"Zone {zi + 1}", zi)
            self.cmb_zone.currentIndexChanged.connect(self._on_zone)
            zr.addWidget(self.cmb_zone, 1)
            nl.addLayout(zr)
        self.btn_home = QPushButton("Home")
        self.btn_home.clicked.connect(self._on_home)
        nl.addWidget(self.btn_home)
        zrow = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        zrow.addWidget(self.btn_zoom_in)
        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        zrow.addWidget(self.btn_zoom_out)
        nl.addLayout(zrow)
        trow = QHBoxLayout()
        self.btn_zoom_box = QPushButton("Zoom Box")
        self.btn_zoom_box.setCheckable(True)
        self.btn_zoom_box.clicked.connect(self._on_toggle_zoom_box)
        trow.addWidget(self.btn_zoom_box)
        self.btn_pan = QPushButton("Pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(self._on_toggle_pan)
        trow.addWidget(self.btn_pan)
        nl.addLayout(trow)
        return nav

    def _build_buttons(self):
        box = QGroupBox()
        bl = QGridLayout(box)
        from utils.export_figure_ui import make_export_button
        self.btn_export = make_export_button(lambda: self.fig, parent=self)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("confirmBtn")        # accent starts here
        self.btn_apply.clicked.connect(self._recompute)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.setObjectName("confirmBtn")
        self.btn_confirm.setEnabled(False)                # gated on Apply
        self.btn_confirm.clicked.connect(self._on_confirm)
        for b in (self.btn_export, self.btn_apply, self.btn_cancel, self.btn_confirm):
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setMinimumWidth(90)
        bl.addWidget(self.btn_export, 0, 0)
        bl.addWidget(self.btn_apply, 0, 1)
        bl.addWidget(self.btn_cancel, 1, 0)
        bl.addWidget(self.btn_confirm, 1, 1)
        return box

    def _apply_local_style(self):
        self.setStyleSheet(self.styleSheet() + """
            QGroupBox { border-radius: 2px; }
        """)

    # ── callbacks ──

    def _on_preset(self, name):
        preset = jp.ASLS_PRESETS.get(name)
        if not preset:
            return
        self.le_lambda.setText(_fmt(preset["asls_lambda"]))
        self.le_p.setText(_fmt(preset["asls_p"]))
        self.le_margin.setText(_fmt(preset["asls_margin"]))
        self.sp_iter.setValue(int(preset["asls_iter"]))

    def _on_zone(self, idx):
        if 0 <= idx < self._n_zones:
            self._active = idx
            # Keep the not-yet-applied state until the user hits Apply.
            (self._recompute if self._applied else self._show_raw)()

    # ── navigation (synced x-window; y auto-fit with 10% margin) ──

    def _clamp_x(self, x0, x1):
        if not self._x_full:
            return x0, x1
        lo, hi = self._x_full
        span, full = x1 - x0, hi - lo
        if span >= full:
            return lo, hi
        if x0 < lo:
            return lo, lo + span
        if x1 > hi:
            return hi - span, hi
        return x0, x1

    def _autofit_y(self, ax, x0, x1):
        """Set ax's y-limits to the data visible in [x0, x1] plus a 10% margin."""
        ymins, ymaxs = [], []
        for line in ax.get_lines():
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            if xd.size == 0:
                continue
            yv = yd[(xd >= x0) & (xd <= x1)]
            if yv.size:
                ymins.append(np.nanmin(yv))
                ymaxs.append(np.nanmax(yv))
        if not ymins:
            return
        ymin, ymax = min(ymins), max(ymaxs)
        margin = 0.10 * (ymax - ymin) if ymax > ymin else (abs(ymax) * 0.10 or 1.0)
        ax.set_ylim(ymin - margin, ymax + margin)

    def _set_xlim(self, x0, x1):
        x0, x1 = self._clamp_x(x0, x1)
        self._redraw_for_view(x0, x1)     # re-decimate to the visible window
        self.ax.set_xlim(x0, x1)
        self._autofit_y(self.ax, x0, x1)
        self.canvas.draw_idle()

    def _redraw_for_view(self, x0, x1):
        """Re-decimate every plotted series to the visible x-window, so zooming
        in shows real detail (min-max decimation over the whole range would
        otherwise leave only a few coarse points inside a zoomed span)."""
        if not self._plot_series:
            return
        span = x1 - x0
        lo, hi = x0 - 0.02 * span, x1 + 0.02 * span   # margin avoids edge gaps
        for t, y, line in self._plot_series:
            if not len(t):
                continue
            mask = (t >= lo) & (t <= hi)
            if not np.any(mask):
                mask = slice(None)
            rx, ry = minmax_decimate(t[mask], y[mask], MAX_PLOT_POINTS)
            line.set_data(rx, ry)

    def _plot_signal(self, series, title, xlabel):
        """Plot one or more signals into the single axis with view-aware
        decimation. *series* is a list of (t, y, color, linewidth, label)."""
        ax = self.ax
        ax.clear()
        self._plot_series = []
        tmax = 1.0
        for t, y, color, lw, label in series:
            t = np.asarray(t, dtype=float)
            y = np.asarray(y, dtype=float)
            (line,) = ax.plot([], [], color=color, linewidth=lw, label=label)
            self._plot_series.append((t, y, line))
            if len(t):
                tmax = max(tmax, float(t[-1]))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)
        self._x_full = (0.0, tmax if tmax > 0 else 1.0)
        self._redraw_for_view(*self._x_full)
        ax.set_xlim(*self._x_full)
        self._autofit_y(ax, *self._x_full)
        self.canvas.draw_idle()

    def _on_home(self):
        if self._x_full:
            self._set_xlim(*self._x_full)

    def _on_zoom_in(self):
        x0, x1 = self.ax.get_xlim()
        mid, span = (x0 + x1) / 2.0, (x1 - x0) / 4.0
        self._set_xlim(mid - span, mid + span)

    def _on_zoom_out(self):
        x0, x1 = self.ax.get_xlim()
        mid, span = (x0 + x1) / 2.0, (x1 - x0)
        self._set_xlim(mid - span, mid + span)

    def _on_toggle_zoom_box(self, checked):
        self._zb_mode = checked
        if checked:
            self._pan_mode = False
            self.btn_pan.setChecked(False)

    def _on_toggle_pan(self, checked):
        self._pan_mode = checked
        if checked:
            self._zb_mode = False
            self.btn_zoom_box.setChecked(False)

    def _zb_x(self, event):
        """Cursor x in the zoom-box axes' data coords — keeps tracking even when
        the pointer strays outside the axes vertically."""
        if event.xdata is not None:
            return event.xdata
        if event.x is not None and self._zb_ax is not None:
            return float(self._zb_ax.transData.inverted().transform(
                (event.x, 0))[0])
        return None

    def _mpl_press(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if self._zb_mode and event.xdata is not None:
            # Blit the rubber-band: cache the static background once, then only
            # redraw the span per motion — no full-figure redraw, so no lag.
            self._zb_ax = event.inaxes
            self._zb_start = event.xdata
            self._zb_span = event.inaxes.axvspan(
                event.xdata, event.xdata, alpha=0.25, color="#4488cc",
                animated=True)
            self.canvas.draw()
            self._zb_bg = self.canvas.copy_from_bbox(self.fig.bbox)
            self._zb_ax.draw_artist(self._zb_span)
            self.canvas.blit(self.fig.bbox)
        elif self._pan_mode:
            self._pan_start = (event.x, self.ax.get_xlim())

    def _mpl_motion(self, event):
        if self._pan_mode and self._pan_start is not None:
            px0, (xl0, xl1) = self._pan_start
            inv = self.ax.transData.inverted()
            dx = inv.transform((px0, 0))[0] - inv.transform((event.x, 0))[0]
            x0, x1 = self._clamp_x(xl0 + dx, xl1 + dx)
            self.ax.set_xlim(x0, x1)
            self.canvas.draw_idle()
            return
        if self._zb_mode and self._zb_start is not None and self._zb_span is not None:
            xcur = self._zb_x(event)
            if xcur is None:
                return
            self._zb_span.set_x(min(self._zb_start, xcur))
            self._zb_span.set_width(abs(xcur - self._zb_start))
            if self._zb_bg is not None:
                self.canvas.restore_region(self._zb_bg)
                self._zb_ax.draw_artist(self._zb_span)
                self.canvas.blit(self.fig.bbox)
            else:
                self.canvas.draw_idle()

    def _mpl_release(self, event):
        if self._zb_mode and self._zb_start is not None:
            xend = self._zb_x(event)
            if self._zb_span is not None:
                try:
                    self._zb_span.remove()
                except Exception:
                    pass
                self._zb_span = None
            self._zb_bg = None
            if xend is not None and abs(xend - self._zb_start) > 1e-9:
                self._set_xlim(min(self._zb_start, xend),
                               max(self._zb_start, xend))
            else:
                self.canvas.draw_idle()  # clear the leftover rubber-band
            self._zb_start = None
            self._zb_ax = None
            self._zb_mode = False
            self.btn_zoom_box.setChecked(False)
        elif self._pan_mode and self._pan_start is not None:
            self._pan_start = None
            self._set_xlim(*self.ax.get_xlim())  # re-fit y for the new window

    def _read_params(self):
        p = self._params
        p["smooth_enable"] = self.cb_smooth.isChecked()
        p["smooth_width"] = int(self.sp_width.value())
        p["smooth_type"] = int(self.cmb_type.currentData())
        p["ds_enable"] = self.cb_ds.isChecked()
        p["ds_factor"] = int(self.sp_factor.value())
        p["lp_enable"] = self.cb_lp.isChecked()
        p["lp_cutoff"] = float(self.sp_cutoff.value())
        p["lp_pad"] = self.cb_pad.isChecked()
        p["asls_enable"] = self.cb_asls.isChecked()
        p["asls_lambda"] = _to_float(self.le_lambda.text(), p["asls_lambda"])
        p["asls_p"] = _to_float(self.le_p.text(), p["asls_p"])
        p["asls_margin"] = _to_float(self.le_margin.text(), p["asls_margin"])
        p["asls_iter"] = int(self.sp_iter.value())

    def _show_raw(self):
        """Initial view before Apply: just the raw signal."""
        y = np.asarray(self._zones[self._active], dtype=float).ravel()
        if self._fs:
            t = np.arange(y.size) / self._fs
            xlabel = "Time (s)"
        else:
            t = np.arange(y.size)
            xlabel = "Sample"
        self._plot_signal([(t, y, "0.5", 0.6, "raw")], "Raw signal", xlabel)

    def _recompute(self):
        if self._busy:
            return
        self._read_params()
        self._applied = True
        y = np.asarray(self._zones[self._active], dtype=float).ravel()

        # Cancellable progress dialog — the ASLS solve can be slow on long
        # recordings. Only appears if the work actually takes a moment.
        prog = QProgressDialog("Processing…", "Cancel", 0, 100, self)
        prog.setMinimumWidth(380)
        prog.setStyleSheet("""
            QProgressBar {
                border: 1px solid palette(mid);
                border-radius: 2px;
                background: palette(base);
                min-height: 14px;
                text-align: center;
            }
            QProgressBar::chunk { background: #1f6fb5; border-radius: 1px; }
        """)
        prog.setWindowTitle("Preprocess")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(400)

        def report(frac, msg):
            prog.setLabelText(msg)
            prog.setValue(int(frac * 100))
            QApplication.processEvents()
            if prog.wasCanceled():
                raise _Cancelled()

        self._busy = True
        try:
            r = jp.process_1d(y, self._fs, self._params, progress=report)
        except _Cancelled:
            return
        except Exception as exc:
            from utils.app_logger import logger
            logger.warning(f"Preprocess preview failed: {exc}")
            return
        finally:
            self._busy = False
            prog.close()
            prog.deleteLater()

        lp, trend = r["lp"], r["trendline"]
        fs_out = r["fs_out"]
        if self._fs and fs_out:
            t_out = np.arange(lp.size) / fs_out
            xlabel = "Time (s)"
        else:
            t_out = np.arange(lp.size)
            xlabel = "Sample"
        # Always show the signal after the enabled pre-detrend stages
        # (raw -> smoothed -> downsampled -> filtered = lp); overlay the ASLS
        # baseline when detrend is on so the fit can be judged.
        series = [(t_out, lp, "#1f6fb5", 0.8, "signal")]
        if self._params["asls_enable"]:
            series.append((t_out, trend, "m", 1.2, "ASLS baseline"))
            title = "Signal + ASLS baseline"
        else:
            title = "Signal"
        self._plot_signal(series, title, xlabel)
        self._mark_applied()

    def _mark_applied(self):
        if self.btn_confirm.isEnabled():
            return
        self.btn_apply.setObjectName("")
        self.btn_confirm.setEnabled(True)
        for b in (self.btn_apply, self.btn_confirm):
            b.style().unpolish(b)
            b.style().polish(b)

    def _autosave(self):
        root = project_root()
        folder = os.path.join(root, "SavedTemplates", "Downsample")
        os.makedirs(folder, exist_ok=True)
        try:
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(self._params, f, indent=2)
        except Exception:
            pass

    def _on_confirm(self):
        self._read_params()
        self._autosave()
        self.confirmed = True
        self.accept()


def downsample_ui(data, sample_rate, init_params=None):
    """Open the interactive dialog; return the confirmed params dict.

    Raises ValueError if the user closes without confirming (matches the
    other interactive blocks' cancel-stops-pipeline convention).
    """
    dlg = _DownsampleDialog(data, sample_rate, init_params=init_params)
    dlg.exec()
    if not dlg.confirmed:
        raise ValueError("User closed Downsample without confirming.")
    return dlg._params
