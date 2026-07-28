"""Interactive filter configuration and application.

Translates MATLAB FilterUI.m: multi-stage interactive filter design with
time-domain and frequency-domain visualization, supporting lowpass, highpass,
bandpass, and notch filter types with stacked filter chains.

Uses a QDialog with QTabWidget (matching MATLAB uitabgroup layout).
"""

import numpy as np
from scipy import signal as sig

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QWidget,
    QTabWidget,
    QButtonGroup,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from utils.decimate import minmax_decimate


MAX_PLOT_POINTS = 5000


def _reflect_pad(data, pad_len):
    """Reflect-pad a 1-D signal to reduce edge artifacts during filtering."""
    n = len(data)
    if pad_len <= 0:
        return data
    start_pad = data[1 : min(pad_len + 1, n)][::-1]
    if len(start_pad) < pad_len:
        start_pad = np.concatenate(
            [start_pad, np.full(pad_len - len(start_pad), data[0])]
        )
    else:
        start_pad = start_pad[:pad_len]
    end_pad = data[max(0, n - pad_len - 1) : n - 1][::-1]
    if len(end_pad) < pad_len:
        end_pad = np.concatenate([end_pad, np.full(pad_len - len(end_pad), data[-1])])
    else:
        end_pad = end_pad[:pad_len]
    return np.concatenate([start_pad, data, end_pad])


def _apply_single_filter(
    data, fs, ftype, cutoff1, cutoff2, notch_mode, notch_bw, order=4
):
    """Apply a single Butterworth or notch filter with reflect-padding.

    Uses SOS (second-order sections) for numerical stability.
    """
    fs = float(fs)
    nyq = fs / 2
    pad_len = int(round(fs * 0.5))
    padded = _reflect_pad(data, pad_len)

    if ftype == "lowpass":
        wn = min(cutoff1 / nyq, 0.99)
        sos = sig.butter(order, wn, btype="low", output="sos")
        out = sig.sosfiltfilt(sos, padded)
    elif ftype == "highpass":
        wn = max(cutoff1 / nyq, 0.01)
        sos = sig.butter(order, wn, btype="high", output="sos")
        out = sig.sosfiltfilt(sos, padded)
    elif ftype == "bandpass":
        wn_lo = max(cutoff1 / nyq, 0.01)
        wn_hi = min(cutoff2 / nyq, 0.99)
        if wn_lo >= wn_hi:
            wn_lo = wn_hi - 0.01
        sos = sig.butter(order, [wn_lo, wn_hi], btype="band", output="sos")
        out = sig.sosfiltfilt(sos, padded)
    elif ftype == "notch":
        if notch_mode == "freqBW":
            w0 = np.clip(cutoff1, 1, nyq - 1)
            bandwidth = max(notch_bw, 0.1)
            Q = w0 / bandwidth
        else:
            center = (cutoff1 + cutoff2) / 2
            bandwidth = max(cutoff2 - cutoff1, 0.1)
            w0 = np.clip(center, 1, nyq - 1)
            Q = w0 / bandwidth
        Q = max(Q, 0.5)
        b, a = sig.iirnotch(w0, Q, fs=fs)
        out = sig.filtfilt(b, a, padded)
    else:
        out = padded

    return out[pad_len : pad_len + len(data)]


def apply_filter_chain(data, fs, chain):
    """Apply a zero-phase filter chain to a 1-D or 2-D (time x zones) array.

    Chain entries use the Filter UI config schema (type/cutoff1/cutoff2/
    notch_mode/notch_bw).
    """
    x = np.atleast_1d(np.asarray(data, dtype=float))
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        y = x[:, ch]
        for fi in chain:
            y = _apply_single_filter(
                y,
                fs,
                fi["type"],
                fi.get("cutoff1", 0.0),
                fi.get("cutoff2", 0.0),
                fi.get("notch_mode", "freqBW"),
                fi.get("notch_bw", 2.0),
            )
        out[:, ch] = y
    return out[:, 0] if was_1d else out


def _get_window(name, length):
    """Return a scipy window array by name."""
    wmap = {
        "Rectangular": "boxcar",
        "Hamming": "hamming",
        "Hanning": "hann",
        "Blackman": "blackman",
        "Bartlett": "bartlett",
    }
    return sig.get_window(wmap.get(name, "hamming"), length)


def _compute_psd(data, fs, win_name, seg_len, overlap, nfft):
    """Compute PSD via Welch's method."""
    win = _get_window(win_name, seg_len)
    f, psd = sig.welch(
        data, fs, window=win, nperseg=seg_len, noverlap=overlap, nfft=nfft
    )
    return f, psd


def _compute_fft_mag(data, fs, win_name, nfft):
    """Compute single-sided FFT magnitude spectrum via averaged segments.

    Splits data into overlapping segments (like Welch) but returns magnitude
    instead of power density. This gives sharp peaks while averaging out noise.
    Capped at nfft points per segment for performance.
    """
    N = len(data)
    seg_len = min(nfft, N)
    win = _get_window(win_name, seg_len)
    win_sum = np.sum(win)
    overlap = seg_len // 2
    step = seg_len - overlap

    # Average magnitude over segments
    n_segs = max(1, (N - seg_len) // step + 1)
    fft_len = seg_len
    mag_sum = np.zeros(fft_len // 2 + 1)
    for i in range(n_segs):
        start = i * step
        segment = data[start : start + seg_len]
        if len(segment) < seg_len:
            break
        X = np.fft.rfft(segment * win, n=fft_len)
        mag_sum += np.abs(X)
    mag = mag_sum * 2.0 / (win_sum * n_segs)
    f = np.fft.rfftfreq(fft_len, d=1.0 / fs)
    return f, mag


# ──────────────────── Filter Tab Widget ────────────────────


class _FilterTab(QWidget):
    """One tab in the filter dialog – plots + controls for a single filter stage."""

    def __init__(
        self, input_data, fs, zone_num, total_zones, default_cutoff, parent_dialog
    ):
        super().__init__()
        self.input_data = input_data.copy()
        self.current_data = input_data.copy()
        self.fs = fs
        self.nyq = fs / 2
        self.zone_num = zone_num
        self.total_zones = total_zones
        self.parent_dialog = parent_dialog

        # Filter state
        self.filter_type = "lowpass"
        self.cutoff1 = default_cutoff
        self.cutoff2 = min(default_cutoff * 2, self.nyq - 1)
        self.notch_mode = "freqBW"
        self.notch_freq = 60.0
        self.notch_bw = 2.0
        self.bp_mode = "lohi"  # 'lohi' or 'freqBW'
        self.bp_center = default_cutoff
        self.bp_bw = min(default_cutoff, self.nyq - 1)

        # Spectrum state
        self.spec_min = 0
        self.spec_max = min(2000, self.nyq)
        self.y_log = True
        self.spec_mode = "psd"  # 'psd' or 'fft'

        # Precompute cached data (input never changes per tab)
        N = len(input_data)
        self._t = np.arange(N) / fs
        self._td_input, self._yd_input = minmax_decimate(
            self._t, input_data, MAX_PLOT_POINTS
        )

        # PSD params
        self._seg_len = min(int(2 ** np.ceil(np.log2(fs))), N // 4)
        self._seg_len = max(self._seg_len, 256)
        self._overlap = self._seg_len // 2
        self._nfft = max(self._seg_len, min(int(2 ** np.ceil(np.log2(N))), 2**17))

        # Cache input PSD and FFT (computed once with default window)
        self._cached_win_name = "Hamming"
        self._f_input, self._psd_input = _compute_psd(
            input_data,
            fs,
            self._cached_win_name,
            self._seg_len,
            self._overlap,
            self._nfft,
        )
        self._f_input_fft, self._mag_input_fft = _compute_fft_mag(
            input_data, fs, self._cached_win_name, self._nfft
        )

        self._build_ui()
        self._apply_filter()
        self._update_plots()

    def _refresh_input_cache(self, win_name):
        """Recompute cached input spectra if window changed."""
        if win_name != self._cached_win_name:
            self._cached_win_name = win_name
            self._f_input, self._psd_input = _compute_psd(
                self.input_data,
                self.fs,
                win_name,
                self._seg_len,
                self._overlap,
                self._nfft,
            )
            self._f_input_fft, self._mag_input_fft = _compute_fft_mag(
                self.input_data, self.fs, win_name, self._nfft
            )

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Left: plots ──
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 7), tight_layout=True)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        self.ax_time = self.fig.add_subplot(2, 1, 1)
        self.ax_spec = self.fig.add_subplot(2, 1, 2)
        plot_layout.addWidget(self.canvas)

        # Hidden matplotlib toolbar drives Pan / Zoom Box. Each drag acts only
        # on the axes it starts in, so the two plots are never synced.
        self._nav = NavigationToolbar2QT(self.canvas, self)
        self._nav.setVisible(False)
        # Non-drag nav buttons (Home/Zoom In/Out) act on the last-touched plot.
        self._active_ax = self.ax_time
        self.canvas.mpl_connect("axes_enter_event", self._on_axes_enter)

        # ── Right: controls ──
        ctrl_widget = QWidget()
        ctrl_widget.setFixedWidth(260)
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)

        # --- Information group ---
        from utils.sr_info import make_sr_info_group

        ctrl_layout.addWidget(
            make_sr_info_group(self.fs, getattr(self.parent_dialog, "ds_factor", None))
        )

        # --- Filter Type group ---
        type_group = QGroupBox("Filter Type")
        type_lay = QVBoxLayout(type_group)
        self._type_bg = QButtonGroup(self)
        self._radio_lp = QRadioButton("Lowpass")
        self._radio_hp = QRadioButton("Highpass")
        self._radio_bp = QRadioButton("Bandpass")
        self._radio_notch = QRadioButton("Notch")
        self._radio_lp.setChecked(True)
        for rb in (self._radio_lp, self._radio_hp, self._radio_bp, self._radio_notch):
            self._type_bg.addButton(rb)
            type_lay.addWidget(rb)
        self._type_bg.buttonClicked.connect(self._on_type_changed)
        ctrl_layout.addWidget(type_group)

        # --- Cutoff fields ---
        cut_group = QGroupBox("Cutoff")
        cut_lay = QVBoxLayout(cut_group)

        # Bandpass / Notch mode row (shared: Lower/Upper vs Freq+BW)
        mode_row = QHBoxLayout()
        self._mode_bg = QButtonGroup(self)
        self._radio_lohi = QRadioButton("Lower/Upper")
        self._radio_freqbw = QRadioButton("Freq+BW")
        self._radio_lohi.setChecked(True)
        self._mode_bg.addButton(self._radio_lohi)
        self._mode_bg.addButton(self._radio_freqbw)
        mode_row.addWidget(self._radio_lohi)
        mode_row.addWidget(self._radio_freqbw)
        cut_lay.addLayout(mode_row)
        self._radio_lohi.setVisible(False)
        self._radio_freqbw.setVisible(False)
        self._mode_bg.buttonClicked.connect(self._on_mode_changed)

        row1 = QHBoxLayout()
        self._lbl_cut1 = QLabel("Cutoff (Hz):")
        self._edit_cut1 = QLineEdit(f"{self.cutoff1:.1f}")
        self._edit_cut1.setFixedWidth(90)
        row1.addWidget(self._lbl_cut1)
        row1.addWidget(self._edit_cut1)
        cut_lay.addLayout(row1)

        row2 = QHBoxLayout()
        self._lbl_cut2 = QLabel("Upper (Hz):")
        self._edit_cut2 = QLineEdit(f"{self.cutoff2:.1f}")
        self._edit_cut2.setFixedWidth(90)
        row2.addWidget(self._lbl_cut2)
        row2.addWidget(self._edit_cut2)
        cut_lay.addLayout(row2)
        self._lbl_cut2.setVisible(False)
        self._edit_cut2.setVisible(False)

        # BW row (used by both notch Freq+BW and bandpass Freq+BW)
        bw_row = QHBoxLayout()
        self._lbl_bw = QLabel("BW (Hz):")
        self._edit_bw = QLineEdit(f"{self.notch_bw:.1f}")
        self._edit_bw.setFixedWidth(90)
        bw_row.addWidget(self._lbl_bw)
        bw_row.addWidget(self._edit_bw)
        cut_lay.addLayout(bw_row)
        self._lbl_bw.setVisible(False)
        self._edit_bw.setVisible(False)

        ctrl_layout.addWidget(cut_group)

        # --- Spectrum Controls ---
        spec_group = QGroupBox("Spectrum")
        spec_lay = QVBoxLayout(spec_group)

        # Mode: PSD / FFT
        r = QHBoxLayout()
        r.addWidget(QLabel("Mode:"))
        self._spec_mode_bg = QButtonGroup(self)
        self._radio_psd = QRadioButton("PSD")
        self._radio_fft = QRadioButton("FFT")
        self._radio_psd.setChecked(True)
        self._spec_mode_bg.addButton(self._radio_psd)
        self._spec_mode_bg.addButton(self._radio_fft)
        r.addWidget(self._radio_psd)
        r.addWidget(self._radio_fft)
        spec_lay.addLayout(r)
        self._spec_mode_bg.buttonClicked.connect(self._on_spec_mode_changed)

        # Min Freq
        r = QHBoxLayout()
        r.addWidget(QLabel("Min Freq (Hz):"))
        self._edit_spec_min = QLineEdit(f"{self.spec_min:.0f}")
        self._edit_spec_min.setFixedWidth(70)
        r.addWidget(self._edit_spec_min)
        spec_lay.addLayout(r)

        # Max Freq
        r = QHBoxLayout()
        r.addWidget(QLabel("Max Freq (Hz):"))
        self._edit_spec_max = QLineEdit(f"{self.spec_max:.0f}")
        self._edit_spec_max.setFixedWidth(70)
        r.addWidget(self._edit_spec_max)
        spec_lay.addLayout(r)

        # Window
        r = QHBoxLayout()
        r.addWidget(QLabel("Window:"))
        self._combo_window = QComboBox()
        self._combo_window.addItems(
            ["Rectangular", "Hamming", "Hanning", "Blackman", "Bartlett"]
        )
        self._combo_window.setCurrentIndex(1)  # Hamming default
        r.addWidget(self._combo_window)
        spec_lay.addLayout(r)

        # Freq Res (display only)
        r = QHBoxLayout()
        r.addWidget(QLabel("Freq Res (Hz):"))
        self._lbl_freq_res = QLabel("0.00")
        self._lbl_freq_res.setStyleSheet(
            "background: #e8e8e8; padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px;"
        )
        r.addWidget(self._lbl_freq_res)
        spec_lay.addLayout(r)

        # Y-Scale
        r = QHBoxLayout()
        r.addWidget(QLabel("Y-Scale:"))
        self._yscale_bg = QButtonGroup(self)
        self._radio_log = QRadioButton("Log")
        self._radio_linear = QRadioButton("Linear")
        self._radio_log.setChecked(True)
        self._yscale_bg.addButton(self._radio_log)
        self._yscale_bg.addButton(self._radio_linear)
        r.addWidget(self._radio_log)
        r.addWidget(self._radio_linear)
        spec_lay.addLayout(r)
        self._yscale_bg.buttonClicked.connect(self._on_yscale_changed)

        ctrl_layout.addWidget(spec_group)

        # --- Apply button ---
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.clicked.connect(self._on_apply)
        ctrl_layout.addWidget(self._btn_apply)

        # --- Navigation group ---
        nav_group = QGroupBox("Navigation")
        nav_lay = QVBoxLayout(nav_group)
        self._btn_home = QPushButton("Home")
        self._btn_home.clicked.connect(self._on_home)
        nav_lay.addWidget(self._btn_home)

        zoom_row = QHBoxLayout()
        self._btn_zoom_in = QPushButton("Zoom In")
        self._btn_zoom_in.clicked.connect(lambda: self._zoom_active(0.8))
        zoom_row.addWidget(self._btn_zoom_in)
        self._btn_zoom_out = QPushButton("Zoom Out")
        self._btn_zoom_out.clicked.connect(lambda: self._zoom_active(1.25))
        zoom_row.addWidget(self._btn_zoom_out)
        nav_lay.addLayout(zoom_row)

        tool_row = QHBoxLayout()
        self._btn_zoom_box = QPushButton("Zoom Box")
        self._btn_zoom_box.setCheckable(True)
        self._btn_zoom_box.clicked.connect(self._on_toggle_zoom_box)
        tool_row.addWidget(self._btn_zoom_box)
        self._btn_pan = QPushButton("Pan")
        self._btn_pan.setCheckable(True)
        self._btn_pan.clicked.connect(self._on_toggle_pan)
        tool_row.addWidget(self._btn_pan)
        nav_lay.addLayout(tool_row)

        ctrl_layout.addWidget(nav_group)

        ctrl_layout.addStretch()

        # --- Bottom buttons ---
        self._btn_additional = QPushButton("Additional Filter")
        self._btn_additional.clicked.connect(self._on_additional)
        ctrl_layout.addWidget(self._btn_additional)

        bottom_row = QHBoxLayout()
        from utils.export_figure_ui import make_export_button

        self._btn_export = make_export_button(lambda: self.fig, parent=self)
        bottom_row.addWidget(self._btn_export)

        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.clicked.connect(self._on_confirm)
        bottom_row.addWidget(self._btn_confirm)
        ctrl_layout.addLayout(bottom_row)

        # ── Assemble ──
        layout.addWidget(plot_widget, stretch=3)
        layout.addWidget(ctrl_widget, stretch=0)

    # ── Callbacks ──

    def _on_type_changed(self, btn):
        ftype_map = {
            self._radio_lp: "lowpass",
            self._radio_hp: "highpass",
            self._radio_bp: "bandpass",
            self._radio_notch: "notch",
        }
        self.filter_type = ftype_map.get(btn, "lowpass")
        is_notch = self.filter_type == "notch"
        is_bp = self.filter_type == "bandpass"
        has_mode = is_notch or is_bp

        # Save notch freq when leaving notch, restore when entering
        was_notch = hasattr(self, "_prev_type_notch") and self._prev_type_notch
        if was_notch and not is_notch:
            try:
                self.notch_freq = float(self._edit_cut1.text())
            except ValueError:
                pass
            self._edit_cut1.setText(f"{self.cutoff1:.1f}")
        self._prev_type_notch = is_notch

        # Show/hide mode toggle
        self._radio_lohi.setVisible(has_mode)
        self._radio_freqbw.setVisible(has_mode)

        if is_bp:
            if self.bp_mode == "freqBW":
                self._radio_freqbw.setChecked(True)
            else:
                self._radio_lohi.setChecked(True)
            self._update_bp_controls()
        elif is_notch:
            self._edit_cut1.setText(f"{self.notch_freq:.1f}")
            if self.notch_mode == "freqBW":
                self._radio_freqbw.setChecked(True)
            else:
                self._radio_lohi.setChecked(True)
            self._update_notch_controls()
        else:
            self._lbl_cut1.setText("Cutoff (Hz):")
            self._lbl_cut2.setVisible(False)
            self._edit_cut2.setVisible(False)
            self._lbl_bw.setVisible(False)
            self._edit_bw.setVisible(False)

    def _on_mode_changed(self, btn):
        is_freqbw = btn == self._radio_freqbw
        if self.filter_type == "notch":
            self.notch_mode = "freqBW" if is_freqbw else "band"
            self._update_notch_controls()
        elif self.filter_type == "bandpass":
            self.bp_mode = "freqBW" if is_freqbw else "lohi"
            self._update_bp_controls()

    def _update_notch_controls(self):
        if self.notch_mode == "band":
            self._lbl_cut1.setText("Lower (Hz):")
            self._lbl_cut2.setVisible(True)
            self._edit_cut2.setVisible(True)
            self._lbl_bw.setVisible(False)
            self._edit_bw.setVisible(False)
        else:
            self._lbl_cut1.setText("Freq (Hz):")
            self._lbl_cut2.setVisible(False)
            self._edit_cut2.setVisible(False)
            self._lbl_bw.setVisible(True)
            self._edit_bw.setVisible(True)
            self._edit_bw.setText(f"{self.notch_bw:.1f}")

    def _update_bp_controls(self):
        if self.bp_mode == "freqBW":
            self._lbl_cut1.setText("Center (Hz):")
            self._lbl_cut2.setVisible(False)
            self._edit_cut2.setVisible(False)
            self._lbl_bw.setVisible(True)
            self._edit_bw.setVisible(True)
            self._edit_bw.setText(f"{self.bp_bw:.1f}")
        else:
            self._lbl_cut1.setText("Lower (Hz):")
            self._lbl_cut2.setText("Upper (Hz):")
            self._lbl_cut2.setVisible(True)
            self._edit_cut2.setVisible(True)
            self._lbl_bw.setVisible(False)
            self._edit_bw.setVisible(False)

    def _on_spec_mode_changed(self, btn):
        self.spec_mode = "psd" if btn == self._radio_psd else "fft"
        self._update_spectrum()
        self.canvas.draw_idle()

    def _on_yscale_changed(self, btn):
        self.y_log = btn == self._radio_log
        self._update_spectrum()
        self.canvas.draw_idle()

    def _on_apply(self):
        self._apply_filter()
        self._update_plots()

    # ── Navigation ──

    def _on_axes_enter(self, event):
        """Remember which plot the cursor is over — the target for Home/Zoom."""
        if event.inaxes in (self.ax_time, self.ax_spec):
            self._active_ax = event.inaxes

    def _on_home(self):
        """Reset the last-touched plot to its default view; leaves the other
        plot untouched by re-running only that axes' plot routine."""
        if self._active_ax is self.ax_spec:
            self._update_spectrum()
        else:
            self._update_time_plot()
        self.canvas.draw_idle()

    def _zoom_active(self, factor):
        """Scale the last-touched plot about its center (factor < 1 zooms in).
        Log axes scale geometrically so the log-PSD zoom stays centered."""
        ax = self._active_ax or self.ax_time
        for get_lim, set_lim, scale in (
            (ax.get_xlim, ax.set_xlim, ax.get_xscale()),
            (ax.get_ylim, ax.set_ylim, ax.get_yscale()),
        ):
            lo, hi = get_lim()
            if scale == "log" and lo > 0 and hi > 0:
                c = np.sqrt(lo * hi)
                set_lim(c * (lo / c) ** factor, c * (hi / c) ** factor)
            else:
                c = (lo + hi) / 2.0
                half = (hi - lo) / 2.0 * factor
                set_lim(c - half, c + half)
        self.canvas.draw_idle()

    def _on_toggle_zoom_box(self):
        self._nav.zoom()
        self._sync_nav_buttons()

    def _on_toggle_pan(self):
        self._nav.pan()
        self._sync_nav_buttons()

    def _sync_nav_buttons(self):
        """Mirror the toolbar's active mode onto the checkable buttons so
        Pan and Zoom Box stay mutually exclusive."""
        mode = self._nav.mode  # _Mode(str): '' | 'pan/zoom' | 'zoom rect'
        self._btn_pan.setChecked(mode == "pan/zoom")
        self._btn_zoom_box.setChecked(mode == "zoom rect")

    def _on_additional(self):
        self._apply_filter()
        self._update_plots()
        self.parent_dialog.add_filter_tab(self.current_data)

    def _on_confirm(self):
        self._apply_filter()
        self.parent_dialog.confirm()

    # ── Filter logic ──

    def _read_cutoffs(self):
        try:
            c1 = float(self._edit_cut1.text())
        except ValueError:
            c1 = self.cutoff1
        try:
            c2 = float(self._edit_cut2.text())
        except ValueError:
            c2 = self.cutoff2
        try:
            bw = float(self._edit_bw.text())
        except ValueError:
            bw = self.notch_bw if self.filter_type == "notch" else self.bp_bw

        if self.filter_type == "notch" and self.notch_mode == "freqBW":
            # Freq+BW mode: compute lower/upper from center freq and BW
            bw = max(bw, 0.1)
            self.notch_bw = bw
            freq = np.clip(c1, 1, self.nyq - 1)
            self.cutoff1 = freq
            self.cutoff2 = freq  # not used directly; BW is used
        elif self.filter_type == "bandpass" and self.bp_mode == "freqBW":
            # Freq+BW mode: compute lower/upper from center and BW
            bw = max(bw, 1)
            self.bp_bw = bw
            center = np.clip(c1, 1, self.nyq - 1)
            self.bp_center = center
            self.cutoff1 = max(center - bw / 2, 1)
            self.cutoff2 = min(center + bw / 2, self.nyq - 1)
        else:
            c1 = np.clip(c1, 1, self.nyq - 1)
            c2 = np.clip(c2, c1 + 1, self.nyq - 1)
            self.cutoff1 = c1
            self.cutoff2 = c2

    def _apply_filter(self):
        self._read_cutoffs()
        try:
            self.current_data = _apply_single_filter(
                self.input_data,
                self.fs,
                self.filter_type,
                self.cutoff1,
                self.cutoff2,
                self.notch_mode,
                self.notch_bw,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            from utils.app_logger import logger

            logger.warning(f"Filter failed: {e}")
            self.current_data = self.input_data.copy()

    def _update_plots(self):
        self._update_time_plot()
        self._update_spectrum()
        self.canvas.draw_idle()

    def _update_time_plot(self):
        ax = self.ax_time
        ax.clear()

        # Input uses cached downsampled data
        ax.plot(
            self._td_input,
            self._yd_input,
            color="k",
            linewidth=0.5,
            alpha=0.6,
            label="Input",
        )

        td_f, yd_f = minmax_decimate(self._t, self.current_data, MAX_PLOT_POINTS)
        ax.plot(td_f, yd_f, color="r", linewidth=0.8, label="Filtered")

        zone_label = (
            f"Zone {self.zone_num}/{self.total_zones} - "
            if self.total_zones > 1
            else ""
        )
        ax.set_title(f"{zone_label}Time Domain")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim([0, self._t[-1]])
        ax.legend(loc="best")
        ax.grid(True)

    def _update_spectrum(self):
        ax = self.ax_spec
        ax.clear()

        # Read spectrum controls
        try:
            s_min = float(self._edit_spec_min.text())
        except ValueError:
            s_min = self.spec_min
        try:
            s_max = float(self._edit_spec_max.text())
        except ValueError:
            s_max = self.spec_max
        s_min = max(0, s_min)
        s_max = min(self.nyq, max(s_max, s_min + 1))
        self.spec_min = s_min
        self.spec_max = s_max

        win_name = self._combo_window.currentText()
        self._refresh_input_cache(win_name)

        if self.spec_mode == "fft":
            self._draw_fft(ax, win_name, s_min, s_max)
        else:
            self._draw_psd(ax, win_name, s_min, s_max)

        ax.set_xlim([s_min, s_max])
        ax.set_xlabel("Frequency (Hz)")
        ax.legend(loc="best")
        ax.grid(True)

        freq_res = self.fs / self._seg_len
        self._lbl_freq_res.setText(f"{freq_res:.4f}")

    def _draw_psd(self, ax, win_name, s_min, s_max):
        """Draw Welch PSD (smooth overview of power distribution)."""
        f_orig, psd_orig = self._f_input, self._psd_input

        f_filt, psd_filt = _compute_psd(
            self.current_data,
            self.fs,
            win_name,
            self._seg_len,
            self._overlap,
            self._nfft,
        )

        if self.y_log:
            ax.semilogy(
                f_orig,
                np.maximum(psd_orig, 1e-20),
                color="grey",
                linewidth=1.5,
                label="Input",
            )
            ax.semilogy(
                f_filt,
                np.maximum(psd_filt, 1e-20),
                color="k",
                linewidth=1.5,
                label="Filtered",
            )
            ax.set_ylabel("PSD (V\u00b2/Hz, log)")
        else:
            ax.plot(f_orig, psd_orig, color="grey", linewidth=1.5, label="Input")
            ax.plot(f_filt, psd_filt, color="k", linewidth=1.5, label="Filtered")
            ax.set_ylabel("PSD (V\u00b2/Hz)")
        ax.set_title("Power Spectral Density")

    def _draw_fft(self, ax, win_name, s_min, s_max):
        """Draw FFT magnitude spectrum (sharp peaks for identifying frequencies)."""
        f_orig, mag_orig = self._f_input_fft, self._mag_input_fft

        f_filt, mag_filt = _compute_fft_mag(
            self.current_data, self.fs, win_name, self._nfft
        )

        if self.y_log:
            ax.semilogy(
                f_orig,
                np.maximum(mag_orig, 1e-20),
                color="grey",
                linewidth=0.8,
                label="Input",
            )
            ax.semilogy(
                f_filt,
                np.maximum(mag_filt, 1e-20),
                color="k",
                linewidth=0.8,
                label="Filtered",
            )
            ax.set_ylabel("Magnitude (log)")
        else:
            ax.plot(f_orig, mag_orig, color="grey", linewidth=0.8, label="Input")
            ax.plot(f_filt, mag_filt, color="k", linewidth=0.8, label="Filtered")
            ax.set_ylabel("Magnitude")
        ax.set_title("FFT Magnitude Spectrum")

    def disable_controls(self):
        """Disable all controls on this tab (for previous filter stages)."""
        for w in self.findChildren(QPushButton):
            w.setEnabled(False)
        for w in self.findChildren(QLineEdit):
            w.setEnabled(False)
        for w in self.findChildren(QRadioButton):
            w.setEnabled(False)
        for w in self.findChildren(QComboBox):
            w.setEnabled(False)

    def enable_controls(self):
        """Re-enable all controls on this tab."""
        for w in self.findChildren(QPushButton):
            w.setEnabled(True)
        for w in self.findChildren(QLineEdit):
            w.setEnabled(True)
        for w in self.findChildren(QRadioButton):
            w.setEnabled(True)
        for w in self.findChildren(QComboBox):
            w.setEnabled(True)

    def get_info(self):
        """Return a dict describing the filter applied in this tab."""
        if self.filter_type == "bandpass" and self.bp_mode == "freqBW":
            bp_desc = f"Bandpass: {self.bp_center:.3f} Hz (BW={self.bp_bw:.1f} Hz)"
        else:
            bp_desc = f"Bandpass: {self.cutoff1:.3f} - {self.cutoff2:.3f} Hz"

        if self.filter_type == "notch" and self.notch_mode == "freqBW":
            notch_desc = f"Notch: {self.cutoff1:.3f} Hz (BW={self.notch_bw:.1f} Hz)"
        else:
            notch_desc = f"Notch: {self.cutoff1:.3f} - {self.cutoff2:.3f} Hz"

        desc_map = {
            "lowpass": f"Lowpass cutoff: {self.cutoff1:.3f} Hz",
            "highpass": f"Highpass cutoff: {self.cutoff1:.3f} Hz",
            "bandpass": bp_desc,
            "notch": notch_desc,
        }
        return {
            "type": self.filter_type,
            "cutoff1": self.cutoff1,
            "cutoff2": self.cutoff2,
            "notch_mode": self.notch_mode,
            "notch_bw": self.notch_bw,
            "bp_mode": self.bp_mode,
            "bp_center": self.bp_center,
            "bp_bw": self.bp_bw,
            "description": desc_map.get(self.filter_type, "Unknown"),
        }

    def set_state(self, info):
        """Restore filter params + widgets from a get_info() dict (used by
        the multi-file dialog to replay the chain when the preview file/zone
        changes)."""
        self.filter_type = info.get("type", "lowpass")
        self.cutoff1 = float(info.get("cutoff1", self.cutoff1))
        self.cutoff2 = float(info.get("cutoff2", self.cutoff2))
        self.notch_mode = info.get("notch_mode", "freqBW")
        self.notch_bw = float(info.get("notch_bw", 2.0))
        self.bp_mode = info.get("bp_mode", "lohi")
        self.bp_center = float(info.get("bp_center", self.bp_center))
        self.bp_bw = float(info.get("bp_bw", self.bp_bw))

        radio = {
            "lowpass": self._radio_lp,
            "highpass": self._radio_hp,
            "bandpass": self._radio_bp,
            "notch": self._radio_notch,
        }[self.filter_type]
        if self.filter_type == "notch":
            self.notch_freq = self.cutoff1
        radio.setChecked(True)
        self._on_type_changed(radio)

        if self.filter_type in ("notch", "bandpass"):
            mode = self.notch_mode if self.filter_type == "notch" else self.bp_mode
            mode_btn = self._radio_freqbw if mode == "freqBW" else self._radio_lohi
            mode_btn.setChecked(True)
            self._on_mode_changed(mode_btn)

        if self.filter_type == "bandpass" and self.bp_mode == "freqBW":
            self._edit_cut1.setText(f"{self.bp_center:.1f}")
            self._edit_bw.setText(f"{self.bp_bw:.1f}")
        elif self.filter_type == "notch" and self.notch_mode == "freqBW":
            self._edit_cut1.setText(f"{self.cutoff1:.1f}")
            self._edit_bw.setText(f"{self.notch_bw:.1f}")
        else:
            self._edit_cut1.setText(f"{self.cutoff1:.1f}")
            self._edit_cut2.setText(f"{self.cutoff2:.1f}")


# ──────────────────── Filter Dialog ────────────────────


class _FilterDialog(QDialog):
    """Multi-stage filter dialog with tabs, matching MATLAB FilterUI."""

    def __init__(
        self,
        data,
        fs,
        zone_num,
        total_zones,
        default_cutoff,
        ds_factor=None,
        parent=None,
    ):
        super().__init__(parent)
        self.fs = fs
        self.zone_num = zone_num
        self.total_zones = total_zones
        self.default_cutoff = default_cutoff
        self.ds_factor = ds_factor

        title = "Filtering"
        if total_zones > 1:
            title = f"Filtering - Zone {zone_num} of {total_zones}"
        self.setWindowTitle(title)
        self.resize(1400, 800)

        self.confirmed = False
        self.tabs = []
        self.original_data = data

        from utils.dialog_style import apply_dialog_style

        apply_dialog_style(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close)
        layout.addWidget(self.tab_widget)

        self.add_filter_tab(data)

    def add_filter_tab(self, input_data):
        """Add a new filter stage tab."""
        # Disable controls on previous tab
        if self.tabs:
            prev = self.tabs[-1]
            prev.disable_controls()
            prev_info = prev.get_info()
            idx = len(self.tabs)
            self.tab_widget.setTabText(idx - 1, f"{idx}. {prev_info['description']}")

        tab = _FilterTab(
            input_data,
            self.fs,
            self.zone_num,
            self.total_zones,
            self.default_cutoff,
            self,
        )
        self.tabs.append(tab)
        stage = len(self.tabs)
        self.tab_widget.addTab(tab, f"Filter {stage}")
        self.tab_widget.setCurrentWidget(tab)

    def _on_tab_close(self, index):
        """Remove the filter tab at *index* and all subsequent tabs.

        The chain after the removed tab is invalid, so we discard those tabs
        and re-enable the new last tab so the user can continue editing.
        If only one tab remains open, closing is not allowed.
        """
        if len(self.tabs) <= 1:
            return  # keep at least one tab

        # Remove tabs from index to end (reverse order to keep indices stable)
        for i in range(len(self.tabs) - 1, index - 1, -1):
            self.tab_widget.removeTab(i)
            self.tabs.pop(i)

        if self.tabs:
            # Re-enable the new last tab and make it active
            last = self.tabs[-1]
            last.enable_controls()
            self.tab_widget.setCurrentWidget(last)
            # Restore active tab title
            stage = len(self.tabs)
            self.tab_widget.setTabText(stage - 1, f"Filter {stage}")
        else:
            # All tabs removed – create a fresh one with original data
            self.add_filter_tab(self.original_data)

    def confirm(self):
        self.confirmed = True
        self.accept()

    def get_results(self):
        """Return (filtered_data, info_list)."""
        last_tab = self.tabs[-1]
        info_list = [t.get_info() for t in self.tabs]
        return last_tab.current_data, info_list


# ──────────────────── Public API ────────────────────


def filter_ui(data, sample_rate, filter_history=None, initial_cutoff=0, ds_factor=None):
    """Interactive filter UI - lets user design and apply a filter chain.

    Parameters
    ----------
    data : np.ndarray
        1-D or 2-D array (time x channels).
    sample_rate : float
        Sampling rate in Hz (effective rate of *data*).
    filter_history : list[dict], optional
        Previously applied filters to accumulate in the returned info.
    initial_cutoff : float, optional
        Initial lowpass cutoff in Hz.
    ds_factor : float, optional
        Downsample factor applied upstream; shown in the Information group
        (base SR = sample_rate * ds_factor).

    Returns
    -------
    filtered_data : np.ndarray
        Filtered data (same shape as input).
    filter_info : list[dict]
        List of dicts describing each applied filter.
    """
    if filter_history is None:
        filter_history = []

    data = np.atleast_1d(data).copy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    num_channels = data.shape[1]
    nyq = sample_rate / 2
    if initial_cutoff > 0:
        default_cutoff = min(initial_cutoff, nyq - 1)
    else:
        default_cutoff = min(1000, nyq - 1)

    filtered_data = np.zeros_like(data)
    all_info = list(filter_history)

    from utils.app_logger import logger

    for ch in range(num_channels):
        dlg = _FilterDialog(
            data[:, ch],
            sample_rate,
            ch + 1,
            num_channels,
            default_cutoff,
            ds_factor=ds_factor,
        )
        dlg.exec()

        if not dlg.confirmed:
            raise RuntimeError("User closed Filter UI without confirming.")

        filt_ch, info_ch = dlg.get_results()
        filtered_data[:, ch] = filt_ch

        if ch == 0:
            all_info = list(filter_history) + info_ch

        descs = ", ".join(fi["description"] for fi in info_ch)
        if num_channels > 1:
            logger.info(f"Filter zone {ch + 1}/{num_channels}: {descs}")
        else:
            logger.info(f"Applied filters: {descs}")

    if num_channels > 1:
        logger.info("All zones filtered.")
    else:
        filtered_data = filtered_data.ravel()

    return filtered_data, all_info
