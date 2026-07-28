"""Causal filter design UI and application.

Same interaction model as Filter UI (stacked filter tabs, time + spectrum
preview), but every stage is applied forward-only (sosfilt/lfilter with
steady-state initial conditions), never filtfilt — so the output is exactly
what a streaming front-end would produce sample by sample. Use this to
prepare training data for real-time models: a chain designed here can be
replayed causally on a live stream with identical results.

Input is a list of (name, 2-D array time x zones) files; the chain is
designed on one selectable file/zone and the confirmed config is applied to
all files by the runner (see runners/runCausalFilterUI.py).

Also hosts the shared multi-file dialog machinery: _MultiFileFilterDialog
is the zero-phase variant, used by Filter UI when fed a multi-file struct
(see runners/runFilterUI.py).
"""

import numpy as np
from scipy import signal as sig

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit

from processing.filter_ui import _FilterDialog, _FilterTab


# ──────────────────── Causal application ────────────────────


def apply_causal_filter(
    data, fs, ftype, cutoff1, cutoff2, notch_mode, notch_bw, order=2
):
    """Apply one causal filter stage to a 1-D signal.

    Butterworth stages use SOS + sosfilt; the notch uses iirnotch + lfilter.
    Initial conditions are the steady-state response to data[0], so the
    output does not ramp from zero (no fake startup transient).
    """
    fs = float(fs)
    nyq = fs / 2
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return x

    if ftype == "lowpass":
        wn = np.clip(cutoff1 / nyq, 1e-3, 0.99)
        sos = sig.butter(int(order), wn, btype="low", output="sos")
    elif ftype == "highpass":
        wn = np.clip(cutoff1 / nyq, 0.01, 0.999)
        sos = sig.butter(int(order), wn, btype="high", output="sos")
    elif ftype == "bandpass":
        wn_lo = max(cutoff1 / nyq, 0.01)
        wn_hi = min(cutoff2 / nyq, 0.99)
        if wn_lo >= wn_hi:
            wn_lo = wn_hi - 0.01
        sos = sig.butter(int(order), [wn_lo, wn_hi], btype="band", output="sos")
    elif ftype == "notch":
        if notch_mode == "freqBW":
            w0 = np.clip(cutoff1, 1, nyq - 1)
            bandwidth = max(notch_bw, 0.1)
        else:
            w0 = np.clip((cutoff1 + cutoff2) / 2, 1, nyq - 1)
            bandwidth = max(cutoff2 - cutoff1, 0.1)
        Q = max(w0 / bandwidth, 0.5)
        b, a = sig.iirnotch(w0, Q, fs=fs)
        zi = sig.lfilter_zi(b, a) * x[0]
        y, _ = sig.lfilter(b, a, x, zi=zi)
        return y
    else:
        return x

    zi = sig.sosfilt_zi(sos) * x[0]
    y, _ = sig.sosfilt(sos, x, zi=zi)
    return y


def apply_causal_chain(data, fs, chain):
    """Apply a filter chain causally to a 1-D or 2-D (time x zones) array.

    Chain entries use the Filter UI config schema (type/cutoff1/cutoff2/
    notch_mode/notch_bw) plus an optional 'order' (default 2).
    """
    x = np.atleast_1d(np.asarray(data, dtype=float))
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        y = x[:, ch]
        for fi in chain:
            y = apply_causal_filter(
                y,
                fs,
                fi["type"],
                fi.get("cutoff1", 0.0),
                fi.get("cutoff2", 0.0),
                fi.get("notch_mode", "freqBW"),
                fi.get("notch_bw", 2.0),
                order=int(fi.get("order", 2)),
            )
        out[:, ch] = y
    return out[:, 0] if was_1d else out


# ──────────────────── Tab (causal variant) ────────────────────


class _CausalFilterTab(_FilterTab):
    """Filter UI tab that applies its stage causally and exposes an
    order field (hidden for notch — the biquad order is fixed)."""

    def __init__(self, input_data, fs, default_cutoff, parent_dialog, order=2):
        self.order = int(order)
        super().__init__(input_data, fs, 1, 1, default_cutoff, parent_dialog)

    def _build_ui(self):
        super()._build_ui()
        cut_group = self._edit_cut1.parent()
        row = QHBoxLayout()
        self._lbl_order = QLabel("Order:")
        self._edit_order = QLineEdit(str(self.order))
        self._edit_order.setFixedWidth(90)
        row.addWidget(self._lbl_order)
        row.addWidget(self._edit_order)
        cut_group.layout().addLayout(row)

    def _on_type_changed(self, btn):
        super()._on_type_changed(btn)
        is_notch = self.filter_type == "notch"
        self._lbl_order.setVisible(not is_notch)
        self._edit_order.setVisible(not is_notch)

    def _read_order(self):
        try:
            self.order = max(1, min(8, int(float(self._edit_order.text()))))
        except ValueError:
            pass
        self._edit_order.setText(str(self.order))

    def _apply_filter(self):
        self._read_cutoffs()
        self._read_order()
        try:
            self.current_data = apply_causal_filter(
                self.input_data,
                self.fs,
                self.filter_type,
                self.cutoff1,
                self.cutoff2,
                self.notch_mode,
                self.notch_bw,
                order=self.order,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            from utils.app_logger import logger

            logger.warning(f"Causal filter failed: {e}")
            self.current_data = self.input_data.copy()

    def get_info(self):
        info = super().get_info()
        info["order"] = self.order
        info["causal"] = True
        return info

    def set_state(self, info):
        self.order = int(info.get("order", self.order))
        super().set_state(info)
        self._edit_order.setText(str(self.order))


# ──────────────────── Dialog (multi-file) ────────────────────


class _MultiFileFilterDialog(_FilterDialog):
    """Filter dialog with a preview file/zone selector. The chain is
    designed on the selected signal; switching the preview replays the
    current chain onto the new signal. Base class filters zero-phase
    (plain _FilterTab stages); see _CausalFilterDialog for the causal
    variant."""

    WINDOW_TITLE = "Filtering"
    NOTE = "Zero-phase filters — confirmed chain is applied to all {n} file(s)"

    def __init__(self, files, fs, default_cutoff, ds_factor=None, parent=None):
        self._files = files  # list of (name, 2-D array time x zones)
        self._file_idx = 0
        self._zone_idx = 0
        super().__init__(
            self._preview_signal(),
            fs,
            1,
            1,
            default_cutoff,
            ds_factor=ds_factor,
            parent=parent,
        )
        self.setWindowTitle(self.WINDOW_TITLE)
        self._insert_preview_row()

    def _preview_signal(self):
        arr = self._files[self._file_idx][1]
        z = min(self._zone_idx, arr.shape[1] - 1)
        return np.ascontiguousarray(arr[:, z], dtype=float)

    def _insert_preview_row(self):
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)

        row.addWidget(QLabel("Preview file:"))
        self._file_combo = QComboBox()
        for name, _ in self._files:
            self._file_combo.addItem(name)
        self._file_combo.setCurrentIndex(0)
        self._file_combo.currentIndexChanged.connect(self._on_preview_changed)
        row.addWidget(self._file_combo, 1)

        self._zone_combo = QComboBox()
        self._zone_lbl = QLabel("Zone:")
        row.addWidget(self._zone_lbl)
        row.addWidget(self._zone_combo)
        self._refresh_zone_combo()
        self._zone_combo.currentIndexChanged.connect(self._on_preview_changed)

        note = QLabel(self.NOTE.format(n=len(self._files)))
        note.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(note)

        self.layout().insertLayout(0, row)

    def _refresh_zone_combo(self):
        n_zones = self._files[self._file_idx][1].shape[1]
        self._zone_combo.blockSignals(True)
        self._zone_combo.clear()
        for z in range(n_zones):
            self._zone_combo.addItem(f"{z + 1}")
        self._zone_idx = min(self._zone_idx, n_zones - 1)
        self._zone_combo.setCurrentIndex(self._zone_idx)
        self._zone_combo.blockSignals(False)
        multi = n_zones > 1
        self._zone_lbl.setVisible(multi)
        self._zone_combo.setVisible(multi)

    def _make_tab(self, input_data):
        return _FilterTab(input_data, self.fs, 1, 1, self.default_cutoff, self)

    def add_filter_tab(self, input_data):
        if self.tabs:
            prev = self.tabs[-1]
            prev.disable_controls()
            prev_info = prev.get_info()
            idx = len(self.tabs)
            self.tab_widget.setTabText(idx - 1, f"{idx}. {prev_info['description']}")
        tab = self._make_tab(input_data)
        self.tabs.append(tab)
        stage = len(self.tabs)
        self.tab_widget.addTab(tab, f"Filter {stage}")
        self.tab_widget.setCurrentWidget(tab)

    def _on_preview_changed(self, _index=None):
        states = [t.get_info() for t in self.tabs]
        file_changed = self._file_combo.currentIndex() != self._file_idx
        self._file_idx = self._file_combo.currentIndex()
        if file_changed:
            # Rebuild zone options for the new file (clamps the zone index).
            self._refresh_zone_combo()
        self._zone_idx = self._zone_combo.currentIndex()

        active = self.tab_widget.currentIndex()
        old_tabs = self.tabs
        self.tab_widget.clear()
        self.tabs = []
        for t in old_tabs:
            t.deleteLater()
        data = self._preview_signal()
        self.original_data = data

        # Replay the chain stage by stage on the new preview signal.
        for st in states:
            self.add_filter_tab(data)
            tab = self.tabs[-1]
            tab.set_state(st)
            tab._apply_filter()
            tab._update_plots()
            data = tab.current_data
        if not self.tabs:
            self.add_filter_tab(data)
        if 0 <= active < len(self.tabs):
            self.tab_widget.setCurrentIndex(active)


class _CausalFilterDialog(_MultiFileFilterDialog):
    """Multi-file filter dialog with causal (forward-only) stages."""

    WINDOW_TITLE = "Causal Filtering"
    NOTE = (
        "Causal (forward-only) filters — confirmed chain is applied to all {n} file(s)"
    )

    def _make_tab(self, input_data):
        return _CausalFilterTab(input_data, self.fs, self.default_cutoff, self)


# ──────────────────── Public API ────────────────────


def multifile_filter_ui(
    files, sample_rate, ds_factor=None, initial_cutoff=0, causal=True
):
    """Interactive filter design over a set of files.

    Parameters
    ----------
    files : list of (name, np.ndarray)
        One entry per file; arrays are 1-D or 2-D (time x zones).
    sample_rate : float
        Sampling rate in Hz.
    causal : bool
        True -> forward-only stages (apply with apply_causal_chain);
        False -> zero-phase stages (apply with filter_ui.apply_filter_chain).

    Returns
    -------
    chain : list[dict]
        Filter chain config (Filter UI schema; causal chains add 'order').
    """
    prepared = []
    for name, arr in files:
        a = np.atleast_1d(np.asarray(arr, dtype=float))
        if a.ndim == 1:
            a = a[:, None]
        prepared.append((str(name), a))
    if not prepared:
        raise ValueError("No files with numeric data to filter.")

    nyq = sample_rate / 2
    if initial_cutoff > 0:
        default_cutoff = min(initial_cutoff, nyq - 1)
    else:
        default_cutoff = min(1000, nyq - 1)

    cls = _CausalFilterDialog if causal else _MultiFileFilterDialog
    dlg = cls(prepared, sample_rate, default_cutoff, ds_factor=ds_factor)
    dlg.exec()
    if not dlg.confirmed:
        raise RuntimeError(f"User closed {dlg.WINDOW_TITLE} UI without confirming.")

    _, chain = dlg.get_results()
    return chain


def causal_filter_ui(files, sample_rate, ds_factor=None, initial_cutoff=0):
    return multifile_filter_ui(
        files,
        sample_rate,
        ds_factor=ds_factor,
        initial_cutoff=initial_cutoff,
        causal=True,
    )
