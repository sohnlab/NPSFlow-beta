"""Wavelet Editor: pure helpers and the interactive QDialog.

See docs/superpowers/specs/2026-04-24-wavelet-editor-design.md
"""

import json
import os

import numpy as np

from utils.paths import saved_templates_dir


def derive_segments(tp_settings):
    """Return the initial segment list derived from a TPsettings struct.

    Each segment is a dict with keys: width (int), amplitude (float),
    orig_width (int), orig_amplitude (float). Segments span
    [peak_locations[i], peak_locations[i+1]) within pulse_template_rec.

    Raises ValueError if the struct lacks the data needed to define
    at least one segment.
    """
    if tp_settings is None:
        raise ValueError("TPsettings is None")

    rec = tp_settings.get("pulse_template_rec")
    if rec is None or len(np.asarray(rec)) == 0:
        raise ValueError("TPsettings.pulse_template_rec missing or empty")
    rec = np.asarray(rec).ravel()

    peaks = tp_settings.get("peak_locations")
    if peaks is None:
        raise ValueError("TPsettings.peak_locations missing")
    peaks = np.asarray(peaks).ravel().astype(int)

    # Pulse_template_rec is built from breakpoints [0, *peak_locations, len(rec)],
    # giving len(peak_locations) + 1 segments. peak_locations stores only the
    # interior peaks, so we prepend 0 and append len(rec) here to recover the
    # full segment list.
    n = len(rec)
    edges = [0]
    edges.extend(int(p) for p in peaks if 0 < int(p) < n)
    edges.append(n)
    breaks = np.asarray(sorted(set(edges)), dtype=int)

    if len(breaks) < 2:
        raise ValueError(
            "Cannot derive any segment: pulse_template_rec is empty after "
            "removing duplicate / out-of-range peak_locations.")

    segments = []
    for i in range(len(breaks) - 1):
        start = int(breaks[i])
        end = int(breaks[i + 1])
        width = end - start
        if width <= 0:
            raise ValueError(
                f"Non-positive segment width at index {i}: "
                f"breaks[{i}]={start}, breaks[{i+1}]={end}")
        amplitude = float(rec[start])
        segments.append({
            "width": int(width),
            "amplitude": amplitude,
            "orig_width": int(width),
            "orig_amplitude": amplitude,
        })
    return segments


def reconstruct(segments, trim=None):
    """Build the full step-function vector then apply trim.

    segments: list of dicts with at least 'width' and 'amplitude'.
    trim: dict {'left_sample': int, 'right_sample': int} into the
        pre-trim vector. None means no trim.

    Returns a 1-D float numpy array.
    """
    if not segments:
        return np.zeros(0, dtype=float)
    parts = [np.full(int(s["width"]), float(s["amplitude"]), dtype=float)
             for s in segments]
    full = np.concatenate(parts)
    if trim is None:
        return full
    left = int(trim.get("left_sample", 0))
    right = int(trim.get("right_sample", len(full)))
    left = max(0, min(left, len(full)))
    right = max(left, min(right, len(full)))
    return full[left:right]


def apply_trim_to_segments(segments, trim):
    """Return (kept_widths, kept_amplitudes) after applying trim.

    Segments fully outside the trim range are dropped. A segment that
    straddles a trim edge contributes only its kept portion.
    """
    widths = []
    amps = []
    cursor = 0
    left = int(trim.get("left_sample", 0)) if trim else 0
    right = int(trim.get("right_sample",
                         sum(int(s["width"]) for s in segments))) if trim \
        else sum(int(s["width"]) for s in segments)
    for s in segments:
        w = int(s["width"])
        seg_start = cursor
        seg_end = cursor + w
        cursor = seg_end
        kept_start = max(seg_start, left)
        kept_end = min(seg_end, right)
        kept = kept_end - kept_start
        if kept <= 0:
            continue
        widths.append(int(kept))
        amps.append(float(s["amplitude"]))
    return widths, amps


NORMALIZATION_MODES = ("none", "l2", "wavelet", "peak1", "peak_signed")


def normalize(vector, mode):
    """Apply one of the normalization modes from the spec.

    Returns (normalized_vector, applied) where applied is False when the
    requested normalization was skipped (e.g., all-zero input). The
    requested mode is what callers should record in outputs regardless.
    """
    v = np.asarray(vector, dtype=float)
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    if mode == "none" or v.size == 0:
        return v.copy(), True
    if mode == "l2":
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return v.copy(), False
        return v / n, True
    if mode == "wavelet":
        c = v - float(np.mean(v))
        n = float(np.linalg.norm(c))
        if n == 0.0:
            return v.copy(), False
        return c / n, True
    if mode == "peak1":
        m = float(np.max(np.abs(v)))
        if m == 0.0:
            return v.copy(), False
        return v / m, True
    # peak_signed: divide by the signed value at argmax(|v|)
    idx = int(np.argmax(np.abs(v)))
    denom = float(v[idx])
    if denom == 0.0:
        return v.copy(), False
    return v / denom, True


def save_state(segments, normalization, n_segments_source, filepath,
               kept_rows=None):
    """Atomically write editor state to *filepath* (JSON).

    Casts all numpy scalars to Python ints/floats per project convention.
    `kept_rows` (optional) records which original-segment indices survived
    deletion in the dialog, so the next session can restore the same set.
    """
    payload = {
        "segments": [
            {"width": int(s["width"]), "amplitude": float(s["amplitude"])}
            for s in segments
        ],
        "normalization": str(normalization),
        "n_segments_source": int(n_segments_source),
    }
    if kept_rows is not None:
        payload["kept_rows"] = [int(k) for k in kept_rows]
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, filepath)


def load_state(filepath):
    """Load editor state from JSON. Returns the payload dict as-is."""
    with open(filepath, "r") as f:
        return json.load(f)


from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFileDialog, QMessageBox, QWidget,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure


_NORM_LABELS = [
    ("none",         "None"),
    ("l2",           "Unit L2 norm"),
    ("wavelet",      "Zero-mean + unit L2 (wavelet)"),
    ("peak1",        "Peak magnitude = 1"),
    ("peak_signed",  "Dominant peak = +1"),
]


class WaveletEditorDialog(QDialog):
    """Interactive editor for the rectangularized template.

    Internal state:
      self._segments       : list of {width, amplitude, orig_width, orig_amplitude}
      self._norm_mode      : str (one of NORMALIZATION_MODES)
      self._selected_rows  : set[int] of segment indices currently selected
                             in the table; rendered as a yellow highlight
                             on the plot.

    Trimming is done by selecting one or more rows in the table and
    pressing Delete (or the Delete button); the segments are removed
    from the output entirely. There is no separate trim window.
    """

    def __init__(self, segments, norm_mode="wavelet", current_path=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wavelet Editor")
        self.resize(1200, 700)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._original_segments = [dict(s) for s in segments]
        self._segments = [dict(s) for s in segments]
        self._norm_mode = norm_mode
        self._selected_rows = set()
        self._current_path = current_path
        if current_path:
            self.setWindowTitle(
                f"Wavelet Editor — {os.path.basename(current_path)}")

        self._build_ui()
        self._drag = None  # dict describing the in-progress drag, or None
        self._cid_press = self._canvas.mpl_connect(
            "button_press_event", self._on_press)
        self._cid_move = self._canvas.mpl_connect(
            "motion_notify_event", self._on_motion)
        self._cid_release = self._canvas.mpl_connect(
            "button_release_event", self._on_release)
        self._hit_px = 6  # pixel hit radius
        self._refresh_table()
        self._update_confirm_enabled()
        self._redraw()

    # --- UI construction ---------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left: plot
        self._fig = Figure(figsize=(8, 6))
        style_mpl_figure(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax_norm = None  # secondary axis, created when overlay is shown
        self._canvas = FigureCanvas(self._fig)
        main_layout.addWidget(self._canvas, stretch=4)

        # Right: side panel — Segments → Normalization → Saved settings →
        # Confirm/Close, mirroring the Pulse Detection (BC) layout.
        panel = QWidget()
        panel.setFixedWidth(280)
        panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        panel.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(panel)

        # 1. Segments
        segments_box = QGroupBox("Segments")
        segments_layout = QVBoxLayout(segments_box)
        seg_buttons = QHBoxLayout()
        self._delete_btn = QPushButton("Delete selected")
        self._delete_btn.clicked.connect(self._on_delete_selected)
        self._delete_btn.setEnabled(False)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._on_reset)
        seg_buttons.addWidget(self._delete_btn)
        seg_buttons.addWidget(self._reset_btn)
        segments_layout.addLayout(seg_buttons)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["#", "Width", "Amplitude"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.itemChanged.connect(self._on_table_edited)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        segments_layout.addWidget(self._table, 1)
        panel_layout.addWidget(segments_box, 1)

        panel_layout.addSpacing(5)

        # 2. Normalization
        norm_box = QGroupBox("Normalization")
        norm_layout = QVBoxLayout(norm_box)
        self._norm_combo = QComboBox()
        for key, label in _NORM_LABELS:
            self._norm_combo.addItem(label, key)
        self._norm_combo.setCurrentIndex(
            [k for k, _ in _NORM_LABELS].index(self._norm_mode))
        self._norm_combo.currentIndexChanged.connect(self._on_norm_changed)
        norm_layout.addWidget(self._norm_combo)
        panel_layout.addWidget(norm_box)

        panel_layout.addSpacing(5)

        # 3. Saved settings
        settings_box = QGroupBox("Saved settings")
        settings_layout = QHBoxLayout(settings_box)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn = QPushButton("Save as…")
        self._save_as_btn.clicked.connect(self._on_save_as)
        self._load_btn = QPushButton("Load…")
        self._load_btn.clicked.connect(self._on_load)
        settings_layout.addWidget(self._save_btn)
        settings_layout.addWidget(self._save_as_btn)
        settings_layout.addWidget(self._load_btn)
        panel_layout.addWidget(settings_box)

        panel_layout.addSpacing(10)

        # 4. Confirm / Close
        bottom = QHBoxLayout()
        cancel = QPushButton("Close")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(cancel)
        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setObjectName("confirmBtn")
        self._confirm_btn.setDefault(True)
        self._confirm_btn.clicked.connect(self.accept)
        bottom.addWidget(self._confirm_btn)
        panel_layout.addLayout(bottom)

    # --- Table sync --------------------------------------------------------

    def _refresh_table(self):
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._segments))
        for i, s in enumerate(self._segments):
            idx = QTableWidgetItem(str(i + 1))
            idx.setFlags(idx.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 0, idx)
            self._table.setItem(i, 1, QTableWidgetItem(str(int(s["width"]))))
            self._table.setItem(i, 2,
                                QTableWidgetItem(f"{float(s['amplitude']):.6g}"))
        self._table.blockSignals(False)

    def _on_table_edited(self, item):
        row = item.row()
        col = item.column()
        if row >= len(self._segments):
            return
        try:
            if col == 1:
                w = int(float(item.text()))
                if w < 1:
                    raise ValueError
                self._segments[row]["width"] = w
            elif col == 2:
                self._segments[row]["amplitude"] = float(item.text())
        except ValueError:
            self._refresh_table()
            return
        self._update_confirm_enabled()
        self._redraw()

    def _on_table_selection(self):
        rows = {idx.row() for idx in
                self._table.selectionModel().selectedRows()}
        # Fall back to current cell row when no full-row selection exists
        # (e.g. while the user is mid-edit).
        if not rows:
            current = self._table.currentRow()
            if current >= 0:
                rows = {current}
        self._selected_rows = {r for r in rows
                               if 0 <= r < len(self._segments)}
        self._delete_btn.setEnabled(bool(self._selected_rows))
        self._redraw()

    # --- Normalization / Reset / Delete ----------------------------------

    def _on_norm_changed(self, _idx):
        self._norm_mode = self._norm_combo.currentData()
        self._redraw()

    def _on_reset(self):
        for s in self._segments:
            s["width"] = int(s["orig_width"])
            s["amplitude"] = float(s["orig_amplitude"])
        self._refresh_table()
        self._update_confirm_enabled()
        self._redraw()

    def _on_delete_selected(self):
        if not self._selected_rows:
            return
        self._segments = [s for i, s in enumerate(self._segments)
                          if i not in self._selected_rows]
        self._selected_rows = set()
        self._delete_btn.setEnabled(False)
        self._refresh_table()
        self._update_confirm_enabled()
        self._redraw()

    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                and self._table.hasFocus() and self._selected_rows):
            self._on_delete_selected()
            return
        super().keyPressEvent(event)

    def _update_confirm_enabled(self):
        self._confirm_btn.setEnabled(len(self._segments) > 0)

    # --- Save / Save As / Load ------------------------------------------

    def _saved_settings_dir(self):
        return saved_templates_dir("Wavelet")

    def _kept_rows(self):
        """Indices into self._original_segments that survived deletion."""
        kept = []
        j = 0
        for i, o in enumerate(self._original_segments):
            if j >= len(self._segments):
                break
            s = self._segments[j]
            if (int(s.get("orig_width", -1)) == int(o["orig_width"])
                    and float(s.get("orig_amplitude", float("nan")))
                    == float(o["orig_amplitude"])):
                kept.append(i)
                j += 1
        return kept

    def _write_state(self, path):
        save_state(self._segments, self._norm_mode,
                   n_segments_source=len(self._original_segments),
                   filepath=path,
                   kept_rows=self._kept_rows())

    def _on_save(self):
        if not self._current_path:
            self._on_save_as()
            return
        try:
            self._write_state(self._current_path)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self._show_status(f"Saved → {os.path.basename(self._current_path)}")

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Wavelet Template",
            self._saved_settings_dir(), "JSON (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            self._write_state(path)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self._current_path = path
        self._show_status(f"Saved → {os.path.basename(path)}")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Wavelet Template",
            self._saved_settings_dir(), "JSON (*.json)")
        if not path:
            return
        try:
            loaded = load_state(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        if loaded.get("n_segments_source") != len(self._original_segments):
            QMessageBox.warning(
                self, "Segment count mismatch",
                f"Saved file has n_segments_source="
                f"{loaded.get('n_segments_source')} but current template "
                f"has {len(self._original_segments)} segments. "
                "Cannot apply this file.")
            return
        # Rebuild from original, then apply edits + kept_rows + normalization
        try:
            new_segments = [dict(s) for s in self._original_segments]
            for s, ls in zip(new_segments, loaded.get("segments", [])):
                s["width"] = int(ls["width"])
                s["amplitude"] = float(ls["amplitude"])
            kept = loaded.get("kept_rows")
            if kept is not None:
                kept_set = {int(k) for k in kept}
                new_segments = [s for i, s in enumerate(new_segments)
                                if i in kept_set]
        except (KeyError, TypeError, ValueError) as e:
            QMessageBox.warning(self, "Load failed",
                                f"Malformed saved state: {e}")
            return

        self._segments = new_segments
        if "normalization" in loaded:
            self._norm_mode = loaded["normalization"]
            try:
                idx = [k for k, _ in _NORM_LABELS].index(self._norm_mode)
                self._norm_combo.blockSignals(True)
                self._norm_combo.setCurrentIndex(idx)
                self._norm_combo.blockSignals(False)
            except ValueError:
                pass
        self._current_path = path
        self._selected_rows = set()
        self._delete_btn.setEnabled(False)
        self._refresh_table()
        self._update_confirm_enabled()
        self._redraw()
        self._show_status(f"Loaded {os.path.basename(path)}")

    def _show_status(self, msg):
        # Surface in the window title since there's no dedicated status bar.
        base = "Wavelet Editor"
        if self._current_path:
            base = f"Wavelet Editor — {os.path.basename(self._current_path)}"
        self.setWindowTitle(f"{base}    [{msg}]")

    # --- Plotting ----------------------------------------------------------

    def _redraw(self):
        self._ax.clear()
        if self._ax_norm is not None:
            self._ax_norm.remove()
            self._ax_norm = None

        full = reconstruct(self._segments, trim=None)
        x = np.arange(len(full))
        self._ax.step(x, full, where="post", color="#444", linewidth=1.5,
                      label="edited (raw)")

        # Highlight selected segments
        if self._selected_rows and self._segments:
            cursor = 0
            for i, s in enumerate(self._segments):
                w = int(s["width"])
                if i in self._selected_rows:
                    self._ax.axvspan(cursor, cursor + w,
                                     color="#ffeb3b", alpha=0.35, zorder=0)
                cursor += w

        # Normalized overlay (over the full kept signal)
        if self._norm_mode != "none" and len(full) > 0:
            norm_v, _ok = normalize(full, self._norm_mode)
            self._ax_norm = self._ax.twinx()
            self._ax_norm.step(np.arange(len(norm_v)), norm_v, where="post",
                               color="#d32f2f", linewidth=0.9, alpha=0.8,
                               label="normalized")
            self._ax_norm.set_ylabel("normalized")
            # Keep _ax on top for mouse-event capture, transparent so the
            # twinx data shows through.
            self._ax.set_zorder(self._ax_norm.get_zorder() + 1)
            self._ax.patch.set_visible(False)

        self._ax.set_xlabel("sample")
        self._ax.set_ylabel("amplitude")
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # --- Public accessors -------------------------------------------------

    def result_state(self):
        """Return (segments, norm_mode) reflecting current edits."""
        return ([dict(s) for s in self._segments], self._norm_mode)

    # --- Drag interactions ------------------------------------------------

    def _segment_extents_samples(self):
        """Return list of (start_sample, end_sample) per segment."""
        out = []
        cursor = 0
        for s in self._segments:
            w = int(s["width"])
            out.append((cursor, cursor + w))
            cursor += w
        return out

    def _data_x_to_pixels(self, xdata):
        return self._ax.transData.transform((xdata, 0))[0]

    def _hit_test(self, event):
        """Return ('boundary'|'top', payload) or None.

        Priority: boundaries > tops.
        """
        if event.inaxes is not self._ax or event.x is None:
            return None
        ext = self._segment_extents_samples()
        if not ext:
            return None
        # Internal boundaries (between segments i and i+1)
        for i in range(len(ext) - 1):
            px = self._data_x_to_pixels(ext[i][1])
            if abs(event.x - px) <= self._hit_px:
                return ("boundary", i)
        # Top edges (segment whose x range contains the cursor and whose
        # amplitude is near the cursor's y)
        for i, (a, b) in enumerate(ext):
            if a <= event.xdata <= b:
                amp = self._segments[i]["amplitude"]
                py = self._ax.transData.transform((0, amp))[1]
                if abs(event.y - py) <= self._hit_px:
                    return ("top", i)
                break
        return None

    def _on_press(self, event):
        if event.button != 1:
            return
        hit = self._hit_test(event)
        if hit is None:
            return
        kind, payload = hit
        self._drag = {"kind": kind, "payload": payload,
                      "x0": event.xdata, "y0": event.ydata}
        if kind == "boundary":
            i = payload
            self._drag["w_left0"] = int(self._segments[i]["width"])
            self._drag["w_right0"] = int(self._segments[i + 1]["width"])

    def _on_motion(self, event):
        if self._drag is None or event.inaxes is not self._ax:
            return
        kind = self._drag["kind"]
        if kind == "top":
            self._segments[self._drag["payload"]]["amplitude"] = \
                float(event.ydata)
        elif kind == "boundary":
            i = self._drag["payload"]
            dx = int(round(event.xdata - self._drag["x0"]))
            new_left = max(1, self._drag["w_left0"] + dx)
            new_right = max(1, self._drag["w_right0"] - dx)
            # If clamping kicked in on one side, also clamp the other so
            # total width is preserved.
            if new_left + new_right != \
                    self._drag["w_left0"] + self._drag["w_right0"]:
                if new_left == 1:
                    new_right = self._drag["w_left0"] + \
                        self._drag["w_right0"] - 1
                if new_right == 1:
                    new_left = self._drag["w_left0"] + \
                        self._drag["w_right0"] - 1
            self._segments[i]["width"] = new_left
            self._segments[i + 1]["width"] = new_right
        else:
            return
        self._refresh_table()
        self._redraw()

    def _on_release(self, event):
        self._drag = None
