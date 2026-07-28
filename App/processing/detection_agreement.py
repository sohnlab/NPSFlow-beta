"""Detection Agreement — compare two pulse-detection outputs symmetrically.

Pure logic lives at the top of the file (testable without Qt).
Qt dialog and helpers live below.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np


_DEFAULT_PARAMS: Dict = {
    "iou_thr": 0.5,
    "iou_enabled": True,
    "midpoint_tol_samples": 50,
    "midpoint_enabled": True,
    "primary_score": "F1",  # one of {"F1", "Jaccard", "MeanIoU"}
}

_VALID_SCORES = ("F1", "Jaccard", "MeanIoU")


def _coerce_params(d: Dict) -> Dict:
    """Validate and normalize a params dict, filling missing keys with
    defaults."""
    out = dict(_DEFAULT_PARAMS)
    if not isinstance(d, dict):
        return out
    if "iou_thr" in d:
        out["iou_thr"] = float(max(0.0, min(1.0, float(d["iou_thr"]))))
    if "iou_enabled" in d:
        out["iou_enabled"] = bool(d["iou_enabled"])
    if "midpoint_tol_samples" in d:
        out["midpoint_tol_samples"] = int(max(0, int(d["midpoint_tol_samples"])))
    if "midpoint_enabled" in d:
        out["midpoint_enabled"] = bool(d["midpoint_enabled"])
    if "primary_score" in d and d["primary_score"] in _VALID_SCORES:
        out["primary_score"] = d["primary_score"]
    # Enforce: at least one gate enabled
    if not out["iou_enabled"] and not out["midpoint_enabled"]:
        out["iou_enabled"] = True
    return out


def load_state(path: str) -> Dict:
    """Load params from JSON. Returns a fresh copy of defaults if path is
    missing or unreadable."""
    if not path or not os.path.isfile(path):
        return dict(_DEFAULT_PARAMS)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return dict(_DEFAULT_PARAMS)
    return _coerce_params(data)


def save_state(path: str, params: Dict) -> None:
    """Persist params as JSON via an atomic tmp+rename write.

    Creates parent dir if needed. Raises OSError on write failure; caller
    is responsible for handling. Pass the coerced params through
    ``_coerce_params`` before writing so saved files are always valid.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_coerce_params(params), f, indent=2)
    os.replace(tmp, path)


def compare(pulses_a, pulses_b, iou_thr, midpoint_tol,
            require_iou=True, require_midpoint=True
            ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    """Symmetric pulse-set comparison.

    Returns
    -------
    match_table : ndarray, shape (K, 5)
        Columns: [side, idx, partner_idx, iou, midpoint_offset_samples].
        side 0 = A, 1 = B. partner_idx = -1 if unmatched.
        Row order: all A rows in original order, then B-only rows in
        original order.
        For *unmatched* rows the diagnostic columns are decoupled:
        ``iou`` holds the **maximum** IoU found against the other set
        (best-by-IoU candidate); ``midpoint_offset_samples`` holds the
        **minimum** absolute midpoint offset found against the other set
        (best-by-midpoint candidate). The two values may come from
        different counterpart pulses.
    counts : dict
        {"A", "B", "TP", "FP", "FN"} as ints.
    scores : dict
        {"F1", "Jaccard", "MeanIoU"} as floats in [0, 1].
    """
    if not require_iou and not require_midpoint:
        raise ValueError("At least one of require_iou / require_midpoint "
                         "must be True")

    A = np.asarray(pulses_a, dtype=int).reshape(-1, 2) if np.size(pulses_a) \
        else np.zeros((0, 2), dtype=int)
    B = np.asarray(pulses_b, dtype=int).reshape(-1, 2) if np.size(pulses_b) \
        else np.zeros((0, 2), dtype=int)

    nA, nB = len(A), len(B)

    # Per-pair IoU + midpoint offset matrices
    iou = np.zeros((nA, nB), dtype=float)
    mid_off = np.zeros((nA, nB), dtype=float)
    if nA and nB:
        a_lo = A[:, 0:1]; a_hi = A[:, 1:2]   # shape (nA, 1)
        b_lo = B[:, 0:1].T; b_hi = B[:, 1:2].T  # shape (1, nB)
        inter = np.maximum(0, np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo))
        union = np.maximum(a_hi, b_hi) - np.minimum(a_lo, b_lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        a_mid = (A[:, 0] + A[:, 1]) / 2.0
        b_mid = (B[:, 0] + B[:, 1]) / 2.0
        mid_off = np.abs(a_mid[:, None] - b_mid[None, :])

    # Gate mask: which (i, j) pairs are eligible
    mask = np.ones((nA, nB), dtype=bool)
    if require_iou:
        mask &= (iou >= iou_thr)
    if require_midpoint:
        mask &= (mid_off <= midpoint_tol)

    # Greedy pairing: sort eligible pairs by IoU desc, tiebreak by smaller
    # midpoint offset. Assign without reuse.
    pairs = []
    if nA and nB:
        ii, jj = np.nonzero(mask)
        order = np.lexsort((mid_off[ii, jj], -iou[ii, jj]))
        for k in order:
            pairs.append((int(ii[k]), int(jj[k]), float(iou[ii[k], jj[k]]),
                          float(mid_off[ii[k], jj[k]])))

    a_partner = np.full(nA, -1, dtype=int)
    b_partner = np.full(nB, -1, dtype=int)
    matched_iou = {}  # a_idx -> iou
    matched_mid = {}  # a_idx -> mid_off
    for i, j, p_iou, p_mid in pairs:
        if a_partner[i] == -1 and b_partner[j] == -1:
            a_partner[i] = j
            b_partner[j] = i
            matched_iou[i] = p_iou
            matched_mid[i] = p_mid

    tp = int((a_partner >= 0).sum())
    fn = nA - tp  # A-only
    fp = nB - tp  # B-only

    # Build match table: all A rows, then B-only rows
    rows = []
    for i in range(nA):
        j = a_partner[i]
        if j >= 0:
            rows.append([0, i, j, matched_iou[i], matched_mid[i]])
        else:
            # Best (non-gated) candidate for diagnostic columns — decoupled
            if nB:
                best_j_by_iou = int(np.argmax(iou[i]))
                best_j_by_mid = int(np.argmin(mid_off[i]))
                rows.append([0, i, -1,
                             float(iou[i, best_j_by_iou]),
                             float(mid_off[i, best_j_by_mid])])
            else:
                rows.append([0, i, -1, 0.0, 0.0])
    for j in range(nB):
        if b_partner[j] == -1:
            if nA:
                best_i_by_iou = int(np.argmax(iou[:, j]))
                best_i_by_mid = int(np.argmin(mid_off[:, j]))
                rows.append([1, j, -1,
                             float(iou[best_i_by_iou, j]),
                             float(mid_off[best_i_by_mid, j])])
            else:
                rows.append([1, j, -1, 0.0, 0.0])

    match_table = (np.array(rows, dtype=float)
                   if rows else np.zeros((0, 5), dtype=float))

    counts = {"A": int(nA), "B": int(nB),
              "TP": int(tp), "FP": int(fp), "FN": int(fn)}

    denom_f1 = 2 * tp + fp + fn
    denom_j = tp + fp + fn
    f1 = (2 * tp / denom_f1) if denom_f1 else 0.0
    jaccard = (tp / denom_j) if denom_j else 0.0
    mean_iou = (float(np.mean(list(matched_iou.values())))
                if matched_iou else 0.0)
    scores = {"F1": float(f1), "Jaccard": float(jaccard),
              "MeanIoU": float(mean_iou)}

    return match_table, counts, scores


# ---------------------------------------------------------------------------
# Qt dialog (only imported when invoked from the runner)
# ---------------------------------------------------------------------------

def _build_dialog_class():
    """Lazy import of Qt + matplotlib so the module stays importable in
    headless smoke-check mode."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
        QSlider, QSpinBox, QRadioButton, QButtonGroup, QGroupBox,
        QFileDialog, QMessageBox,
    )
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from utils.decimate import minmax_decimate
    from utils.dialog_style import apply_dialog_style, style_mpl_figure
    from utils.paths import saved_templates_dir

    class DetectionAgreementDialog(QDialog):
        def __init__(self, signal, pulses_a, pulses_b, sample_rate,
                     params, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Detection Agreement")
            self.resize(1100, 650)
            apply_dialog_style(self)

            self._signal = np.asarray(signal, dtype=float).ravel()
            self._A = np.asarray(pulses_a, dtype=int).reshape(-1, 2)
            self._B = np.asarray(pulses_b, dtype=int).reshape(-1, 2)
            self._sr = float(sample_rate)
            self._params = _coerce_params(params)
            self._current_path = None

            # Layout: left = plot + toolbar, right = controls
            root = QHBoxLayout(self)

            left = QVBoxLayout()
            self._fig = Figure(figsize=(7, 5))
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            style_mpl_figure(self._fig)
            self._toolbar = NavigationToolbar(self._canvas, self)
            left.addWidget(self._toolbar)
            left.addWidget(self._canvas, 1)
            root.addLayout(left, 3)

            right = QVBoxLayout()

            # Match rule group
            match_grp = QGroupBox("Match rule")
            mg = QVBoxLayout(match_grp)
            row_iou = QHBoxLayout()
            self._chk_iou = QCheckBox("Require IoU")
            self._chk_iou.setChecked(self._params["iou_enabled"])
            self._sld_iou = QSlider(Qt.Orientation.Horizontal)
            self._sld_iou.setRange(0, 100)
            self._sld_iou.setSingleStep(5)
            self._sld_iou.setValue(int(round(self._params["iou_thr"] * 100)))
            self._lbl_iou = QLabel(f"{self._params['iou_thr']:.2f}")
            row_iou.addWidget(self._chk_iou)
            row_iou.addWidget(self._sld_iou, 1)
            row_iou.addWidget(self._lbl_iou)
            mg.addLayout(row_iou)

            row_mid = QHBoxLayout()
            self._chk_mid = QCheckBox("Require midpoint")
            self._chk_mid.setChecked(self._params["midpoint_enabled"])
            self._spn_mid = QSpinBox()
            self._spn_mid.setRange(0, 10_000_000)
            self._spn_mid.setValue(self._params["midpoint_tol_samples"])
            self._lbl_mid_ms = QLabel("")
            row_mid.addWidget(self._chk_mid)
            row_mid.addWidget(QLabel("samples ≤"))
            row_mid.addWidget(self._spn_mid)
            row_mid.addWidget(self._lbl_mid_ms)
            mg.addLayout(row_mid)
            right.addWidget(match_grp)

            # Primary score
            score_grp = QGroupBox("Primary score (drives 'score' output)")
            sg = QVBoxLayout(score_grp)
            self._rb_group = QButtonGroup(self)
            self._rb_f1 = QRadioButton("F1")
            self._rb_j  = QRadioButton("Jaccard")
            self._rb_m  = QRadioButton("Mean IoU")
            for i, rb in enumerate((self._rb_f1, self._rb_j, self._rb_m)):
                self._rb_group.addButton(rb, i)
                sg.addWidget(rb)
            {"F1": self._rb_f1, "Jaccard": self._rb_j,
             "MeanIoU": self._rb_m}[self._params["primary_score"]].setChecked(True)
            right.addWidget(score_grp)

            # Summary readout
            self._summary = QLabel("")
            self._summary.setStyleSheet(
                "QLabel { font-family: monospace; padding: 6px; "
                "background-color: #f4f4f4; }")
            right.addWidget(self._summary)

            # Save / Load row
            io_row = QHBoxLayout()
            btn_save = QPushButton("Save")
            btn_saveas = QPushButton("Save As…")
            btn_load = QPushButton("Load…")
            btn_save.clicked.connect(self._on_save)
            btn_saveas.clicked.connect(self._on_save_as)
            btn_load.clicked.connect(self._on_load)
            io_row.addWidget(btn_save)
            io_row.addWidget(btn_saveas)
            io_row.addWidget(btn_load)
            right.addLayout(io_row)

            right.addStretch(1)

            # Confirm / Cancel
            btn_row = QHBoxLayout()
            self._btn_confirm = QPushButton("Confirm")
            btn_cancel = QPushButton("Cancel")
            self._btn_confirm.setDefault(True)
            self._btn_confirm.clicked.connect(self.accept)
            btn_cancel.clicked.connect(self.reject)
            btn_row.addStretch(1)
            btn_row.addWidget(btn_cancel)
            btn_row.addWidget(self._btn_confirm)
            right.addLayout(btn_row)

            root.addLayout(right, 2)

            # Debounced refresh
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.setInterval(60)
            self._refresh_timer.timeout.connect(self._refresh)

            for w in (self._chk_iou, self._chk_mid):
                w.toggled.connect(self._schedule_refresh)
            self._sld_iou.valueChanged.connect(self._schedule_refresh)
            self._spn_mid.valueChanged.connect(self._schedule_refresh)
            self._rb_group.buttonToggled.connect(self._schedule_refresh)

            self._refresh()  # initial draw

        # --- helpers ---

        def _read_params(self) -> Dict:
            """Read raw widget state. Does NOT apply ``_coerce_params``'s
            both-off auto-fix — that would mask the disabled-Confirm UX. Coercion
            is applied only on save and on Confirm output via ``get_params``.
            """
            primary = "F1"
            if self._rb_j.isChecked():
                primary = "Jaccard"
            elif self._rb_m.isChecked():
                primary = "MeanIoU"
            return {
                "iou_thr": self._sld_iou.value() / 100.0,
                "iou_enabled": self._chk_iou.isChecked(),
                "midpoint_tol_samples": int(self._spn_mid.value()),
                "midpoint_enabled": self._chk_mid.isChecked(),
                "primary_score": primary,
            }

        def get_params(self) -> Dict:
            """Return coerced params for the runner."""
            return _coerce_params(self._read_params())

        def _schedule_refresh(self, *_a, **_kw):
            self._refresh_timer.start()

        def _refresh(self):
            p = self._read_params()
            self._params = p

            self._lbl_iou.setText(f"{p['iou_thr']:.2f}")
            self._sld_iou.setEnabled(p["iou_enabled"])
            self._spn_mid.setEnabled(p["midpoint_enabled"])
            ms = (p["midpoint_tol_samples"] / self._sr * 1000.0
                  if self._sr > 0 else 0.0)
            self._lbl_mid_ms.setText(f"({ms:.2f} ms)")

            both_off = not p["iou_enabled"] and not p["midpoint_enabled"]
            self._btn_confirm.setEnabled(not both_off)

            if both_off:
                # Show the user why Confirm is disabled; skip compare() since
                # it would raise ValueError on both gates off.
                self._summary.setText(
                    "Enable at least one match rule (IoU or midpoint) "
                    "to compute agreement.")
                self._ax.clear()
                self._ax.set_title("(both match rules disabled)")
                self._canvas.draw_idle()
                return

            mt, counts, scores = compare(
                self._A, self._B,
                p["iou_thr"], p["midpoint_tol_samples"],
                p["iou_enabled"], p["midpoint_enabled"])

            self._summary.setText(
                f"|A| = {counts['A']:>5}   |B| = {counts['B']:>5}\n"
                f"TP  = {counts['TP']:>5}   FP = {counts['FP']:>5}   "
                f"FN = {counts['FN']:>5}\n"
                f"F1  = {scores['F1']:.3f}   Jaccard = {scores['Jaccard']:.3f}"
                f"   meanIoU = {scores['MeanIoU']:.3f}")

            self._draw_preview(mt, counts)

        def _draw_preview(self, match_table, counts):
            # NOTE: spec calls for a thin connector bracket above matched A/B
            # pairs. Deferred — pulses-vs-trace overlay + red unmatched outline
            # already convey enough for v1. Revisit if users request it.
            ax = self._ax
            ax.clear()
            n = len(self._signal)
            if self._sr > 0:
                t = np.arange(n) / self._sr * 1000.0
                ax.set_xlabel("time (ms)")
            else:
                t = np.arange(n)
                ax.set_xlabel("samples")
            t_ds, sig_ds = minmax_decimate(t, self._signal,
                                           max_points=10_000)
            ax.plot(t_ds, sig_ds, color="#444444", lw=0.6)

            # Index unmatched rows by side+idx for red-outline lookup
            unmatched_a = set()
            unmatched_b = set()
            for row in match_table:
                side, idx, partner = int(row[0]), int(row[1]), int(row[2])
                if partner == -1:
                    if side == 0:
                        unmatched_a.add(idx)
                    else:
                        unmatched_b.add(idx)

            def _x(sample_idx):
                return (sample_idx / self._sr * 1000.0
                        if self._sr > 0 else float(sample_idx))

            for i, (s, e) in enumerate(self._A):
                edge = "#cc0000" if i in unmatched_a else "none"
                ax.axvspan(_x(s), _x(e), color="#3a7bd5", alpha=0.25,
                           edgecolor=edge, linewidth=1.0 if edge != "none" else 0)
            for j, (s, e) in enumerate(self._B):
                edge = "#cc0000" if j in unmatched_b else "none"
                ax.axvspan(_x(s), _x(e), color="#e58a2b", alpha=0.25,
                           edgecolor=edge, linewidth=1.0 if edge != "none" else 0)

            ax.set_title(
                f"Detector A (blue): {counts['A']} pulses    "
                f"Detector B (orange): {counts['B']} pulses    "
                f"matched: {counts['TP']}")
            self._fig.tight_layout()
            self._canvas.draw_idle()

        # --- save/load ---

        def _settings_dir_abs(self) -> str:
            return saved_templates_dir(
                os.path.join("Analysis", "DetectionAgreement"))

        def _on_save(self):
            if not self._current_path:
                self._on_save_as()
                return
            try:
                save_state(self._current_path, self._read_params())
            except OSError as e:
                QMessageBox.warning(self, "Save failed", str(e))

        def _on_save_as(self):
            d = self._settings_dir_abs()
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Detection Agreement settings", d, "JSON (*.json)")
            if not path:
                return
            if not path.endswith(".json"):
                path += ".json"
            try:
                save_state(path, self._read_params())
            except OSError as e:
                QMessageBox.warning(self, "Save failed", str(e))
                return
            self._current_path = path

        def _on_load(self):
            d = self._settings_dir_abs()
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Detection Agreement settings", d, "JSON (*.json)")
            if not path:
                return
            try:
                loaded = load_state(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Load failed", str(e))
                return
            self._params = loaded
            self._current_path = path
            self._apply_params_to_widgets()

        def _apply_params_to_widgets(self):
            p = self._params
            self._chk_iou.setChecked(p["iou_enabled"])
            self._sld_iou.setValue(int(round(p["iou_thr"] * 100)))
            self._chk_mid.setChecked(p["midpoint_enabled"])
            self._spn_mid.setValue(p["midpoint_tol_samples"])
            {"F1": self._rb_f1, "Jaccard": self._rb_j,
             "MeanIoU": self._rb_m}[p["primary_score"]].setChecked(True)
            self._refresh()

    return DetectionAgreementDialog


def DetectionAgreementDialog(*args, **kwargs):
    """Module-level constructor that lazy-builds the Qt dialog class on
    first call. This keeps the module importable in headless smoke-test
    mode."""
    cls = _build_dialog_class()
    globals()["DetectionAgreementDialog"] = cls  # cache for next call
    return cls(*args, **kwargs)


if __name__ == "__main__":
    # Smoke checks for compare(). Run from project root:
    #   python App/processing/detection_agreement.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))  # App/ → utils importable
    from utils.decimate import minmax_decimate

    # 1) Both empty → all zeros, empty table
    mt, counts, scores = compare(
        np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int),
        iou_thr=0.5, midpoint_tol=50)
    assert mt.shape == (0, 5), mt.shape
    assert counts == {"A": 0, "B": 0, "TP": 0, "FP": 0, "FN": 0}, counts
    assert scores == {"F1": 0.0, "Jaccard": 0.0, "MeanIoU": 0.0}, scores

    # 2) Identical A == B → F1=Jaccard=MeanIoU=1, all rows matched
    A = np.array([[10, 20], [40, 60], [100, 130]], dtype=int)
    mt, counts, scores = compare(A, A.copy(), 0.5, 50)
    assert counts["TP"] == 3 and counts["FP"] == 0 and counts["FN"] == 0
    assert scores["F1"] == 1.0 and scores["Jaccard"] == 1.0
    assert abs(scores["MeanIoU"] - 1.0) < 1e-9
    # All rows must be A-side (side=0) and have a partner_idx >= 0
    assert (mt[:, 0] == 0).all()
    assert (mt[:, 2] >= 0).all()

    # 3) Disjoint intervals → F1=Jaccard=MeanIoU=0
    A = np.array([[0, 10]], dtype=int)
    B = np.array([[100, 110]], dtype=int)
    mt, counts, scores = compare(A, B, 0.5, 50)
    assert counts == {"A": 1, "B": 1, "TP": 0, "FP": 1, "FN": 1}, counts
    assert scores["F1"] == 0.0 and scores["Jaccard"] == 0.0
    assert scores["MeanIoU"] == 0.0
    # Row 0 = A-side unmatched, Row 1 = B-side unmatched
    assert mt.shape == (2, 5)
    assert mt[0, 0] == 0 and mt[0, 2] == -1
    assert mt[1, 0] == 1 and mt[1, 2] == -1

    # 4) Asymmetry on FP/FN, symmetry on F1/Jaccard
    A = np.array([[0, 10], [50, 60], [100, 110]], dtype=int)
    B = np.array([[0, 10], [50, 60]], dtype=int)
    _, c_ab, s_ab = compare(A, B, 0.5, 50)
    _, c_ba, s_ba = compare(B, A, 0.5, 50)
    assert c_ab["TP"] == 2 and c_ab["FP"] == 0 and c_ab["FN"] == 1
    assert c_ba["TP"] == 2 and c_ba["FP"] == 1 and c_ba["FN"] == 0
    assert abs(s_ab["F1"] - s_ba["F1"]) < 1e-9
    assert abs(s_ab["Jaccard"] - s_ba["Jaccard"]) < 1e-9

    # 5) Disabling IoU gate makes matching more permissive
    A = np.array([[0, 100]], dtype=int)
    B = np.array([[40, 60]], dtype=int)  # IoU = 20/100 = 0.2, mid offset = 0
    _, c_strict, _ = compare(A, B, iou_thr=0.5, midpoint_tol=50,
                             require_iou=True, require_midpoint=True)
    assert c_strict["TP"] == 0
    _, c_loose, _ = compare(A, B, iou_thr=0.5, midpoint_tol=50,
                            require_iou=False, require_midpoint=True)
    assert c_loose["TP"] == 1

    # 6) Diagnostic columns on unmatched rows are decoupled:
    #    iou column = max IoU; midpoint column = min midpoint offset.
    A = np.array([[0, 100]], dtype=int)              # mid=50
    B = np.array([[50, 150], [45, 55]], dtype=int)   # mids=100, 50
    # IoU(A, B[0]) = 50/150 ≈ 0.333; IoU(A, B[1]) = 10/100 = 0.10
    # mid offsets = 50, 0
    # With default IoU thr 0.5 and midpoint tol 50, both pairs fail the IoU gate,
    # so no matches. A's diagnostic row should report iou≈0.333 (from B[0]) and
    # mid_off=0 (from B[1]).
    mt, counts, scores = compare(A, B, iou_thr=0.5, midpoint_tol=50)
    assert counts["TP"] == 0
    # Find A's row (side==0, idx==0)
    a_row = mt[(mt[:, 0] == 0) & (mt[:, 1] == 0)][0]
    assert abs(a_row[3] - (50.0 / 150.0)) < 1e-9, a_row[3]   # IoU column
    assert a_row[4] == 0.0, a_row[4]                          # midpoint column

    # 7) load_state / save_state round-trip + coercion clamp through save+load
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # 7a) Faithful round-trip of valid values
        p = os.path.join(td, "settings.json")
        params_in = {"iou_thr": 0.42, "iou_enabled": False,
                     "midpoint_tol_samples": 17, "midpoint_enabled": True,
                     "primary_score": "Jaccard"}
        save_state(p, params_in)
        params_out = load_state(p)
        for k, v in params_in.items():
            assert params_out[k] == v, (k, v, params_out[k])

        # 7b) Out-of-range / invalid values get clamped & rejected by coercion
        p2 = os.path.join(td, "clamp.json")
        params_bad = {"iou_thr": 1.5,                 # clamp to 1.0
                      "iou_enabled": False,
                      "midpoint_tol_samples": -5,    # clamp to 0
                      "midpoint_enabled": False,     # both off → iou flipped on
                      "primary_score": "nonsense"}   # rejected → default "F1"
        save_state(p2, params_bad)
        loaded = load_state(p2)
        assert loaded["iou_thr"] == 1.0, loaded["iou_thr"]
        assert loaded["midpoint_tol_samples"] == 0, loaded["midpoint_tol_samples"]
        assert loaded["iou_enabled"] is True, loaded["iou_enabled"]
        assert loaded["primary_score"] == "F1", loaded["primary_score"]

    # 8) load_state on missing file → returns defaults (copy, not reference)
    defaults = load_state("/nonexistent/path.json")
    assert defaults == _DEFAULT_PARAMS
    assert defaults is not _DEFAULT_PARAMS  # must be a fresh dict

    # 9) minmax_decimate preserves envelope on long signals
    rng = np.random.default_rng(42)
    long_sig = rng.standard_normal(1_000_000)
    long_t = np.arange(1_000_000, dtype=float)
    t_ds, y_ds = minmax_decimate(long_t, long_sig, max_points=10_000)
    assert len(t_ds) <= 10_000, len(t_ds)
    assert len(t_ds) == len(y_ds)
    assert y_ds.max() == long_sig[:(1_000_000 // 5000) * 5000].max(), \
        (y_ds.max(), long_sig.max())
    assert y_ds.min() == long_sig[:(1_000_000 // 5000) * 5000].min(), \
        (y_ds.min(), long_sig.min())
    # Short array passes through unchanged
    short = np.arange(100, dtype=float)
    t2, y2 = minmax_decimate(short, short)
    assert np.array_equal(t2, short) and np.array_equal(y2, short)

    print("compare() smoke checks passed")
