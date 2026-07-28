"""PySide6 dialog for selecting and running analysis methods on segment data.

Supported methods:
  - dR/R (fractional resistance change)
  - Amplitude distribution
  - Peak counting
  - Custom (user-defined Python expression)
"""

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QPushButton, QTextEdit, QGroupBox, QDialogButtonBox,
)


class AnalysisMethodDialog(QDialog):
    """Dialog for selecting an analysis method and viewing results."""

    METHODS = [
        "dR/R (Fractional Change)",
        "Amplitude Distribution",
        "Peak Counting",
        "Custom Expression",
    ]

    def __init__(self, segment_data, sample_rate=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Method Selector")
        self.setMinimumSize(600, 450)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self.segment_data = segment_data
        self.sample_rate = sample_rate
        self.result = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(self.METHODS)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        layout.addLayout(method_row)

        # Custom expression area
        self._custom_group = QGroupBox("Custom Expression")
        custom_layout = QVBoxLayout(self._custom_group)
        custom_layout.addWidget(QLabel(
            "Available variables: segment_values, sample_rate, np"))
        self._custom_edit = QTextEdit()
        self._custom_edit.setPlainText(
            "# Example: compute mean of each segment value array\n"
            "result = [np.mean(v) for v in segment_values]")
        self._custom_edit.setMaximumHeight(120)
        custom_layout.addWidget(self._custom_edit)
        self._custom_group.setVisible(False)
        layout.addWidget(self._custom_group)

        # Run button
        btn_run = QPushButton("Run Analysis")
        btn_run.clicked.connect(self._run_analysis)
        layout.addWidget(btn_run)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        results_layout.addWidget(self._results_text)
        layout.addWidget(results_group)

        # Bottom buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_method_changed(self, index):
        self._custom_group.setVisible(index == 3)

    def _get_segment_values(self):
        """Extract numeric value arrays from segment_data."""
        values = []
        if isinstance(self.segment_data, dict):
            for key, seg in self.segment_data.items():
                if isinstance(seg, dict) and "values" in seg:
                    vals = seg["values"]
                    if isinstance(vals, list):
                        values.extend(vals)
                    else:
                        values.append(np.asarray(vals))
                elif isinstance(seg, (np.ndarray, list)):
                    values.append(np.asarray(seg))
        elif isinstance(self.segment_data, list):
            for item in self.segment_data:
                if isinstance(item, dict) and "segmentValue" in item:
                    values.append(np.asarray(item["segmentValue"]))
                elif isinstance(item, np.ndarray):
                    values.append(item)
        return values

    def _run_analysis(self):
        method_idx = self._method_combo.currentIndex()
        segment_values = self._get_segment_values()

        if not segment_values:
            self._results_text.setPlainText("No segment values found.")
            return

        try:
            if method_idx == 0:
                # dR/R
                results = self._compute_dr_r(segment_values)
            elif method_idx == 1:
                # Amplitude distribution
                results = self._compute_amplitude_distribution(segment_values)
            elif method_idx == 2:
                # Peak counting
                results = self._compute_peak_counting(segment_values)
            elif method_idx == 3:
                # Custom
                results = self._run_custom(segment_values)
            else:
                results = {"error": "Unknown method"}

            self.result = results
            self._display_results(results)

        except Exception as exc:
            self._results_text.setPlainText(f"Error: {exc}")
            self.result = {"error": str(exc)}

    def _compute_dr_r(self, segment_values):
        """Fractional resistance change: dR/R = (R_segment - R_baseline) / R_baseline."""
        results = []
        for i, sv in enumerate(segment_values):
            sv = np.asarray(sv, dtype=float)
            if len(sv) < 2:
                continue
            baseline = np.mean(sv[:max(1, len(sv) // 10)])
            if abs(baseline) < 1e-30:
                baseline = np.mean(sv)
            dr_r = (sv - baseline) / baseline if abs(baseline) > 1e-30 else sv * 0
            results.append({
                "segment_index": i,
                "mean_dR_R": float(np.mean(dr_r)),
                "std_dR_R": float(np.std(dr_r)),
                "baseline": float(baseline),
            })
        return {"method": "dR/R", "segments": results}

    def _compute_amplitude_distribution(self, segment_values):
        """Compute amplitude statistics for each segment."""
        results = []
        for i, sv in enumerate(segment_values):
            sv = np.asarray(sv, dtype=float)
            results.append({
                "segment_index": i,
                "mean": float(np.mean(sv)),
                "std": float(np.std(sv)),
                "min": float(np.min(sv)),
                "max": float(np.max(sv)),
                "median": float(np.median(sv)),
            })
        return {"method": "Amplitude Distribution", "segments": results}

    def _compute_peak_counting(self, segment_values):
        """Count peaks (local maxima) in each segment."""
        from scipy.signal import find_peaks
        results = []
        for i, sv in enumerate(segment_values):
            sv = np.asarray(sv, dtype=float)
            peaks, props = find_peaks(sv)
            results.append({
                "segment_index": i,
                "n_peaks": len(peaks),
                "peak_indices": peaks.tolist(),
            })
        return {"method": "Peak Counting", "segments": results}

    def _run_custom(self, segment_values):
        """Run user-defined custom expression."""
        code = self._custom_edit.toPlainText()
        local_ns = {
            "segment_values": segment_values,
            "sample_rate": self.sample_rate,
            "np": np,
        }
        exec(code, {"__builtins__": __builtins__}, local_ns)
        result = local_ns.get("result", None)
        return {"method": "Custom", "result": result}

    def _display_results(self, results):
        """Format results dict for display."""
        lines = [f"Method: {results.get('method', 'Unknown')}"]
        if "segments" in results:
            for seg in results["segments"]:
                parts = [f"  Segment {seg.get('segment_index', '?')}:"]
                for k, v in seg.items():
                    if k == "segment_index":
                        continue
                    if isinstance(v, float):
                        parts.append(f"    {k} = {v:.6g}")
                    else:
                        parts.append(f"    {k} = {v}")
                lines.extend(parts)
        elif "result" in results:
            lines.append(f"  Result: {results['result']}")
        self._results_text.setPlainText("\n".join(lines))

    def _on_ok(self):
        if self.result is None:
            self._run_analysis()
        self.accept()


def analysis_method_selector(segment_data, sample_rate=1):
    """Show the analysis method dialog and return results.

    Parameters
    ----------
    segment_data : dict or list
        Segment data from pulse slicing or feature extraction.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    result : dict
        Analysis results including method name and per-segment data.
    """
    dlg = AnalysisMethodDialog(segment_data, sample_rate)
    dlg.exec()
    return dlg.result if dlg.result is not None else {}
