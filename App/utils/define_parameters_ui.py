"""Define Parameters dialog — lets user define named variables and global settings."""

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QDoubleSpinBox,
    QCheckBox, QHeaderView,
)
from PySide6.QtCore import Qt


class DefineParametersDialog(QDialog):
    """Dialog for defining named variables and global parameters."""

    def __init__(self, variables=None, global_sample_rate=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Parameters")
        self.resize(500, 450)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._variables = list(variables or [])
        self._global_sample_rate = global_sample_rate
        self.confirmed = False

        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # --- Global Parameters ---
        grp_global = QGroupBox("Global Parameters")
        gl = QVBoxLayout(grp_global)

        self._chk_sample_rate = QCheckBox("Define Sample Rate")
        self._chk_sample_rate.setChecked(self._global_sample_rate is not None)
        self._chk_sample_rate.stateChanged.connect(self._on_sr_toggle)
        gl.addWidget(self._chk_sample_rate)

        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("Sample Rate (Hz):"))
        self._spin_sr = QDoubleSpinBox()
        self._spin_sr.setDecimals(0)
        self._spin_sr.setRange(1, 1e9)
        self._spin_sr.setValue(self._global_sample_rate or 250000)
        self._spin_sr.setEnabled(self._global_sample_rate is not None)
        sr_row.addWidget(self._spin_sr, stretch=1)
        gl.addLayout(sr_row)

        layout.addWidget(grp_global)

        # --- User Variables ---
        grp_vars = QGroupBox("User Variables")
        vl = QVBoxLayout(grp_vars)

        lbl = QLabel("Define variables to store in the reference pool:")
        lbl.setStyleSheet("font-size: 11px; color: #666;")
        vl.addWidget(lbl)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Name", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.cellChanged.connect(self._on_cell_changed)
        vl.addWidget(self._table)

        layout.addWidget(grp_vars, stretch=1)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Confirm")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._on_confirm)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _on_sr_toggle(self, state):
        self._spin_sr.setEnabled(state == Qt.CheckState.Checked.value)

    def _populate_table(self):
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for var in self._variables:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(var.get("name", "")))
            self._table.setItem(row, 1, QTableWidgetItem(str(var.get("value", ""))))
        # Add blank row
        self._add_blank_row()
        self._table.blockSignals(False)

    def _add_blank_row(self):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(""))
        self._table.setItem(row, 1, QTableWidgetItem(""))

    def _on_cell_changed(self, row, col):
        # If user typed in the last row, add a new blank row
        if row == self._table.rowCount() - 1:
            name = (self._table.item(row, 0).text().strip()
                    if self._table.item(row, 0) else "")
            value = (self._table.item(row, 1).text().strip()
                     if self._table.item(row, 1) else "")
            if name or value:
                self._table.blockSignals(True)
                self._add_blank_row()
                self._table.blockSignals(False)

    def _collect_variables(self):
        """Collect non-empty variable definitions from the table."""
        variables = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            value_item = self._table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            value_str = value_item.text().strip() if value_item else ""
            if not name:
                continue
            # Try to parse as number
            value = self._parse_value(value_str)
            variables.append({"name": name, "value": value, "value_str": value_str})
        return variables

    @staticmethod
    def _parse_value(s):
        """Parse a string value into a number or keep as string."""
        if not s:
            return 0.0
        try:
            # Try int first
            if "." not in s and "e" not in s.lower():
                return int(s)
            return float(s)
        except ValueError:
            pass
        # Try numpy expression
        try:
            val = eval(s, {"__builtins__": {}, "np": np, "pi": np.pi, "inf": np.inf})
            return val
        except Exception:
            return s

    def _on_confirm(self):
        self._variables = self._collect_variables()
        if self._chk_sample_rate.isChecked():
            self._global_sample_rate = self._spin_sr.value()
        else:
            self._global_sample_rate = None
        self.confirmed = True
        self.accept()

    def get_results(self):
        return {
            "variables": self._variables,
            "globalSampleRate": self._global_sample_rate,
        }


def define_parameters_dialog(variables=None, global_sample_rate=None, parent=None):
    """Open the Define Parameters dialog and return results."""
    dlg = DefineParametersDialog(variables, global_sample_rate, parent)
    dlg.exec()
    if not dlg.confirmed:
        raise ValueError("User closed Define Parameters without confirming.")
    return dlg.get_results()
