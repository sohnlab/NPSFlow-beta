"""Generic formula editor dialog for analysis blocks."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QTextEdit, QComboBox, QPushButton, QDialogButtonBox, QGroupBox,
)

from utils.dialog_style import apply_dialog_style


class FormulaEditorDialog(QDialog):
    """Generic formula editor for analysis blocks.

    ``unit_fields`` optionally adds a dropdown per entry (e.g. input-unit
    selectors). Each entry is a dict ``{"key", "label", "options", "current",
    "group"}``; entries are grouped into side-by-side boxes by ``group`` (in
    first-seen order). The chosen values are read back via ``get_unit_values()``.
    """

    def __init__(self, title="Edit Formula", current_formula="",
                 default_formula="", variables=None, unit_fields=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} — Edit Formula")
        self.setMinimumWidth(520)
        self.setMinimumHeight(250)
        apply_dialog_style(self)

        self._formula = current_formula or default_formula
        self._default = default_formula
        self._unit_combos = {}  # key -> QComboBox

        layout = QVBoxLayout(self)

        # --- unit selectors, grouped side by side (Input Units | Output Unit) ---
        if unit_fields:
            order, by_group = [], {}
            for uf in unit_fields:
                g = uf.get("group", "Units")
                if g not in by_group:
                    by_group[g] = []
                    order.append(g)
                by_group[g].append(uf)

            unit_row = QHBoxLayout()
            for g in order:
                box = QGroupBox(g)
                box_form = QFormLayout(box)
                for uf in by_group[g]:
                    combo = QComboBox()
                    combo.addItems([str(o) for o in uf.get("options", [])])
                    idx = combo.findText(str(uf.get("current", "")))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    box_form.addRow(f"{uf.get('label', uf['key'])}:", combo)
                    self._unit_combos[uf["key"]] = combo
                unit_row.addWidget(box)
            unit_row.addStretch()
            layout.addLayout(unit_row)

        # --- variables list ---
        if variables:
            var_group = QGroupBox("Available Variables")
            var_layout = QVBoxLayout(var_group)
            for name, display, desc in variables:
                text = f"\u2022  {name}"
                if desc:
                    text += f"  — {desc}"
                lbl = QLabel(text)
                var_layout.addWidget(lbl)
            hint = QLabel("Use Python / NumPy syntax (e.g. np.sqrt, **, np.log, np.pi).")
            hint.setStyleSheet("color: #666; font-style: italic;")
            var_layout.addWidget(hint)
            layout.addWidget(var_group)

        # --- formula editor ---
        self._editor = QTextEdit()
        self._editor.setPlainText(self._formula)
        self._editor.setAcceptRichText(False)
        self._editor.setMaximumHeight(80)
        layout.addWidget(self._editor)

        # --- reset button ---
        reset_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_formula)
        reset_layout.addWidget(reset_btn)
        reset_layout.addStretch()
        layout.addLayout(reset_layout)

        # --- OK / Cancel ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = btn_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setObjectName("confirmBtn")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _reset_formula(self):
        self._editor.setPlainText(self._default)

    def get_formula(self) -> str:
        return self._editor.toPlainText().strip()

    def get_unit_values(self) -> dict:
        return {k: c.currentText() for k, c in self._unit_combos.items()}
