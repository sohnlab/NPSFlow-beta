"""Runner for UserChoice block - shows a QMessageBox with Yes/No question."""

from PySide6.QtWidgets import QMessageBox


def run(inputs, params, block):
    message = params.get("message", "Continue?")
    title = block.display_name if hasattr(block, "display_name") else "Choice"

    reply = QMessageBox.question(
        None, title, message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes)

    choice = "Yes" if reply == QMessageBox.StandardButton.Yes else "No"
    return {"choice": choice}
