"""Welcome panel shown when no canvas tabs are open."""

import os
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import recent_files
from theme import theme


def _relative_time(ts):
    delta = datetime.now() - datetime.fromtimestamp(ts)
    s = int(delta.total_seconds())
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        return f"{s // 3600} hr ago"
    if s < 86400 * 7:
        return f"{s // 86400} days ago"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


class WelcomePanel(QWidget):
    """Blank-state view shown when all canvas tabs are closed."""

    newRequested = Signal()
    openRequested = Signal()
    openFileRequested = Signal(str)

    def __init__(self, root_dir, parent=None):
        super().__init__(parent)
        self._root_dir = root_dir
        self._build_ui()
        self.apply_theme()
        self.refresh()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 60, 60, 60)
        outer.setSpacing(24)

        title = QLabel("NPSflow")
        f = QFont()
        f.setPointSize(28)
        f.setBold(True)
        title.setFont(f)
        outer.addWidget(title)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._new_btn = QPushButton("New Workflow")
        self._open_btn = QPushButton("Open Workflow...")
        for b in (self._new_btn, self._open_btn):
            b.setMinimumHeight(36)
            b.setMinimumWidth(160)
        self._new_btn.clicked.connect(self.newRequested.emit)
        self._open_btn.clicked.connect(self.openRequested.emit)
        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        header_row = QHBoxLayout()
        header = QLabel("Recent")
        hf = QFont()
        hf.setPointSize(13)
        hf.setBold(True)
        header.setFont(hf)
        header_row.addWidget(header)
        header_row.addStretch()
        self._clear_btn = QPushButton("Clear History")
        self._clear_btn.clicked.connect(self._on_clear)
        header_row.addWidget(self._clear_btn)
        outer.addLayout(header_row)

        self._divider = QFrame()
        self._divider.setFrameShape(QFrame.Shape.HLine)
        outer.addWidget(self._divider)

        self._list = QListWidget()
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.itemActivated.connect(self._on_item_activated)
        outer.addWidget(self._list, 1)

    def apply_theme(self):
        p = theme.palette
        self.setStyleSheet(
            f"WelcomePanel {{ background: {p.window_bg}; color: {p.text}; }}"
        )
        for btn in (self._new_btn, self._open_btn):
            btn.setStyleSheet(
                f"QPushButton {{ background: {p.button_bg}; color: {p.button_text};"
                f" border: 1px solid {p.border}; border-radius: 4px;"
                f" padding: 6px 14px; }}"
                f"QPushButton:hover {{ background: {p.button_bg_hover}; }}"
                f"QPushButton:pressed {{ background: {p.button_bg_press}; }}"
            )
        self._clear_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {p.text_muted};"
            f" border: 1px solid {p.border}; border-radius: 4px;"
            f" padding: 3px 10px; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover};"
            f" color: {p.text}; }}"
            f"QPushButton:pressed {{ background: {p.button_bg_press}; }}"
        )
        self._divider.setStyleSheet(f"color: {p.border};")
        self._list.setStyleSheet(
            f"QListWidget {{ background: {p.window_bg}; color: {p.text};"
            f" border: none; outline: 0; }}"
            f"QListWidget::item {{ padding: 8px 4px;"
            f" border-bottom: 1px solid {p.border}; }}"
            f"QListWidget::item:selected,"
            f"QListWidget::item:selected:active,"
            f"QListWidget::item:selected:!active {{"
            f" background: {p.accent}; color: {p.accent_text}; }}"
        )

    def refresh(self):
        self._list.clear()
        paths = recent_files.load(self._root_dir)
        self._clear_btn.setVisible(bool(paths))
        if not paths:
            empty = QListWidgetItem("No recent workflows yet.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(empty)
            return

        for path in paths:
            exists = os.path.isfile(path)
            name = os.path.basename(path)
            if exists:
                parent = os.path.basename(os.path.dirname(path)) or "/"
                subtitle = f"{parent} · {_relative_time(os.path.getmtime(path))}"
            else:
                subtitle = "(file not found)"
            item = QListWidgetItem(f"{name}\n{subtitle}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            if not exists:
                item.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(item)

    def _on_clear(self):
        recent_files.clear(self._root_dir)
        self.refresh()

    def _on_item_activated(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        if not os.path.isfile(path):
            recent_files.remove(self._root_dir, path)
            self.refresh()
            return
        self.openFileRequested.emit(path)
