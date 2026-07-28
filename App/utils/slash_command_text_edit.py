"""QTextEdit with a `/` slash-command completer.

Typing `/` opens a popup listing known commands; further typing filters
the list; Enter / click inserts the chosen command, replacing the `/…`
token under the cursor. Escape dismisses the popup. The popup never
shows when the `/` is part of an existing word (e.g. inside a path).
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QCompleter, QTextEdit


SLASH_COMMANDS = [
    ("/lastpath", "Use the block's last opened file or folder"),
    ("/lastSettings", "Use the block's previously saved settings"),
]


class SlashCommandTextEdit(QTextEdit):
    def __init__(self, commands=None, parent=None):
        super().__init__(parent)
        cmds = commands if commands is not None else SLASH_COMMANDS
        # Display each entry as "trigger — description" so users see context.
        self._display_items = [f"{trig}    {desc}" for trig, desc in cmds]
        self._triggers = [trig for trig, _ in cmds]

        self._completer = QCompleter(self._display_items, self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        self._completer.activated.connect(self._insert_completion)

    def _current_token(self):
        """Return (token, start_col_in_block) for the slash token immediately
        before the cursor, or (None, -1) if there isn't one. A slash token is
        a `/` followed by non-whitespace, preceded by start-of-line or
        whitespace (so `/foo` inside a path like `dir/foo` is ignored).
        """
        tc = self.textCursor()
        text = tc.block().text()
        col = tc.positionInBlock()
        i = col - 1
        while i >= 0 and not text[i].isspace():
            if text[i] == '/':
                if i == 0 or text[i - 1].isspace():
                    return text[i:col], i
                return None, -1
            i -= 1
        return None, -1

    def _insert_completion(self, completion):
        # `completion` is one of the display items; resolve back to its trigger.
        try:
            idx = self._display_items.index(completion)
        except ValueError:
            return
        trigger = self._triggers[idx]

        token, start = self._current_token()
        if token is None:
            return
        tc = self.textCursor()
        block_pos = tc.block().position()
        tc.setPosition(block_pos + start)
        tc.setPosition(block_pos + start + len(token),
                       QTextCursor.MoveMode.KeepAnchor)
        tc.insertText(trigger)
        self.setTextCursor(tc)

    def keyPressEvent(self, e):
        popup = self._completer.popup()
        if popup.isVisible():
            if e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return,
                           Qt.Key.Key_Tab, Qt.Key.Key_Escape,
                           Qt.Key.Key_Up, Qt.Key.Key_Down):
                e.ignore()
                return

        super().keyPressEvent(e)

        token, _ = self._current_token()
        if token is None:
            popup.hide()
            return

        self._completer.setCompletionPrefix(token)
        model = self._completer.completionModel()
        if model.rowCount() == 0:
            popup.hide()
            return
        popup.setCurrentIndex(model.index(0, 0))
        rect = self.cursorRect()
        rect.setWidth(
            popup.sizeHintForColumn(0)
            + popup.verticalScrollBar().sizeHint().width() + 16)
        self._completer.complete(rect)

    def focusInEvent(self, e):
        self._completer.setWidget(self)
        super().focusInEvent(e)
