"""FloatingParamEditor — in-canvas floating panel for generic block parameters.

Replaces the modal ``QDialog`` produced by ``_edit_block_parameters``. Parented
to the canvas widget, so it floats over the scene; draggable by the header,
dismissed with Esc / Cancel / X, commits on OK like a dialog would.

The widget is intentionally small in scope: it handles the generic case (a
``defaultParameters`` dict with optional ``parameterDefinitions`` metadata).
Bespoke editors (Text, Zone Selection, DefineParameters, PlatformDetection,
NotchFilter, FilterRows, formula, SubPipeline, Reference, DataBus, Mean,
Unpacker) keep their existing dialogs — those are richer than a flat form.
"""

from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QCheckBox, QScrollArea,
)

from theme import theme


class _ResizeGrip(QWidget):
    """Small diagonal-stripes handle in the bottom-right that resizes its
    parent panel. Clamps the new size to ``parent.minimum*`` and to the
    parent's canvas bounds (so the panel never extends past the canvas)."""

    SIZE = 14

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._start_global = None
        self._start_size = None

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = QColor(theme.palette.text_muted)
        p.setPen(QPen(col, 1))
        # Three diagonal hash marks tucked into the corner.
        for i in (3, 6, 9):
            p.drawLine(self.width() - i - 1, self.height() - 1,
                       self.width() - 1, self.height() - i - 1)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._start_global = ev.globalPosition().toPoint()
            self._start_size = self.parent().size()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._start_global is None:
            super().mouseMoveEvent(ev)
            return
        panel = self.parent()
        delta = ev.globalPosition().toPoint() - self._start_global
        new_w = max(panel.minimumWidth(), self._start_size.width() + delta.x())
        new_h = max(panel.minimumHeight(), self._start_size.height() + delta.y())
        canvas = getattr(panel, "_canvas", None)
        if canvas is not None:
            pp = panel.pos()
            new_w = min(new_w, canvas.width() - pp.x() - 4)
            new_h = min(new_h, canvas.height() - pp.y() - 4)
        panel.resize(new_w, new_h)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._start_global is not None and ev.button() == Qt.MouseButton.LeftButton:
            self._start_global = None
            self._start_size = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class FloatingParamEditor(QFrame):
    """Drag-able floating parameter form overlaying the canvas."""

    # Class-level handle so opening a new editor closes any prior one.
    _active = None

    def __init__(self, block, defn, canvas, on_apply, on_cancel=None):
        super().__init__(canvas)
        self.setObjectName("floatingParamEditor")
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._block = block
        self._defn = defn or {}
        self._canvas = canvas
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        self._drag_offset = None  # mouse-anchor for header drag
        # Per-key widgets — built once at construction.
        self._widgets = {}  # key -> (widget, default_val, kind)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ----- header / drag handle -----
        self._header = QWidget(self)
        self._header.setObjectName("fpHeader")
        self._header.setCursor(Qt.CursorShape.SizeAllCursor)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(10, 6, 6, 6)
        hl.setSpacing(6)
        self._title = QLabel(block.display_name or block.definition_name or "Block")
        self._title.setObjectName("fpTitle")
        hl.addWidget(self._title, 1)
        self._close_btn = QPushButton("×", self._header)
        self._close_btn.setObjectName("fpClose")
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.setToolTip("Close (Esc)")
        self._close_btn.clicked.connect(self._cancel)
        hl.addWidget(self._close_btn)
        outer.addWidget(self._header)

        # ----- form body (scrolls for tall forms) -----
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget(self._scroll)
        self._form = QFormLayout(body)
        self._form.setContentsMargins(12, 10, 12, 10)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._scroll.setWidget(body)
        outer.addWidget(self._scroll, 1)

        self._build_form()

        # ----- footer buttons -----
        footer = QWidget(self)
        footer.setObjectName("fpFooter")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(8)
        fl.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._commit)
        fl.addWidget(self._cancel_btn)
        fl.addWidget(self._ok_btn)
        outer.addWidget(footer)

        self.apply_theme()
        self.setMinimumWidth(320)
        self.setMinimumHeight(140)
        # Cap height so very-long forms scroll instead of pushing off-canvas.
        self.adjustSize()
        max_h = min(560, max(180, canvas.height() - 40))
        self.resize(self.width(), min(self.height(), max_h))
        # Bottom-right resize grip; sits on top of the footer.
        self._resize_grip = _ResizeGrip(self)
        self._reposition_resize_grip()
        self._resize_grip.raise_()

    # ----- public API -----

    @classmethod
    def open_for(cls, block, defn, canvas, on_apply, on_cancel=None):
        """Close any existing editor and open a new one for ``block``.

        Returns the new editor, or None if there are no editable parameters.
        """
        if cls._active is not None:
            try:
                cls._active._cancel(silent=True)
            except Exception:
                pass
            cls._active = None

        editor = cls(block, defn, canvas, on_apply, on_cancel)
        if not editor._widgets:
            # Nothing to edit — release immediately.
            editor.setParent(None)
            editor.deleteLater()
            return None
        cls._active = editor
        editor._position_near_block()
        editor.show()
        editor.raise_()
        editor.setFocus(Qt.FocusReason.PopupFocusReason)
        return editor

    def apply_theme(self):
        p = theme.palette
        self.setStyleSheet(
            f"QFrame#floatingParamEditor {{ background: {p.window_bg};"
            f" border: 1px solid {p.border}; border-radius: 6px; }}"
            f"QWidget#fpHeader {{ background: {p.panel_bg};"
            f" border-bottom: 1px solid {p.border};"
            f" border-top-left-radius: 6px; border-top-right-radius: 6px; }}"
            f"QWidget#fpFooter {{ background: {p.panel_bg};"
            f" border-top: 1px solid {p.border};"
            f" border-bottom-left-radius: 6px; border-bottom-right-radius: 6px; }}"
            f"QLabel#fpTitle {{ color: {p.text}; font-weight: bold; }}"
            f"QPushButton#fpClose {{ background: transparent;"
            f" color: {p.text_muted}; border: none; font-size: 16px;"
            f" padding: 0; margin: 0; }}"
            f"QPushButton#fpClose:hover {{ color: {p.text}; }}"
            # Form fields: line edits / combos / spins inherit the app palette,
            # which the theme module already set. Buttons get the standard
            # theme button style here.
            f"QPushButton {{ background: {p.button_bg}; color: {p.button_text};"
            f" border: 1px solid {p.border}; border-radius: 4px;"
            f" padding: 4px 12px; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover}; }}"
            f"QPushButton:pressed {{ background: {p.button_bg_press}; }}"
            f"QPushButton:default {{ border-color: {p.accent}; }}"
            f"QPushButton#fpClose:hover {{ background: transparent; }}"
            f"QLabel {{ color: {p.text}; background: transparent; }}"
            f"QLineEdit, QComboBox {{ background: {p.base_bg}; color: {p.text};"
            f" border: 1px solid {p.border}; border-radius: 3px;"
            f" padding: 2px 6px; }}"
            f"QLineEdit:focus, QComboBox:focus {{ border-color: {p.accent}; }}"
        )
        # The class-name selector above doesn't catch `self` (an instance of
        # QFrame with objectName set); ensure the outer frame paints too.
        self.setAutoFillBackground(True)
        # Grip repaints from theme.palette.text_muted; force an update so a
        # live theme swap is reflected on the corner.
        grip = getattr(self, "_resize_grip", None)
        if grip is not None:
            grip.update()

    # ----- form construction -----

    def _build_form(self):
        defaults = self._defn.get("defaultParameters", {}) or {}
        param_defs = self._defn.get("parameterDefinitions", []) or []
        pd_map = {pd.get("name"): pd for pd in param_defs}
        skip = {"displayText"}
        editable = [(k, v) for k, v in defaults.items() if k not in skip]
        if not editable:
            return

        for key, default_val in editable:
            current = self._block.parameters.get(key, default_val)
            pd = pd_map.get(key, {})
            label = pd.get("displayName", key)
            self._add_field(key, label, current, default_val, pd)

    def _add_field(self, key, label, current, default_val, pd):
        ptype = pd.get("type")

        if ptype == "choice" and "options" in pd:
            combo = QComboBox()
            combo.addItems([str(o) for o in pd["options"]])
            idx = combo.findText(str(current))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            self._form.addRow(f"{label}:", combo)
            self._widgets[key] = (combo, default_val, "choice")
            return

        if ptype == "bool" or isinstance(default_val, bool):
            cb = QCheckBox()
            cb.setChecked(bool(current))
            self._form.addRow(f"{label}:", cb)
            self._widgets[key] = (cb, default_val, "bool")
            return

        line = QLineEdit(str(current))
        line.setClearButtonEnabled(True)
        self._form.addRow(f"{label}:", line)
        self._widgets[key] = (line, default_val, "text")

    # ----- commit / cancel -----

    def _read_value(self, widget, default_val, kind):
        if kind == "choice":
            return widget.currentText()
        if kind == "bool":
            return bool(widget.isChecked())
        text = widget.text()
        if isinstance(default_val, bool):
            return text.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default_val, int) and not isinstance(default_val, bool):
            try:
                return int(float(text))
            except (TypeError, ValueError):
                return default_val
        if isinstance(default_val, float):
            try:
                return float(text)
            except (TypeError, ValueError):
                return default_val
        try:
            val = float(text)
            return int(val) if val == int(val) else val
        except (TypeError, ValueError):
            return text

    def _commit(self):
        changes = {}
        for key, (widget, default_val, kind) in self._widgets.items():
            new_val = self._read_value(widget, default_val, kind)
            if self._block.parameters.get(key) != new_val:
                changes[key] = new_val
        for key, val in changes.items():
            self._block.parameters[key] = val
        self._close()
        if self._on_apply is not None:
            try:
                self._on_apply(self._block, changes)
            except Exception:
                import traceback; traceback.print_exc()

    def _cancel(self, silent=False):
        cb = self._on_cancel
        self._close()
        if not silent and cb is not None:
            try:
                cb(self._block)
            except Exception:
                import traceback; traceback.print_exc()

    def _close(self):
        if FloatingParamEditor._active is self:
            FloatingParamEditor._active = None
        # Drop the event filter on the canvas viewport before teardown so we
        # don't re-enter eventFilter after deleteLater().
        target = getattr(self, "_filter_target", None)
        if target is not None:
            try:
                target.removeEventFilter(self)
            except RuntimeError:
                pass
            self._filter_target = None
        self.hide()
        self.setParent(None)
        self.deleteLater()

    # ----- resize -----

    def _reposition_resize_grip(self):
        g = getattr(self, "_resize_grip", None)
        if g is None:
            return
        g.move(self.width() - g.width() - 1, self.height() - g.height() - 1)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_resize_grip()

    # ----- positioning -----

    def _position_near_block(self):
        """Center the editor over the canvas (clamped into view)."""
        canvas = self._canvas
        cw, ch = canvas.width(), canvas.height()
        x = (cw - self.width()) // 2
        y = (ch - self.height()) // 2
        x = max(8, min(int(x), cw - self.width() - 8))
        y = max(8, min(int(y), ch - self.height() - 8))
        self.move(x, y)

    # ----- input events -----

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self._cancel()
            ev.accept()
            return
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Don't commit on Enter while typing in a QLineEdit — the default
            # button takes care of that case. Pass through.
            super().keyPressEvent(ev)
            return
        super().keyPressEvent(ev)

    # Drag the panel by pressing-and-holding on the header. _drag_offset is
    # the mouse position in editor-local coords at the start of the drag,
    # i.e. the point on the editor that should stay glued to the cursor.
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            pos_in_header = self._header.mapFrom(self, ev.position().toPoint())
            if self._header.rect().contains(pos_in_header):
                self._drag_offset = ev.position().toPoint()
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_offset is not None:
            cursor_in_canvas = self._canvas.mapFromGlobal(
                ev.globalPosition().toPoint())
            new_x = cursor_in_canvas.x() - self._drag_offset.x()
            new_y = cursor_in_canvas.y() - self._drag_offset.y()
            cw, ch = self._canvas.width(), self._canvas.height()
            new_x = max(0, min(new_x, cw - self.width()))
            new_y = max(0, min(new_y, ch - self.height()))
            self.move(new_x, new_y)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_offset is not None and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    # ----- outside-click-to-close -----

    def showEvent(self, ev):
        super().showEvent(ev)
        # QGraphicsView is a QAbstractScrollArea — real mouse events land on
        # its viewport() widget, not on the view itself. Watch the viewport
        # so we actually see outside-canvas clicks.
        target = self._canvas.viewport() if self._canvas is not None else None
        if target is not None:
            target.installEventFilter(self)
            self._filter_target = target
        else:
            self._filter_target = None

    def eventFilter(self, obj, event):
        if (obj is getattr(self, "_filter_target", None)
                and event.type() == QEvent.Type.MouseButtonPress):
            # Cancel but do NOT consume — let the canvas keep handling the
            # click normally (select a different block, start a pan, etc.).
            self._cancel()
        return super().eventFilter(obj, event)


class FloatingMessage(QFrame):
    """Small floating notice on the canvas — drop-in for QMessageBox.information.

    Same look / drag / Esc / outside-click behaviour as ``FloatingParamEditor``,
    but with no form. One slot for a title, one for a body message, an OK
    button.
    """

    _active = None

    def __init__(self, title, message, canvas):
        super().__init__(canvas)
        self.setObjectName("floatingParamEditor")  # share QSS with the editor
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._canvas = canvas
        self._drag_offset = None
        self._filter_target = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ----- header (drag handle + close X) -----
        self._header = QWidget(self)
        self._header.setObjectName("fpHeader")
        self._header.setCursor(Qt.CursorShape.SizeAllCursor)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(10, 6, 6, 6)
        hl.setSpacing(6)
        self._title = QLabel(title or "")
        self._title.setObjectName("fpTitle")
        hl.addWidget(self._title, 1)
        self._close_btn = QPushButton("×", self._header)
        self._close_btn.setObjectName("fpClose")
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.setToolTip("Close (Esc)")
        self._close_btn.clicked.connect(self._close)
        hl.addWidget(self._close_btn)
        outer.addWidget(self._header)

        # ----- body -----
        body = QWidget(self)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 14, 14, 12)
        bl.setSpacing(8)
        self._body_label = QLabel(message or "")
        self._body_label.setWordWrap(True)
        self._body_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._body_label)
        outer.addWidget(body, 1)

        # ----- footer (OK button) -----
        footer = QWidget(self)
        footer.setObjectName("fpFooter")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(8)
        fl.addStretch(1)
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._close)
        fl.addWidget(self._ok_btn)
        outer.addWidget(footer)

        self.apply_theme()
        self.setMinimumWidth(260)
        self.setMaximumWidth(420)
        self.adjustSize()

    # ----- public API -----

    @classmethod
    def show_for(cls, title, message, canvas, near_block=None):
        """Close any prior message and pop a new one. Returns the message."""
        if cls._active is not None:
            try:
                cls._active._close()
            except Exception:
                pass
            cls._active = None

        msg = cls(title, message, canvas)
        cls._active = msg
        msg._position(near_block)
        msg.show()
        msg.raise_()
        msg.setFocus(Qt.FocusReason.PopupFocusReason)
        return msg

    def apply_theme(self):
        # Reuse FloatingParamEditor's stylesheet builder so message + editor
        # stay visually identical.
        FloatingParamEditor.apply_theme(self)

    # ----- positioning -----

    def _position(self, near_block=None):
        canvas = self._canvas
        cw, ch = canvas.width(), canvas.height()
        x = (cw - self.width()) // 2
        y = (ch - self.height()) // 2
        x = max(8, min(int(x), cw - self.width() - 8))
        y = max(8, min(int(y), ch - self.height() - 8))
        self.move(x, y)

    # ----- close / event filter / drag -----

    def _close(self):
        if FloatingMessage._active is self:
            FloatingMessage._active = None
        target = getattr(self, "_filter_target", None)
        if target is not None:
            try:
                target.removeEventFilter(self)
            except RuntimeError:
                pass
            self._filter_target = None
        self.hide()
        self.setParent(None)
        self.deleteLater()

    def showEvent(self, ev):
        super().showEvent(ev)
        target = self._canvas.viewport() if self._canvas is not None else None
        if target is not None:
            target.installEventFilter(self)
            self._filter_target = target

    def eventFilter(self, obj, event):
        if (obj is getattr(self, "_filter_target", None)
                and event.type() == QEvent.Type.MouseButtonPress):
            self._close()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._close()
            ev.accept()
            return
        super().keyPressEvent(ev)

    # Drag via header (same math as FloatingParamEditor).
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            pos_in_header = self._header.mapFrom(self, ev.position().toPoint())
            if self._header.rect().contains(pos_in_header):
                self._drag_offset = ev.position().toPoint()
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_offset is not None:
            cursor_in_canvas = self._canvas.mapFromGlobal(
                ev.globalPosition().toPoint())
            new_x = cursor_in_canvas.x() - self._drag_offset.x()
            new_y = cursor_in_canvas.y() - self._drag_offset.y()
            cw, ch = self._canvas.width(), self._canvas.height()
            new_x = max(0, min(new_x, cw - self.width()))
            new_y = max(0, min(new_y, ch - self.height()))
            self.move(new_x, new_y)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_offset is not None and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class FloatingShell(QFrame):
    """Generic floating in-canvas panel: header + caller-built body + footer.

    Use when an editor needs a layout richer than the generic param form
    (Edit Text, Edit Data Bus, etc.). The shell owns the drag handle, Esc
    handling, outside-click-to-close filter, theme styles, and OK/Cancel
    plumbing; the body widget is yours to populate.
    """

    _active = None

    def __init__(self, title, canvas, body, on_ok=None, on_cancel=None,
                 ok_text="OK", cancel_text="Cancel", show_cancel=True,
                 show_footer=True, close_on_outside_click=True):
        super().__init__(canvas)
        self._close_on_outside_click = close_on_outside_click
        # Reuse FloatingParamEditor's objectName so apply_theme below gets the
        # same stylesheet without duplication.
        self.setObjectName("floatingParamEditor")
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._canvas = canvas
        self._on_ok = on_ok
        self._on_cancel = on_cancel
        self._drag_offset = None
        self._filter_target = None
        self._committed = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._header = QWidget(self)
        self._header.setObjectName("fpHeader")
        self._header.setCursor(Qt.CursorShape.SizeAllCursor)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(10, 6, 6, 6)
        hl.setSpacing(6)
        self._title = QLabel(title or "")
        self._title.setObjectName("fpTitle")
        hl.addWidget(self._title, 1)
        self._close_btn = QPushButton("×", self._header)
        self._close_btn.setObjectName("fpClose")
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.setToolTip("Close (Esc)")
        self._close_btn.clicked.connect(self._cancel)
        hl.addWidget(self._close_btn)
        outer.addWidget(self._header)

        # Body inside a scroll area in case it's tall
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._body = body
        self._scroll.setWidget(self._body)
        outer.addWidget(self._scroll, 1)

        # Footer (omitted entirely for live-edit panels — close via X / Esc).
        self._cancel_btn = None
        self._ok_btn = None
        if show_footer:
            footer = QWidget(self)
            footer.setObjectName("fpFooter")
            fl = QHBoxLayout(footer)
            fl.setContentsMargins(10, 6, 10, 8)
            fl.setSpacing(8)
            fl.addStretch(1)
            if show_cancel:
                self._cancel_btn = QPushButton(cancel_text)
                self._cancel_btn.clicked.connect(self._cancel)
                fl.addWidget(self._cancel_btn)
            self._ok_btn = QPushButton(ok_text)
            self._ok_btn.setDefault(True)
            self._ok_btn.clicked.connect(self._commit)
            fl.addWidget(self._ok_btn)
            outer.addWidget(footer)

        self.apply_theme()
        self.setMinimumWidth(360)
        self.setMinimumHeight(160)
        self.adjustSize()
        max_h = min(640, max(200, canvas.height() - 40))
        self.resize(max(self.width(), 360),
                    min(self.height(), max_h))
        self._resize_grip = _ResizeGrip(self)
        self._reposition_resize_grip()
        self._resize_grip.raise_()

    # ----- public API -----

    @classmethod
    def open(cls, title, canvas, body, on_ok=None, on_cancel=None,
             near_block=None, ok_text="OK", cancel_text="Cancel",
             show_cancel=True, show_footer=True,
             close_on_outside_click=True):
        if cls._active is not None:
            try:
                cls._active._cancel(silent=True)
            except Exception:
                pass
            cls._active = None

        shell = cls(title, canvas, body, on_ok, on_cancel,
                    ok_text=ok_text, cancel_text=cancel_text,
                    show_cancel=show_cancel, show_footer=show_footer,
                    close_on_outside_click=close_on_outside_click)
        cls._active = shell
        shell._position(near_block)
        shell.show()
        shell.raise_()
        shell.setFocus(Qt.FocusReason.PopupFocusReason)
        return shell

    def apply_theme(self):
        # Reuse FloatingParamEditor's stylesheet builder so all floating
        # surfaces stay visually identical.
        FloatingParamEditor.apply_theme(self)

    # ----- lifecycle -----

    def _commit(self):
        self._committed = True
        cb = self._on_ok
        self._close()
        if cb is not None:
            try:
                cb()
            except Exception:
                import traceback; traceback.print_exc()

    def _cancel(self, silent=False):
        cb = self._on_cancel
        self._close()
        if not silent and cb is not None:
            try:
                cb()
            except Exception:
                import traceback; traceback.print_exc()

    def _close(self):
        if FloatingShell._active is self:
            FloatingShell._active = None
        target = getattr(self, "_filter_target", None)
        if target is not None:
            try:
                target.removeEventFilter(self)
            except RuntimeError:
                pass
            self._filter_target = None
        self.hide()
        self.setParent(None)
        self.deleteLater()

    # ----- resize -----

    def _reposition_resize_grip(self):
        g = getattr(self, "_resize_grip", None)
        if g is None:
            return
        g.move(self.width() - g.width() - 1, self.height() - g.height() - 1)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_resize_grip()

    # ----- positioning -----

    def _position(self, near_block=None):
        canvas = self._canvas
        cw, ch = canvas.width(), canvas.height()
        x = (cw - self.width()) // 2
        y = (ch - self.height()) // 2
        x = max(8, min(int(x), cw - self.width() - 8))
        y = max(8, min(int(y), ch - self.height() - 8))
        self.move(x, y)

    # ----- input events (drag + Esc + outside-click) -----

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._close_on_outside_click:
            return
        target = self._canvas.viewport() if self._canvas is not None else None
        if target is not None:
            target.installEventFilter(self)
            self._filter_target = target

    def eventFilter(self, obj, event):
        if (obj is getattr(self, "_filter_target", None)
                and event.type() == QEvent.Type.MouseButtonPress):
            self._cancel()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self._cancel()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            pos_in_header = self._header.mapFrom(self, ev.position().toPoint())
            if self._header.rect().contains(pos_in_header):
                self._drag_offset = ev.position().toPoint()
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_offset is not None:
            cursor_in_canvas = self._canvas.mapFromGlobal(
                ev.globalPosition().toPoint())
            new_x = cursor_in_canvas.x() - self._drag_offset.x()
            new_y = cursor_in_canvas.y() - self._drag_offset.y()
            cw, ch = self._canvas.width(), self._canvas.height()
            new_x = max(0, min(new_x, cw - self.width()))
            new_y = max(0, min(new_y, ch - self.height()))
            self.move(new_x, new_y)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_offset is not None and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)
