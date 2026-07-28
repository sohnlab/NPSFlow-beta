"""NPSWorkflowApp - Main application for the NPSView workflow editor.

Creates the main window with toolbar, block palette, canvas, and status bar.
Orchestrates interaction between all components.
"""

import os
import sys
import json
import glob
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QPushButton,
    QLabel,
    QTextEdit,
    QSplitter,
    QMenu,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QColorDialog,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QTabWidget,
    QTabBar,
    QStackedWidget,
    QSizePolicy,
    QStyle,
)
from PySide6.QtCore import Qt, QSize, QEventLoop
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QKeySequence,
    QPixmap,
    QPainter,
    QPen,
    QBrush,
    QIcon,
)

from block_registry import BlockRegistry
from block_palette import BlockPalettePanel
from workflow_canvas import WorkflowCanvas, BlockGraphicsItem, WireGraphicsItem
from workflow_engine import WorkflowEngine
from data_inspector import DataInspector
from block_node import BlockNode
from wire_connection import WireConnection
from theme import theme, set_theme, get_theme_name, build_qpalette
from floating_param_editor import FloatingParamEditor, FloatingMessage, FloatingShell
from port import Port
import recent_files
from utils import recent_paths
from utils.paths import map_state_paths, resolve_project_path, to_project_relative
from welcome_panel import WelcomePanel


_CHOICE_LABELS = {
    "moving_average": "Moving Avg",
    "gaussian": "Gaussian",
    "median": "Median",
    "savgol": "Savitzky-Golay",
    "samples_to_ms": "samples \u2192 ms",
    "samples_to_s": "samples \u2192 s",
    "ms_to_s": "ms \u2192 s",
    "s_to_ms": "s \u2192 ms",
    "V_to_mV": "V \u2192 mV",
    "mV_to_V": "mV \u2192 V",
    "nm_to_um": "nm \u2192 \u00b5m",
    "um_to_nm": "\u00b5m \u2192 nm",
    "custom_multiply": "\u00d7 factor",
    "custom_divide": "\u00f7 factor",
}


def _build_display_text(params, param_defs):
    """Build a compact display text string from parameter values and definitions."""
    # If there's a single choice param with a label mapping, just use that
    choice_defs = [pd for pd in param_defs if pd.get("type") == "choice"]
    if len(choice_defs) == 1:
        val = params.get(choice_defs[0]["name"], "")
        mapped = _CHOICE_LABELS.get(val)
        if mapped:
            return mapped

    values = []
    unit = ""
    for pd in param_defs:
        val = params.get(pd["name"], "")
        if pd.get("type") == "choice":
            val = _CHOICE_LABELS.get(val, val)
        elif isinstance(val, float) and val == int(val):
            val = int(val)
        values.append(str(val))
        label = pd.get("displayName", "")
        if "(" in label and ")" in label:
            unit = label[label.index("(") + 1 : label.index(")")]
    suffix = f" {unit}" if unit else ""
    return " ".join(values) + suffix


def _themed_combo_qss():
    """Flat QComboBox: white field, themed border, gray drop-down button.

    The down-arrow itself is painted in code (see ThemedComboBox) — QSS
    `image: url(...)` requires a real file URL which is fragile across
    Qt's resource lookups.
    """
    p = theme.palette
    return (
        # White text-field; drop-down area stays transparent (no separate
        # button), the triangle is drawn by ThemedComboBox.paintEvent.
        f"QComboBox {{ background: {p.base_bg}; color: {p.text};"
        f" border: 1px solid {p.border}; border-radius: 4px;"
        f" min-height: 22px; padding: 0; }}"
        f"QComboBox:hover {{ border-color: {p.accent}; }}"
        f"QComboBox QLineEdit {{ background: transparent; color: {p.text};"
        f" border: none; margin: 0; }}"
        f"QComboBox::drop-down {{ subcontrol-origin: border;"
        f" subcontrol-position: top right; width: 20px;"
        f" background: transparent; border: none; }}"
        f"QComboBox QAbstractItemView {{ background: {p.base_bg};"
        f" color: {p.text}; border: 1px solid {p.border};"
        f" selection-background-color: {p.accent};"
        f" selection-color: {p.accent_text}; }}"
    )


class ThemedComboBox(QComboBox):
    """QComboBox that paints its own down-triangle on top of Qt's drop-down
    area. Avoids the QSS `image: url(...)` plumbing entirely.

    Text-input left padding is applied via QLineEdit.setTextMargins() because
    QSS padding on `QComboBox QLineEdit` is ignored when the line-edit is the
    combobox's internal editor."""

    LEFT_TEXT_PAD = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arrow_color = QColor(theme.palette.text_muted)
        self._apply_text_margins()

    def setEditable(self, editable):
        super().setEditable(editable)
        self._apply_text_margins()

    def _apply_text_margins(self):
        le = self.lineEdit()
        if le is not None:
            le.setTextMargins(self.LEFT_TEXT_PAD, 0, 0, 0)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Hand-paint the triangle inside the drop-down area (rightmost ~20 px).
        from PySide6.QtCore import QPointF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(self._arrow_color))
        painter.setBrush(QBrush(self._arrow_color))
        w = self.width()
        h = self.height()
        cx = w - 10  # center inside the 20-px drop-down area
        cy = h / 2  # widget vertical center
        half_w = 4
        half_h = 3  # makes the triangle vertically symmetric
        pts = [
            QPointF(cx - half_w, cy - half_h),
            QPointF(cx + half_w, cy - half_h),
            QPointF(cx, cy + half_h),
        ]
        from PySide6.QtGui import QPolygonF

        painter.drawPolygon(QPolygonF(pts))
        painter.end()


def _make_grid_pattern_icon(kind, size=18):
    """Build a small icon depicting a grid, dot pattern, or empty canvas."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    fg = QColor(60, 110, 175)  # blue
    if kind == "grid":
        pen = QPen(fg, 1)
        p.setPen(pen)
        step = (size - 4) / 3.0
        for i in range(4):
            x = 2 + step * i
            p.drawLine(x, 2, x, size - 2)
            p.drawLine(2, x, size - 2, x)
    elif kind == "dots":
        p.setBrush(QBrush(fg))
        p.setPen(Qt.PenStyle.NoPen)
        step = (size - 4) / 3.0
        for i in range(4):
            for j in range(4):
                cx = 2 + step * i
                cy = 2 + step * j
                p.drawEllipse(int(cx) - 1, int(cy) - 1, 2, 2)
    else:  # "none"
        pen = QPen(fg, 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawRect(2, 2, size - 5, size - 5)
    p.end()
    return QIcon(pm)


def _make_block_shape_icon(shape, w=26, h=16):
    """Build an icon showing a block with the specified corner style."""
    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Blue palette to match the active theme accent.
    p.setBrush(QBrush(QColor(140, 180, 230)))
    p.setPen(QPen(QColor(60, 110, 175), 1.2))
    if shape == "rectangular":
        radius = 0
    elif shape == "soft":
        radius = 3
    elif shape == "rounded":
        radius = 7
    else:
        radius = 0
    p.drawRoundedRect(2, 2, w - 4, h - 4, radius, radius)
    p.end()
    return QIcon(pm)


def _make_wire_routing_icon(mode, w=26, h=16):
    """Icon showing a wire routed straight ('normal') or detouring a block
    ('auto')."""
    from PySide6.QtGui import QPainterPath

    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(60, 110, 175), 1.4))
    y = h / 2.0
    path = QPainterPath()
    path.moveTo(2, y)
    if mode == "auto":
        # S-curve detouring up and over a small block in the middle.
        path.cubicTo(w * 0.35, y, w * 0.30, 3, w * 0.5, 3)
        path.cubicTo(w * 0.70, 3, w * 0.65, y, w - 2, y)
        p.drawPath(path)
        p.setBrush(QBrush(QColor(140, 180, 230)))
        p.setPen(QPen(QColor(60, 110, 175), 1))
        p.drawRect(int(w * 0.5) - 3, int(y) - 1, 6, 5)
    else:  # "normal": gentle straight-through bezier
        path.cubicTo(w * 0.4, y, w * 0.6, y, w - 2, y)
        p.drawPath(path)
    p.end()
    return QIcon(pm)


class NPSWorkflowApp(QMainWindow):
    """Main application window for the NPSView workflow editor."""

    def __init__(self):
        super().__init__()
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.app_dir)

        self.app_settings = {
            "fontSize": 11,
            "portFontSize": 9,
            "consoleFontSize": 9,
            "annotationFontSize": 12,
            "theme": "light",
            "gridPattern": "grid",
            "gridSize": 20,
            "nodeShape": "rectangular",
            "canvasBg": "",  # hex like "#f4f4f4"; empty = follow theme
            "wireRouting": "normal",  # "normal" (straight) | "auto" (route around blocks)
        }
        self._block_menu_config = None  # built after registry scan

        self._current_file_path_fallback = ""
        self._dirty_fallback = False
        self._tabs = []  # list of {"canvas": WorkflowCanvas, "file_path": str}
        self._canvas_fallback = None
        self.inspector = DataInspector()

        self._setup_ui()

        from utils.app_logger import connect_to_app

        connect_to_app(self)

        self._init_engine()
        self._create_palette()
        self._init_block_menu_config()
        self._create_block_menu()
        self._create_view_menu()
        self._load_most_recent()

        self.log("NPSflow initialized.")

    def _setup_ui(self):
        """Create the main application window."""
        self.setWindowTitle("NPSflow - Workflow Editor")
        self.resize(1400, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._create_toolbar()

        # Horizontal splitter: palette (left) | canvas+log (right)
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.h_splitter)

        # Palette placeholder (created after engine init in _create_palette)
        self._palette_placeholder = QWidget()
        self.h_splitter.addWidget(self._palette_placeholder)

        # Right side: vertical splitter with canvas + log
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.h_splitter.addWidget(v_splitter)

        # Welcome panel lives as the pinned Home tab (always tab 0).
        self._welcome_panel = WelcomePanel(self.root_dir)
        self._welcome_panel.newRequested.connect(self.new_workflow)
        self._welcome_panel.openRequested.connect(self.open_workflow)
        self._welcome_panel.openFileRequested.connect(self._open_workflow_path)

        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.setMovable(True)
        self._tab_widget.tabCloseRequested.connect(self._close_tab)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        # Insert Home as the first, pinned tab; show a home icon, no label.
        home_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon)
        self._tab_widget.addTab(self._welcome_panel, home_icon, "")
        self._tab_widget.setTabToolTip(0, "Home")
        bar = self._tab_widget.tabBar()
        for side in (QTabBar.ButtonPosition.RightSide, QTabBar.ButtonPosition.LeftSide):
            btn = bar.tabButton(0, side)
            if btn is not None:
                btn.hide()
                bar.setTabButton(0, side, None)
        # If the user drags another tab past Home, snap it back to slot 0.
        bar.tabMoved.connect(self._on_tab_moved)
        v_splitter.addWidget(self._tab_widget)

        # Status bar + Log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(5, 2, 5, 2)
        log_layout.setSpacing(2)

        # Header row: status label + debug toggle
        log_header = QHBoxLayout()
        log_header.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(theme.qss.status_label)
        log_header.addWidget(self.status_label)
        log_header.addStretch()
        self._debug_mode = False
        self._debug_toggle = QPushButton("Debug")
        self._debug_toggle.setCheckable(True)
        self._debug_toggle.setFixedSize(50, 18)
        self._debug_toggle.setStyleSheet(theme.qss.small_button_checkable)
        self._debug_toggle.toggled.connect(self._on_debug_toggled)
        log_header.addWidget(self._debug_toggle)

        self._copy_log_btn = QPushButton("Copy")
        self._copy_log_btn.setFixedSize(50, 18)
        self._copy_log_btn.setStyleSheet(theme.qss.small_button)
        self._copy_log_btn.clicked.connect(self._copy_log_to_clipboard)
        log_header.addWidget(self._copy_log_btn)

        self._clear_log_btn = QPushButton("Clear")
        self._clear_log_btn.setFixedSize(50, 18)
        self._clear_log_btn.setStyleSheet(theme.qss.small_button)
        self._clear_log_btn.clicked.connect(self._clear_log)
        log_header.addWidget(self._clear_log_btn)

        log_layout.addLayout(log_header)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        _mono = "Menlo" if sys.platform == "darwin" else "Consolas"
        self.log_area.setFont(QFont(_mono, self.app_settings["consoleFontSize"]))
        log_layout.addWidget(self.log_area)

        # Internal log storage: list of (message, is_debug)
        self._log_entries = []

        v_splitter.addWidget(log_widget)
        v_splitter.setSizes([700, 150])

        self.h_splitter.setSizes([180, 1220])

    def _create_toolbar(self):
        """Create the top toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File actions
        toolbar.addAction(self._make_action("New", self.new_workflow, "Ctrl+N"))
        toolbar.addAction(self._make_action("Open", self.open_workflow, "Ctrl+O"))
        toolbar.addAction(self._make_action("Save", self.save_workflow, "Ctrl+S"))
        toolbar.addAction(
            self._make_action("Save As", self.save_workflow_as, "Ctrl+Shift+S")
        )
        toolbar.addSeparator()

        # Edit actions
        toolbar.addAction(self._make_action("Delete", self.delete_selected))
        toolbar.addAction(self._make_action("Clear", self.clear_canvas))
        toolbar.addAction(self._make_action("Reset State", self.clear_block_states))
        toolbar.addSeparator()

        toolbar.addAction(
            self._make_action("Undo", lambda: self.canvas.undo(), "Ctrl+Z")
        )
        toolbar.addAction(
            self._make_action("Redo", lambda: self.canvas.redo(), "Ctrl+Y")
        )
        toolbar.addAction(self._make_action("Versions", self._show_version_history))
        toolbar.addSeparator()

        # Keep keyboard shortcuts for layout actions (no toolbar buttons)
        self.addAction(
            self._make_action("", lambda: self.canvas.auto_layout(), "Ctrl+A")
        )
        self.addAction(
            self._make_action("", lambda: self.canvas.fit_to_content(), "Ctrl+0")
        )

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Debug toggle
        self._debug_action = QAction("Debug", self)
        self._debug_action.setCheckable(True)
        self._debug_action.toggled.connect(self._on_debug_mode_toggled)
        toolbar.addAction(self._debug_action)
        toolbar.addSeparator()

        # Dark Mode toggle — also added to View menu later; sharing one QAction
        # keeps toolbar + menu states in sync automatically.
        self._dark_mode_action = QAction("Dark Mode", self, checkable=True)
        self._dark_mode_action.setChecked(get_theme_name() == "dark")
        self._dark_mode_action.toggled.connect(self._on_dark_mode_toggled)
        toolbar.addAction(self._dark_mode_action)
        toolbar.addSeparator()

        toolbar.addAction(self._make_action("?", self.show_shortcuts_help))
        toolbar.addAction(self._make_action("Settings", self.open_settings))

    def _make_action(self, text, slot, shortcut=None):
        """Create a QAction."""
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        return action

    # Home tab is pinned at tab-widget index 0 and is NOT in self._tabs.
    # Workflow tab N (0-based in self._tabs) lives at tab-widget index N + 1.
    HOME_TAB_INDEX = 0

    def _workflow_index(self, tab_index):
        """Map a tab-widget index to its index in self._tabs.

        Returns None for the Home tab or out-of-range indices.
        """
        if tab_index is None or tab_index <= self.HOME_TAB_INDEX:
            return None
        wi = tab_index - 1
        return wi if 0 <= wi < len(self._tabs) else None

    def _tab_index_for_workflow(self, workflow_index):
        """Inverse of _workflow_index: workflow N \u2192 tab-widget index N + 1."""
        return workflow_index + 1

    @property
    def canvas(self):
        """Return the active tab's canvas, or None when Home is active."""
        tw = getattr(self, "_tab_widget", None)
        if tw is not None:
            wi = self._workflow_index(tw.currentIndex())
            if wi is not None:
                return self._tabs[wi]["canvas"]
        return self._canvas_fallback

    @property
    def current_file_path(self):
        tw = getattr(self, "_tab_widget", None)
        if tw is not None:
            wi = self._workflow_index(tw.currentIndex())
            if wi is not None:
                return self._tabs[wi]["file_path"]
        return self._current_file_path_fallback

    @current_file_path.setter
    def current_file_path(self, value):
        tw = getattr(self, "_tab_widget", None)
        if tw is not None:
            idx = tw.currentIndex()
            wi = self._workflow_index(idx)
            if wi is not None:
                self._tabs[wi]["file_path"] = value
                label = os.path.basename(value) if value else "Untitled"
                if self._tabs[wi].get("dirty"):
                    label += " \u2022"
                tw.setTabText(idx, label)
                return
        self._current_file_path_fallback = value

    @property
    def _dirty(self):
        tw = getattr(self, "_tab_widget", None)
        if tw is not None:
            wi = self._workflow_index(tw.currentIndex())
            if wi is not None:
                return self._tabs[wi].get("dirty", False)
        return self._dirty_fallback

    @_dirty.setter
    def _dirty(self, value):
        tw = getattr(self, "_tab_widget", None)
        if tw is not None:
            wi = self._workflow_index(tw.currentIndex())
            if wi is not None:
                self._tabs[wi]["dirty"] = value
                return
        self._dirty_fallback = value

    def _init_engine(self):
        """Initialize the block registry and workflow engine."""
        self.registry = BlockRegistry()

        # Add project dir to Python path for imports
        if self.app_dir not in sys.path:
            sys.path.insert(0, self.app_dir)

        self.registry.scan_block_defs()

        self._create_engine()

        # Start on the Home tab; user creates/opens a workflow from there.
        self._current_file_path_fallback = ""
        self._canvas_fallback = None

        # Debug stepping state
        self._debug_loop = None  # QEventLoop used to pause execution
        self._debug_panel = None
        self._create_debug_panel()
        self._run_btn = None
        self._create_run_button()

    def _setup_canvas_callbacks(self, canvas):
        """Wire up all callbacks on a canvas instance."""
        canvas.block_double_click_callback = self._on_block_double_click
        canvas.block_edit_callback = self._on_block_cmd_click
        canvas.block_context_menu_callback = self._on_block_context_menu
        canvas.canvas_context_menu_callback = self._on_canvas_context_menu
        canvas.wire_change_callback = self._on_wire_changed
        canvas.delete_callback = self.delete_selected
        canvas.state_changed_callback = self._on_state_changed
        canvas.drop_block_callback = self._add_block_at_position
        canvas.resize_callback = self._reposition_floating_panels

    def _on_tab_moved(self, from_idx, to_idx):
        """Keep the Home tab pinned at index 0."""
        bar = self._tab_widget.tabBar()
        if from_idx == self.HOME_TAB_INDEX:
            bar.moveTab(to_idx, self.HOME_TAB_INDEX)
        elif to_idx == self.HOME_TAB_INDEX:
            bar.moveTab(self.HOME_TAB_INDEX, from_idx)

    def _add_canvas_tab(self, file_path="", label="Untitled"):
        """Create a new workflow canvas tab and return its index in self._tabs."""
        canvas = WorkflowCanvas(self.registry, self)
        # Seed the new canvas with the current app-level settings so freshly
        # loaded annotations / blocks pick up the active Block Style, grid,
        # and font sizes (otherwise the canvas's __init__ defaults win until
        # something nudges _apply_settings).
        s = self.app_settings
        canvas.base_font_size = s["fontSize"]
        canvas.base_port_font_size = s["portFontSize"]
        canvas.base_annotation_font_size = s["annotationFontSize"]
        canvas.base_node_shape = s["nodeShape"]
        canvas.auto_route_on_create = s.get("wireRouting") == "auto"
        canvas.set_grid_settings(s["gridPattern"], s["gridSize"])
        bg_hex = s.get("canvasBg") or theme.palette.canvas_bg
        canvas.set_theme(QColor(bg_hex), QColor(theme.palette.canvas_grid))
        canvas._recent_load_paths = []  # Load Data recent-files list (per workflow)
        self._setup_canvas_callbacks(canvas)
        tab_info = {"canvas": canvas, "file_path": file_path, "dirty": False}
        self._tabs.append(tab_info)
        wi = len(self._tabs) - 1
        tab_idx = self._tab_widget.addTab(canvas, label)
        self._tab_widget.setCurrentIndex(tab_idx)
        canvas.reset_view()
        return wi

    def _close_tab(self, index):
        """Close a canvas tab. Home tab (index 0) is non-closable."""
        wi = self._workflow_index(index)
        if wi is None:
            return
        tab = self._tabs[wi]

        # Prompt to save if tab has unsaved changes
        if tab.get("dirty"):
            if tab.get("is_super_block"):
                title = "Save Super Block"
                msg = "Save changes to the super block template before closing?"
            else:
                name = os.path.basename(tab.get("file_path", "")) or "Untitled"
                title = "Save Workflow"
                msg = f'Save changes to "{name}" before closing?'
            reply = QMessageBox.question(
                self,
                title,
                msg,
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                # Switch to the tab so _save_to_file operates on the right one
                self._tab_widget.setCurrentIndex(index)
                if tab.get("is_super_block"):
                    self._save_super_block(tab, tab.get("file_path", ""))
                else:
                    self.save_workflow()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        canvas = tab["canvas"]
        canvas.setScene(None)
        self._tabs.pop(wi)
        self._tab_widget.removeTab(index)

    def _on_tab_changed(self, index):
        """Update engine callbacks when switching tabs; refresh Home on entry."""
        if index == self.HOME_TAB_INDEX:
            self._welcome_panel.refresh()
            return
        wi = self._workflow_index(index)
        if wi is None:
            return
        canvas = self._tabs[wi]["canvas"]
        if not hasattr(canvas, "_recent_load_paths"):
            canvas._recent_load_paths = []
        recent_paths.bind(canvas._recent_load_paths)
        if hasattr(self, "engine"):
            self.engine.wire_remove_callback = canvas.remove_wire
        if getattr(self, "_debug_panel", None) is not None:
            self._debug_panel.setParent(canvas)
        if getattr(self, "_run_btn", None) is not None:
            self._run_btn.setParent(canvas)
            self._run_btn.show()
        self._reposition_floating_panels()
        canvas.resize_callback = self._reposition_floating_panels
        if hasattr(self, "block_palette"):
            self.block_palette._canvas_ref = canvas
            self.block_palette.refresh_current_view()
        # Reflect the new active canvas's errors in the floating Issues panel.
        self._refresh_issues_panel()

    def _create_debug_panel(self):
        """Create the floating debug control panel (hidden by default)."""
        panel = QWidget(self.canvas)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        self._dbg_label = QLabel("Paused")
        self._dbg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._dbg_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._dbg_step_btn = QPushButton("Step")
        self._dbg_step_btn.setToolTip("Run the next block (F10)")
        self._dbg_step_btn.setShortcut(QKeySequence("F10"))
        self._dbg_step_btn.clicked.connect(self._debug_step)
        btn_row.addWidget(self._dbg_step_btn)

        self._dbg_continue_btn = QPushButton("Continue")
        self._dbg_continue_btn.setToolTip("Run remaining blocks (F5)")
        self._dbg_continue_btn.setShortcut(QKeySequence("F5"))
        self._dbg_continue_btn.clicked.connect(self._debug_continue)
        btn_row.addWidget(self._dbg_continue_btn)

        self._dbg_stop_btn = QPushButton("Stop")
        self._dbg_stop_btn.setToolTip("Stop execution (Shift+F5)")
        self._dbg_stop_btn.setShortcut(QKeySequence("Shift+F5"))
        self._dbg_stop_btn.clicked.connect(self._debug_stop)
        btn_row.addWidget(self._dbg_stop_btn)

        outer.addLayout(btn_row)

        self._debug_panel = panel
        self._apply_debug_panel_theme()
        panel.adjustSize()
        panel.hide()

    def _apply_debug_panel_theme(self):
        """Apply theme-driven styles to the floating debug panel + buttons."""
        if not getattr(self, "_debug_panel", None):
            return
        p = theme.palette
        btn_style = (
            f"QPushButton {{ background: {p.button_bg}; color: {p.button_text};"
            f" border: 1px solid {p.border}; border-radius: 4px;"
            f" padding: 4px 14px; font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover}; }}"
            f"QPushButton:pressed {{ background: {p.button_bg_press}; }}"
            f"QPushButton:disabled {{ color: {p.text_muted};"
            f" background: {p.alt_base_bg}; }}"
        )
        for btn in (self._dbg_step_btn, self._dbg_continue_btn, self._dbg_stop_btn):
            btn.setStyleSheet(btn_style)
        self._dbg_label.setStyleSheet(
            f"QLabel {{ color: {p.text}; font-size: 12px; font-weight: bold;"
            f" border: none; background: transparent; }}"
        )
        # objectName-scoped selector keeps the bg rule from leaking into
        # children (which would otherwise paint over their own backgrounds).
        self._debug_panel.setObjectName("debugPanel")
        self._debug_panel.setStyleSheet(
            f"QWidget#debugPanel {{ background: {p.panel_bg};"
            f" border: 1px solid {p.border}; border-radius: 6px; }}"
        )

    def _create_run_button(self):
        """Create the floating Run All button on the canvas."""
        btn = QPushButton(self.canvas)
        btn.setFixedSize(36, 36)
        btn.setToolTip("Run All")
        btn.clicked.connect(self.run_all)
        btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(76, 175, 80, 220);"
            "  border: none; border-radius: 18px;"
            "  color: white; font-size: 18px; font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(56, 142, 60, 240);"
            "}"
            "QPushButton:pressed {"
            "  background: rgba(46, 125, 50, 255);"
            "}"
        )
        btn.setText("\u25b6")  # ▶ play triangle
        btn.move(12, 12)
        btn.raise_()
        self._run_btn = btn

    def _reposition_floating_panels(self):
        """Reposition all floating panels on the canvas."""
        dbg = getattr(self, "_debug_panel", None)
        if dbg is not None:
            dbg.adjustSize()
            pw = dbg.width()
            x = (self.canvas.width() - pw) // 2
            dbg.move(max(x, 0), 8)
        btn = getattr(self, "_run_btn", None)
        if btn is not None:
            btn.move(12, 12)
            btn.raise_()

    def _on_debug_mode_toggled(self, checked):
        """Toggle debug stepping mode."""
        self.engine.debug_mode = checked
        self.engine._debug_continue = False
        if checked:
            # Auto-enable debug console messages
            if not self._debug_mode:
                self._debug_toggle.setChecked(True)
            self.log("Debug mode ON — pipeline will pause before each block.")
        else:
            self.log("Debug mode OFF.")
            # If currently paused, resume and finish
            if self._debug_loop and self._debug_loop.isRunning():
                self.engine._debug_continue = True
                self._debug_loop.quit()
            if self._debug_panel:
                self._debug_panel.hide()

    def _debug_pause(self, block):
        """Called by the engine before running a block in debug mode.
        Blocks execution until the user clicks Step/Continue/Stop."""
        # Highlight the block about to run and lock selection
        self.canvas.select_block(block)
        self.canvas._debug_locked_block = block
        self.canvas.scroll_to_block(block)
        bid = getattr(block, "block_id", None)
        bid_tag = f"B{bid}" if bid is not None else ""
        label = f"Next: {block.display_name}" + (f"  [{bid_tag}]" if bid_tag else "")
        self._dbg_label.setText(label)
        self._debug_panel.show()
        self._reposition_debug_panel()
        self.status_label.setText(f"Debug: paused before '{block.display_name}'")
        QApplication.processEvents()

        # Block execution here using a local event loop
        self._debug_loop = QEventLoop()
        self._debug_loop.exec()
        self._debug_loop = None

    def _debug_step(self):
        """Run the next block, then pause again."""
        self.canvas._debug_locked_block = None
        if self._debug_loop and self._debug_loop.isRunning():
            self._debug_loop.quit()

    def _debug_continue(self):
        """Run all remaining blocks without pausing."""
        self.canvas._debug_locked_block = None
        self.engine._debug_continue = True
        self._debug_panel.hide()
        if self._debug_loop and self._debug_loop.isRunning():
            self._debug_loop.quit()

    def _debug_stop(self):
        """Stop execution entirely."""
        self.canvas._debug_locked_block = None
        self.engine.stop_requested = True
        self.engine._debug_continue = False
        self._debug_panel.hide()
        if self._debug_loop and self._debug_loop.isRunning():
            self._debug_loop.quit()

    def _create_palette(self):
        """Create the block palette panel and replace the placeholder."""
        self.block_palette = BlockPalettePanel(self.registry)
        self.block_palette._canvas_ref = self.canvas
        self.block_palette._sb_delete_zone.delete_requested = (
            self._on_super_block_delete
        )

        # Replace placeholder with actual palette
        self.h_splitter.replaceWidget(0, self.block_palette)
        self._palette_placeholder.deleteLater()
        self.h_splitter.setSizes([180, 1220])

    _MENU_CONFIG_FILE = "block_menu_config.json"

    def _menu_config_path(self):
        return os.path.join(self.root_dir, self._MENU_CONFIG_FILE)

    def _init_block_menu_config(self):
        """Load saved block menu config, falling back to defaults.

        After loading, any blocks that were added or removed from the
        registry since the config was saved are reconciled automatically.
        """
        from utils.block_menu_editor import build_default_menu_config

        path = self._menu_config_path()
        loaded = None
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    loaded = json.load(f)
            except Exception:
                loaded = None

        if loaded:
            self._block_menu_config = self._reconcile_menu_config(loaded)
        else:
            self._block_menu_config = build_default_menu_config(self.registry)
        self._apply_menu_config()

    def _save_block_menu_config(self):
        """Persist the current block menu config to disk."""
        try:
            with open(self._menu_config_path(), "w") as f:
                json.dump(self._block_menu_config, f, indent=2)
        except Exception as e:
            self.log(f"Warning: could not save block menu config: {e}")

    def _reconcile_menu_config(self, config):
        """Merge a saved config with the current registry.

        - Blocks that no longer exist in the registry are removed.
        - New blocks in the registry that aren't in the config are added to
          their category's section (visible in the palette), or a new section
          if none exists — never the hidden Unused bucket.
        """
        from utils.block_menu_editor import add_missing_blocks

        # Remove blocks that no longer exist or are hidden (reroute, no name)
        def _keep(def_name):
            if not self.registry.has(def_name):
                return False
            defn = self.registry.get(def_name)
            return not defn.get("isReroute") and bool(defn.get("displayName"))

        for entry in config:
            entry["blocks"] = [
                b for b in entry.get("blocks", []) if _keep(b["defName"])
            ]
            for sc in entry.get("subcategories", []):
                sc["blocks"] = [b for b in sc["blocks"] if _keep(b["defName"])]

        # New registry blocks → their category section (visible).
        return add_missing_blocks(config, self.registry)

    # Category ordering and labels matching MATLAB version
    CATEGORY_ORDER = [
        "DataIO",
        "Variables",
        "FlowControl",
        "Preprocessing",
        "Filters",
        "Template",
        "Detection",
        "FeatureExtraction",
        "Analysis",
        "Utility",
    ]
    CATEGORY_LABELS = {
        "DataIO": "IO",
        "Variables": "Variables",
        "FlowControl": "Flow Control",
        "Preprocessing": "Preprocessing",
        "Filters": "Filters",
        "Template": "Template",
        "Detection": "Detection",
        "FeatureExtraction": "Feature Extraction",
        "Analysis": "Analysis",
        "Utility": "Utility",
    }
    # DataIO block names classified as "Input"
    _IO_INPUT_KEYWORDS = {
        "load",
        "import",
        "read",
        "constant",
        "toggle",
        "unpack",
        "filepath",
        "text",
    }

    def _populate_block_menu(self, parent_menu, action_factory):
        """Populate a menu with block categories and items.

        Args:
            parent_menu: QMenu to populate.
            action_factory: callable(def_name) that returns the callback
                            to invoke when a block item is selected.
        """
        groups = self.registry.list_by_category()
        used = set()

        for cat in self.CATEGORY_ORDER:
            if cat not in groups:
                continue
            used.add(cat)
            label = self.CATEGORY_LABELS.get(cat, cat)
            cat_menu = parent_menu.addMenu(label)
            defs = sorted(
                [d for d in groups[cat] if not d.get("isReroute")],
                key=lambda d: d["displayName"],
            )

            if cat == "DataIO":
                # Split into Input / Output submenus (matches MATLAB)
                input_defs = [
                    d
                    for d in defs
                    if any(kw in d["name"].lower() for kw in self._IO_INPUT_KEYWORDS)
                ]
                output_defs = [d for d in defs if d not in input_defs]

                if input_defs:
                    in_menu = cat_menu.addMenu("Input")
                    for defn in input_defs:
                        in_menu.addAction(
                            defn["displayName"], action_factory(defn["name"])
                        )
                if output_defs:
                    out_menu = cat_menu.addMenu("Output")
                    for defn in output_defs:
                        out_menu.addAction(
                            defn["displayName"], action_factory(defn["name"])
                        )
            else:
                for defn in defs:
                    cat_menu.addAction(
                        defn["displayName"], action_factory(defn["name"])
                    )

        # Include any categories not in the predefined order
        for cat in sorted(groups.keys()):
            if cat in used:
                continue
            cat_menu = parent_menu.addMenu(cat.replace("_", " "))
            for defn in sorted(
                [d for d in groups[cat] if not d.get("isReroute")],
                key=lambda d: d["displayName"],
            ):
                cat_menu.addAction(defn["displayName"], action_factory(defn["name"]))

    def _create_block_menu(self):
        """Build hierarchical block menu from registry."""
        menubar = self.menuBar()
        block_menu = menubar.addMenu("Add Block")
        self._populate_block_menu(
            block_menu,
            lambda dn: lambda: self._add_block_from_menu(dn),
        )

    def _create_view_menu(self):
        """Build the View menu (theme toggle, dock toggles).

        Reuses the toolbar's ``_dark_mode_action`` and the issues dock's
        ``toggleViewAction`` so menu + toolbar + dock close button all stay
        in sync automatically.
        """
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._dark_mode_action)
        view_menu.addSeparator()
        self._show_issues_action = QAction("Show Issues", self)
        self._show_issues_action.triggered.connect(self._show_issues_panel)
        view_menu.addAction(self._show_issues_action)

    # ------------------------------------------------------------------
    # Issues panel
    # ------------------------------------------------------------------

    def _show_issues_panel(self):
        """Open the floating Issues panel near the first errored block.

        No-op if the current canvas has no errored blocks.
        """
        from PySide6.QtWidgets import QListWidget, QListWidgetItem

        canvas = self.canvas
        blocks = canvas.blocks if canvas is not None else []
        errored = [b for b in blocks if getattr(b, "status", None) == "error"]
        if not errored:
            return

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 10)
        body_layout.setSpacing(6)

        n = len(errored)
        summary = QLabel("1 error." if n == 1 else f"{n} errors.")
        summary.setStyleSheet(f"color: {theme.palette.text}; font-weight: bold;")
        body_layout.addWidget(summary)

        lst = QListWidget()
        lst.setAlternatingRowColors(True)
        lst.setWordWrap(True)
        lst.setTextElideMode(Qt.TextElideMode.ElideNone)
        for b in errored:
            msg = (b.error_message or "Block failed").strip()
            item = QListWidgetItem(f"{b.display_name}\n{msg}")
            item.setData(Qt.ItemDataRole.UserRole, b)
            item.setToolTip(msg)
            lst.addItem(item)
        lst.setCurrentRow(0)
        lst.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        lst.customContextMenuRequested.connect(
            lambda pos: self._issues_context_menu(lst, pos)
        )
        body_layout.addWidget(lst, 1)

        # Action row: Re-run + Close. Jump is gone — opening the panel now
        # auto-centers the canvas on the errored block, so a dedicated Jump
        # is redundant.
        actions = QHBoxLayout()
        actions.addStretch(1)
        btn_rerun = QPushButton("Re-run from here")
        btn_close = QPushButton("Close")
        actions.addWidget(btn_rerun)
        actions.addWidget(btn_close)
        body_layout.addLayout(actions)

        def _selected():
            it = lst.currentItem()
            return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

        def _rerun():
            b = _selected()
            if b is None:
                return
            # Close the panel before re-running so it doesn't sit over the
            # block whose error just got cleared.
            shell = FloatingShell._active
            if shell is not None:
                shell._cancel(silent=True)
            self._run_from_block(b)

        def _close():
            shell = FloatingShell._active
            if shell is not None:
                shell._cancel(silent=True)

        btn_rerun.clicked.connect(_rerun)
        btn_close.clicked.connect(_close)

        # Open the panel first (it auto-centers in the canvas), then scroll
        # the canvas so the errored block sits just to the upper-left of the
        # panel — that way the user sees the block AND the error details
        # side by side instead of one covering the other.
        shell = FloatingShell.open(
            "Issues",
            canvas,
            body,
            near_block=errored[0],
            show_footer=False,
        )
        self._focus_block_beside_panel(errored[0], shell)

    def _issues_context_menu(self, lst, pos):
        """Right-click menu on an Issues row: copy one error or all of them."""
        item = lst.itemAt(pos)
        menu = QMenu(lst)

        def _msg(it):
            b = it.data(Qt.ItemDataRole.UserRole)
            return (getattr(b, "error_message", "") or "Block failed").strip()

        if item is not None:
            menu.addAction("Copy", lambda: QApplication.clipboard().setText(_msg(item)))
        if lst.count() > 1:
            menu.addAction(
                "Copy All",
                lambda: QApplication.clipboard().setText(
                    "\n\n".join(_msg(lst.item(i)) for i in range(lst.count()))
                ),
            )
        if menu.isEmpty():
            return
        menu.exec(lst.mapToGlobal(pos))

    def _refresh_issues_panel(self):
        """Auto-show on first error; close when all clear; rebuild only when
        the set of errored block ids actually changes (avoids tearing down
        the panel on every unrelated status transition)."""
        canvas = self.canvas
        blocks = canvas.blocks if canvas is not None else []
        errored = [b for b in blocks if getattr(b, "status", None) == "error"]
        ids = tuple(b.id for b in errored)
        if ids == getattr(self, "_last_issue_ids", ()):
            return
        self._last_issue_ids = ids

        shell = FloatingShell._active
        is_open_issues = shell is not None and shell._title.text() == "Issues"

        if errored and not is_open_issues:
            self._show_issues_panel()
        elif not errored and is_open_issues:
            shell._cancel(silent=True)
        elif errored and is_open_issues:
            shell._cancel(silent=True)
            self._show_issues_panel()

    def _show_version_history(self):
        """Open a dialog listing past saved versions of the current workflow.

        Each entry was captured the last time the workflow was saved over an
        existing file — the *previous* state is what gets stored, so the
        most recent entry shows the state just before the latest save.
        Restoring replaces the canvas with the chosen snapshot; the current
        state is pushed onto the undo stack first so the user can roll back
        if they pick the wrong one.
        """
        from PySide6.QtWidgets import QListWidget, QListWidgetItem

        canvas = self.canvas
        history = list(getattr(canvas, "_workflow_history", []) or [])
        if not history:
            QMessageBox.information(
                self,
                "Version History",
                "No previous versions saved yet — history is captured on each "
                "Save over an existing file.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Version History")
        try:
            from utils.dialog_style import apply_dialog_style

            apply_dialog_style(dlg)
        except Exception:
            pass
        dlg.resize(520, 400)
        layout = QVBoxLayout(dlg)
        layout.addWidget(
            QLabel(
                f"{len(history)} previous version(s). "
                "Restore replaces the current canvas — undo to revert."
            )
        )

        lst = QListWidget()
        for entry in history:
            ts = entry.get("timestamp", "(unknown)")
            msg = entry.get("message", "") or ""
            label = f"{ts}    {msg}".strip()
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            lst.addItem(item)
        lst.setCurrentRow(0)
        layout.addWidget(lst, 1)

        btns = QHBoxLayout()
        btn_restore = QPushButton("Restore")
        btn_delete = QPushButton("Delete")
        btn_close = QPushButton("Close")
        btns.addWidget(btn_restore)
        btns.addWidget(btn_delete)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        layout.addLayout(btns)

        def _selected():
            it = lst.currentItem()
            return it.data(Qt.ItemDataRole.UserRole) if it else None

        def _restore():
            entry = _selected()
            if entry is None:
                return
            state = entry.get("state")
            if not isinstance(state, dict):
                return
            # Push current state to undo before clobbering it.
            self.canvas.push_undo()
            map_state_paths(state, resolve_project_path)
            self.canvas.load_from_struct(state)
            self._apply_workflow_view_settings(state)
            self._refresh_dynamic_ports()
            self._dirty = True
            self._update_tab_dirty_indicator()
            self.log(f"Restored version from {entry.get('timestamp', '')}.")
            dlg.accept()

        def _delete():
            entry = _selected()
            if entry is None:
                return
            history.remove(entry)
            self.canvas._workflow_history = history
            self._dirty = True
            self._update_tab_dirty_indicator()
            row = lst.currentRow()
            lst.takeItem(row)

        btn_restore.clicked.connect(_restore)
        btn_delete.clicked.connect(_delete)
        btn_close.clicked.connect(dlg.reject)
        dlg.exec()

    def _focus_block_beside_panel(self, block, shell):
        """Scroll the canvas so ``block`` sits just outside the upper-left
        corner of ``shell`` (which is already positioned over the canvas).
        Used by the Issues panel so the failed block is visible next to its
        error details instead of being hidden behind the panel.
        """
        from PySide6.QtCore import QPointF

        canvas = self.canvas
        if canvas is None or block not in canvas.blocks or shell is None:
            return
        canvas.select_block(block)

        bx, by = block.position
        bw, bh = block.size
        block_scene_cx = bx + bw / 2
        block_scene_cy = -by + bh / 2

        # Shell is parented to the QGraphicsView itself; use the view's own
        # geometry rather than viewport() so the coordinate frames match.
        vp_w = canvas.width()
        vp_h = canvas.height()
        panel_geom = shell.geometry()
        # Block's intended center in viewport pixels: hug the panel's
        # left edge with a small gap, vertically aligned with the panel top.
        gap = 24
        target_vx = panel_geom.left() - bw / 2 - gap
        target_vy = panel_geom.top() + bh / 2
        # Keep the block fully inside the viewport.
        target_vx = max(bw / 2 + 8, min(target_vx, vp_w - bw / 2 - 8))
        target_vy = max(bh / 2 + 8, min(target_vy, vp_h - bh / 2 - 8))

        scale = canvas.transform().m11() or 1.0
        # Find the scene point that, placed at viewport center, lands the
        # block center at (target_vx, target_vy).
        dx_view = target_vx - vp_w / 2
        dy_view = target_vy - vp_h / 2
        center_x = block_scene_cx - dx_view / scale
        center_y = block_scene_cy - dy_view / scale
        canvas.centerOn(QPointF(center_x, center_y))

    # ------------------------------------------------------------------
    # Properties panel
    # ------------------------------------------------------------------

    def _on_dark_mode_toggled(self, checked):
        """Activate light/dark theme, restyle known widgets, persist preference."""
        name = "dark" if checked else "light"
        set_theme(name)
        app = QApplication.instance()
        if app is not None:
            app.setPalette(build_qpalette())
        self._apply_theme_to_widgets()
        try:
            recent_files.set_setting(self.root_dir, "theme", name)
        except Exception:
            pass

    def _apply_theme_to_widgets(self):
        """Re-apply theme-driven styles to all known UI surfaces."""
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(theme.qss.status_label)
        if hasattr(self, "_debug_toggle"):
            self._debug_toggle.setStyleSheet(theme.qss.small_button_checkable)
        if hasattr(self, "_copy_log_btn"):
            self._copy_log_btn.setStyleSheet(theme.qss.small_button)
        if hasattr(self, "_clear_log_btn"):
            self._clear_log_btn.setStyleSheet(theme.qss.small_button)

        # Block palette panel
        if hasattr(self, "block_palette") and self.block_palette is not None:
            self.block_palette.apply_theme()

        # Any open floating editor / shell / message picks up the new theme.
        for cls in (FloatingParamEditor, FloatingShell, FloatingMessage):
            active = getattr(cls, "_active", None)
            if active is not None:
                try:
                    active.apply_theme()
                except Exception:
                    pass

        # Floating debug controls panel
        self._apply_debug_panel_theme()

        # Welcome panel (blank-state view)
        if getattr(self, "_welcome_panel", None) is not None:
            self._welcome_panel.apply_theme()

        # All canvases (every open tab + fallback) — re-skin and repaint.
        # apply_theme() resets canvas_bg to the theme default; honour any
        # user override (Settings → Background) immediately after.
        custom_bg = (
            self.app_settings.get("canvasBg", "")
            if hasattr(self, "app_settings")
            else ""
        )
        for tab in getattr(self, "_tabs", []):
            c = tab.get("canvas")
            if c is not None:
                c.apply_theme()
                if custom_bg:
                    c.set_theme(QColor(custom_bg), QColor(theme.palette.canvas_grid))
                c.scene().update()
        fb = getattr(self, "_canvas_fallback", None)
        if fb is not None:
            fb.apply_theme()
            if custom_bg:
                fb.set_theme(QColor(custom_bg), QColor(theme.palette.canvas_grid))
            fb.scene().update()

        # Re-render log entries (debug-span HTML uses theme.text_muted).
        if hasattr(self, "_log_entries"):
            self._render_log()

    # --- Actions ---

    def _add_block_from_menu(self, def_name):
        """Add a block at the last mouse position, or view center."""
        mouse_pos = self.canvas._last_mouse_pos
        if mouse_pos and mouse_pos != (0, 0):
            pos = (mouse_pos[0] - 70, mouse_pos[1] + 40)
        else:
            center = self.canvas.get_view_center()
            pos = (center[0] - 70, center[1] + 40)
        block = self.canvas.add_block(def_name, pos)
        block.node_shape = self.app_settings["nodeShape"]
        self._init_block_after_add(block)
        self.canvas.update_block_graphics(block)
        self.log(f"Added block: {def_name}")

    def _is_boundary_block(self, block):
        """Check if a block is a super block boundary block (Inputs/Outputs)."""
        return block.id in ("_sb_inputs", "_sb_outputs")

    def delete_selected(self):
        if self.canvas._multi_drag_blocks:
            blocks = [
                b
                for b in self.canvas._multi_drag_blocks
                if not self._is_boundary_block(b)
            ]
            self.canvas._deselect_multi()
            if blocks:
                self.canvas.push_undo()
                for block in blocks:
                    self.canvas._remove_block_no_undo(block)
                self.block_palette.refresh_current_view()
                self.log(f"Deleted {len(blocks)} blocks.")
        elif self.canvas.selected_block:
            if self._is_boundary_block(self.canvas.selected_block):
                self.log("Cannot delete boundary block.")
                return
            self._delete_block(self.canvas.selected_block)
        elif self.canvas.selected_wire:
            self.canvas.delete_selected()
            self.log("Deleted wire.")
        elif self.canvas.selected_annotation:
            self.canvas.delete_selected()
            self.log("Deleted annotation.")
        elif self.canvas.selected_text_label:
            self.canvas.delete_selected()
            self.log("Deleted text label.")
        else:
            self.log("Nothing selected to delete.")

    def _delete_block(self, block):
        name = block.display_name
        self.canvas.remove_block(block)
        self.block_palette.refresh_current_view()
        self.log(f"Deleted block: {name}")

    def _duplicate_blocks(self, blocks):
        """Clone the given blocks with an offset, preserving parameters,
        port settings, and internal wire connections."""
        import copy as _copy

        offset = 30
        self.canvas.push_undo()
        block_map = {}  # old block id -> new block
        new_blocks = []
        src_set = {id(b) for b in blocks}

        for src in blocks:
            if self._is_boundary_block(src):
                continue
            new_pos = (src.position[0] + offset, src.position[1] - offset)
            new_block = self.canvas.registry.create_block(src.definition_name, new_pos)
            new_block.block_id = self.canvas._next_block_id
            self.canvas._next_block_id += 1
            new_block.node_shape = src.node_shape
            new_block.size = src.size
            new_block.color = src.color
            new_block.parameters = _copy.deepcopy(src.parameters)
            if src.is_custom_name:
                new_block.display_name = src.display_name
                new_block.is_custom_name = True

            # Rebuild ports from source block to preserve dynamic ports
            new_block.input_ports = []
            for i, p in enumerate(src.input_ports):
                np_ = Port(
                    name=p.name,
                    direction="input",
                    type_=p.type,
                    required=p.required,
                    description=p.description,
                )
                np_.display_name = p.display_name
                np_.parent_block = new_block
                np_.index = i + 1
                new_block.input_ports.append(np_)

            new_block.output_ports = []
            for i, p in enumerate(src.output_ports):
                np_ = Port(
                    name=p.name,
                    direction="output",
                    type_=p.type,
                    required=getattr(p, "required", True),
                    description=p.description,
                )
                np_.display_name = p.display_name
                np_.parent_block = new_block
                np_.index = i + 1
                new_block.output_ports.append(np_)

            self.canvas.blocks.append(new_block)
            item = BlockGraphicsItem(new_block, self.canvas)
            self.canvas._scene.addItem(item)
            self.canvas._block_items[new_block.id] = item
            self._init_block_after_add(new_block)

            block_map[id(src)] = new_block
            new_blocks.append(new_block)

        # Duplicate internal wires (both endpoints in the duplicated set)
        for wire in self.canvas.wires:
            sp = wire.source_port
            dp = wire.dest_port
            if not sp or not dp:
                continue
            src_blk = sp.parent_block
            dst_blk = dp.parent_block
            if id(src_blk) not in block_map or id(dst_blk) not in block_map:
                continue
            new_src = block_map[id(src_blk)]
            new_dst = block_map[id(dst_blk)]
            new_sp = new_src.get_output_port(sp.name)
            new_dp = new_dst.get_input_port(dp.name)
            if new_sp and new_dp:
                w = WireConnection(new_sp, new_dp)
                w.color = new_src.color
                self.canvas.wires.append(w)
                wi = WireGraphicsItem(w, self.canvas)
                self.canvas._scene.addItem(wi)
                self.canvas._wire_items[w.id] = wi

        # Update graphics and select duplicated blocks
        if new_blocks:
            for b in new_blocks:
                self.canvas.update_block_graphics(b)
            self.canvas._update_scene_rect()
            self.canvas._minimap.update()
            self.canvas._deselect_multi()
            for b in new_blocks:
                b.set_selected(True)
                self.canvas._multi_drag_blocks.append(b)
                self.canvas.update_block_graphics(b)
            if len(new_blocks) == 1:
                self.canvas.select_block(new_blocks[0])
        if len(new_blocks) == 1:
            self.log(f"Duplicated block: {new_blocks[0].display_name}")
        elif new_blocks:
            self.log(f"Duplicated {len(new_blocks)} blocks.")

    def clear_canvas(self):
        reply = QMessageBox.question(
            self,
            "Clear Canvas",
            "Clear all blocks and wires from the canvas?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.push_undo()
            self.canvas.blocks.clear()
            self.canvas.wires.clear()
            self.canvas.redraw()
            self.block_palette.refresh_current_view()
            self.log("Canvas cleared.")

    def clear_block_states(self):
        """Reset every block on the canvas to a fresh state: parameters
        revert to the definition defaults, cached outputs/status are cleared.
        Port arrangement and names are preserved via the saved port defs."""
        blocks = list(self.canvas.blocks)
        if not blocks:
            self.log("No blocks on canvas to reset.")
            return
        reply = QMessageBox.question(
            self,
            "Reset Block States",
            f"Reset state on all {len(blocks)} blocks?\n\n"
            "Parameters revert to defaults and cached outputs are cleared. "
            "Port arrangement, names, and wires are preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.canvas.push_undo()
        count = 0
        for block in blocks:
            port_defs_in = block.parameters.get("inputPortDefs")
            port_defs_out = block.parameters.get("outputPortDefs")

            defn = self.registry.definitions.get(block.definition_name)
            new_params = {}
            if defn:
                defaults = defn.get("defaultParameters") or {}
                for k, v in defaults.items():
                    new_params[k] = v
            if port_defs_in is not None:
                new_params["inputPortDefs"] = port_defs_in
            if port_defs_out is not None:
                new_params["outputPortDefs"] = port_defs_out
            block.parameters = new_params

            block.output_data = {}
            block.status = "pending"
            block.previous_status = "pending"
            block.error_message = ""
            for p in block.output_ports:
                p.has_data = False

            self.canvas.update_block_graphics(block)
            count += 1

        self.canvas.redraw()
        self.log(f"Reset state on {count} block{'s' if count != 1 else ''}.")

    def new_workflow(self):
        self._add_canvas_tab()
        self._add_start_block()
        self.log("New workflow tab created.")

    def open_workflow(self):
        workflows_dir = os.path.join(self.root_dir, "workflows")
        os.makedirs(workflows_dir, exist_ok=True)

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Workflow", workflows_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        self._open_workflow_path(filepath)

    def _open_workflow_path(self, filepath):
        """Open a workflow file by path. Used by the picker and the welcome panel."""
        # Check if already open in a tab
        for i, tab in enumerate(self._tabs):
            if tab["file_path"] == filepath:
                self._tab_widget.setCurrentIndex(self._tab_index_for_workflow(i))
                return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            label = self._load_workflow_data(data, filepath)
            self.log(f"Loaded: {label}")
            if "viewState" not in data:
                self.canvas.scroll_to_start()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading workflow:\n{e}")

    def _load_workflow_data(self, data, filepath):
        """Shared load sequence: new tab, canvas restore, palette state."""
        label = os.path.basename(filepath)
        map_state_paths(data, resolve_project_path)
        self._add_canvas_tab(file_path=filepath, label=label)
        self.canvas.load_from_struct(data)
        self.canvas._workflow_history = list(data.get("history") or [])
        self._apply_workflow_view_settings(data)
        self._refresh_dynamic_ports()
        self._restore_recent_blocks(data)
        self._restore_recent_load_paths(data)
        if hasattr(self, "block_palette"):
            self.block_palette.refresh_current_view()
        recent_files.push(self.root_dir, filepath)
        return label

    def _apply_workflow_view_settings(self, data):
        """Restore per-workflow visual settings persisted in the file."""
        changed = False
        style = data.get("blockStyle")
        if style and style != self.app_settings.get("nodeShape"):
            self.app_settings["nodeShape"] = style
            changed = True
        pattern = data.get("gridPattern")
        if pattern and pattern != self.app_settings.get("gridPattern"):
            self.app_settings["gridPattern"] = pattern
            changed = True
        size = data.get("gridSize")
        if isinstance(size, (int, float)) and int(size) != self.app_settings.get(
            "gridSize"
        ):
            self.app_settings["gridSize"] = int(size)
            changed = True
        bg = data.get("canvasBg")
        if bg is not None and bg != self.app_settings.get("canvasBg", ""):
            self.app_settings["canvasBg"] = bg
            changed = True
        routing = data.get("wireRouting")
        if routing and routing != self.app_settings.get("wireRouting"):
            self.app_settings["wireRouting"] = routing
            changed = True
        if changed:
            self._apply_settings()

    def _restore_recent_blocks(self, data):
        """Restore the palette's recent-blocks list from a loaded workflow.

        Called from every workflow load path so the recent list persists with
        the file rather than living only for the session. A missing/empty
        ``recentBlocks`` leaves the current list untouched (older files)."""
        if not hasattr(self, "block_palette"):
            return
        recent = data.get("recentBlocks", [])
        if recent:
            cap = self.block_palette.MAX_RECENT
            self.block_palette._recent_names = list(recent)[:cap]
            self.block_palette._refresh_recent_grid()

    def _restore_recent_load_paths(self, data):
        """Restore the workflow's shared recent-files list (Load Data block).

        Always replaces the active canvas's list — a missing key resets it to
        empty so recent files never carry over from another workflow. Older
        files predating the shared list are migrated by collecting each Load
        Data block's ``lastPath`` across the live state and the version history
        (newest-first, deduped), so the picker shows a real "last N" on first
        open rather than just the single most-recent path."""
        paths = [
            p for p in (data.get("recentLoadPaths") or []) if isinstance(p, str) and p
        ]
        if not paths:
            seen = set()
            states = [data] + [s.get("state", {}) for s in (data.get("history") or [])]
            for st in states:
                for b in st.get("blocks", []):
                    lp = (b.get("parameters") or {}).get("lastPath")
                    if isinstance(lp, str) and lp and lp not in seen:
                        seen.add(lp)
                        paths.append(lp)
        self.canvas._recent_load_paths = paths[: recent_paths.MAX_RECENT]
        recent_paths.bind(self.canvas._recent_load_paths)

    def save_workflow(self):
        if self.canvas is None:
            return
        if not self.current_file_path:
            self.save_workflow_as()
        else:
            self._save_to_file(self.current_file_path)

    def save_workflow_as(self):
        if self.canvas is None:
            return
        workflows_dir = os.path.join(self.root_dir, "workflows")
        os.makedirs(workflows_dir, exist_ok=True)

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow",
            os.path.join(workflows_dir, "my_pipeline.json"),
            "JSON Files (*.json)",
        )
        if not filepath:
            return

        self._save_to_file(filepath)
        self.current_file_path = filepath

    def _save_to_file(self, filepath):
        try:
            # Check if current tab is a super block
            wi = self._workflow_index(self._tab_widget.currentIndex())
            tab = self._tabs[wi] if wi is not None else {}
            if tab.get("is_super_block"):
                self._save_super_block(tab, filepath)
                return

            data = self.canvas.to_struct()
            data["version"] = "1.0"
            data["name"] = "NPS Pipeline"
            data["created"] = datetime.now().isoformat()
            data["recentBlocks"] = list(self.block_palette._recent_names)
            data["recentLoadPaths"] = list(
                getattr(self.canvas, "_recent_load_paths", [])
            )
            data["blockStyle"] = self.app_settings["nodeShape"]
            data["gridPattern"] = self.app_settings["gridPattern"]
            data["gridSize"] = self.app_settings["gridSize"]
            data["canvasBg"] = self.app_settings.get("canvasBg", "")
            data["wireRouting"] = self.app_settings.get("wireRouting", "normal")

            # Build the version history: take the previously-saved state
            # off disk (if any), turn it into a snapshot, and prepend it
            # to the in-memory history. Cap at MAX_HISTORY entries so the
            # workflow JSON doesn't grow unbounded.
            MAX_HISTORY = 20
            history = list(getattr(self.canvas, "_workflow_history", []) or [])
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r") as f:
                        prev = json.load(f)
                    prev.pop("history", None)
                    snapshot = {
                        "timestamp": prev.get("created", datetime.now().isoformat()),
                        "message": "",
                        "state": prev,
                    }
                    history = [snapshot] + history
                except Exception:
                    pass
            history = history[:MAX_HISTORY]
            data["history"] = history
            self.canvas._workflow_history = history

            # Persist file paths project-relative so the JSON carries no
            # machine-specific prefixes; resolved back on load/restore.
            map_state_paths(data, to_project_relative)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._dirty = False
            self._update_tab_dirty_indicator()
            self.log(f"Saved: {filepath}")
            recent_files.push(self.root_dir, filepath)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving:\n{e}")

    def _save_super_block(self, tab, filepath):
        """Save a super block template from the current canvas tab."""
        canvas = tab["canvas"]
        template = tab.get("template_data", {})

        boundary_ids = {"_sb_inputs", "_sb_outputs"}

        # Separate inner blocks from boundary blocks
        inner_blocks = [b for b in canvas.blocks if b.id not in boundary_ids]
        inner_block_ids = {b.id for b in inner_blocks}

        # Inner wires: both ends are inner blocks
        inner_wires = []
        for w in canvas.wires:
            src = w.source_port.parent_block if w.source_port else None
            dst = w.dest_port.parent_block if w.dest_port else None
            if src and dst and src.id in inner_block_ids and dst.id in inner_block_ids:
                inner_wires.append(w)

        template["innerBlocks"] = [b.to_struct() for b in inner_blocks]
        template["innerWires"] = [w.to_struct() for w in inner_wires]

        # Read exposed inputs from Inputs block's output ports (wired)
        in_block = next((b for b in canvas.blocks if b.id == "_sb_inputs"), None)
        exposed_inputs = []
        mapped_inner_inputs = set()  # (block_id, port_name) already mapped
        if in_block:
            for port in in_block.output_ports:
                if port.name in ("addOutput",):
                    continue
                for wire in port.connections:
                    dst = wire.dest_port
                    if (
                        dst
                        and dst.parent_block
                        and dst.parent_block.id in inner_block_ids
                    ):
                        exposed_inputs.append(
                            {
                                "name": port.display_name or port.name,
                                "innerBlock": dst.parent_block.id,
                                "innerPort": dst.name,
                                "type": port.type or "any",
                            }
                        )
                        mapped_inner_inputs.add((dst.parent_block.id, dst.name))
                        break

        # Also detect inner block input ports with no internal wire and no
        # boundary wire — these are unwired exposed inputs that should be
        # included so they appear as ports on the super block
        internal_dest = set()
        for w in inner_wires:
            if w.dest_port:
                internal_dest.add((w.dest_port.parent_block.id, w.dest_port.name))
        seen_names = {ei["name"] for ei in exposed_inputs}
        for b in inner_blocks:
            for p in b.input_ports:
                if p.name in ("addInput",):
                    continue
                key = (b.id, p.name)
                if key not in internal_dest and key not in mapped_inner_inputs:
                    ename = p.display_name or p.name
                    base = ename
                    i = 1
                    while ename in seen_names:
                        ename = f"{base}_{i}"
                        i += 1
                    seen_names.add(ename)
                    exposed_inputs.append(
                        {
                            "name": ename,
                            "innerBlock": b.id,
                            "innerPort": p.name,
                            "type": p.type or "any",
                        }
                    )

        # Read exposed outputs from Outputs block's input ports (wired)
        out_block = next((b for b in canvas.blocks if b.id == "_sb_outputs"), None)
        exposed_outputs = []
        if out_block:
            for port in out_block.input_ports:
                if port.name in ("addInput",):
                    continue
                for wire in port.connections:
                    src = wire.source_port
                    if (
                        src
                        and src.parent_block
                        and src.parent_block.id in inner_block_ids
                    ):
                        exposed_outputs.append(
                            {
                                "name": port.display_name or port.name,
                                "innerBlock": src.parent_block.id,
                                "innerPort": src.name,
                                "type": port.type or "any",
                            }
                        )
                        break

        template["exposedInputs"] = exposed_inputs
        template["exposedOutputs"] = exposed_outputs

        with open(filepath, "w") as f:
            json.dump(template, f, indent=2, default=str)

        tab["template_data"] = template
        self._dirty = False
        self._update_tab_dirty_indicator()

        # Re-scan registry to update the block definition
        self.registry._scan_sub_pipelines()
        self._init_block_menu_config()
        self._apply_menu_config()
        self.log(f"Super Block saved: {filepath}")

    def _begin_pipeline_run(self):
        """Block re-entrant runs: processEvents between blocks lets a second
        Run click arrive mid-run, which would hot-reload the modules currently
        executing and replace the engine under them. Returns False if a run
        is already in progress."""
        if getattr(self, "_pipeline_running", False):
            self.log("Run ignored — a pipeline is already running.")
            return False
        self._pipeline_running = True
        btn = getattr(self, "_run_btn", None)
        if btn is not None:
            btn.setEnabled(False)
        return True

    def _end_pipeline_run(self):
        self._pipeline_running = False
        btn = getattr(self, "_run_btn", None)
        if btn is not None:
            btn.setEnabled(True)

    def run_all(self):
        if not self._begin_pipeline_run():
            return
        try:
            self.reload_modules()
            self.engine._debug_continue = False
            self.log("Starting pipeline execution...")
            self.status_label.setText("Running pipeline...")
            QApplication.processEvents()
            self.engine.run_all(self.canvas.blocks, self.canvas.wires)
            self.status_label.setText("Pipeline complete.")
        except Exception as e:
            self.status_label.setText(f"Pipeline failed: {e}")
            self.log(f"Pipeline error: {e}")
        finally:
            self._end_pipeline_run()
            if self._debug_panel:
                self._debug_panel.hide()

    def _create_engine(self):
        """Create the WorkflowEngine and (re)wire its callbacks.

        Called at startup and from ``reload_modules`` so engine-level code
        changes take effect without an app restart. A fresh import picks up a
        reloaded ``workflow_engine`` module; transient state (debug mode, the
        active canvas's wire-removal callback) is carried over.
        """

        prev = getattr(self, "engine", None)
        engine = WorkflowEngine(self.registry)
        engine.status_callback = self._on_block_status_changed
        engine.log_callback = self.log_debug
        engine.debug_log_callback = self.log_debug
        engine.debug_pause_callback = self._debug_pause
        if prev is not None:
            engine.debug_mode = getattr(prev, "debug_mode", False)
            # Carry over run state established by the last full run so a
            # partial re-run (Re-run from here / single block), which doesn't
            # re-execute upstream blocks, still sees the sample rate, reference
            # variables, and Data Bus pools. run_all resets these itself, so a
            # full run is unaffected.
            engine.global_vars = dict(getattr(prev, "global_vars", {}) or {})
            engine.reference_store = dict(getattr(prev, "reference_store", {}) or {})
            engine.data_bus_store = dict(getattr(prev, "data_bus_store", {}) or {})
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "remove_wire"):
            engine.wire_remove_callback = canvas.remove_wire
        self.engine = engine

    def reload_modules(self):
        """Hot-reload all processing, runners, utils, and blockdefs modules,
        then rescan block definitions and rebuild the palette and menus."""
        import importlib

        count = 0
        failures = []
        for mod_name in list(sys.modules):
            if mod_name.startswith(("processing.", "runners.", "utils.", "blockdefs.")):
                try:
                    importlib.reload(sys.modules[mod_name])
                    count += 1
                except Exception as e:
                    failures.append(f"{mod_name}: {e}")
        if failures:
            # A failed reload keeps the stale module running; without this
            # message an edit with a syntax error silently "does nothing".
            self.log(
                f"[reload] {len(failures)} module(s) failed to reload "
                "(stale code still active):"
            )
            for msg in failures:
                self.log(f"  {msg}")

        # Reload the engine module and recreate the engine so engine-level
        # changes apply without an app restart. Fail-safe: keep the existing
        # engine if anything goes wrong (no worse than before).
        try:
            if "workflow_engine" in sys.modules:
                importlib.reload(sys.modules["workflow_engine"])
            self._create_engine()
        except Exception as e:
            self.log(f"[reload] engine recreate failed, keeping current: {e}")

        # Rescan block definitions
        self.registry.definitions.clear()
        self.registry.scan_block_defs()

        # Rebuild menu config, palette, and menu bar
        self._init_block_menu_config()
        self._apply_menu_config()
        menubar = self.menuBar()
        menubar.clear()
        self._create_block_menu()
        self._create_view_menu()

        # Reloading utils.recent_paths reset its module-level binding to the
        # active workflow's shared recent-files list; re-establish it so the
        # Load Data picker and add_recent keep working after every reload
        # (run_all reloads before each run).
        try:
            from utils import recent_paths as _rp

            canvas = getattr(self, "canvas", None)
            if canvas is not None:
                if not hasattr(canvas, "_recent_load_paths"):
                    canvas._recent_load_paths = []
                _rp.bind(canvas._recent_load_paths)
        except Exception:
            pass

        self.log(
            f"Reloaded {count} modules and {len(self.registry.definitions)} block definitions."
        )
        if failures:
            self.status_label.setText(
                f"Reloaded {count} modules — {len(failures)} failed (see log)"
            )
        else:
            self.status_label.setText(f"Reloaded {count} modules.")

    def _reload_block(self, block):
        """Reload a single block's definition and runner, then update it on canvas."""
        import importlib

        def_name = block.definition_name

        # Reload relevant modules for this block
        for mod_name in list(sys.modules):
            if mod_name.startswith(("blockdefs.", "runners.")):
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass

        # Rescan block definitions
        self.registry.definitions.clear()
        self.registry.scan_block_defs()

        # Re-scanning reset every definition to its category-scan color
        # (assign_dynamic_colors). Re-apply the palette's section colors so the
        # registry matches what the palette and canvas show — otherwise blocks
        # added after a reload pick up the wrong (scan) color.
        try:
            self.block_palette._apply_section_colors()
        except Exception:
            pass

        try:
            defn = self.registry.get(def_name)
        except KeyError:
            self.log(f"Block definition '{def_name}' not found after reload.")
            return

        # Save wire connections by port name
        wire_map_in = {}  # input port name -> wire
        wire_map_out = {}  # output port name -> [wire, ...]
        for wire in self.canvas.wires:
            if wire.dest_port and wire.dest_port.parent_block is block:
                wire_map_in[wire.dest_port.name] = wire
            if wire.source_port and wire.source_port.parent_block is block:
                wire_map_out.setdefault(wire.source_port.name, []).append(wire)

        # Preserve user customizations
        saved_custom_name = block.is_custom_name
        saved_display_name = block.display_name
        saved_params = dict(block.parameters)
        saved_position = block.position
        saved_node_shape = block.node_shape
        saved_font_size = block.font_size
        saved_port_font_size = block.port_font_size
        saved_marker_scale = block.marker_scale
        saved_color = block.color
        saved_gradient_end = block.gradient_end

        # Re-initialize the block from the (possibly updated) definition
        block.init_from_definition(defn)

        # Restore user customizations
        if saved_custom_name:
            block.display_name = saved_display_name
            block.is_custom_name = True
        block.position = saved_position
        block.node_shape = saved_node_shape
        block.font_size = saved_font_size
        block.port_font_size = saved_port_font_size
        block.marker_scale = saved_marker_scale
        # Keep the on-canvas color/gradient — reload updates logic and ports,
        # not appearance (the rescanned defn carries only the scan color).
        block.color = saved_color
        block.gradient_end = saved_gradient_end

        # Merge saved parameters (keep user values, add new defaults).
        # Parameters listed in defn["resetOnReload"] are runtime state and
        # are intentionally NOT restored — they go back to their defaults.
        reset_keys = set(defn.get("resetOnReload") or [])
        for key, val in saved_params.items():
            if key in reset_keys:
                continue
            if key in block.parameters or key in (defn.get("defaultParameters") or {}):
                block.parameters[key] = val

        # Clear cached output data
        block.output_data = {}
        block.status = "pending"
        block.error_message = ""
        for p in block.output_ports:
            p.has_data = False

        # Reconnect wires by matching port names
        new_in_ports = {p.name: p for p in block.input_ports}
        new_out_ports = {p.name: p for p in block.output_ports}

        wires_to_remove = []
        for wire in self.canvas.wires:
            if wire.dest_port and wire.dest_port.parent_block is block:
                if wire.dest_port.name in new_in_ports:
                    new_port = new_in_ports[wire.dest_port.name]
                    wire.dest_port = new_port
                    new_port.connections.append(wire)
                else:
                    wires_to_remove.append(wire)
            if wire.source_port and wire.source_port.parent_block is block:
                if wire.source_port.name in new_out_ports:
                    new_port = new_out_ports[wire.source_port.name]
                    wire.source_port = new_port
                    new_port.connections.append(wire)
                else:
                    wires_to_remove.append(wire)

        for wire in wires_to_remove:
            self.canvas.remove_wire(wire)

        # Update graphics
        self.canvas.update_block_graphics(block)
        self.log(f"Reloaded block: {block.display_name} ({def_name})")

    def _add_start_block(self):
        """Add a Start block at the view center and focus on it."""
        center = self.canvas.get_view_center()
        block = self.canvas.add_block("StartBlock", (center[0] - 60, center[1] + 15))
        block.node_shape = self.app_settings["nodeShape"]
        self.canvas.update_block_graphics(block)
        self.canvas.scroll_to_block(block)

    def _load_most_recent(self):
        """Open the most recently modified workflow on startup, if one exists.

        With the Home tab as the startup view, missing/empty workflows
        folder is a no-op (user stays on Home and uses New/Open).
        """
        workflows_dir = os.path.join(self.root_dir, "workflows")
        if not os.path.isdir(workflows_dir):
            return

        json_files = glob.glob(os.path.join(workflows_dir, "*.json"))
        if not json_files:
            return

        most_recent = max(json_files, key=os.path.getmtime)
        try:
            with open(most_recent, "r") as f:
                data = json.load(f)
            label = self._load_workflow_data(data, most_recent)
            self.log(f"Loaded recent: {label}")
        except Exception as e:
            self.log(f"Failed to load recent workflow: {e}")

    # --- Block interaction ---

    def _on_block_status_changed(self, block):
        # Only update blocks that belong to the current canvas
        if block not in self.canvas.blocks:
            return

        # Refresh the floating Issues panel before sampling scroll position.
        self._refresh_issues_panel()

        # Save scroll position before updating graphics — closing an
        # interactive dialog can cause Qt to fire resize/layout events
        # that shift the viewport.
        h = self.canvas.horizontalScrollBar().value()
        v = self.canvas.verticalScrollBar().value()

        self.canvas.update_block_graphics(block)
        self.status_label.setText(f"{block.display_name}: {block.status}")

        if block.status == "running":
            self.canvas.ensure_block_visible_right(block)
        elif block.status in ("done", "error"):
            # Restore scroll position to prevent camera jump after
            # dialog close — only when the view wasn't intentionally
            # moved by ensure_block_visible_right above.
            self.canvas.horizontalScrollBar().setValue(h)
            self.canvas.verticalScrollBar().setValue(v)
            if block.status == "error":
                self.log(f"  [{block.display_name}] ERROR: {block.error_message}")

        QApplication.processEvents()

    def _on_wire_changed(self, block1, block2):
        """Called after a wire is added or removed."""
        # Skip dynamic port rebuild during wire replacement (old wire removed,
        # new wire about to be added to the same port)
        if getattr(self.canvas, "_replacing_wire", False):
            return

        names = []
        if block1:
            names.append(block1.display_name)
        if block2:
            names.append(block2.display_name)
        self.log_debug(f"Wire changed: {' <-> '.join(names)}")
        blocks = []
        if block1:
            blocks.append(block1)
        if block2:
            blocks.append(block2)

        for blk in blocks:
            # Check block's live ports for addInput pattern (most reliable)
            has_add_input_port = any(p.name == "addInput" for p in blk.input_ports)

            if blk.definition_name and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isPause"):
                    self._update_junction_ports(blk)
                if defn.get("isPriority"):
                    self._update_priority_ports(blk)

            if has_add_input_port:
                self._update_dynamic_ports(blk)

            # Mirror input ports as output ports for FilterRows blocks
            if blk.definition_name and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isMirrorPorts"):
                    self._mirror_dynamic_outputs(blk)

    @staticmethod
    def _has_add_input(defn):
        """Check if a block definition uses the addInput dynamic port pattern."""
        return any(inp.get("name") == "addInput" for inp in defn.get("inputs", []))

    def _refresh_dynamic_ports(self):
        """Re-sort dynamic input ports on all blocks after workflow load."""
        for blk in self.canvas.blocks:
            if any(p.name == "addInput" for p in blk.input_ports):
                self._update_dynamic_ports(blk)
            elif not blk.definition_name or not self.registry.has(blk.definition_name):
                continue
            else:
                defn = self.registry.get(blk.definition_name)
                if self._has_add_input(defn):
                    self._update_dynamic_ports(blk)

            # Mirror outputs for FilterRows blocks
            if blk.definition_name and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isMirrorPorts"):
                    self._mirror_dynamic_outputs(blk)

            # Ensure Data Bus / Reference output port names are sequential
            if blk.definition_name and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isDataBus") or defn.get("isReference"):
                    prefix = "ref_" if defn.get("isReference") else "out"
                    seq = 0
                    for p in blk.output_ports:
                        if p.name == "Run":
                            continue
                        seq += 1
                        p.name = f"{prefix}{seq}"
                    blk.parameters["outputPortDefs"] = blk.port_defs("output")

                # Restore Data Bus display names from pool parameter
                if defn.get("isDataBus") and "pool" in blk.parameters:
                    self._update_data_bus_display_name(blk)
                    self.canvas.update_block_graphics(blk)

    def _update_junction_ports(self, block):
        """Update Junction block ports based on connections."""
        data_ports = []
        pause_port = None

        for p in block.input_ports:
            if p.name == "Pause":
                pause_port = p
            elif p.name == "addInput":
                if p.is_connected:
                    wire = p.connections[0]
                    label = wire.source_port.display_name
                    data_ports.append({"port": p, "wire": wire, "label": label})
            else:
                if p.is_connected:
                    wire = p.connections[0]
                    label = wire.source_port.display_name
                    data_ports.append({"port": p, "wire": wire, "label": label})

        # Rebuild input ports
        block.input_ports = []
        idx = 1

        if pause_port is None:
            pause_port = Port("Pause", "input", "logical", False, "Pause workflow")
        pause_port.parent_block = block
        pause_port.index = idx
        block.input_ports.append(pause_port)
        idx += 1

        for i, dp in enumerate(data_ports):
            p = dp["port"]
            p.name = f"in{i + 1}"
            p.display_name = dp["label"]
            p.preserve_case = True
            p.parent_block = block
            p.index = idx
            block.input_ports.append(p)
            dp["wire"].dest_port = p
            idx += 1

        add_input = Port("addInput", "input", "any", False, "Connect to add")
        add_input.display_name = "Add input"
        add_input.parent_block = block
        add_input.index = idx
        block.input_ports.append(add_input)

        # Rebuild output ports (mirror data inputs + Choice)
        old_wires = {}
        for p in block.output_ports:
            if p.is_connected:
                old_wires[p.name] = list(p.connections)

        block.output_ports = []
        choice_port = Port("Choice", "output", "any", True, "User choice")
        choice_port.parent_block = block
        choice_port.index = 1
        if "Choice" in old_wires:
            for w in old_wires["Choice"]:
                w.source_port = choice_port
                choice_port.add_connection(w)
        block.output_ports.append(choice_port)

        for i, dp in enumerate(data_ports):
            out_name = f"out{i + 1}"
            p = Port(out_name, "output", "any", True, "")
            p.display_name = dp["label"]
            p.preserve_case = True
            p.parent_block = block
            p.index = i + 2
            if out_name in old_wires:
                for w in old_wires[out_name]:
                    w.source_port = p
                    p.add_connection(w)
            block.output_ports.append(p)

        # Save dynamic port definitions for persistence
        block.parameters["inputPortDefs"] = block.port_defs("input")
        block.parameters["outputPortDefs"] = block.port_defs("output")

        # Resize
        block.resize_to_fit_ports()
        self.canvas.update_block_graphics(block)

    def _update_priority_ports(self, block):
        """Update Priority block output ports."""
        data_ports = []
        for p in block.output_ports:
            if p.name == "addOutput":
                if p.is_connected:
                    data_ports.append({"port": p, "wires": list(p.connections)})
            else:
                if p.is_connected:
                    data_ports.append({"port": p, "wires": list(p.connections)})

        is_boundary = block.id == "_sb_inputs"
        block.output_ports = []
        for i, dp in enumerate(data_ports):
            p = dp["port"]
            p.name = f"out{i + 1}"
            if not is_boundary:
                p.display_name = str(i + 1)
            elif p.display_name in ("Add input", "addOutput", str(i + 1)):
                # Auto-label from connected destination port
                for w in dp["wires"]:
                    if w.dest_port and w.dest_port.display_name:
                        p.display_name = w.dest_port.display_name
                        break
            p.parent_block = block
            p.index = i + 1
            for w in dp["wires"]:
                w.source_port = p
            block.output_ports.append(p)

        add_label = "Add input" if is_boundary else "Add output"
        add_output = Port("addOutput", "output", "any", False, "Connect to add")
        add_output.display_name = add_label
        add_output.parent_block = block
        add_output.index = len(block.output_ports) + 1
        block.output_ports.append(add_output)

        # Save dynamic port definitions for persistence
        block.parameters["outputPortDefs"] = block.port_defs("output")

        block.resize_to_fit_ports()
        self.canvas.update_block_graphics(block)

    def _update_dynamic_ports(self, block):
        """Generic dynamic port update for blocks with addInput placeholder."""
        # Identify fixed ports from the block definition — these are never
        # renumbered or removed (e.g. "Run" on Data Bus, "filename" on SaveFile)
        defn = (
            self.registry.get(block.definition_name)
            if self.registry.has(block.definition_name)
            else {}
        )
        fixed_input_names = {
            inp["name"] for inp in defn.get("inputs", []) if inp["name"] != "addInput"
        }

        # --- PlotData: paired x/y ports per series ---
        if defn.get("isPlotData"):
            self._update_plot_data_ports(block)
            return

        # Separate fixed ports from dynamic ones
        fixed_ports = []
        existing_ports = []
        new_ports = []
        for p in block.input_ports:
            if p.name in fixed_input_names:
                fixed_ports.append(p)
            elif p.name == "addInput":
                if p.is_connected:
                    wire = p.connections[0]
                    label = wire.source_port.display_name
                    new_ports.append(
                        {"port": p, "wire": wire, "label": label, "is_new": True}
                    )
            else:
                if p.is_connected:
                    wire = p.connections[0]
                    existing_ports.append(
                        {
                            "port": p,
                            "wire": wire,
                            "label": p.display_name,
                            "is_new": False,
                        }
                    )

        # Keep existing ports in their current order, append new ones at the end
        data_ports = existing_ports + new_ports

        is_filter_config = defn.get("isFilterConfig", False)

        # Rebuild input port list: fixed ports first, then dynamic, then addInput
        block.input_ports = []
        idx = 0
        for p in fixed_ports:
            idx += 1
            p.parent_block = block
            p.index = idx
            block.input_ports.append(p)

        for i, dp in enumerate(data_ports):
            idx += 1
            p = dp["port"]
            p.name = f"in{i + 1}"
            if is_filter_config:
                p.display_name = f"Filter {i + 1}"
            elif dp["is_new"]:
                p.display_name = dp["label"]
            # else: keep existing display_name (may have been user-renamed)
            p.preserve_case = True  # data/variable name — show verbatim
            p.parent_block = block
            p.index = idx
            block.input_ports.append(p)
            dp["wire"].dest_port = p

        add_label = "Add output" if block.id == "_sb_outputs" else "Add input"
        add_input = Port("addInput", "input", "any", False, "Connect to add")
        add_input.display_name = add_label
        add_input.parent_block = block
        add_input.index = len(block.input_ports) + 1
        block.input_ports.append(add_input)

        # Save dynamic port definitions so they persist across save/load
        block.parameters["inputPortDefs"] = block.port_defs("input")

        block.resize_to_fit_ports()
        self.canvas.update_block_graphics(block)

    def _mirror_dynamic_outputs(self, block):
        """Mirror input data ports as output ports for blocks with isMirrorPorts."""
        data_inputs = [
            p
            for p in block.input_ports
            if p.name not in ("addInput",) and p.is_connected
        ]

        # Map existing output ports by name to preserve wires
        old_wires = {}
        for p in block.output_ports:
            if p.is_connected:
                old_wires[p.name] = list(p.connections)

        block.output_ports = []
        for i, inp in enumerate(data_inputs):
            out_name = inp.name.replace("in", "out", 1)
            p = Port(out_name, "output", "any", True, "")
            p.display_name = inp.display_name
            p.preserve_case = inp.preserve_case
            p.parent_block = block
            p.index = i + 1
            if out_name in old_wires:
                for w in old_wires[out_name]:
                    w.source_port = p
                    p.add_connection(w)
            block.output_ports.append(p)

        # Remove wires from output ports that no longer exist
        kept = {p.name for p in block.output_ports}
        for name, wires in old_wires.items():
            if name not in kept:
                for w in wires:
                    self.canvas.remove_wire(w)

        # Save output port definitions for persistence
        block.parameters["outputPortDefs"] = block.port_defs("output")

        block.resize_to_fit_ports()
        self.canvas.update_block_graphics(block)

        # FilterRows: auto-populate column from first input if empty
        if block.parameters.get("column") is not None:
            col = block.parameters.get("column", "").strip()
            if not col and data_inputs:
                block.parameters["column"] = data_inputs[0].display_name

    def _update_plot_data_ports(self, block):
        """Dynamic port update for PlotData blocks with paired x/y series."""
        # Group existing ports by series number
        series_ports = {}  # {series_num: {"x": port, "y": port}}
        new_from_add = []

        for p in block.input_ports:
            if p.name == "addInput":
                if p.is_connected:
                    wire = p.connections[0]
                    new_from_add.append({"port": p, "wire": wire})
            elif len(p.name) > 1 and p.name[0] in ("x", "y") and p.name[1:].isdigit():
                sn = int(p.name[1:])
                series_ports.setdefault(sn, {})[p.name[0]] = p

        # Keep series that have at least one connected port
        active_series_nums = sorted(
            sn
            for sn, sp in series_ports.items()
            if (sp.get("x") and sp["x"].is_connected)
            or (sp.get("y") and sp["y"].is_connected)
        )

        # Rebuild port list
        block.input_ports = []
        idx = 0
        series_num = 0

        def _port_label(prefix, num, port):
            """Build 'x1: varName' when connected, 'x1:' when empty."""
            base = f"{prefix}{num}:"
            if port and port.is_connected:
                wire = port.connections[0]
                if wire.source_port and wire.source_port.display_name:
                    return f"{base} {wire.source_port.display_name}"
            return base

        for sn in active_series_nums:
            series_num += 1
            sp = series_ports[sn]

            # x port
            xp = sp.get("x")
            if xp is None:
                xp = Port(f"x{series_num}", "input", "any", False, "X data")
            xp.name = f"x{series_num}"
            xp.display_name = _port_label("x", series_num, xp)
            idx += 1
            xp.parent_block = block
            xp.index = idx
            block.input_ports.append(xp)

            # y port
            yp = sp.get("y")
            if yp is None:
                yp = Port(f"y{series_num}", "input", "any", False, "Y data")
            yp.name = f"y{series_num}"
            yp.display_name = _port_label("y", series_num, yp)
            idx += 1
            yp.parent_block = block
            yp.index = idx
            block.input_ports.append(yp)

        # New series from addInput connections
        for ns in new_from_add:
            series_num += 1
            p = ns["port"]
            wire = ns["wire"]
            p.name = f"x{series_num}"
            src_label = ""
            if wire.source_port and wire.source_port.display_name:
                src_label = f" {wire.source_port.display_name}"
            p.display_name = f"x{series_num}:{src_label}"
            idx += 1
            p.parent_block = block
            p.index = idx
            block.input_ports.append(p)
            wire.dest_port = p

            # Companion y port
            yp = Port(f"y{series_num}", "input", "any", False, "Y data")
            yp.display_name = f"y{series_num}:"
            idx += 1
            yp.parent_block = block
            yp.index = idx
            block.input_ports.append(yp)

        # If no series exist at all, ensure at least series 1 (x1/y1) is shown
        if series_num == 0:
            series_num = 1
            xp = Port("x1", "input", "any", False, "X data")
            xp.display_name = "x1:"
            xp.parent_block = block
            xp.index = 1
            block.input_ports.append(xp)

            yp = Port("y1", "input", "any", False, "Y data")
            yp.display_name = "y1:"
            yp.parent_block = block
            yp.index = 2
            block.input_ports.append(yp)
            idx = 2

        # Fresh "Add series" placeholder
        add_input = Port("addInput", "input", "any", False, "Connect to add series")
        add_input.display_name = "Add series"
        add_input.parent_block = block
        add_input.index = len(block.input_ports) + 1
        block.input_ports.append(add_input)

        # Persist
        block.parameters["inputPortDefs"] = block.port_defs("input")

        block.resize_to_fit_ports()
        self.canvas.update_block_graphics(block)

    def _run_single_block(self, block):
        """Run a single block, resetting its status first to allow re-run."""
        if not self._begin_pipeline_run():
            return
        try:
            self.reload_modules()
            self.engine._apply_reuse_mode(self.canvas.blocks, self.canvas.wires)
            block.status = "pending"
            block.output_data = {}
            self.engine._running_blocks = set()
            self.engine.run_block(block, self.canvas.wires)
        finally:
            self._end_pipeline_run()

    def _run_from_block(self, block):
        """Reload modules once, then run from block and all downstream."""
        if not self._begin_pipeline_run():
            return
        try:
            self.reload_modules()
            self.engine.run_from_block(block, self.canvas.blocks, self.canvas.wires)
        finally:
            self._end_pipeline_run()

    def _run_to_block(self, block):
        """Reload modules once, then smart-run up to block: reuse done blocks,
        run only the un-run cone from the earliest un-run ancestor to here."""
        if not self._begin_pipeline_run():
            return
        try:
            self.reload_modules()
            self.engine.run_to_block(block, self.canvas.blocks, self.canvas.wires)
        finally:
            self._end_pipeline_run()

    def _on_block_double_click(self, block):
        """Double-click: run the block."""
        try:
            defn = self.registry.get(block.definition_name)
            is_start = defn.get("isStart", False)
            is_toggle = defn.get("isToggle", False)
        except (KeyError, AttributeError):
            is_start = False
            is_toggle = False

        try:
            if is_start:
                self.run_all()
            elif is_toggle:
                block.parameters.setdefault("value", False)
                block.parameters["value"] = not block.parameters["value"]
                block.output_data = {"value": block.parameters["value"]}
                self.canvas.update_block_graphics(block)
                self.log(f'Toggle "{block.display_name}" = {block.parameters["value"]}')
            else:
                if block.status == "done":
                    # Already run: re-run only this block; upstream is reused.
                    self.log(f"Re-running block: {block.display_name}")
                    self._run_single_block(block)
                else:
                    # Un-run or errored: search back and run the un-run chain
                    # (from the earliest un-run ancestor) up to this block.
                    self.log(f"Running to block: {block.display_name}")
                    self._run_to_block(block)
                self.canvas.update_block_graphics(block)
        except Exception as e:
            self.log(f"Error running {block.display_name}: {e}")
            import traceback

            traceback.print_exc()

    def _on_block_cmd_click(self, block):
        """Cmd+click (macOS) / Ctrl+click: edit the block parameters."""
        try:
            defn = self.registry.get(block.definition_name)
        except (KeyError, AttributeError):
            defn = {}

        try:
            if defn.get("noEditor"):
                FloatingMessage.show_for(
                    block.display_name,
                    "Double-click the block to open its UI.",
                    self.canvas,
                    near_block=block,
                )
            elif defn.get("isSubPipeline"):
                self._edit_sub_pipeline(block, defn)
            elif defn.get("isReference"):
                self._edit_reference_outputs(block)
            elif defn.get("isUnpacker"):
                self._edit_unpack_outputs(block)
            elif defn.get("isDataBus"):
                self._edit_data_bus_outputs(block)
            elif defn.get("isMeanBlock"):
                self._edit_mean_block(block)
            elif defn.get("isPacker"):
                self._rename_dynamic_inputs(block)
            elif defn.get("isCompile"):
                self._edit_compile_table(block)
            elif defn.get("isParameterized") or defn.get("defaultParameters"):
                self._edit_block_parameters(block)
            else:
                FloatingMessage.show_for(
                    block.display_name,
                    "No editable content.",
                    self.canvas,
                    near_block=block,
                )
        except Exception as e:
            self.log(f"Error editing {block.display_name}: {e}")
            import traceback

            traceback.print_exc()

    def _safe_action(self, func):
        """Wrap a menu action to prevent crashes from unhandled exceptions."""

        def wrapper():
            try:
                func()
            except Exception as e:
                self.log(f"Error: {e}")
                import traceback

                traceback.print_exc()

        return wrapper

    def _on_block_context_menu(self, block, global_pos):
        """Show right-click context menu for a block."""
        menu = QMenu(self)

        # Boundary blocks (Inputs/Outputs) get a simplified menu
        if self._is_boundary_block(block):
            sa = self._safe_action
            ports = (
                block.output_ports if block.id == "_sb_inputs" else block.input_ports
            )
            editable = [p for p in ports if p.name not in ("addOutput", "addInput")]
            for port in editable:
                menu.addAction(
                    f'Rename "{port.display_name}"',
                    sa(lambda p=port: self._rename_boundary_port(p, block)),
                )
            menu.exec(global_pos)
            return

        try:
            defn = self.registry.get(block.definition_name)
            is_start = defn.get("isStart", False)
        except (KeyError, AttributeError):
            defn = {}
            is_start = False

        sa = self._safe_action
        if is_start:
            menu.addAction(
                "Run Pipeline", sa(lambda: self._on_block_double_click(block))
            )
        else:
            menu.addAction("Run This Block", sa(lambda: self._run_single_block(block)))
            menu.addAction("Run From Here", sa(lambda: self._run_from_block(block)))
            menu.addAction("Run to Here", sa(lambda: self._run_to_block(block)))

        menu.addSeparator()
        menu.addAction("Data Inspector", sa(lambda: self._inspect_block(block)))
        menu.addAction("Description", sa(lambda: self._show_block_description(block)))

        # Edit
        if defn.get("isMirrorPorts"):
            menu.addAction("Rename Ports", sa(lambda: self._rename_mirror_ports(block)))
        elif defn.get("isAddToReference") or defn.get("isPacker"):
            menu.addAction(
                "Rename Inputs", sa(lambda: self._rename_dynamic_inputs(block))
            )
        if defn.get("isDeviceGeometry"):
            menu.addAction(
                "Open Geometry UI…", sa(lambda: self._open_device_geometry(block))
            )
        elif defn.get("noEditor"):
            pass  # block opts out of the generic parameter editor
        elif defn.get("isReference"):
            menu.addAction("Edit Node", sa(lambda: self._edit_reference_outputs(block)))
        elif defn.get("isUnpacker"):
            menu.addAction("Edit Node", sa(lambda: self._edit_unpack_outputs(block)))
        elif defn.get("isDataBus"):
            menu.addAction("Edit Node", sa(lambda: self._edit_data_bus_outputs(block)))
        elif defn.get("isMeanBlock"):
            menu.addAction("Edit Node", sa(lambda: self._edit_mean_block(block)))
        elif defn.get("isSubPipeline"):
            menu.addAction(
                "Edit Template", sa(lambda: self._edit_sub_pipeline(block, defn))
            )
        elif defn.get("isCompile"):
            menu.addAction(
                "Edit Variables…", sa(lambda: self._edit_compile_table(block))
            )
            menu.addAction("View Table", sa(lambda: self._view_compile_table(block)))
        elif defn.get("isParameterized") or defn.get("defaultParameters"):
            menu.addAction("Edit Node", sa(lambda: self._edit_block_parameters(block)))

        in_reorderable = [p for p in block.input_ports if p.name != "addInput"]
        out_reorderable = [p for p in block.output_ports if p.name != "addOutput"]
        if len(in_reorderable) > 1:
            sub = menu.addMenu("Rearrange Inputs")
            sub.addAction(
                "Auto", sa(lambda: self._auto_rearrange_ports(block, "input"))
            )
            sub.addAction("Manual", sa(lambda: self._rearrange_ports(block, "input")))
        if len(out_reorderable) > 1:
            sub = menu.addMenu("Rearrange Outputs")
            sub.addAction(
                "Auto", sa(lambda: self._auto_rearrange_ports(block, "output"))
            )
            sub.addAction("Manual", sa(lambda: self._rearrange_ports(block, "output")))

        menu.addSeparator()
        menu.addAction("Rename", sa(lambda: self._rename_block(block)))
        # Super blocks get a presets-with-custom picker (vibrant gradients);
        # regular blocks keep the plain color dialog.
        if defn.get("isSubPipeline"):
            self._build_sub_pipeline_color_submenu(menu, block)
        else:
            menu.addAction("Change Color", sa(lambda: self._change_block_color(block)))

        # Disable/Enable
        if block.status == "skipped":
            menu.addAction("Enable", sa(lambda: self._toggle_disable(block)))
        else:
            menu.addAction("Disable", sa(lambda: self._toggle_disable(block)))

        # Sub-pipeline creation (when multiple blocks selected)
        if self.canvas._multi_drag_blocks and len(self.canvas._multi_drag_blocks) > 1:
            menu.addSeparator()
            menu.addAction(
                "Create Super Block", sa(lambda: self._create_sub_pipeline())
            )

        menu.addSeparator()
        if self.canvas._multi_drag_blocks and len(self.canvas._multi_drag_blocks) > 1:
            menu.addAction(
                "Duplicate",
                sa(
                    lambda: self._duplicate_blocks(list(self.canvas._multi_drag_blocks))
                ),
            )
        else:
            menu.addAction("Duplicate", sa(lambda: self._duplicate_blocks([block])))
        menu.addAction("Reload Block", sa(lambda: self._reload_block(block)))
        menu.addAction("Delete", sa(lambda: self._delete_block(block)))

        menu.exec(global_pos)

    def _show_block_description(self, block):
        """Open a floating panel describing the block: purpose + inputs/outputs."""
        try:
            defn = self.registry.get(block.definition_name)
        except (KeyError, AttributeError):
            defn = {}
        title = (
            block.display_name
            or defn.get("displayName")
            or block.definition_name
            or "Block"
        )
        body = QLabel()
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        body.setText(self._build_description_html(defn, block))
        body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        body.setContentsMargins(12, 10, 12, 12)
        body.setStyleSheet(
            f"QLabel {{ color: {theme.palette.text}; background: transparent;"
            f" font-size: 12px; }}"
        )
        FloatingShell.open(
            title, self.canvas, body, near_block=block, show_footer=False
        )

    def _build_description_html(self, defn, block):
        """Rich-text body for the Description panel: purpose paragraph followed
        by Inputs / Outputs lists auto-derived from the block definition."""
        import html as _html
        from block_descriptions import get_description

        p = theme.palette
        name = defn.get("name") or block.definition_name or ""
        purpose = get_description(name) or "No description available yet."

        def rows(ports, mark_optional):
            out = []
            for prt in ports or []:
                nm = prt.get("displayName") or prt.get("name") or ""
                if prt.get("name") in (
                    "addInput",
                    "addOutput",
                    "addSeries",
                ) or nm.lower() in ("add input", "add output", "add series"):
                    continue  # skip the dynamic "add port" affordances
                label = _html.escape(nm)
                opt = (
                    ""
                    if (not mark_optional or prt.get("required"))
                    else f" <span style='color:{p.text_muted}'>(optional)</span>"
                )
                desc = prt.get("description")
                if desc:
                    out.append(
                        f"<li><b>{label}</b>{opt} &mdash; {_html.escape(desc)}</li>"
                    )
                else:
                    out.append(f"<li><b>{label}</b>{opt}</li>")
            return out

        def section(header, items):
            head = (
                f"<p style='margin:10px 0 2px 0; color:{p.accent}'><b>{header}</b></p>"
            )
            if items:
                return head + (
                    "<ul style='margin:2px 0 0 0; "
                    "-qt-list-indent:1'>" + "".join(items) + "</ul>"
                )
            return head + f"<div style='color:{p.text_muted}'>none</div>"

        parts = [
            f"<div style='line-height:140%'>{_html.escape(purpose)}</div>",
            section("Inputs", rows(defn.get("inputs"), True)),
            section("Outputs", rows(defn.get("outputs"), False)),
        ]
        return "".join(parts)

    def _on_canvas_context_menu(self, canvas_pt, global_pos):
        """Show right-click menu for empty canvas."""
        menu = QMenu(self)

        # Add Block submenu
        add_menu = menu.addMenu("Add Block")
        self._populate_block_menu(
            add_menu,
            lambda dn: lambda: self._add_block_at_position(dn, canvas_pt),
        )

        menu.addSeparator()
        menu.addAction("Undo", self.canvas.undo)
        menu.addAction("Redo", self.canvas.redo)
        menu.addSeparator()
        menu.addAction("Auto Layout", self.canvas.auto_layout)
        menu.addAction("Fit View", self.canvas.fit_to_content)
        menu.addAction("Tidy all wires", self.canvas.tidy_all_wires)
        menu.addSeparator()
        menu.addAction("Run All", self.run_all)
        menu.addSeparator()
        menu.addAction("Clear Canvas", self.clear_canvas)

        menu.exec(global_pos)

    def _init_block_after_add(self, block):
        """Post-creation initialization for special block types."""
        if block.definition_name and self.registry.has(block.definition_name):
            defn = self.registry.get(block.definition_name)
            if defn.get("isDataBus"):
                self._update_data_bus_display_name(block)

    def _add_block_at_position(self, def_name, pos):
        block = self.canvas.add_block(def_name, pos)
        block.node_shape = self.app_settings["nodeShape"]
        self._init_block_after_add(block)
        self.canvas.update_block_graphics(block)
        self.block_palette.record_use(def_name)
        self.block_palette.refresh_current_view()
        self.log(f"Added block: {def_name}")

    def _inspect_block(self, block):
        if block.status == "error":
            QMessageBox.warning(
                self,
                "Block Error",
                f'Block "{block.display_name}" error:\n{block.error_message}',
            )
            return
        # Gather input data from upstream wires
        input_data = {}
        for wire in self.canvas.wires:
            if wire.dest_port and wire.dest_port.parent_block is block:
                port_name = wire.dest_port.name
                if wire.source_port and wire.source_port.parent_block:
                    src = wire.source_port.parent_block
                    src_port = wire.source_port.name
                    if src.output_data and src_port in src.output_data:
                        input_data[port_name] = src.output_data[src_port]
        output_data = block.output_data or {}
        if output_data or input_data:
            self.inspector.inspect(block.display_name, output_data, input_data)
        else:
            self.log(f'Block "{block.display_name}" has no data yet.')

    def _rearrange_ports(self, block, direction):
        """Dialog to reorder input or output ports via drag-and-drop list."""
        from PySide6.QtWidgets import QListWidget, QAbstractItemView

        label = "Inputs" if direction == "input" else "Outputs"
        all_ports = block.input_ports if direction == "input" else block.output_ports

        # Reorderable ports exclude the fixed addInput/addOutput tail.
        fixed_suffix = "addInput" if direction == "input" else "addOutput"
        reorderable = [p for p in all_ports if p.name != fixed_suffix]

        if len(reorderable) < 2:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Rearrange {label} — {block.display_name}")
        dlg.setMinimumWidth(300)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Drag to reorder:"))

        list_widget = QListWidget(dlg)
        list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        for port in reorderable:
            list_widget.addItem(port.display_name or port.name)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_names = [list_widget.item(i).text() for i in range(list_widget.count())]
            port_map = {}
            for port in reorderable:
                key = port.display_name or port.name
                port_map[key] = port
            ordered = [port_map[name] for name in new_names]
            self.canvas.push_undo()
            self._apply_port_order(block, direction, ordered)

    def _apply_port_order(self, block, direction, ordered):
        """Commit a new order of reorderable ports: reindex, re-append the
        fixed addInput/addOutput tail, rewrite port defs, refresh graphics
        and wires. Shared by Manual (drag) and Auto rearrange."""
        fixed_suffix = "addInput" if direction == "input" else "addOutput"
        all_ports = block.input_ports if direction == "input" else block.output_ports
        tail_ports = [p for p in all_ports if p.name == fixed_suffix]

        new_ports = []
        for i, port in enumerate(ordered):
            port.index = i + 1
            new_ports.append(port)
        for tp in tail_ports:
            tp.index = len(new_ports) + 1
            new_ports.append(tp)

        if direction == "input":
            block.input_ports = new_ports
            param_key = "inputPortDefs"
        else:
            block.output_ports = new_ports
            param_key = "outputPortDefs"

        block.parameters[param_key] = block.port_defs(direction)

        self.canvas.update_block_graphics(block)
        # Update connected wires
        for wire in self.canvas.wires:
            if (wire.source_port and wire.source_port.parent_block is block) or (
                wire.dest_port and wire.dest_port.parent_block is block
            ):
                if wire.id in self.canvas._wire_items:
                    self.canvas._wire_items[wire.id].update_path()

    def _auto_rearrange_ports(self, block, direction):
        """Reorder a block's ports to reduce wire crossings by matching the
        vertical order of the ports they connect to on the neighbor block(s)
        (barycenter heuristic). The `run` port is pinned to the top and
        unconnected ports drop to the bottom; the fixed addInput/addOutput
        tail stays last (handled by _apply_port_order)."""
        fixed_suffix = "addInput" if direction == "input" else "addOutput"
        all_ports = block.input_ports if direction == "input" else block.output_ports
        reorderable = [p for p in all_ports if p.name != fixed_suffix]
        if len(reorderable) < 2:
            return

        def other_ys(port):
            ys = []
            for wire in port.connections:
                other = wire.dest_port if wire.source_port is port else wire.source_port
                if other is not None and other.parent_block is not None:
                    ys.append(other.position[1])
            return ys

        run_ports = [p for p in reorderable if (p.name or "").lower() == "run"]
        rest = [p for p in reorderable if (p.name or "").lower() != "run"]

        # Split the rest into connected vs unconnected, preserving original order.
        connected, unconnected = [], []
        for i, p in enumerate(rest):
            ys = other_ys(p)
            if ys:
                connected.append((sum(ys) / len(ys), i, p))
            else:
                unconnected.append(p)

        # Connected ports follow the neighbours' vertical order (top of a block
        # = largest scene-y, so sort by barycenter descending; original index
        # breaks ties). Unconnected ports keep their order, at the bottom.
        connected.sort(key=lambda t: (-t[0], t[1]))

        # run pinned to top; then connected (matched); then unconnected.
        ordered = run_ports + [p for _, _, p in connected] + unconnected

        if ordered == reorderable:
            self.log(
                f'"{block.display_name}" '
                f"{direction}s already arranged to reduce crossings."
            )
            return

        self.canvas.push_undo()
        self._apply_port_order(block, direction, ordered)
        self.log(f'Auto-arranged {direction}s for "{block.display_name}".')

    def _rename_block(self, block):
        name, ok = QInputDialog.getText(
            self, "Rename Block", "New name:", text=block.display_name
        )
        if ok and name:
            block.display_name = name
            block.is_custom_name = True
            self.canvas.update_block_graphics(block)

    def _rename_boundary_port(self, port, block):
        """Rename a super block boundary port."""
        new_name, ok = QInputDialog.getText(
            self, "Rename Port", "Port name:", text=port.display_name
        )
        if ok and new_name.strip():
            port.display_name = new_name.strip()
            self.canvas.update_block_graphics(block)

    def _view_compile_table(self, block):
        """Open the CompileTable viewer against the block's last output."""
        table = (block.output_data or {}).get("table")
        if not table:
            QMessageBox.information(
                self, "Compile Table", "No table to view yet — run the pipeline first."
            )
            return
        from processing.compile_table_view import show_compile_table

        show_compile_table(table, parent=self)

    def _edit_compile_table(self, block):
        """Floating editor: rename Compile Table input variables (column names)
        and attach a free-text note (shown as the port's hover tooltip)."""
        from PySide6.QtWidgets import QGridLayout

        data_ports = [p for p in block.input_ports if p.name != "addInput"]
        if not data_ports:
            FloatingMessage.show_for(
                block.display_name,
                "Connect inputs first, then edit their labels and notes.",
                self.canvas,
                near_block=block,
            )
            return

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 10)
        body_layout.addWidget(
            QLabel(
                "Rename each input variable (used as the table column name) "
                "and add an optional note."
            )
        )

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        grid.addWidget(QLabel("Port"), 0, 0)
        grid.addWidget(QLabel("Label"), 0, 1)
        grid.addWidget(QLabel("Note"), 0, 2)

        fields = []  # (port, label_edit, note_edit)
        for i, p in enumerate(data_ports):
            grid.addWidget(QLabel(p.name), i + 1, 0)
            label_edit = QLineEdit(p.display_name)
            note_edit = QLineEdit(p.description or "")
            note_edit.setPlaceholderText("optional note")
            grid.addWidget(label_edit, i + 1, 1)
            grid.addWidget(note_edit, i + 1, 2)
            fields.append((p, label_edit, note_edit))

        body_layout.addWidget(grid_host)
        body_layout.addStretch(1)

        def _commit():
            for p, label_edit, note_edit in fields:
                new_label = label_edit.text().strip()
                if new_label:
                    p.display_name = new_label
                p.preserve_case = True  # variable name — show verbatim
                p.description = note_edit.text().strip()
            block.parameters["inputPortDefs"] = block.port_defs("input")
            self.canvas.update_block_graphics(block)
            if getattr(self.canvas, "state_changed_callback", None):
                self.canvas.state_changed_callback()

        FloatingShell.open(
            "Edit Compile Table", self.canvas, body, on_ok=_commit, near_block=block
        )

    def _change_block_color(self, block):
        color = QColorDialog.getColor(
            QColor(
                int(block.color[0] * 255),
                int(block.color[1] * 255),
                int(block.color[2] * 255),
            ),
            self,
            "Block Color",
        )
        if color.isValid():
            block.color = (color.redF(), color.greenF(), color.blueF())
            self.canvas.update_block_graphics(block)

    # Vibrant preset bases for super blocks. _make_block_brush rotates each
    # into a complementary endpoint to produce the gradient.
    _SUB_PIPELINE_COLOR_PRESETS = [
        ("Purple", (0.55, 0.40, 0.85)),
        ("Pink", (0.85, 0.30, 0.65)),
        ("Blue", (0.30, 0.55, 0.85)),
        ("Teal", (0.20, 0.75, 0.65)),
        ("Orange", (0.95, 0.55, 0.25)),
        ("Lime", (0.55, 0.80, 0.30)),
        ("Crimson", (0.85, 0.20, 0.35)),
    ]

    def _build_sub_pipeline_color_submenu(self, parent_menu, block):
        """Attach a "Change Color" sub-menu of vibrant gradient presets
        (plus a Custom… option) for a super block."""
        from PySide6.QtCore import QRect, QSize
        from PySide6.QtGui import QPainter, QPixmap, QIcon
        from block_palette import _make_block_brush

        sub = parent_menu.addMenu("Change Color")
        ico_sz = QSize(48, 16)
        for name, rgb in self._SUB_PIPELINE_COLOR_PRESETS:
            pix = QPixmap(ico_sz)
            pix.fill(Qt.GlobalColor.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            brush, qc = _make_block_brush(
                rgb, QRect(0, 0, ico_sz.width(), ico_sz.height()), "SubPipeline_preview"
            )
            p.setBrush(brush)
            p.setPen(qc.darker(140))
            p.drawRoundedRect(0, 0, ico_sz.width() - 1, ico_sz.height() - 1, 3, 3)
            p.end()
            act = sub.addAction(QIcon(pix), name)
            act.triggered.connect(
                lambda _checked=False, b=block, c=rgb: self._set_sub_pipeline_color(
                    b, c
                )
            )
        sub.addSeparator()
        sub.addAction("Custom…", lambda: self._open_sub_pipeline_custom_picker(block))

    def _set_sub_pipeline_color(self, block, rgb):
        block.color = tuple(rgb)
        # Reset explicit end-color so the auto-shift formula governs again.
        block.gradient_end = None
        self.canvas.update_block_graphics(block)

    def _open_sub_pipeline_custom_picker(self, block):
        """Two-color picker for super-block gradients: user chooses the
        start and end colors of the gradient explicitly."""
        from PySide6.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QGridLayout,
            QPushButton,
            QLabel,
            QDialogButtonBox,
        )
        from PySide6.QtGui import QPainter, QPixmap, QLinearGradient, QBrush

        # Start with the current colors; if no explicit end, seed the picker
        # with the auto-shifted endpoint so users see what the renderer was
        # producing and can tweak from there.
        r, g, b = block.color
        if block.gradient_end is not None:
            r2, g2, b2 = block.gradient_end
        else:
            r2 = min(0.95, max(0.1, 1.0 - r * 0.5))
            g2 = min(0.95, max(0.1, g * 0.4 + 0.3))
            b2 = min(0.95, max(0.1, b * 0.5 + 0.4))
        state = {"start": (r, g, b), "end": (r2, g2, b2)}

        dlg = QDialog(self)
        dlg.setWindowTitle("Custom Gradient")
        # Compact two-row layout. Fixed column widths so the preview's span
        # (cols 1..3) lines up exactly with the start swatch's left edge and
        # the end swatch's right edge.
        SWATCH_W = 80
        SWATCH_H = 22
        END_LBL_W = 30
        H_SPACING = 8
        PREVIEW_W = SWATCH_W + H_SPACING + END_LBL_W + H_SPACING + SWATCH_W
        v = QVBoxLayout(dlg)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(H_SPACING)
        grid.setVerticalSpacing(8)
        # Columns: [Left label][Start swatch][End label][End swatch]
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 0)
        grid.setColumnStretch(4, 1)  # trailing stretch so the grid stays left-packed

        preview = QLabel()
        preview.setFixedSize(PREVIEW_W, 22)

        def render_preview():
            pix = QPixmap(PREVIEW_W, 22)
            pix.fill(Qt.GlobalColor.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            grad = QLinearGradient(0, 0, PREVIEW_W, 0)
            sr, sg, sb = state["start"]
            er, eg, eb = state["end"]
            grad.setColorAt(0.0, QColor(int(sr * 255), int(sg * 255), int(sb * 255)))
            grad.setColorAt(1.0, QColor(int(er * 255), int(eg * 255), int(eb * 255)))
            p.setBrush(QBrush(grad))
            p.setPen(QPen(QColor(0, 0, 0, 90), 1))
            p.drawRoundedRect(0, 0, PREVIEW_W - 1, 21, 4, 4)
            p.end()
            preview.setPixmap(pix)

        def make_swatch(key):
            btn = QPushButton()
            btn.setFixedSize(SWATCH_W, SWATCH_H)

            def refresh_btn():
                rr, gg, bb = state[key]
                btn.setStyleSheet(
                    f"QPushButton {{ background: rgb({int(rr * 255)},{int(gg * 255)},"
                    f"{int(bb * 255)}); border: 1px solid rgba(0,0,0,90);"
                    f" border-radius: 4px; }}"
                )

            def pick():
                rr, gg, bb = state[key]
                col = QColorDialog.getColor(
                    QColor(int(rr * 255), int(gg * 255), int(bb * 255)),
                    dlg,
                    f"Gradient {key.capitalize()}",
                )
                if col.isValid():
                    state[key] = (col.redF(), col.greenF(), col.blueF())
                    refresh_btn()
                    render_preview()

            btn.clicked.connect(pick)
            refresh_btn()
            return btn

        start_lbl = QLabel("Start")
        start_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        end_lbl = QLabel("End")
        end_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prev_lbl = QLabel("Preview")
        prev_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        end_lbl.setFixedWidth(END_LBL_W)

        grid.addWidget(start_lbl, 0, 0)
        grid.addWidget(make_swatch("start"), 0, 1)
        grid.addWidget(end_lbl, 0, 2)
        grid.addWidget(make_swatch("end"), 0, 3)
        grid.addWidget(prev_lbl, 1, 0)
        grid.addWidget(
            preview, 1, 1, 1, 3
        )  # span cols 1..3 (matches the row-1 swatches)
        v.addLayout(grid)
        render_preview()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            block.color = tuple(state["start"])
            block.gradient_end = tuple(state["end"])
            self.canvas.update_block_graphics(block)

    def _collect_reference_var_labels(self):
        """Collect all variable labels from AddToReference blocks on the canvas."""
        labels = []
        for blk in self.canvas.blocks:
            if not blk.definition_name or not self.registry.has(blk.definition_name):
                continue
            defn = self.registry.get(blk.definition_name)
            if not defn.get("isAddToReference"):
                continue
            for port in blk.input_ports:
                if port.name == "addInput":
                    continue
                label = port.display_name
                if label and label not in labels:
                    labels.append(label)
        return labels

    def _rename_dynamic_inputs(self, block):
        """Dialog to rename input variable names on AddToReference (or similar) blocks."""
        data_ports = [p for p in block.input_ports if p.name != "addInput"]
        if not data_ports:
            QMessageBox.information(
                self,
                "No Inputs",
                "Connect wires to add inputs first, then rename them.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Rename Inputs")
        dialog.setMinimumWidth(300)
        dlg_layout = QVBoxLayout(dialog)
        dlg_layout.addWidget(QLabel("Edit variable names for each input:"))

        form = QFormLayout()
        fields = []
        for port in data_ports:
            field = QLineEdit(port.display_name)
            form.addRow(f"Input {port.name}:", field)
            fields.append((port, field))
        dlg_layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlg_layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        for port, field in fields:
            new_name = field.text().strip()
            if new_name:
                port.display_name = new_name

        # Persist renamed labels
        block.parameters["inputPortDefs"] = block.port_defs("input")
        self.canvas.update_block_graphics(block)

    def _rename_mirror_ports(self, block):
        """Rename input/output ports on blocks with mirrored dynamic ports."""
        data_ports = [p for p in block.input_ports if p.name != "addInput"]
        if not data_ports:
            QMessageBox.information(
                self, "No Ports", "Connect wires to add ports first."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Rename Ports")
        dialog.setMinimumWidth(300)
        dlg_layout = QVBoxLayout(dialog)

        form = QFormLayout()
        fields = []
        for port in data_ports:
            field = QLineEdit(port.display_name)
            form.addRow(f"{port.name}:", field)
            fields.append((port, field))
        dlg_layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlg_layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        for port, field in fields:
            new_name = field.text().strip()
            if new_name:
                port.display_name = new_name
                # Update the matching output port
                out_name = port.name.replace("in", "out", 1)
                for op in block.output_ports:
                    if op.name == out_name:
                        op.display_name = new_name
                        break

        block.parameters["inputPortDefs"] = block.port_defs("input")
        block.parameters["outputPortDefs"] = block.port_defs("output")
        self.canvas.update_block_graphics(block)

    def _edit_reference_outputs(self, block):
        """Dialog to select which reference variables to expose as output ports."""
        available = self._collect_reference_var_labels()
        if not available:
            QMessageBox.information(
                self,
                "No Variables",
                "No variables found. Add variables using 'Add to Ref' blocks first,\n"
                "then run the pipeline before editing Reference outputs.",
            )
            return

        current_visible = block.parameters.get("visibleOutputs", [])
        aliases = block.parameters.get("outputAliases", {})

        # Build dialog with checkboxes and rename fields
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Reference Outputs")
        dialog.setMinimumWidth(350)
        dlg_layout = QVBoxLayout(dialog)

        dlg_layout.addWidget(
            QLabel("Select variables to output and optionally rename:")
        )

        from PySide6.QtWidgets import QCheckBox, QGridLayout

        grid = QGridLayout()
        grid.addWidget(QLabel("Variable"), 0, 0)
        grid.addWidget(QLabel("Display Name"), 0, 1)

        checkboxes = []
        rename_fields = []
        for i, label in enumerate(available):
            cb = QCheckBox(label)
            cb.setChecked(label in current_visible)
            grid.addWidget(cb, i + 1, 0)
            checkboxes.append((label, cb))

            rename = QLineEdit(aliases.get(label, label))
            grid.addWidget(rename, i + 1, 1)
            rename_fields.append((label, rename))

        dlg_layout.addLayout(grid)

        buttons = QDialogButtonBox()
        cancel_btn = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setDefault(True)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlg_layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Gather selections
        visible = [label for label, cb in checkboxes if cb.isChecked()]
        new_aliases = {}
        for label, field in rename_fields:
            alias = field.text().strip()
            if alias and alias != label:
                new_aliases[label] = alias

        block.parameters["visibleOutputs"] = visible
        block.parameters["outputAliases"] = new_aliases

        self._sync_reference_outputs(block, visible, new_aliases)
        self.canvas.update_block_graphics(block)

    def _sync_reference_outputs(self, block, visible, aliases):
        """Update Reference block output ports based on selected variables."""
        old_port_map = {p.description: p for p in block.output_ports}

        new_ports = []
        for i, var_label in enumerate(visible):
            display = aliases.get(var_label, var_label)
            # Reuse existing port if it matches the same variable label
            if var_label in old_port_map:
                p = old_port_map[var_label]
                p.name = f"ref_{i + 1}"
                p.display_name = display
                p.index = i + 1
            else:
                p = Port(
                    name=f"ref_{i + 1}",
                    direction="output",
                    type_="any",
                    required=True,
                    description=var_label,
                )
                p.display_name = display
                p.parent_block = block
                p.index = i + 1
            new_ports.append(p)

        # Remove wires for ports no longer present
        kept_descriptions = {var for var in visible}
        for desc, port in old_port_map.items():
            if desc not in kept_descriptions:
                for wire in list(port.connections):
                    self.canvas.remove_wire(wire)

        block.output_ports = new_ports

        # Clear placeholder text once outputs are configured
        if new_ports:
            block.parameters.pop("displayText", None)
        else:
            block.parameters["displayText"] = "Edit to add output"

        # Save port definitions for serialization
        block.parameters["outputPortDefs"] = block.port_defs("output")

        # Resize
        block.resize_to_fit_ports()

    def _collect_data_bus_pools(self):
        """Collect all pool names used by Data Bus blocks on the canvas."""
        pools = set()
        for blk in self.canvas.blocks:
            if not blk.definition_name or not self.registry.has(blk.definition_name):
                continue
            defn = self.registry.get(blk.definition_name)
            if defn.get("isDataBus"):
                pools.add(blk.parameters.get("pool", "Default"))
        return sorted(pools)

    def _collect_data_bus_var_labels(self, pool=None):
        """Collect all variable labels stored in Data Bus input ports across the canvas.
        If pool is given, only collect from blocks in that pool.
        Also includes variables from Define Parameters and Add to Ref blocks."""
        labels = []
        for blk in self.canvas.blocks:
            if not blk.definition_name or not self.registry.has(blk.definition_name):
                continue
            defn = self.registry.get(blk.definition_name)
            if defn.get("isDataBus"):
                if pool is not None and blk.parameters.get("pool", "Default") != pool:
                    continue
                for port in blk.input_ports:
                    if port.name in ("Run", "addInput"):
                        continue
                    if port.is_connected and port.display_name not in labels:
                        labels.append(port.display_name)
            elif defn.get("isDefineParameters"):
                for var in blk.parameters.get("variables", []):
                    name = var.get("name", "")
                    if name and name not in labels:
                        labels.append(name)
            elif defn.get("isAddToReference"):
                for port in blk.input_ports:
                    if port.name in ("addInput",):
                        continue
                    if port.is_connected and port.display_name not in labels:
                        labels.append(port.display_name)
        return labels

    def _update_data_bus_display_name(self, block):
        """Set the Data Bus block's display name to 'Data Bus - PoolName'."""
        pool = block.parameters.get("pool", "Default")
        block.display_name = f"Data Bus - {pool}"
        block.is_custom_name = True

    def _sync_unpack_output_ports(self, block, selected_paths):
        """Rebuild an Unpack block's output ports from the user-selected
        paths so the editor reflects the new layout without running the
        pipeline. Preserves existing ports (and their wires) whose names
        still appear in the new selection; removes wires for ports that
        no longer exist. Mirrors the port-management logic in
        WorkflowEngine._execute_unpack.
        """
        if not selected_paths:
            return

        old_port_map = {p.name: p for p in block.output_ports}
        new_fields = list(selected_paths)

        removed_names = set(old_port_map.keys()) - set(new_fields)
        for name in removed_names:
            port = old_port_map[name]
            for wire in list(port.connections):
                self.canvas.remove_wire(wire)

        new_ports = []
        for i, field_name in enumerate(new_fields):
            if field_name in old_port_map and field_name not in removed_names:
                p = old_port_map[field_name]
            else:
                p = Port(
                    name=field_name,
                    direction="output",
                    type_="any",
                    required=True,
                    description=f"Field: {field_name}",
                )
                p.parent_block = block
            p.index = i + 1
            new_ports.append(p)

        block.output_ports = new_ports

        block.resize_to_fit_ports()

        block.parameters["outputPortDefs"] = [
            {"name": f, "type": "any", "description": f"Field: {f}"} for f in new_fields
        ]

    def _edit_unpack_outputs(self, block):
        """Floating editor for the Unpack Variable block: a Data Inspector-
        style structure tree of the input variable. Each node has a checkbox;
        checked nodes become output ports.

        The structure is taken from the last successful run's
        `structShape` cache (no values stored — just type/size/children).
        """
        from PySide6.QtWidgets import (
            QTreeWidget,
            QTreeWidgetItem,
            QHeaderView,
        )

        shape = block.parameters.get("structShape")
        current_paths = list(block.parameters.get("selectedPaths") or [])

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 10)
        body_layout.setSpacing(8)

        intro = QLabel(
            "Check the fields you want as output ports. Checking a parent "
            "outputs the whole sub-struct as one port; checking children "
            "drills in. For a 2-D array, its columns appear as col0, col1, … "
            "so you can output individual columns. Mix freely."
        )
        intro.setWordWrap(True)
        body_layout.addWidget(intro)

        if not shape:
            hint = QLabel(
                "Run the pipeline once to populate the variable structure, "
                "then reopen this editor.\n"
                "(Connect the input wire and run a single block via "
                "right-click → Run This Block if you don't want to run the "
                "whole pipeline.)"
            )
            hint.setWordWrap(True)
            body_layout.addWidget(hint)

        tree = QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels(["Name", "Size", "Type"])
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setStretchLastSection(True)
        tree.setColumnWidth(0, 260)
        tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        body_layout.addWidget(tree, 1)

        def add_nodes(parent_item, children_dict, prefix):
            for key, node in children_dict.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                item = QTreeWidgetItem(
                    [str(key), str(node.get("size", "")), str(node.get("type", ""))]
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if path in current_paths
                    else Qt.CheckState.Unchecked,
                )
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                if parent_item is None:
                    tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                grandchildren = node.get("children")
                if grandchildren:
                    add_nodes(item, grandchildren, path)

        if shape and isinstance(shape, dict):
            children = shape.get("children")
            if children:
                add_nodes(None, children, "")
            else:
                item = QTreeWidgetItem(
                    ["value", str(shape.get("size", "")), str(shape.get("type", ""))]
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if "value" in current_paths
                    else Qt.CheckState.Unchecked,
                )
                item.setData(0, Qt.ItemDataRole.UserRole, "value")
                tree.addTopLevelItem(item)

        tree.expandAll()

        btn_row = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_none = QPushButton("Select none")

        def _walk_set(state):
            stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
            while stack:
                it = stack.pop()
                if it is None:
                    continue
                it.setCheckState(0, state)
                for j in range(it.childCount()):
                    stack.append(it.child(j))

        select_all.clicked.connect(lambda: _walk_set(Qt.CheckState.Checked))
        select_none.clicked.connect(lambda: _walk_set(Qt.CheckState.Unchecked))
        btn_row.addWidget(select_all)
        btn_row.addWidget(select_none)
        btn_row.addStretch(1)
        body_layout.addLayout(btn_row)

        def _commit():
            selected = []

            def collect(item):
                if item is None:
                    return
                if item.checkState(0) == Qt.CheckState.Checked:
                    path = item.data(0, Qt.ItemDataRole.UserRole)
                    if path:
                        selected.append(path)
                for j in range(item.childCount()):
                    collect(item.child(j))

            for i in range(tree.topLevelItemCount()):
                collect(tree.topLevelItem(i))

            if selected:
                block.parameters["selectedPaths"] = selected
                # selectedPaths takes priority; legacy keys cleared so the
                # next run uses the new list rather than depth flattening.
                block.parameters.pop("visibleOutputs", None)
            else:
                block.parameters.pop("selectedPaths", None)

            self._sync_unpack_output_ports(block, selected)
            self.canvas.update_block_graphics(block)
            self.log(
                f"Unpack '{block.display_name}': {len(selected)} field(s) selected."
            )
            if getattr(self.canvas, "state_changed_callback", None):
                self.canvas.state_changed_callback()

        shell = FloatingShell.open(
            f"Edit Unpack Variable — {block.display_name}",
            self.canvas,
            body,
            on_ok=_commit,
            near_block=block,
        )
        # FloatingShell wraps the body in a QScrollArea, whose sizeHint
        # doesn't reflect the content — so adjustSize stays narrow. Force
        # a reasonable default size, clamped to the canvas.
        target_w = min(720, max(420, self.canvas.width() - 40))
        target_h = min(640, max(320, self.canvas.height() - 60))
        shell.resize(target_w, target_h)
        shell._position(block)

    def _edit_data_bus_outputs(self, block):
        """Floating editor: pool select, input renames, output variable picker."""
        from PySide6.QtWidgets import QCheckBox, QGridLayout, QGroupBox

        current_pool = block.parameters.get("pool", "Default")

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 10)

        # --- Pool selection ---
        pool_layout = QHBoxLayout()
        pool_layout.addWidget(QLabel("Variable Pool:"))
        pool_combo = ThemedComboBox()
        existing_pools = self._collect_data_bus_pools()
        if current_pool not in existing_pools:
            existing_pools.append(current_pool)
            existing_pools.sort()
        pool_combo.addItems(existing_pools)
        pool_combo.setCurrentText(current_pool)
        pool_combo.setEditable(True)
        pool_combo.setStyleSheet(_themed_combo_qss())
        pool_layout.addWidget(pool_combo)

        del_pool_btn = QPushButton("Delete Pool")
        del_pool_btn.setToolTip(
            "Delete the selected pool and reset all its blocks to 'Default'"
        )

        def _delete_pool():
            pool_name = pool_combo.currentText().strip()
            if not pool_name or pool_name == "Default":
                FloatingMessage.show_for(
                    "Delete Pool",
                    "Cannot delete the 'Default' pool.",
                    self.canvas,
                    near_block=block,
                )
                return
            count = sum(
                1
                for b in self.canvas.blocks
                if b.definition_name
                and self.registry.has(b.definition_name)
                and self.registry.get(b.definition_name).get("isDataBus")
                and b.parameters.get("pool", "Default") == pool_name
            )
            reply = QMessageBox.question(
                self,
                "Delete Pool",
                f"Delete pool '{pool_name}'?\n"
                f"{count} Data Bus block(s) will be reset to 'Default'.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for b in self.canvas.blocks:
                if not b.definition_name or not self.registry.has(b.definition_name):
                    continue
                defn = self.registry.get(b.definition_name)
                if (
                    defn.get("isDataBus")
                    and b.parameters.get("pool", "Default") == pool_name
                ):
                    b.parameters["pool"] = "Default"
                    self._update_data_bus_display_name(b)
                    self.canvas.update_block_graphics(b)
            idx = pool_combo.findText(pool_name)
            if idx >= 0:
                pool_combo.removeItem(idx)
            pool_combo.setCurrentText("Default")

        del_pool_btn.clicked.connect(_delete_pool)
        pool_layout.addWidget(del_pool_btn)
        body_layout.addLayout(pool_layout)

        # --- Rename Inputs ---
        data_ports = [p for p in block.input_ports if p.name not in ("Run", "addInput")]
        input_rename_fields = []
        if data_ports:
            in_group = QGroupBox("Inputs")
            # Fixed vertical sizing — the group reports its content height as
            # the maximum, so extra body height goes to the trailing stretch
            # instead of stretching rows inside the group.
            in_group.setSizePolicy(
                in_group.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed
            )
            in_grid = QGridLayout(in_group)
            in_grid.addWidget(QLabel("Port"), 0, 0)
            in_grid.addWidget(QLabel("Name"), 0, 1)
            for i, p in enumerate(data_ports):
                in_grid.addWidget(QLabel(p.name), i + 1, 0)
                field = QLineEdit(p.display_name)
                in_grid.addWidget(field, i + 1, 1)
                input_rename_fields.append((p, field))
            body_layout.addWidget(in_group)

        # --- Output variable selection (rebuilt when pool changes) ---
        out_group = QGroupBox("Outputs")
        out_group.setSizePolicy(
            out_group.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed
        )
        out_layout = QVBoxLayout(out_group)
        var_host = QWidget()
        grid = QGridLayout(var_host)
        grid.addWidget(QLabel("Variable"), 0, 0)
        grid.addWidget(QLabel("Display Name"), 0, 1)
        out_layout.addWidget(var_host)
        body_layout.addWidget(out_group)

        # Trailing stretch absorbs extra vertical space when the user
        # resizes the panel, so groups stay compact at their content height.
        body_layout.addStretch(1)

        current_visible = []
        aliases = {}
        for p in block.output_ports:
            if p.name == "Run":
                continue
            var_label = p.description if p.description else p.display_name
            current_visible.append(var_label)
            if p.display_name != var_label:
                aliases[var_label] = p.display_name

        checkboxes = []
        rename_fields = []

        def _rebuild_var_grid():
            for _, cb in checkboxes:
                cb.setParent(None)
            for _, rf in rename_fields:
                rf.setParent(None)
            checkboxes.clear()
            rename_fields.clear()

            pool_name = pool_combo.currentText().strip() or "Default"
            available = self._collect_data_bus_var_labels(pool_name)
            for i, label in enumerate(available):
                cb = QCheckBox(label)
                cb.setChecked(label in current_visible)
                grid.addWidget(cb, i + 1, 0)
                checkboxes.append((label, cb))

                rename = QLineEdit(aliases.get(label, label))
                grid.addWidget(rename, i + 1, 1)
                rename_fields.append((label, rename))

        _rebuild_var_grid()
        pool_combo.currentTextChanged.connect(lambda _: _rebuild_var_grid())

        def _commit():
            new_pool = pool_combo.currentText().strip() or "Default"
            block.parameters["pool"] = new_pool
            self._update_data_bus_display_name(block)

            # Capture variable renames (old label -> new label) before applying,
            # so the rename can propagate to every Data Bus block in the pool.
            renames = {}
            for port, field in input_rename_fields:
                new_name = field.text().strip()
                if new_name and new_name != port.display_name:
                    renames[port.display_name] = new_name

            for port, field in input_rename_fields:
                new_name = field.text().strip()
                if new_name:
                    port.display_name = new_name
            block.parameters["inputPortDefs"] = block.port_defs("input")

            visible = [label for label, cb in checkboxes if cb.isChecked()]
            new_aliases = {}
            for label, field in rename_fields:
                alias = field.text().strip()
                if alias and alias != label:
                    new_aliases[label] = alias
            # Follow the renames so this block's own outputs (and any aliases)
            # track the new variable names.
            if renames:
                visible = [renames.get(lbl, lbl) for lbl in visible]
                new_aliases = {renames.get(k, k): v for k, v in new_aliases.items()}

            self._sync_data_bus_outputs(block, visible, new_aliases)

            # Propagate the renames to every other Data Bus block in the pool.
            if renames:
                self._propagate_data_bus_rename(new_pool, renames, exclude=block)

            self.canvas.update_block_graphics(block)
            if getattr(self.canvas, "state_changed_callback", None):
                self.canvas.state_changed_callback()

        FloatingShell.open(
            "Edit Data Bus", self.canvas, body, on_ok=_commit, near_block=block
        )

    def _sync_data_bus_outputs(self, block, visible, aliases):
        """Update Data Bus output ports based on selected variables."""
        # Map existing data output ports by description (variable label)
        old_port_map = {}
        for p in block.output_ports:
            if p.name == "Run":
                continue
            key = p.description if p.description else p.display_name
            old_port_map[key] = p

        # Preserve the fixed Run port
        run_port = None
        for p in block.output_ports:
            if p.name == "Run":
                run_port = p
                break
        if run_port is None:
            run_port = Port("Run", "output", "any", False, "Run signal")
            run_port.display_name = "Run"
            run_port.parent_block = block

        new_ports = [run_port]
        run_port.index = 1

        for i, var_label in enumerate(visible):
            display = aliases.get(var_label, var_label)
            if var_label in old_port_map:
                p = old_port_map[var_label]
                p.name = f"out{i + 1}"
                p.display_name = display
                p.index = i + 2
            else:
                p = Port(
                    name=f"out{i + 1}",
                    direction="output",
                    type_="any",
                    required=True,
                    description=var_label,
                )
                p.display_name = display
                p.parent_block = block
                p.index = i + 2
            p.preserve_case = True  # variable name — show verbatim
            new_ports.append(p)

        # Remove wires for ports no longer present
        kept_descriptions = set(visible)
        for desc, port in old_port_map.items():
            if desc not in kept_descriptions:
                for wire in list(port.connections):
                    self.canvas.remove_wire(wire)

        block.output_ports = new_ports

        # Save port definitions for serialization
        block.parameters["outputPortDefs"] = block.port_defs("output")

        # Resize
        block.resize_to_fit_ports()

    def _propagate_data_bus_rename(self, pool, renames, exclude=None):
        """Apply variable renames {old_label: new_label} to every Data Bus block
        in `pool`, so a rename made in one block updates all readers/writers of
        that variable instead of orphaning it as a new one.

        Writer inputs match on display_name; reader outputs match on the
        canonical key (description, falling back to display_name).  A custom
        output alias (display_name != description) is preserved.
        """
        for b in self.canvas.blocks:
            if b is exclude:
                continue
            if not b.definition_name or not self.registry.has(b.definition_name):
                continue
            defn = self.registry.get(b.definition_name)
            if not defn.get("isDataBus"):
                continue
            if b.parameters.get("pool", "Default") != pool:
                continue

            changed = False
            for p in b.input_ports:
                if p.name in ("Run", "addInput"):
                    continue
                if p.display_name in renames:
                    p.display_name = renames[p.display_name]
                    changed = True
            for p in b.output_ports:
                if p.name == "Run":
                    continue
                key = p.description if p.description else p.display_name
                if key in renames:
                    new = renames[key]
                    if p.description:
                        if p.display_name == p.description:  # no custom alias
                            p.display_name = new
                        p.description = new
                    else:
                        p.display_name = new
                    changed = True

            if not changed:
                continue
            # Inline (not port_defs): this path is driven by duck-typed stub
            # blocks in test_databus_rename.
            b.parameters["inputPortDefs"] = [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "description": p.description,
                    "displayName": p.display_name,
                }
                for p in b.input_ports
            ]
            b.parameters["outputPortDefs"] = [
                {
                    "name": p.name,
                    "type": "any",
                    "description": p.description,
                    "displayName": p.display_name,
                }
                for p in b.output_ports
            ]
            self.canvas.update_block_graphics(b)

    def _edit_mean_block(self, block):
        """Dialog to configure Mean block: rename inputs and select mode."""
        from PySide6.QtWidgets import QGridLayout, QGroupBox

        current_mode = block.parameters.get("mode", "column")

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Mean Block")
        dialog.setMinimumWidth(340)
        dlg_layout = QVBoxLayout(dialog)

        # --- Rename Inputs ---
        data_ports = [p for p in block.input_ports if p.name != "addInput"]
        input_rename_fields = []
        if data_ports:
            in_group = QGroupBox("Inputs")
            in_grid = QGridLayout(in_group)
            in_grid.addWidget(QLabel("Port"), 0, 0)
            in_grid.addWidget(QLabel("Name"), 0, 1)
            for i, p in enumerate(data_ports):
                in_grid.addWidget(QLabel(p.name), i + 1, 0)
                field = QLineEdit(p.display_name)
                in_grid.addWidget(field, i + 1, 1)
                input_rename_fields.append((p, field))
            dlg_layout.addWidget(in_group)

        # --- Mode selection ---
        from PySide6.QtWidgets import QRadioButton, QButtonGroup

        grp = QGroupBox("Mode (for a 2-D matrix or multiple inputs)")
        grp_layout = QVBoxLayout(grp)
        btn_group = QButtonGroup(dialog)
        rb_col = QRadioButton("Column-wise (axis 0): mean of each column")
        rb_row = QRadioButton("Row-wise (axis 1): mean of each row")
        btn_group.addButton(rb_col)
        btn_group.addButton(rb_row)
        if current_mode == "row":
            rb_row.setChecked(True)
        else:
            rb_col.setChecked(True)
        grp_layout.addWidget(rb_col)
        grp_layout.addWidget(rb_row)
        note = QLabel(
            "1-D input (row or column vector) → one scalar.\n"
            "2-D matrix → the selected means as a column vector [N×1]."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.palette.text_muted}; font-size: 11px;")
        grp_layout.addWidget(note)
        dlg_layout.addWidget(grp)

        # --- Buttons ---
        buttons = QDialogButtonBox()
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setDefault(True)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlg_layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Apply input renames
        for port, field in input_rename_fields:
            new_name = field.text().strip()
            if new_name:
                port.display_name = new_name
        block.parameters["inputPortDefs"] = block.port_defs("input")

        # Apply mode
        block.parameters["mode"] = "row" if rb_row.isChecked() else "column"
        self.canvas.update_block_graphics(block)

    def _toggle_disable(self, block):
        if block.status == "skipped":
            block.status = block.previous_status
        else:
            block.previous_status = block.status
            block.status = "skipped"
        self.canvas.update_block_graphics(block)

    def _edit_filter_rows(self, block):
        """Custom parameter editor for FilterRows with column dropdown."""
        # Collect connected input port display names
        data_ports = [
            p for p in block.input_ports if p.name != "addInput" and p.is_connected
        ]
        column_names = [p.display_name for p in data_ports]

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit {block.display_name}")
        dlg.setMinimumWidth(300)
        form = QFormLayout(dlg)

        # Column dropdown
        col_combo = QComboBox(dlg)
        col_combo.addItems(column_names)
        current_col = block.parameters.get("column", "")
        idx = col_combo.findText(current_col)
        if idx >= 0:
            col_combo.setCurrentIndex(idx)
        elif current_col:
            col_combo.addItem(current_col)
            col_combo.setCurrentText(current_col)
        form.addRow("Column:", col_combo)

        # Operator dropdown
        op_combo = QComboBox(dlg)
        op_combo.addItems([">", "<", ">=", "<=", "=", "!="])
        op_combo.setCurrentText(block.parameters.get("operator", ">"))
        form.addRow("Operator:", op_combo)

        # Threshold
        val_edit = QLineEdit(str(block.parameters.get("value", 0)), dlg)
        form.addRow("Threshold:", val_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            block.parameters["column"] = col_combo.currentText()
            block.parameters["operator"] = op_combo.currentText()
            try:
                val = float(val_edit.text())
                block.parameters["value"] = int(val) if val == int(val) else val
            except ValueError:
                block.parameters["value"] = val_edit.text()
            self.canvas.update_block_graphics(block)

    def _edit_block_parameters(self, block):
        """Parameter editor dialog."""
        try:
            defn = self.registry.get(block.definition_name)
            defaults = defn.get("defaultParameters", {})
        except (KeyError, AttributeError):
            defaults = {}

        # Text/display blocks get a custom dialog with both fields
        if "valueString" in defaults and "displayText" in defaults:
            self._edit_text_block(block, defaults)
            return

        # Zone Selection gets an interactive zone picker
        if "selectedZones" in defaults:
            self._edit_zone_selection(block)
            return

        # DefineParameters gets a custom variable editor
        if defn.get("isDefineParameters"):
            self._edit_define_parameters(block)
            return

        # Platform Detection gets a custom dialog with checkbox + slider
        if defn.get("isPlatformDetection"):
            self._edit_platform_detection(block)
            return

        # Notch filter gets a combined dialog with harmonics checkboxes
        if defn.get("isNotchFilter"):
            self._edit_notch_block(block, defn)
            return

        # Blocks with a formula parameter get the formula editor
        if "formula" in defaults:
            self._edit_formula_block(block, defn)
            return

        # FilterRows gets a custom dialog with column dropdown
        if defn.get("isFilterRows"):
            self._edit_filter_rows(block)
            return

        # Generic parameter editor — floating, in-canvas panel instead of a
        # native QDialog. Routes back through _on_param_editor_applied for
        # display-text recomputation + repaint + dirty marking.
        FloatingParamEditor.open_for(
            block,
            defn,
            self.canvas,
            on_apply=lambda b, _changes: self._on_param_editor_applied(b),
        )

    def _on_param_editor_applied(self, block):
        """Called after the floating editor commits parameter changes."""
        try:
            defn = self.registry.get(block.definition_name)
        except (KeyError, AttributeError):
            defn = {}
        defaults = defn.get("defaultParameters", {}) or {}
        param_defs = defn.get("parameterDefinitions", []) or []

        if "displayText" in defaults and "valueString" in block.parameters:
            block.parameters["displayText"] = str(block.parameters["valueString"])
        elif "displayText" in defaults and param_defs:
            block.parameters["displayText"] = _build_display_text(
                block.parameters, param_defs
            )
        self.canvas.update_block_graphics(block)
        if getattr(self.canvas, "state_changed_callback", None):
            self.canvas.state_changed_callback()

    def _open_device_geometry(self, block):
        """Open the Device Geometry UI directly (without running the pipeline)
        to view/edit the geometry; persist the result to the block on OK."""
        from utils.device_geometry_dialog import DeviceGeometryDialog

        dialog = DeviceGeometryDialog(dict(block.parameters))
        if not dialog.exec():
            return
        result = dialog.get_result()
        block.parameters["components"] = result["components"]
        block.parameters["ch_height"] = result["ch_height"]
        block.parameters["De_np"] = result["De_np"]
        block.parameters["electrode_pairs"] = result.get("electrode_pairs", [])
        # Auto-save to SavedTemplates/Geometry/temp.json (mirrors the runner).
        try:
            folder = os.path.join(self.root_dir, "SavedTemplates", "Geometry")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        self.canvas.update_block_graphics(block)
        if getattr(self.canvas, "state_changed_callback", None):
            self.canvas.state_changed_callback()

    def _edit_formula_block(self, block, defn):
        """Open the formula editor for any block with a formula parameter."""
        from utils.formula_editor_dialog import FormulaEditorDialog

        defaults = defn.get("defaultParameters", {})
        default_formula = defaults.get("formula", "")
        current = block.parameters.get("formula", default_formula)
        # Variables come from the block's actual input ports (handles legacy /
        # dynamic ports), not just the registry definition.
        port_names = {p.name for p in block.input_ports}
        variables = []
        for p in block.input_ports:
            if p.name == "addInput":
                continue
            variables.append((p.name, p.display_name or p.name, p.description or ""))
        # A time-based block still carrying a legacy SegmentWidth formula is
        # stale — present the current default so OK saves a working formula.
        if (
            "Time" in port_names
            and "SegmentWidth" in (current or "")
            and "Time" not in (current or "")
        ):
            current = default_formula
        # Choice parameters (e.g. input-unit selectors) become dropdowns.
        unit_fields = []
        for pd in defn.get("parameterDefinitions", []):
            if pd.get("type") != "choice":
                continue
            key = pd.get("name")
            unit_fields.append(
                {
                    "key": key,
                    "label": pd.get("displayName", key),
                    "options": pd.get("options", []),
                    "current": block.parameters.get(key, defaults.get(key)),
                    "group": pd.get("group", "Units"),
                }
            )
        dlg = FormulaEditorDialog(
            title=block.display_name,
            current_formula=current,
            default_formula=default_formula,
            variables=variables,
            unit_fields=unit_fields,
            parent=self,
        )
        if dlg.exec():
            block.parameters["formula"] = dlg.get_formula()
            for key, val in dlg.get_unit_values().items():
                block.parameters[key] = val

    def _edit_platform_detection(self, block):
        """Custom editor for Platform Detection with checkbox + slider."""
        from PySide6.QtWidgets import QCheckBox, QSlider

        defaults = {"mzBatchSize": 7, "useCurrentCol": True, "currentCol": 1}
        p = {k: block.parameters.get(k, v) for k, v in defaults.items()}

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit {block.display_name}")
        dlg.setMinimumWidth(320)
        form = QFormLayout(dlg)

        # Batch Size
        batch_spin = QSpinBox(dlg)
        batch_spin.setRange(1, 100)
        batch_spin.setValue(int(p["mzBatchSize"]))
        form.addRow("Batch Size:", batch_spin)

        # Use Current Column checkbox
        use_cb = QCheckBox(dlg)
        use_cb.setChecked(bool(p["useCurrentCol"]))
        form.addRow("Use Current Column:", use_cb)

        # Current Column slider + value label
        col_label = QLabel(str(int(p["currentCol"])), dlg)
        col_slider = QSlider(Qt.Orientation.Horizontal, dlg)
        col_slider.setRange(1, max(int(p["mzBatchSize"]), 1))
        col_slider.setValue(int(p["currentCol"]))
        col_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        col_slider.setTickInterval(1)
        col_slider.valueChanged.connect(lambda v: col_label.setText(str(v)))

        col_row = QHBoxLayout()
        col_row.addWidget(col_slider, 1)
        col_row.addWidget(col_label)
        form.addRow("Current Column:", col_row)

        # Enable/disable slider based on checkbox
        def _toggle_col(checked):
            col_slider.setEnabled(checked)
            col_label.setEnabled(checked)

        _toggle_col(use_cb.isChecked())
        use_cb.toggled.connect(_toggle_col)

        # Update slider range when batch size changes
        def _update_range(val):
            col_slider.setRange(1, max(val, 1))

        batch_spin.valueChanged.connect(_update_range)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            block.parameters["mzBatchSize"] = batch_spin.value()
            block.parameters["useCurrentCol"] = use_cb.isChecked()
            block.parameters["currentCol"] = col_slider.value()

    def _edit_notch_block(self, block, defn):
        """Combined dialog for Notch filter with mode selector."""
        from PySide6.QtWidgets import (
            QCheckBox,
            QGroupBox,
            QGridLayout,
            QRadioButton,
            QButtonGroup,
        )

        defaults = defn.get("defaultParameters", {})
        current_mode = block.parameters.get("mode", "freqBW")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit {block.display_name}")
        dlg.setMinimumWidth(320)

        layout = QFormLayout(dlg)

        # --- Mode selector ---
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)
        btn_group = QButtonGroup(dlg)
        rb_freqbw = QRadioButton("Freq + BW")
        rb_band = QRadioButton("Lower / Upper")
        btn_group.addButton(rb_freqbw)
        btn_group.addButton(rb_band)
        mode_layout.addWidget(rb_freqbw)
        mode_layout.addWidget(rb_band)
        layout.addRow(mode_group)

        # --- Freq+BW page ---
        page_freqbw = QWidget(dlg)
        freqbw_layout = QFormLayout(page_freqbw)
        freqbw_layout.setContentsMargins(0, 0, 0, 0)

        freq_edit = QLineEdit(dlg)
        freq_edit.setText(str(block.parameters.get("freq", defaults.get("freq", 60.0))))
        freqbw_layout.addRow("Frequency (Hz):", freq_edit)

        bw_edit = QLineEdit(dlg)
        bw_edit.setText(str(block.parameters.get("bw", defaults.get("bw", 2.0))))
        freqbw_layout.addRow("Bandwidth (Hz):", bw_edit)

        # Harmonics
        harm_group = QGroupBox("Harmonics")
        harm_layout = QGridLayout(harm_group)
        current_harmonics = block.parameters.get("harmonics", [1])
        harm_cbs = []
        for i in range(5):
            n = i + 1
            cb = QCheckBox(f"{n}x" if n > 1 else f"{n}x (fundamental)")
            cb.setChecked(n in current_harmonics)
            harm_layout.addWidget(cb, i // 3, i % 3)
            harm_cbs.append((n, cb))
        freqbw_layout.addRow(harm_group)

        # --- Band page ---
        page_band = QWidget(dlg)
        band_layout = QFormLayout(page_band)
        band_layout.setContentsMargins(0, 0, 0, 0)

        lower_edit = QLineEdit(dlg)
        lower_edit.setText(
            str(block.parameters.get("lower", defaults.get("lower", 59.0)))
        )
        band_layout.addRow("Lower (Hz):", lower_edit)

        upper_edit = QLineEdit(dlg)
        upper_edit.setText(
            str(block.parameters.get("upper", defaults.get("upper", 61.0)))
        )
        band_layout.addRow("Upper (Hz):", upper_edit)

        # --- Stacked widget ---
        stack = QStackedWidget(dlg)
        stack.addWidget(page_freqbw)  # index 0
        stack.addWidget(page_band)  # index 1
        layout.addRow(stack)

        def _on_mode_changed():
            stack.setCurrentIndex(0 if rb_freqbw.isChecked() else 1)

        rb_freqbw.toggled.connect(lambda: _on_mode_changed())

        # Set initial mode
        if current_mode == "band":
            rb_band.setChecked(True)
        else:
            rb_freqbw.setChecked(True)
        _on_mode_changed()

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            mode = "freqBW" if rb_freqbw.isChecked() else "band"
            block.parameters["mode"] = mode

            if mode == "band":
                try:
                    block.parameters["lower"] = float(lower_edit.text())
                except ValueError:
                    pass
                try:
                    block.parameters["upper"] = float(upper_edit.text())
                except ValueError:
                    pass
                lo = block.parameters["lower"]
                hi = block.parameters["upper"]
                _lo = int(lo) if lo == int(lo) else lo
                _hi = int(hi) if hi == int(hi) else hi
                block.parameters["displayText"] = f"{_lo}-{_hi} Hz"
            else:
                try:
                    block.parameters["freq"] = float(freq_edit.text())
                except ValueError:
                    pass
                try:
                    block.parameters["bw"] = float(bw_edit.text())
                except ValueError:
                    pass
                selected = [n for n, cb in harm_cbs if cb.isChecked()]
                block.parameters["harmonics"] = selected if selected else [1]

                freq = block.parameters["freq"]
                bw = block.parameters["bw"]
                harmonics = block.parameters["harmonics"]
                _f = int(freq) if freq == int(freq) else freq
                _b = int(bw) if bw == int(bw) else bw
                if len(harmonics) == 1 and harmonics[0] == 1:
                    block.parameters["displayText"] = f"{_f}-{_b} Hz"
                else:
                    h_str = ",".join(str(h) for h in sorted(harmonics))
                    block.parameters["displayText"] = f"{_f}-{_b} Hz x{h_str}"

            self.canvas.update_block_graphics(block)

    def _edit_text_block(self, block, defaults):
        """Floating editor for Text blocks: display label + multiline output."""
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 10)
        body_layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        display_edit = QLineEdit(str(block.parameters.get("displayText", "")))
        form.addRow("Display:", display_edit)
        body_layout.addLayout(form)

        body_layout.addWidget(QLabel("Output:"))
        from utils.slash_command_text_edit import SlashCommandTextEdit

        output_edit = SlashCommandTextEdit()
        output_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        output_edit.setAcceptRichText(False)
        output_edit.setPlainText(str(block.parameters.get("valueString", "")))
        body_layout.addWidget(output_edit, 1)

        def _commit():
            block.parameters["valueString"] = output_edit.toPlainText()
            block.parameters["displayText"] = display_edit.text()
            self.canvas.update_block_graphics(block)
            if getattr(self.canvas, "state_changed_callback", None):
                self.canvas.state_changed_callback()

        FloatingShell.open(
            f"Edit {block.display_name}",
            self.canvas,
            body,
            on_ok=_commit,
            near_block=block,
        )

    def _on_super_block_delete(self, def_name):
        """Handle Super Block deletion from the palette drop zone."""
        if not self.registry.has(def_name):
            return
        defn = self.registry.get(def_name)
        self._delete_super_block_by_defn(defn)

    def _delete_super_block_by_defn(self, defn):
        """Delete a Super Block template and remove it from the registry."""
        filepath = defn.get("subPipelineFile", "")
        name = defn.get("displayName", defn.get("name", "Super Block"))

        reply = QMessageBox.question(
            self,
            "Delete Super Block",
            f"Delete Super Block '{name}'?\n\n"
            f"This will remove the template file and all instances "
            f"of this block on the canvas will become invalid.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete the template file
        if filepath and os.path.isfile(filepath):
            os.remove(filepath)

        # Remove from registry
        reg_name = defn.get("name", "")
        if reg_name in self.registry.definitions:
            del self.registry.definitions[reg_name]

        # Remove all instances from all canvas tabs
        for tab in self._tabs:
            canvas = tab.get("canvas")
            if canvas:
                to_remove = [b for b in canvas.blocks if b.definition_name == reg_name]
                for b in to_remove:
                    canvas.remove_block(b)

        # Close any open edit tab for this template
        for i, tab in enumerate(list(self._tabs)):
            if tab.get("file_path") == filepath and tab.get("is_super_block"):
                self._close_tab(i)
                break

        # Refresh palette
        self._init_block_menu_config()
        self._apply_menu_config()
        self.block_palette._refresh_sb_grid()
        self.log(f"Super Block '{name}' deleted.")

    def _add_super_block_boundary(self, canvas, template):
        """Add boundary interface blocks (Inputs on left, Outputs on right)."""
        from port import Port
        from wire_connection import WireConnection

        if not canvas.blocks:
            return

        exposed_inputs = template.get("exposedInputs", [])
        exposed_outputs = template.get("exposedOutputs", [])

        # Compute bounding box of all inner blocks
        min_x = min(b.position[0] for b in canvas.blocks)
        max_x = max(b.position[0] + b.size[0] for b in canvas.blocks)
        max_y = max(b.position[1] for b in canvas.blocks)
        min_y = min(b.position[1] - b.size[1] for b in canvas.blocks)

        pad = 50
        box_x = min_x - pad - 150
        box_y = max_y + pad
        box_w = (max_x - min_x) + 2 * pad + 300
        box_h = (max_y - min_y) + 2 * pad

        block_map = {b.id: b for b in canvas.blocks}
        center_y = (max_y + min_y) / 2

        # --- Inputs block (left side, full height) ---
        # Has OUTPUT ports that wire to inner block input ports
        in_block = BlockNode("_SuperBlockInputs", (box_x, box_y))
        in_block.display_name = "Inputs"
        in_block.color = (0.3, 0.5, 0.8)
        in_block.size = (130, box_h)
        in_block.id = "_sb_inputs"
        in_block.input_ports = []
        in_block.output_ports = []
        in_block.hide_port_labels = False
        in_block.center_ports = True
        for i, ei in enumerate(exposed_inputs):
            p = Port(
                name=f"out{i + 1}", direction="output", type_=ei.get("type", "any")
            )
            p.display_name = ei["name"]
            p.parent_block = in_block
            p.index = i + 1
            in_block.output_ports.append(p)
        # Dynamic add port
        p_add = Port(name="addOutput", direction="output", type_="any")
        p_add.display_name = "Add input"
        p_add.parent_block = in_block
        p_add.index = len(in_block.output_ports) + 1
        in_block.output_ports.append(p_add)
        canvas.blocks.append(in_block)

        # Wire input interface outputs to inner block inputs
        for i, ei in enumerate(exposed_inputs):
            inner_block = block_map.get(ei.get("innerBlock"))
            if not inner_block:
                continue
            src_port = in_block.output_ports[i]
            dst_port = next(
                (p for p in inner_block.input_ports if p.name == ei["innerPort"]), None
            )
            if src_port and dst_port:
                wire = WireConnection(src_port, dst_port)
                wire.color = in_block.color
                src_port.connections.append(wire)
                dst_port.connections.append(wire)
                canvas.wires.append(wire)

        # --- Outputs block (right side, full height) ---
        # Has INPUT ports that receive wires from inner block output ports
        out_block = BlockNode("_SuperBlockOutputs", (box_x + box_w - 130, box_y))
        out_block.display_name = "Outputs"
        out_block.color = (0.3, 0.5, 0.8)
        out_block.size = (130, box_h)
        out_block.id = "_sb_outputs"
        out_block.input_ports = []
        out_block.output_ports = []
        out_block.hide_port_labels = False
        out_block.center_ports = True
        for i, eo in enumerate(exposed_outputs):
            p = Port(
                name=f"in{i + 1}",
                direction="input",
                type_=eo.get("type", "any"),
                required=False,
            )
            p.display_name = eo["name"]
            p.parent_block = out_block
            p.index = i + 1
            out_block.input_ports.append(p)
        # Dynamic add port
        p_add = Port(name="addInput", direction="input", type_="any", required=False)
        p_add.display_name = "Add output"
        p_add.parent_block = out_block
        p_add.index = len(out_block.input_ports) + 1
        out_block.input_ports.append(p_add)
        canvas.blocks.append(out_block)

        # Wire inner block outputs to output interface inputs
        for i, eo in enumerate(exposed_outputs):
            inner_block = block_map.get(eo.get("innerBlock"))
            if not inner_block:
                continue
            src_port = next(
                (p for p in inner_block.output_ports if p.name == eo["innerPort"]), None
            )
            dst_port = out_block.input_ports[i]
            if src_port and dst_port:
                wire = WireConnection(src_port, dst_port)
                wire.color = out_block.color
                src_port.connections.append(wire)
                dst_port.connections.append(wire)
                canvas.wires.append(wire)

        canvas.redraw()

    def _edit_sub_pipeline(self, block, defn):
        """Open the super block template in a new canvas tab for editing."""
        from subpipeline_manager import load_template

        filepath = defn.get("subPipelineFile", "")
        if not filepath or not os.path.isfile(filepath):
            QMessageBox.warning(
                self, "Super Block", f"Template file not found:\n{filepath}"
            )
            return

        # Check if already open in a tab
        for i, tab in enumerate(self._tabs):
            if tab.get("file_path") == filepath:
                self._tab_widget.setCurrentIndex(self._tab_index_for_workflow(i))
                return

        # Load template and convert to canvas-compatible format
        template = load_template(filepath)
        canvas_data = {
            "blocks": template.get("innerBlocks", []),
            "wires": template.get("innerWires", []),
        }

        # Open in a new tab
        display_name = template.get("displayName", template.get("name", "Super Block"))
        wi = self._add_canvas_tab(
            file_path=filepath, label=f"\u2b22 {display_name}"
        )  # hexagon prefix
        canvas = self._tabs[wi]["canvas"]
        canvas.load_from_struct(canvas_data)

        # Add boundary box and port labels
        self._add_super_block_boundary(canvas, template)

        canvas.reset_view()
        # Mark as super block tab for special save behavior
        self._tabs[wi]["is_super_block"] = True
        self._tabs[wi]["template_data"] = template

    def _create_sub_pipeline(self):
        """Create a sub-pipeline from the currently selected blocks."""
        from subpipeline_manager import create_from_selection

        blocks = list(self.canvas._multi_drag_blocks)
        if len(blocks) < 2:
            return

        name, ok = QInputDialog.getText(self, "Create Super Block", "Super Block name:")
        if not ok or not name.strip():
            return
        display = name.strip()
        # Sanitize for filename — keep only alphanumeric, underscore, hyphen
        import re

        name = re.sub(r"[^\w\-]", "_", display)

        try:
            filepath = create_from_selection(
                blocks, self.canvas.wires, name, display_name=display
            )

            # Re-scan registry and rebuild palette to pick up the new template
            self.registry._scan_sub_pipelines()
            self._init_block_menu_config()
            self._apply_menu_config()
            self.log(
                f"Super Block '{name}' created. Available in palette under Super Block."
            )
        except Exception as e:
            self.log(f"Error creating Super Block: {e}")
            import traceback

            traceback.print_exc()

    def _edit_define_parameters(self, block):
        """Open the Define Parameters dialog for editing."""
        from utils.define_parameters_ui import DefineParametersDialog

        saved_vars = block.parameters.get("variables", [])
        saved_sr = block.parameters.get("globalSampleRate", None)
        dlg = DefineParametersDialog(saved_vars, saved_sr, self)
        if dlg.exec() and dlg.confirmed:
            result = dlg.get_results()
            block.parameters["variables"] = result["variables"]
            block.parameters["globalSampleRate"] = result["globalSampleRate"]
            names = [v["name"] for v in result["variables"]]
            parts = []
            if result["globalSampleRate"] is not None:
                parts.append(f"fs={result['globalSampleRate']:g}")
            parts.extend(names)
            block.parameters["displayText"] = ", ".join(parts) if parts else "(empty)"
            self.canvas.viewport().update()

    def _edit_zone_selection(self, block):
        """Interactive zone picker with a configurable number of zones."""
        from PySide6.QtWidgets import QCheckBox, QGridLayout

        previous = block.parameters.get("selectedZones", [])
        prev_count = block.parameters.get("numZones", 6)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit {block.display_name}")
        dlg.setMinimumWidth(280)
        from utils.dialog_style import apply_dialog_style

        apply_dialog_style(dlg)

        layout = QVBoxLayout(dlg)

        # Number of zones spinner
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Number of zones:"))
        spin = QSpinBox(dlg)
        spin.setRange(1, 50)
        spin.setValue(prev_count)
        top_row.addWidget(spin)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Container for checkboxes
        cb_container = QVBoxLayout()
        layout.addLayout(cb_container)
        checkboxes = []

        def rebuild_checkboxes():
            nonlocal checkboxes
            # Clear old
            for cb in checkboxes:
                cb.setParent(None)
            checkboxes.clear()
            # Remove old grid if present
            while cb_container.count():
                item = cb_container.takeAt(0)
                if item.layout():
                    while item.layout().count():
                        item.layout().takeAt(0)

            n = spin.value()
            grid = QGridLayout()
            n_rows = (n + 1) // 2
            for zi in range(n):
                cb = QCheckBox(f"Zone {zi + 1}")
                if previous:
                    cb.setChecked((zi + 1) in previous)
                else:
                    cb.setChecked(True)
                row = zi % n_rows
                col = zi // n_rows
                grid.addWidget(cb, row, col)
                checkboxes.append(cb)
            cb_container.addLayout(grid)
            dlg.adjustSize()

        spin.valueChanged.connect(rebuild_checkboxes)
        rebuild_checkboxes()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            sel = [i + 1 for i, cb in enumerate(checkboxes) if cb.isChecked()]
            block.parameters["selectedZones"] = sel
            block.parameters["numZones"] = spin.value()
            if sel and len(sel) < spin.value():
                block.parameters["displayText"] = ", ".join(str(z) for z in sel)
            else:
                block.parameters["displayText"] = "(all)"
            self.canvas.update_block_graphics(block)

    def show_shortcuts_help(self):
        from utils.dialog_style import apply_dialog_style

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        apply_dialog_style(dlg)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)

        table = (
            '<table cellpadding="4" style="font-size:13px;">'
            '<tr><td colspan="2"><b>Keyboard</b></td></tr>'
            "<tr><td>Ctrl+Z</td><td>Undo</td></tr>"
            "<tr><td>Ctrl+Y</td><td>Redo</td></tr>"
            "<tr><td>Delete</td><td>Delete selected</td></tr>"
            "<tr><td>Ctrl+A</td><td>Auto-layout</td></tr>"
            "<tr><td>Ctrl+=</td><td>Zoom in</td></tr>"
            "<tr><td>Ctrl+\u2013</td><td>Zoom out</td></tr>"
            "<tr><td>Ctrl+0</td><td>Fit to content</td></tr>"
            "<tr><td>&nbsp;</td><td></td></tr>"
            '<tr><td colspan="2"><b>Mouse</b></td></tr>'
            "<tr><td>Left-click block</td><td>Select and drag</td></tr>"
            "<tr><td>Left-click port</td><td>Draw wire</td></tr>"
            "<tr><td>Double-click</td><td>Run block</td></tr>"
            "<tr><td>Right-click</td><td>Context menu</td></tr>"
            "<tr><td>Scroll wheel</td><td>Zoom</td></tr>"
            "<tr><td>Drag canvas</td><td>Pan</td></tr>"
            "</table>"
        )

        label = QLabel(table)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)

        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)

        dlg.exec()

    def open_settings(self):
        """Floating in-canvas settings panel — every edit applies in real time."""
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 16, 20, 16)
        form_host = QWidget()
        layout = QFormLayout(form_host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        # Compact size for all input widgets so they don't stretch across the
        # panel — leaves visual margin on the right.
        FIELD_W = 110

        def _live(key, getter):
            def handler(_=None):
                self.app_settings[key] = getter()
                self._apply_settings()

            return handler

        # --- Font Size: header row + sub-rows below ---
        font_header = QLabel("Font Size:")
        layout.addRow(font_header)

        font_group = QWidget()
        font_grid = QFormLayout(font_group)
        font_grid.setContentsMargins(20, 0, 0, 0)
        font_grid.setSpacing(6)
        font_grid.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )

        spin_font = QSpinBox()
        spin_font.setFixedWidth(FIELD_W)
        spin_font.setRange(6, 24)
        spin_font.setValue(self.app_settings["fontSize"])
        spin_font.valueChanged.connect(_live("fontSize", lambda: spin_font.value()))
        font_grid.addRow("Title:", spin_font)

        spin_port = QSpinBox()
        spin_port.setFixedWidth(FIELD_W)
        spin_port.setRange(4, 18)
        spin_port.setValue(self.app_settings["portFontSize"])
        spin_port.valueChanged.connect(_live("portFontSize", lambda: spin_port.value()))
        font_grid.addRow("Port:", spin_port)

        spin_console = QSpinBox()
        spin_console.setFixedWidth(FIELD_W)
        spin_console.setRange(6, 18)
        spin_console.setValue(self.app_settings["consoleFontSize"])
        spin_console.valueChanged.connect(
            _live("consoleFontSize", lambda: spin_console.value())
        )
        font_grid.addRow("Console:", spin_console)

        spin_annot = QSpinBox()
        spin_annot.setFixedWidth(FIELD_W)
        spin_annot.setRange(6, 48)
        spin_annot.setValue(self.app_settings["annotationFontSize"])
        spin_annot.valueChanged.connect(
            _live("annotationFontSize", lambda: spin_annot.value())
        )
        font_grid.addRow("Annotation:", spin_annot)

        layout.addRow(font_group)

        # --- Grid Pattern: icon button group ---
        grid_row = self._make_icon_picker(
            options=[
                ("grid", "Grid", _make_grid_pattern_icon("grid")),
                ("dots", "Dots", _make_grid_pattern_icon("dots")),
                ("none", "None", _make_grid_pattern_icon("none")),
            ],
            current=self.app_settings["gridPattern"],
            on_change=lambda v: (
                self.app_settings.__setitem__("gridPattern", v),
                self._apply_settings(),
            ),
        )
        layout.addRow("Grid Pattern:", grid_row)

        spin_grid = QSpinBox()
        spin_grid.setFixedWidth(FIELD_W)
        spin_grid.setRange(10, 200)
        spin_grid.setSingleStep(10)
        spin_grid.setValue(self.app_settings["gridSize"])
        spin_grid.valueChanged.connect(_live("gridSize", lambda: spin_grid.value()))
        layout.addRow("Grid Size:", spin_grid)

        # --- Canvas Background colour ---
        bg_row = QWidget()
        bg_layout = QHBoxLayout(bg_row)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(6)

        bg_swatch = QPushButton(bg_row)
        bg_swatch.setFixedSize(28, 22)
        bg_swatch.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        def _resolved_bg_hex():
            hx = self.app_settings.get("canvasBg") or ""
            return hx if hx else theme.palette.canvas_bg

        def _refresh_swatch():
            c = QColor(_resolved_bg_hex())
            border = "#888" if c.lightness() > 160 else "#444"
            bg_swatch.setStyleSheet(
                f"QPushButton {{ background: {c.name()}; border: 1px solid {border};"
                f" border-radius: 3px; }}"
            )

        _refresh_swatch()

        def _pick_bg():
            initial = QColor(_resolved_bg_hex())
            if not initial.isValid():
                initial = QColor("#f7f7f7")
            # Use Qt's own dialog — macOS's native panel can return
            # uninitialised (black) values when the user accepts without
            # interacting with the swatches, which is the bug we hit.
            dlg = QColorDialog(initial, self)
            dlg.setWindowTitle("Canvas Background")
            dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
            dlg.setCurrentColor(initial)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            color = dlg.currentColor()
            if not color.isValid():
                return
            self.app_settings["canvasBg"] = color.name()
            _refresh_swatch()
            self._apply_settings()

        bg_swatch.clicked.connect(_pick_bg)

        btn_reset = QPushButton("Reset", bg_row)
        btn_reset.setFixedHeight(22)
        btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        def _reset_bg():
            self.app_settings["canvasBg"] = ""
            _refresh_swatch()
            self._apply_settings()

        btn_reset.clicked.connect(_reset_bg)

        bg_layout.addWidget(bg_swatch)
        bg_layout.addWidget(btn_reset)
        bg_layout.addStretch(1)
        layout.addRow("Background:", bg_row)

        # --- Block Style: icon button group ---
        # rectangular: square corners; soft: small radius; rounded: large radius.
        shape_row = self._make_icon_picker(
            options=[
                ("rectangular", "Rectangular", _make_block_shape_icon("rectangular")),
                ("soft", "Soft", _make_block_shape_icon("soft")),
                ("rounded", "Rounded", _make_block_shape_icon("rounded")),
            ],
            current=self.app_settings["nodeShape"],
            on_change=lambda v: (
                self.app_settings.__setitem__("nodeShape", v),
                self._apply_settings(),
            ),
        )
        layout.addRow("Block Style:", shape_row)

        # --- Wire Routing: straight ("normal") vs auto-route around blocks ---
        routing_row = self._make_icon_picker(
            options=[
                (
                    "normal",
                    "Normal — straight wires",
                    _make_wire_routing_icon("normal"),
                ),
                ("auto", "Auto-route around blocks", _make_wire_routing_icon("auto")),
            ],
            current=self.app_settings.get("wireRouting", "normal"),
            on_change=lambda v: (
                self.app_settings.__setitem__("wireRouting", v),
                self._apply_settings(),
            ),
        )
        layout.addRow("Wire Routing:", routing_row)

        btn_menu_editor = QPushButton("Edit Block Menu…")
        btn_menu_editor.clicked.connect(lambda: self._open_block_menu_editor())
        layout.addRow(btn_menu_editor)

        body_layout.addWidget(form_host)
        body_layout.addStretch(1)

        FloatingShell.open("Settings", self.canvas, body, show_footer=False)

    def _make_icon_picker(self, options, current, on_change):
        """Build a row of mutually-exclusive icon buttons.

        options: list of (key, tooltip, QIcon)
        current: currently-selected key
        on_change: callable(key) invoked when the user picks a different option
        """
        from PySide6.QtWidgets import QButtonGroup, QToolButton

        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        group = QButtonGroup(host)
        group.setExclusive(True)
        # Keep the QButtonGroup alive (no Qt parent ownership of widgets here).
        host._btn_group = group
        for key, tip, icon in options:
            btn = QToolButton(host)
            btn.setIcon(icon)
            btn.setIconSize(QSize(20, 16))
            btn.setCheckable(True)
            btn.setAutoRaise(False)
            btn.setToolTip(tip)
            btn.setFixedSize(32, 26)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setChecked(key == current)
            btn.clicked.connect(lambda _checked=False, k=key: on_change(k))
            group.addButton(btn)
            row.addWidget(btn)
        row.addStretch(1)
        return host

    # ------------------------------------------------------------------
    # Settings application
    # ------------------------------------------------------------------

    def _apply_settings(self):
        """Push current app_settings to all canvases, blocks, and console.

        Diff-based: only rebuild what actually changed since the previous call.
        The grid (especially the "dots" pattern over a 10k×10k scene) and the
        per-block graphics are expensive; rebuilding them on every spinbox
        tick caused the multi-hundred-millisecond hitches the user reported.
        """
        s = self.app_settings
        prev = getattr(self, "_last_applied_settings", {}) or {}

        font_changed = s["fontSize"] != prev.get("fontSize") or s[
            "portFontSize"
        ] != prev.get("portFontSize")
        shape_changed = s["nodeShape"] != prev.get("nodeShape")
        annot_changed = s["annotationFontSize"] != prev.get("annotationFontSize")
        grid_changed = s["gridPattern"] != prev.get("gridPattern") or s[
            "gridSize"
        ] != prev.get("gridSize")
        bg_changed = s.get("canvasBg", "") != prev.get("canvasBg", "")

        for tab in self._tabs:
            c = tab["canvas"]
            c.base_font_size = s["fontSize"]
            c.base_port_font_size = s["portFontSize"]
            c.base_annotation_font_size = s["annotationFontSize"]
            c.base_node_shape = s["nodeShape"]
            c.auto_route_on_create = s.get("wireRouting") == "auto"
            for block in c.blocks:
                block.font_size = s["fontSize"]
                block.port_font_size = s["portFontSize"]
                block.node_shape = s["nodeShape"]
            for lbl in getattr(c, "text_labels", []):
                lbl.font_size = s["annotationFontSize"]

            if grid_changed:
                c.set_grid_settings(s["gridPattern"], s["gridSize"])

            if bg_changed:
                hx = s.get("canvasBg") or theme.palette.canvas_bg
                c.set_theme(QColor(hx), QColor(theme.palette.canvas_grid))

            # Pick the lightest update that still covers the changes.
            if font_changed:
                # Fonts permeate every text item inside each block; do the
                # full per-block rebuild.
                scene = c._scene
                for block in c.blocks:
                    old = c._block_items.get(block.id)
                    if old is not None:
                        scene.removeItem(old)
                    item = BlockGraphicsItem(block, c)
                    scene.addItem(item)
                    c._block_items[block.id] = item
            elif shape_changed:
                # Just re-stroke body + header paths on existing block items.
                for block in c.blocks:
                    item = c._block_items.get(block.id)
                    if item is not None and hasattr(item, "apply_shape"):
                        item.apply_shape(block.node_shape)
                # Annotations follow the same shape; their paint() reads
                # canvas.base_node_shape, so just invalidate them.
                for ann_item in getattr(c, "_annotation_items", {}).values():
                    ann_item.update()

            if annot_changed:
                for item in list(getattr(c, "_text_label_items", {}).values()):
                    try:
                        item.update_graphics()
                    except Exception:
                        pass

        # Console font
        current_font = self.log_area.font()
        current_font.setPointSize(s["consoleFontSize"])
        self.log_area.setFont(current_font)

        # Snapshot for the next diff.
        self._last_applied_settings = dict(s)

        # Live-edit settings panel re-applies on each step; route to the debug
        # log so we don't flood the visible console.
        self.log_debug(
            f"Settings applied — font={s['fontSize']}, grid={s['gridPattern']}"
        )

    def _open_block_menu_editor(self):
        """Open the block menu editor dialog."""
        from utils.block_menu_editor import BlockMenuEditorDialog

        dlg = BlockMenuEditorDialog(self._block_menu_config, self.registry, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            config = dlg.result_config()
            if config:
                self._block_menu_config = config
                self._apply_menu_config()
                self._save_block_menu_config()
                self.log("Block menu updated.")

    def _apply_menu_config(self):
        """Rebuild the palette and context menus from _block_menu_config."""
        self.block_palette.rebuild_from_config(self._block_menu_config, self.registry)

    def log(self, message):
        """Append a normal message to the log."""
        self._log_entries.append((message, False))
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )

    def log_debug(self, message):
        """Append a debug message (only visible in debug mode)."""
        self._log_entries.append((message, True))
        if self._debug_mode:
            self.log_area.append(
                f'<span style="color:{theme.palette.text_muted}">{message}</span>'
            )
            self.log_area.verticalScrollBar().setValue(
                self.log_area.verticalScrollBar().maximum()
            )

    def _copy_log_to_clipboard(self):
        """Copy visible log messages to clipboard."""
        lines = []
        for msg, is_debug in self._log_entries:
            if is_debug and not self._debug_mode:
                continue
            lines.append(msg)
        QApplication.clipboard().setText("\n".join(lines))

    def _clear_log(self):
        """Clear all log messages."""
        self._log_entries.clear()
        self.log_area.clear()

    def _on_debug_toggled(self, checked):
        """Toggle debug mode and rebuild the log view."""
        self._debug_mode = checked
        self._render_log()

    def _render_log(self):
        """Re-render all log entries using the current theme + debug mode."""
        self.log_area.clear()
        for msg, is_debug in self._log_entries:
            if is_debug:
                if self._debug_mode:
                    self.log_area.append(
                        f'<span style="color:{theme.palette.text_muted}">{msg}</span>'
                    )
            else:
                self.log_area.append(msg)
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )

    def _on_state_changed(self):
        """Mark current tab as having unsaved changes."""
        self._dirty = True
        self._update_tab_dirty_indicator()
        # Block delete / reset / clear all funnel through here; keep the
        # floating Issues panel in sync since those paths don't fire status_callback.
        self._refresh_issues_panel()

    def _update_tab_dirty_indicator(self):
        """Update the current tab label to show/hide unsaved indicator."""
        idx = self._tab_widget.currentIndex()
        if self._workflow_index(idx) is None:
            return
        text = self._tab_widget.tabText(idx)
        # Strip existing indicator
        clean = text.rstrip(" \u2022")
        if self._dirty:
            self._tab_widget.setTabText(idx, clean + " \u2022")
        else:
            self._tab_widget.setTabText(idx, clean)

    def closeEvent(self, event):
        """Handle window close — prompt to save each unsaved tab."""
        dirty_tabs = [
            (i, tab) for i, tab in enumerate(self._tabs) if tab.get("dirty", False)
        ]
        if not dirty_tabs:
            for tab in self._tabs:
                tab["canvas"].setScene(None)
            self.inspector.close()
            event.accept()
            return

        for i, tab in dirty_tabs:
            if tab.get("is_super_block"):
                title = "Save Super Block"
                name = os.path.basename(tab.get("file_path", "")) or "Super Block"
                msg_text = f'Save changes to "{name}" before closing?'
            else:
                name = os.path.basename(tab.get("file_path", "")) or "Untitled"
                title = "Save Workflow"
                msg_text = f'Save changes to "{name}" before closing?'

            reply = QMessageBox.question(
                self,
                title,
                msg_text,
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._tab_widget.setCurrentIndex(self._tab_index_for_workflow(i))
                if tab.get("is_super_block"):
                    self._save_super_block(tab, tab.get("file_path", ""))
                else:
                    self.save_workflow()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            # Discard: continue to next tab

        for tab in self._tabs:
            tab["canvas"].setScene(None)
        self.inspector.close()
        event.accept()
