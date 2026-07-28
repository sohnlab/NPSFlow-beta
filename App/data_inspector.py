"""DataInspector - Viewer for inspecting block output data."""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QPushButton, QLabel,
    QGroupBox, QHeaderView, QMessageBox, QTableWidget, QTableWidgetItem,
    QStackedWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from theme import theme


def _size_str(val):
    if isinstance(val, np.ndarray):
        return "x".join(str(s) for s in val.shape)
    if isinstance(val, (list, tuple)):
        return str(len(val))
    return "1x1"


def _type_str(val):
    if isinstance(val, np.ndarray):
        return f"ndarray ({val.dtype})"
    return type(val).__name__


def _to_plottable(val):
    """Try to convert val to a 1-D numpy array for plotting. Return None on failure."""
    if isinstance(val, np.ndarray):
        if val.size > 1:
            return val
        return None
    if isinstance(val, (list, tuple)) and len(val) > 1:
        try:
            arr = np.asarray(val, dtype=float)
            if arr.ndim >= 1 and arr.size > 1:
                return arr
        except (ValueError, TypeError):
            pass
    return None


class DataInspector(QMainWindow):
    """Viewer for inspecting block output data.
    Supports numeric arrays, structs, cell arrays with plotting.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Data Inspector")
        self.resize(950, 500)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: variable tree
        left_group = QGroupBox("Output Variables")
        left_layout = QVBoxLayout(left_group)

        self.var_tree = QTreeWidget()
        self.var_tree.setColumnCount(3)
        self.var_tree.setHeaderLabels(["Name", "Size", "Type"])
        self.var_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.var_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.var_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.var_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.var_tree.currentItemChanged.connect(self._on_item_select)
        left_layout.addWidget(self.var_tree)

        left_btn_row = QHBoxLayout()
        btn_plot = QPushButton("Quick Plot")
        btn_plot.clicked.connect(self._quick_plot)
        left_btn_row.addWidget(btn_plot)

        btn_export = QPushButton("Print to Console")
        btn_export.clicked.connect(self._print_to_console)
        left_btn_row.addWidget(btn_export)
        left_layout.addLayout(left_btn_row)

        splitter.addWidget(left_group)

        # Right panel: details + plot
        right_group = QGroupBox("Details")
        right_layout = QVBoxLayout(right_group)

        # Stacked widget: text view (scalars/dicts) or table view (arrays/lists)
        self._detail_stack = QStackedWidget()

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFontFamily("Courier")
        self._detail_stack.addWidget(self.detail_text)  # index 0

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(2)

        self.detail_table = QTableWidget()
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.detail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.detail_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.detail_table)

        # Pagination bar
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        nav_style = (
            f"QPushButton {{ padding: 2px 8px; font-size: 11px; }}"
            f"QPushButton:disabled {{ color: {theme.palette.text_muted}; }}"
        )
        self._nav_first = QPushButton("<<")
        self._nav_first.setStyleSheet(nav_style)
        self._nav_first.setFixedWidth(36)
        self._nav_first.clicked.connect(lambda: self._nav_page(0))
        self._nav_prev = QPushButton("<")
        self._nav_prev.setStyleSheet(nav_style)
        self._nav_prev.setFixedWidth(36)
        self._nav_prev.clicked.connect(lambda: self._nav_page(self._table_page - 1))
        self._nav_next = QPushButton(">")
        self._nav_next.setStyleSheet(nav_style)
        self._nav_next.setFixedWidth(36)
        self._nav_next.clicked.connect(lambda: self._nav_page(self._table_page + 1))
        self._nav_last = QPushButton(">>")
        self._nav_last.setStyleSheet(nav_style)
        self._nav_last.setFixedWidth(36)
        self._nav_last.clicked.connect(lambda: self._nav_page(self._table_total_pages - 1))
        self._nav_label = QLabel("")
        self._nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._nav_label.setStyleSheet(
            f"font-size: 11px; color: {theme.palette.text};")
        nav_row.addWidget(self._nav_first)
        nav_row.addWidget(self._nav_prev)
        nav_row.addWidget(self._nav_label, 1)
        nav_row.addWidget(self._nav_next)
        nav_row.addWidget(self._nav_last)
        self._nav_bar = QWidget()
        self._nav_bar.setLayout(nav_row)
        self._nav_bar.hide()
        table_layout.addWidget(self._nav_bar)

        self._detail_stack.addWidget(table_container)  # index 1

        # Pagination state
        self._table_page = 0
        self._table_total_pages = 1
        self._table_arr = None
        self._table_name = ""
        self._PAGE_SIZE = 100

        right_layout.addWidget(self._detail_stack)

        self.figure = Figure(figsize=(4, 2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        right_layout.addWidget(self.canvas)

        btn_row = QHBoxLayout()
        btn_home = QPushButton("Home")
        btn_home.clicked.connect(self._reset_axes)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(btn_home)
        btn_row.addWidget(btn_close)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right_group)
        splitter.setSizes([300, 450])

    def inspect(self, block_name, output_data, input_data=None):
        """Open or update the inspector with block input and output data."""
        self.current_data = output_data
        self.setWindowTitle(f"Data Inspector - {block_name}")
        self._populate_tree(output_data, input_data)
        self.show()
        self.raise_()

    def _populate_tree(self, data, input_data=None):
        """Build the variable tree, expanding dicts as child items."""
        self.var_tree.clear()

        if not isinstance(data, dict):
            return

        def _attach_children(item, val):
            if isinstance(val, dict):
                for key, sub_val in val.items():
                    child = QTreeWidgetItem([str(key), _size_str(sub_val), _type_str(sub_val)])
                    child.setData(0, Qt.ItemDataRole.UserRole, sub_val)
                    _attach_children(child, sub_val)
                    item.addChild(child)
            elif isinstance(val, (list, tuple)) and val and isinstance(val[0], (dict, np.ndarray)):
                for i, sub_val in enumerate(val):
                    child = QTreeWidgetItem([f"[{i}]", _size_str(sub_val), _type_str(sub_val)])
                    child.setData(0, Qt.ItemDataRole.UserRole, sub_val)
                    _attach_children(child, sub_val)
                    item.addChild(child)

        def _add_entries(parent_item, entries):
            for name, val in entries.items():
                item = QTreeWidgetItem([name, _size_str(val), _type_str(val)])
                item.setData(0, Qt.ItemDataRole.UserRole, val)
                _attach_children(item, val)
                if parent_item is None:
                    self.var_tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)

        # Input data section
        if input_data:
            in_header = QTreeWidgetItem(["Inputs", "", ""])
            in_header.setData(0, Qt.ItemDataRole.UserRole, None)
            font = in_header.font(0)
            font.setBold(True)
            in_header.setFont(0, font)
            in_header.setForeground(0, QColor(100, 100, 100))
            self.var_tree.addTopLevelItem(in_header)
            _add_entries(in_header, input_data)
            in_header.setExpanded(True)

        # Output data section
        out_header = QTreeWidgetItem(["Outputs", "", ""])
        out_header.setData(0, Qt.ItemDataRole.UserRole, None)
        font = out_header.font(0)
        font.setBold(True)
        out_header.setFont(0, font)
        self.var_tree.addTopLevelItem(out_header)
        _add_entries(out_header, data)
        out_header.setExpanded(True)

    def _get_selected_value(self):
        """Return (name, value) for the currently selected tree item, or (None, None)."""
        item = self.var_tree.currentItem()
        if item is None:
            return None, None
        # Build name path for nested items
        name = item.text(0)
        if item.parent() is not None:
            name = f"{item.parent().text(0)}.{name}"
        val = item.data(0, Qt.ItemDataRole.UserRole)
        return name, val

    def _on_item_select(self, current, previous):
        if current is None:
            return
        name, val = self._get_selected_value()
        if val is None:
            return

        # Try to display as table for arrays/lists
        if self._try_show_table(name, val):
            return

        # Fallback: text display for scalars, strings, dicts
        self._detail_stack.setCurrentIndex(0)
        details = [f"Variable: {name}", f"Type:     {type(val).__name__}"]

        if isinstance(val, np.ndarray):
            details.append(f"Shape:    {val.shape}")
            details.append(f"Dtype:    {val.dtype}")
            if val.size == 1:
                details.append(f"\nValue: {val.item()}")
        elif isinstance(val, (int, float, bool)):
            details.append(f"\nValue: {val}")
        elif isinstance(val, str):
            details.append(f'\nValue: "{val}"')
        elif isinstance(val, dict):
            details.append(f"\nKeys: {', '.join(str(k) for k in val.keys())}")

        self.detail_text.setText("\n".join(details))

    def _try_show_table(self, name, val):
        """Show val in the detail table if it's an array or list. Returns True if shown."""
        arr = None
        if isinstance(val, np.ndarray) and val.size > 1:
            arr = val
        elif isinstance(val, (list, tuple)) and len(val) > 1:
            try:
                arr = np.asarray(val, dtype=float)
            except (ValueError, TypeError):
                arr = np.array(val, dtype=object)

        if arr is None:
            return False

        # Flatten 3D+ to 1D
        if arr.ndim > 2:
            arr = arr.ravel()

        # Store for pagination
        self._table_arr = arr
        self._table_name = name
        total_rows = arr.shape[0]
        self._table_total_pages = max(1, (total_rows + self._PAGE_SIZE - 1)
                                      // self._PAGE_SIZE)
        self._table_page = 0

        self._detail_stack.setCurrentIndex(1)
        self._render_table_page()
        return True

    def _render_table_page(self):
        """Fill the table with the current page of data."""
        arr = self._table_arr
        if arr is None:
            return
        name = self._table_name
        page = self._table_page
        ps = self._PAGE_SIZE
        total_rows = arr.shape[0]

        row_start = page * ps
        row_end = min(row_start + ps, total_rows)

        is_float = np.issubdtype(arr.dtype, np.floating)
        fmt = (lambda v: f"{v:.6g}") if is_float else str

        tbl = self.detail_table
        tbl.clear()
        n_show = row_end - row_start

        if arr.ndim == 1:
            tbl.setRowCount(n_show)
            tbl.setColumnCount(2)
            tbl.setHorizontalHeaderLabels(["Index", name])
            for i in range(n_show):
                idx = row_start + i
                tbl.setItem(i, 0, QTableWidgetItem(str(idx)))
                tbl.setItem(i, 1, QTableWidgetItem(fmt(arr[idx])))
        elif arr.ndim == 2:
            cols = arr.shape[1]
            tbl.setRowCount(n_show)
            tbl.setColumnCount(cols + 1)
            headers = ["Index"] + [f"Col {j}" for j in range(cols)]
            tbl.setHorizontalHeaderLabels(headers)
            for i in range(n_show):
                idx = row_start + i
                tbl.setItem(i, 0, QTableWidgetItem(str(idx)))
                for j in range(cols):
                    tbl.setItem(i, j + 1, QTableWidgetItem(fmt(arr[idx, j])))

        tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        for c in range(1, tbl.columnCount()):
            tbl.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeMode.Stretch)

        # Update pagination controls
        has_pages = self._table_total_pages > 1
        self._nav_bar.setVisible(has_pages)
        if has_pages:
            self._nav_label.setText(
                f"{row_start:,}–{row_end - 1:,}  of  {total_rows:,}")
            self._nav_first.setEnabled(page > 0)
            self._nav_prev.setEnabled(page > 0)
            self._nav_next.setEnabled(page < self._table_total_pages - 1)
            self._nav_last.setEnabled(page < self._table_total_pages - 1)

    def _nav_page(self, page):
        """Navigate to a specific page."""
        page = max(0, min(page, self._table_total_pages - 1))
        if page == self._table_page:
            return
        self._table_page = page
        self._render_table_page()

    def _quick_plot(self):
        """Plot the selected variable (or dict field) on the canvas."""
        name, val = self._get_selected_value()
        if val is None:
            QMessageBox.information(self, "No Selection", "Select a variable first.")
            return

        # If it's a dict, expand in tree and prompt to select a field
        if isinstance(val, dict):
            item = self.var_tree.currentItem()
            if item is not None:
                self.var_tree.expandItem(item)
            QMessageBox.information(self, "Select Field",
                                    "Expand and select a field to plot.")
            return

        arr = _to_plottable(val)
        if arr is None:
            QMessageBox.information(self, "Not Plottable",
                                    f"Cannot plot '{name}' (not a numeric array).")
            return

        self._plot_array(arr, name)

    def _plot_array(self, arr, name):
        """Plot a numpy array on the figure canvas."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if arr.ndim == 1:
            ax.plot(arr, ".-")
            ax.set_title(name)
            ax.set_xlabel("Index")
        elif arr.ndim == 2 and arr.shape[1] <= 20:
            for col in range(arr.shape[1]):
                ax.plot(arr[:, col], label=f"Col {col}")
            ax.set_title(name)
            ax.legend()
        else:
            ax.plot(arr.ravel(), ".-")
            ax.set_title(name)
            ax.set_xlabel("Index")

        self.figure.tight_layout()
        self.canvas.draw()

    def _reset_axes(self):
        """Reset axes to fit all plotted data."""
        for ax in self.figure.axes:
            ax.autoscale()
        self.canvas.draw()

    def _print_to_console(self):
        """Print selected variable to console."""
        name, val = self._get_selected_value()
        if val is None:
            QMessageBox.information(self, "No Selection", "Select a variable first.")
            return

        print(f"\n--- {name} ---")
        print(f"Type: {type(val).__name__}")
        if isinstance(val, np.ndarray):
            print(f"Shape: {val.shape}")
        print(val)
        print("---\n")
        QMessageBox.information(self, "Exported",
                                f'Variable "{name}" printed to console.')
