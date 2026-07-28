"""Viewer dialog for the CompileTable block — sortable columns, row
filter, copy-to-clipboard, and CSV export."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFileDialog, QApplication, QMessageBox, QMenu,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

from utils.paths import project_root

from utils.dialog_style import apply_dialog_style


def _fmt_num(x, decimals):
    """Format a single number: ``decimals`` decimal places, or ``None`` for
    auto (6 significant digits, trailing zeros stripped)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:        # NaN
        return ""
    if decimals is None:
        return f"{x:.6g}"
    return f"{x:.{decimals}f}"


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _format_cell(v, decimals=None):
    """Pretty-print a value for display, honoring a per-column decimal count.

    Floats and the numbers inside list/tuple cells are formatted; ints and
    strings render verbatim (ints keep their exact value)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return ""
        return _fmt_num(v, decimals)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple)):
        parts = [_fmt_num(x, decimals) if _is_number(x) else str(x) for x in v]
        return "[" + ", ".join(parts) + "]"
    return str(v)


def _sort_key(v):
    """Numeric sort key for a raw cell value, or None to fall back to text."""
    if isinstance(v, bool):
        return None
    if _is_number(v):
        return float(v) if v == v else None
    if isinstance(v, (list, tuple)) and len(v) == 1 and _is_number(v[0]):
        return float(v[0]) if v[0] == v[0] else None
    return None


class _Cell(QTableWidgetItem):
    """Table item that sorts by its raw value (numeric when possible) so the
    displayed precision never changes the sort order."""

    def __lt__(self, other):
        a = _sort_key(self.data(Qt.ItemDataRole.UserRole))
        b = _sort_key(other.data(Qt.ItemDataRole.UserRole))
        if a is not None and b is not None:
            return a < b
        return self.text() < other.text()


class _CompileTableDialog(QDialog):
    """Sortable / filterable view of a CompileTable result."""

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compiled Table")
        self.resize(900, 560)
        apply_dialog_style(self)
        self._table = table or {}
        self._col_decimals = {}   # column index -> int decimals, or None = auto
        self._build_ui()
        self._populate()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        # Top toolbar: search + export + summary
        top = QHBoxLayout()
        top.setSpacing(8)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter rows (matches any cell)…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        top.addWidget(self._search, 1)

        btn_copy = QPushButton("Copy Selection")
        btn_copy.setToolTip("Copy selected cells to clipboard (tab-delimited)")
        btn_copy.clicked.connect(self._copy_selection)
        top.addWidget(btn_copy)

        btn_csv = QPushButton("Export CSV…")
        btn_csv.clicked.connect(self._export_csv)
        top.addWidget(btn_csv)
        v.addLayout(top)

        # Table
        self._tbl = QTableWidget()
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._tbl.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.setSortingEnabled(True)
        self._tbl.verticalHeader().setVisible(False)
        hdr = self._tbl.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # Right-click a column header to set its displayed decimal places.
        hdr.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hdr.customContextMenuRequested.connect(self._header_menu)
        hdr.setToolTip("Right-click a column header to set its decimal places")
        v.addWidget(self._tbl, 1)

        # Bottom row: count + close
        bot = QHBoxLayout()
        self._count_lbl = QLabel("")
        bot.addWidget(self._count_lbl)
        bot.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot.addWidget(btn_close)
        v.addLayout(bot)

    def _populate(self):
        table = self._table
        cols = list(table.keys())
        n_rows = max((len(v) for v in table.values()), default=0)
        n_cols = len(cols)
        # setSortingEnabled(False) during fill so setItem doesn't trigger sort
        # on every insert — much faster, also keeps the initial row order.
        self._tbl.setSortingEnabled(False)
        self._tbl.clear()
        self._tbl.setColumnCount(n_cols)
        self._tbl.setRowCount(n_rows)
        self._tbl.setHorizontalHeaderLabels(cols)

        for ci, col in enumerate(cols):
            data = table[col]
            dec = self._col_decimals.get(ci)
            for ri in range(n_rows):
                v = data[ri] if ri < len(data) else None
                item = _Cell()
                # Keep the raw value for re-formatting and numeric sorting.
                item.setData(Qt.ItemDataRole.UserRole, v)
                item.setText(_format_cell(v, dec))
                if _is_number(v) or _sort_key(v) is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
                self._tbl.setItem(ri, ci, item)

        # Right-size columns to content (capped by header).
        self._tbl.resizeColumnsToContents()
        for ci in range(n_cols):
            self._tbl.setColumnWidth(
                ci, min(max(80, self._tbl.columnWidth(ci) + 14), 240))

        self._tbl.setSortingEnabled(True)
        self._update_count()

    # ----- per-column decimal places -----

    _DECIMAL_CHOICES = [("Auto", None)] + [(str(n), n) for n in range(0, 9)]

    def _header_menu(self, pos):
        hdr = self._tbl.horizontalHeader()
        ci = hdr.logicalIndexAt(pos)
        if ci < 0:
            return
        hitem = self._tbl.horizontalHeaderItem(ci)
        col_name = hitem.text() if hitem is not None else f"Column {ci}"
        cur = self._col_decimals.get(ci)

        menu = QMenu(self)
        title = menu.addAction(f"Decimal places — {col_name}")
        title.setEnabled(False)
        menu.addSeparator()
        for label, val in self._DECIMAL_CHOICES:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(cur == val)
            act.triggered.connect(
                lambda _checked=False, c=ci, vv=val: self._set_col_decimals(c, vv))

        menu.addSeparator()
        all_menu = menu.addMenu("Apply to all columns")
        for label, val in self._DECIMAL_CHOICES:
            a = all_menu.addAction(label)
            a.triggered.connect(
                lambda _checked=False, vv=val: self._set_all_decimals(vv))

        menu.exec(hdr.mapToGlobal(pos))

    def _set_col_decimals(self, ci, decimals):
        self._col_decimals[ci] = decimals
        self._reformat_column(ci)

    def _set_all_decimals(self, decimals):
        for ci in range(self._tbl.columnCount()):
            self._col_decimals[ci] = decimals
            self._reformat_column(ci)

    def _reformat_column(self, ci):
        dec = self._col_decimals.get(ci)
        for ri in range(self._tbl.rowCount()):
            it = self._tbl.item(ri, ci)
            if it is None:
                continue
            it.setText(_format_cell(it.data(Qt.ItemDataRole.UserRole), dec))

    def _apply_filter(self, text):
        q = text.strip().lower()
        n_rows = self._tbl.rowCount()
        n_cols = self._tbl.columnCount()
        hidden = 0
        for ri in range(n_rows):
            if not q:
                self._tbl.setRowHidden(ri, False)
                continue
            match = False
            for ci in range(n_cols):
                it = self._tbl.item(ri, ci)
                if it is not None and q in it.text().lower():
                    match = True
                    break
            self._tbl.setRowHidden(ri, not match)
            if not match:
                hidden += 1
        self._update_count(hidden)

    def _update_count(self, hidden=0):
        total = self._tbl.rowCount()
        cols = self._tbl.columnCount()
        if hidden:
            self._count_lbl.setText(
                f"{total - hidden} of {total} rows · {cols} cols")
        else:
            self._count_lbl.setText(f"{total} rows · {cols} cols")

    def _copy_selection(self):
        ranges = self._tbl.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        lines = []
        for ri in range(r.topRow(), r.bottomRow() + 1):
            row_cells = []
            for ci in range(r.leftColumn(), r.rightColumn() + 1):
                it = self._tbl.item(ri, ci)
                row_cells.append(it.text() if it is not None else "")
            lines.append("\t".join(row_cells))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _export_csv(self):
        import csv
        import os
        # Default to project Output folder if it exists.
        root = project_root()
        default_folder = os.path.join(root, "Output")
        if not os.path.isdir(default_folder):
            default_folder = os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", os.path.join(default_folder, "compiled_table.csv"),
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        cols = [self._tbl.horizontalHeaderItem(c).text()
                for c in range(self._tbl.columnCount())]
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for ri in range(self._tbl.rowCount()):
                    if self._tbl.isRowHidden(ri):
                        continue
                    row = []
                    for ci in range(self._tbl.columnCount()):
                        it = self._tbl.item(ri, ci)
                        row.append(it.text() if it is not None else "")
                    w.writerow(row)
        except Exception as exc:
            QMessageBox.warning(self, "Export CSV", f"Failed to write:\n{exc}")
            return


def show_compile_table(table, parent=None):
    """Show the dialog modally and return the table unchanged."""
    if QApplication.instance() is None:
        return table
    dlg = _CompileTableDialog(table, parent=parent)
    dlg.exec()
    return table
