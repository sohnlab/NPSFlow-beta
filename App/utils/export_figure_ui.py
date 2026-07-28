"""Export figure UI with editing controls for color, title, axes, fonts, etc.

Public API:
    export_figure_ui(source_fig, info=None)   — open the full editor
    quick_export(source_fig, parent, info)    — one-click save with smart defaults
    make_export_button(get_fig, parent,
                       get_info=None)         — split button (Quick | Customize / Copy)

Last-used folder/format/DPI persist to SavedTemplates/ExportFig/temp.json.
"""

import io
import json
import os
import re
import sys
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QFrame,
    QLineEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QPushButton, QColorDialog,
    QWidget, QFileDialog, QLabel, QToolBox, QGroupBox,
    QToolButton, QMenu, QApplication,
)
from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QColor, QImage, QIcon, QPixmap, QPainter, QPen

from utils.paths import project_root

_SANS = "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"

# Matplotlib linestyle keys and display names
_LINESTYLES = [
    ("-", "Solid"),
    ("--", "Dashed"),
    ("-.", "Dash-dot"),
    (":", "Dotted"),
    ("None", "None"),
]

# Common font families available in matplotlib
_FONT_FAMILIES = [
    "sans-serif",
    "serif",
    "monospace",
    "cursive",
    "fantasy",
]


def _is_data_axes(ax):
    """Return True if *ax* looks like a real data axes, not a widget/button."""
    pos = ax.get_position()
    if pos.height < 0.08:
        return False
    if hasattr(ax, "_colorbar_info"):
        return False
    return True


def _axes_label(ax, idx):
    """Generate a human-readable label for an axes."""
    title = ax.get_title()
    if title:
        return f"Axes {idx + 1}: {title}"
    ylabel = ax.get_ylabel()
    if ylabel:
        return f"Axes {idx + 1}: {ylabel}"
    return f"Axes {idx + 1}"


_SETTINGS_REL = os.path.join("SavedTemplates", "ExportFig", "temp.json")
_FORMAT_FILTERS = [
    ("png", "PNG (*.png)"),
    ("pdf", "PDF (*.pdf)"),
    ("svg", "SVG (*.svg)"),
    ("jpg", "JPEG (*.jpg)"),
]
_SLUG_RE = re.compile(r"[^\w\-]+")


def _settings_path() -> str:
    return os.path.join(project_root(), _SETTINGS_REL)


def load_settings() -> dict:
    """Return persisted settings merged over sensible defaults."""
    out = {
        "folder": os.path.join(project_root(), "Output"),
        "format": "png",
        "dpi": 300,
    }
    try:
        with open(_settings_path()) as f:
            disk = json.load(f)
        if isinstance(disk, dict):
            out.update({k: v for k, v in disk.items() if k in out})
    except (OSError, ValueError):
        pass
    return out


def save_settings(s: dict) -> None:
    path = _settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(s, f, indent=2)
    except OSError:
        pass


def _slugify(s: str) -> str:
    s = _SLUG_RE.sub("_", s).strip("_")
    return s[:60] or "figure"


def _default_filename(source_fig, info=None) -> str:
    """Build a sensible default base name: <upstream-filename>__<axes-title>."""
    parts = []
    if isinstance(info, dict):
        fn = info.get("filename")
        if fn:
            parts.append(_slugify(os.path.splitext(os.path.basename(str(fn)))[0]))
    for ax in source_fig.get_axes():
        title = ax.get_title()
        if title:
            parts.append(_slugify(title))
            break
    return "__".join(parts) if parts else "figure"


def _vector_format(fmt: str) -> bool:
    return fmt.lower() in ("svg", "pdf")


def _build_filter(preferred: str) -> str:
    """File-dialog filter string with preferred format first."""
    pref = next((flt for k, flt in _FORMAT_FILTERS if k == preferred),
                _FORMAT_FILTERS[0][1])
    rest = [flt for k, flt in _FORMAT_FILTERS if k != preferred]
    return ";;".join([pref] + rest + ["All Files (*)"])


def _save_to_path(source_fig, path: str, dpi: int):
    """Save the figure honoring its current facecolor; skip DPI for vectors."""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    facecolor = source_fig.get_facecolor() or "white"
    if _vector_format(ext):
        source_fig.savefig(path, bbox_inches="tight", facecolor=facecolor)
    else:
        source_fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    return ext


def _copy_fig_to_clipboard(source_fig, dpi: int = 200) -> None:
    """Render *source_fig* as a PNG into the system clipboard."""
    buf = io.BytesIO()
    facecolor = source_fig.get_facecolor() or "white"
    source_fig.savefig(buf, format="png", dpi=dpi,
                       bbox_inches="tight", facecolor=facecolor)
    img = QImage.fromData(buf.getvalue(), "PNG")
    QApplication.clipboard().setImage(img)


def quick_export(source_fig, parent=None, info=None) -> str:
    """Save *source_fig* via a pre-filled file dialog. Returns path or ''."""
    s = load_settings()
    folder = s.get("folder") or os.path.join(project_root(), "Output")
    os.makedirs(folder, exist_ok=True)
    fmt = (s.get("format") or "png").lower()
    base = _default_filename(source_fig, info)
    suggested = os.path.join(folder, f"{base}.{fmt}")

    path, _flt = QFileDialog.getSaveFileName(
        parent, "Save Figure", suggested, _build_filter(fmt))
    if not path:
        return ""

    dpi = int(s.get("dpi", 300))
    ext = _save_to_path(source_fig, path, dpi) or fmt
    save_settings({"folder": os.path.dirname(path), "format": ext, "dpi": dpi})
    return path


def make_export_button(get_fig, parent=None, get_info=None, label="Export"):
    """Split button: main click = quick_export; menu = Customize / Copy."""
    btn = QToolButton(parent)
    btn.setText(label)
    btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

    # Match a sibling QPushButton's height so the row looks uniform.
    # Parenting + hide() avoids the ref being painted at (0,0) before deletion.
    ref = QPushButton(label, parent)
    ref.setVisible(False)
    ref.ensurePolished()
    btn.setFixedHeight(ref.sizeHint().height())
    ref.deleteLater()

    def _info():
        return get_info() if get_info else None

    btn.clicked.connect(
        lambda: quick_export(get_fig(), parent=parent or btn.window(), info=_info()))

    menu = QMenu(btn)
    a_cust = menu.addAction("Customize…")
    a_cust.triggered.connect(lambda: export_figure_ui(get_fig(), info=_info()))
    a_copy = menu.addAction("Copy to Clipboard")
    a_copy.triggered.connect(lambda: _copy_fig_to_clipboard(get_fig()))
    btn.setMenu(menu)
    return btn


def export_figure_ui(source_fig=None, info=None):
    """Open the source figure's main plot in an editable export dialog."""
    if source_fig is None:
        source_fig = plt.gcf()

    all_axes = source_fig.get_axes()
    data_axes = [ax for ax in all_axes if _is_data_axes(ax)]

    if not data_axes:
        from utils.app_logger import logger
        logger.warning("No data axes found in the figure to export.")
        return

    data_axes.sort(key=lambda a: (-a.get_position().y0, a.get_position().x0))

    dlg = _ExportDialog(data_axes, info=info)
    dlg.exec()


def _lock_icon(open_state: bool = False, size: int = 14,
               color: str = "#444") -> QIcon:
    """Render a small padlock as a QIcon. ``open_state=True`` draws the shackle tilted."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(color), 1.5))
    # Body: rounded rectangle in the lower half
    body = QRectF(size * 0.25, size * 0.5, size * 0.5, size * 0.40)
    p.drawRoundedRect(body, 1.5, 1.5)
    # Shackle: half-circle above the body
    shackle = QRectF(size * 0.30, size * 0.18, size * 0.40, size * 0.50)
    if open_state:
        # Tilt the shackle by drawing from the right post upward and left.
        p.drawArc(shackle, 30 * 16, 150 * 16)
    else:
        p.drawArc(shackle, 0, 180 * 16)
    p.end()
    return QIcon(pix)


def _swatch_css(hex_color: str) -> str:
    """Stylesheet for a color-swatch QPushButton — rounded square."""
    return (f"QPushButton {{"
            f" background-color: {hex_color};"
            f" border: 1px solid #888;"
            f" border-radius: 6px;"
            f" padding: 0;"
            f" min-width: 0;"
            f" min-height: 0;"
            f" }}")


def _make_color_btn(initial_hex="#000000"):
    """Create a small rounded color swatch button (2:1 rectangle)."""
    btn = QPushButton()
    btn.setFixedSize(32, 16)
    btn.setStyleSheet(_swatch_css(initial_hex))
    return btn


def _pick_color(parent, current_hex, callback):
    """Open a QColorDialog; call *callback(hex_str)* if accepted."""
    color = QColorDialog.getColor(QColor(current_hex), parent, "Pick Color")
    if color.isValid():
        callback(color.name())


def _make_fs_spin(value=12, lo=6, hi=48):
    """Create a font-size QSpinBox."""
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def _make_font_combo():
    """Create a font family QComboBox."""
    combo = QComboBox()
    for fam in _FONT_FAMILIES:
        combo.addItem(fam)
    return combo


# Conversion helpers for figure size units
_INCH_PER_MM = 1.0 / 25.4


class _AspectCanvasHolder(QWidget):
    """Centers an embedded FigureCanvas and keeps it at a fixed W:H aspect.

    The Qt-Agg backend slaves the figure size to the widget size on every
    resize, so the live preview can't carry an arbitrary export size without
    leaving stale pixels. Instead we letterbox the canvas to the chosen aspect:
    figure == widget == buffer always, so the preview is always clean.
    """

    def __init__(self, canvas, aspect, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._aspect = max(aspect, 1e-6)
        canvas.setParent(self)

    def set_aspect(self, aspect):
        self._aspect = max(aspect, 1e-6)
        self._relayout()

    def resizeEvent(self, ev):
        self._relayout()
        super().resizeEvent(ev)

    def _relayout(self):
        W, H = self.width(), self.height()
        if W <= 0 or H <= 0:
            return
        if W / H > self._aspect:
            h = H
            w = int(round(h * self._aspect))
        else:
            w = W
            h = int(round(w / self._aspect))
        self._canvas.setGeometry((W - w) // 2, (H - h) // 2, w, h)


class _ExportDialog(QDialog):
    def __init__(self, data_axes, info=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Figure")
        self.resize(1100, 650)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._data_axes = data_axes  # all available source axes
        self._info = info  # upstream metadata bundle (e.g. {filename, platform})

        # --- Build the matplotlib figure ---
        self.fig = Figure(figsize=(8, 5), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self._copy_axes(data_axes[0])
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Canonical export size in inches — the source of truth for Save/Copy,
        # decoupled from the preview (the backend slaves the on-screen figure
        # size to the widget). The preview only mirrors this size's aspect.
        self._export_in = (self.fig.get_figwidth(), self.fig.get_figheight())
        self._canvas_holder = _AspectCanvasHolder(
            self.canvas, self._export_in[0] / max(self._export_in[1], 1e-6))
        # Keep the axes tightly laid out at whatever size the holder fits to.
        self.canvas.mpl_connect("resize_event", self._on_canvas_resized)

        left = QVBoxLayout()
        left.addWidget(self.toolbar)
        left.addWidget(self._canvas_holder, stretch=1)

        # --- Settings panel ---
        right = self._build_settings_panel()

        main = QHBoxLayout(self)
        main.addLayout(left, stretch=3)
        main.addWidget(right, stretch=1)

        self._sync_from_axes()

        # Disable scroll-wheel value changes on every spinbox / combobox —
        # wheel events over the panel are scroll gestures, not edits.
        from PySide6.QtWidgets import QAbstractSpinBox
        for w in self.findChildren(QAbstractSpinBox):
            w.wheelEvent = lambda ev: ev.ignore()
        for w in self.findChildren(QComboBox):
            w.wheelEvent = lambda ev: ev.ignore()

    # ------------------------------------------------------------------ copy
    def _copy_axes(self, src_ax):
        ax = self.ax
        ax.clear()
        self._lines = []
        for line in src_ax.get_lines():
            (ln,) = ax.plot(
                line.get_xdata(), line.get_ydata(),
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                label=line.get_label(),
                alpha=line.get_alpha(),
            )
            self._lines.append(ln)

        for coll in src_ax.collections:
            try:
                paths = coll.get_paths()
                if paths:
                    from matplotlib.patches import PathPatch
                    for path in paths:
                        fc = coll.get_facecolor()[0] if len(coll.get_facecolor()) else "none"
                        ec = coll.get_edgecolor()[0] if len(coll.get_edgecolor()) else "none"
                        patch = PathPatch(path, facecolor=fc, edgecolor=ec,
                                          alpha=coll.get_alpha())
                        ax.add_patch(patch)
            except Exception:
                pass

        ax.set_xlim(src_ax.get_xlim())
        ax.set_ylim(src_ax.get_ylim())
        ax.set_xscale(src_ax.get_xscale())
        ax.set_yscale(src_ax.get_yscale())
        grid_on = (src_ax.xaxis.get_gridlines()[0].get_visible()
                   if src_ax.xaxis.get_gridlines() else False)
        ax.grid(grid_on)

        ax.set_title(src_ax.get_title(), fontsize=20, fontweight="bold")
        ax.set_xlabel(src_ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(src_ax.get_ylabel(), fontsize=16)
        try:
            tick_fs = src_ax.xaxis.get_ticklabels()[0].get_fontsize()
        except (IndexError, AttributeError):
            tick_fs = 10
        ax.tick_params(labelsize=tick_fs)

        for spine in ax.spines.values():
            spine.set_visible(True)

        legend = src_ax.get_legend()
        if legend is not None:
            handles, labels = ax.get_legend_handles_labels()
            filtered = [(h, l) for h, l in zip(handles, labels)
                        if not l.startswith("_")]
            if filtered:
                ax.legend(*zip(*filtered), loc="best")

        self.fig.tight_layout()

    # --------------------------------------------------------- settings panel
    def _build_settings_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(350)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        # Axes selector (only shown when multiple axes)
        if len(self._data_axes) > 1:
            ax_row = QHBoxLayout()
            ax_row.addWidget(QLabel("Axes:"))
            self._axes_combo = QComboBox()
            for i, dax in enumerate(self._data_axes):
                self._axes_combo.addItem(_axes_label(dax, i))
            self._axes_combo.currentIndexChanged.connect(self._on_axes_changed)
            ax_row.addWidget(self._axes_combo, stretch=1)
            panel_layout.addLayout(ax_row)

        tabs = QToolBox()
        # Fusion style draws QToolBox tabs as flat rectangles; the macOS native
        # style paints a slanted right edge that QSS can't override.
        from PySide6.QtWidgets import QStyleFactory, QAbstractButton
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            tabs.setStyle(fusion)
        tabs.addItem(self._build_tab_figure(), "Figure")
        tabs.addItem(self._build_tab_labels(), "Labels")
        tabs.addItem(self._build_tab_style(), "Style")
        tabs.addItem(self._build_tab_annotate(), "Annotate")
        # QSS padding on QToolBox::tab is unreliable on macOS — pin a height
        # on each internal header button so labels aren't clipped.
        for tab_btn in tabs.findChildren(QAbstractButton):
            tab_btn.setMinimumHeight(32)
        panel_layout.addWidget(tabs, stretch=1)

        # Export controls — two rows, seeded from persisted settings.
        persisted = load_settings()

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._fmt_combo = QComboBox()
        for key, _flt in _FORMAT_FILTERS:
            self._fmt_combo.addItem(key.upper(), key)
        idx = self._fmt_combo.findData((persisted.get("format") or "png").lower())
        if idx >= 0:
            self._fmt_combo.setCurrentIndex(idx)
        self._fmt_combo.setFixedWidth(80)
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        fmt_row.addWidget(self._fmt_combo)
        fmt_row.addStretch()
        panel_layout.addLayout(fmt_row)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("DPI:"))
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setValue(int(persisted.get("dpi", 300)))
        self._dpi_spin.setFixedWidth(70)
        action_row.addWidget(self._dpi_spin)
        self._on_format_changed()  # apply vector-format DPI disable

        action_row.addStretch()
        btn_copy = QPushButton("Copy")
        btn_copy.setToolTip("Copy figure to clipboard")
        btn_copy.setMinimumWidth(70)
        btn_copy.clicked.connect(self._copy_to_clipboard)
        action_row.addWidget(btn_copy)
        btn_save = QPushButton("Save")
        btn_save.setMinimumWidth(70)
        btn_save.clicked.connect(self._save_figure)
        action_row.addWidget(btn_save)
        panel_layout.addLayout(action_row)

        return panel

    def _on_format_changed(self, *_):
        fmt = self._fmt_combo.currentData() or "png"
        if _vector_format(fmt):
            self._dpi_spin.setEnabled(False)
            self._dpi_spin.setToolTip("DPI does not apply to vector formats")
        else:
            self._dpi_spin.setEnabled(True)
            self._dpi_spin.setToolTip("")

    def _copy_to_clipboard(self):
        self._save_with_export_size(
            lambda: _copy_fig_to_clipboard(self.fig, dpi=self._dpi_spin.value()))

    # ---- helper: tick settings for one axis (X or Y) ----
    def _make_axis_ticks_group(self, axis: str, default_size: int):
        """Build a tick-settings QGroupBox for *axis* ('X' or 'Y').

        Returns (group, show_major_cb, show_minor_cb, n_major_spin,
        n_minor_spin, fs_spin). Spinboxes use 0 as 'auto'.
        """
        group = QGroupBox(f"{axis} Ticks")
        form = QFormLayout(group)
        form.setContentsMargins(8, 14, 8, 8)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        show_major = QCheckBox("Major")
        show_major.setChecked(True)
        show_minor = QCheckBox("Minor")
        show_minor.setChecked(False)
        show_row = QHBoxLayout()
        show_row.addWidget(show_major)
        show_row.addWidget(show_minor)
        show_row.addStretch()
        form.addRow("Show:", show_row)

        n_major = QSpinBox()
        n_major.setRange(0, 30)
        n_major.setValue(0)
        n_major.setSpecialValueText("auto")
        n_major.setFixedWidth(70)

        n_minor = QSpinBox()
        n_minor.setRange(0, 10)
        n_minor.setValue(0)
        n_minor.setFixedWidth(70)
        n_minor.setToolTip("Minor ticks between each pair of major ticks (0 = off)")

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N major:"))
        n_row.addWidget(n_major)
        n_row.addSpacing(8)
        n_row.addWidget(QLabel("N minor:"))
        n_row.addWidget(n_minor)
        n_row.addStretch()
        form.addRow(n_row)

        fs_spin = _make_fs_spin(default_size, lo=6, hi=36)
        fs_spin.setFixedWidth(60)
        fs_row = QHBoxLayout()
        fs_row.addWidget(fs_spin)
        fs_row.addStretch()
        form.addRow("Size:", fs_row)

        for w in (show_major, show_minor, n_major, n_minor, fs_spin):
            sig = (w.stateChanged if isinstance(w, QCheckBox) else w.valueChanged)
            sig.connect(self._apply_all_labels)

        return group, show_major, show_minor, n_major, n_minor, fs_spin

    # ---- helper: one label group (text, modifiers, size + color) ----
    def _make_label_group(self, heading, text_edit, fs_spin,
                          color_btn, tex_cb, bold_cb=None, italic_cb=None):
        """Build a label group as a QGroupBox: text, B/I/TeX, size + color.

        Font family is set once at the top of the Labels tab and applies to
        all groups, so this helper no longer takes a per-group font combo.
        """
        group = QGroupBox(heading)
        form = QFormLayout(group)
        form.setContentsMargins(8, 14, 8, 8)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Text:", text_edit)

        # B/I/TeX modifiers on one row.
        mods_row = QHBoxLayout()
        if bold_cb is not None:
            bold_cb.setFixedWidth(30)
            mods_row.addWidget(bold_cb)
        if italic_cb is not None:
            italic_cb.setFixedWidth(26)
            mods_row.addWidget(italic_cb)
        tex_cb.setFixedWidth(44)
        mods_row.addWidget(tex_cb)
        mods_row.addStretch()
        form.addRow("Style:", mods_row)

        # Size as the form label; spinbox + Color + swatch as the value.
        fs_spin.setFixedWidth(60)
        sc_row = QHBoxLayout()
        sc_row.addWidget(fs_spin)
        sc_row.addSpacing(12)
        sc_row.addWidget(QLabel("Color:"))
        sc_row.addWidget(color_btn)
        sc_row.addStretch()
        form.addRow("Size:", sc_row)

        return group

    def _build_tab_labels(self):
        """Labels tab — Title, X Label, Y Label, Tick Labels."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Shared font family for Title / X Label / Y Label / Tick Labels.
        self._labels_font = _make_font_combo()
        self._labels_font.setMaximumWidth(160)
        self._labels_font.currentIndexChanged.connect(self._apply_all_labels)
        shared_font_row = QHBoxLayout()
        shared_font_row.addWidget(QLabel("Font:"))
        shared_font_row.addWidget(self._labels_font)
        shared_font_row.addStretch()
        layout.addLayout(shared_font_row)
        layout.addSpacing(4)

        # Title
        self._title_edit = QLineEdit()
        self._title_fs = _make_fs_spin(20)
        self._title_color_btn = _make_color_btn("#000000")
        self._title_color_hex = "#000000"
        self._title_color_btn.clicked.connect(lambda: _pick_color(
            self, self._title_color_hex, self._set_title_color))
        self._title_tex = QCheckBox("TeX")
        self._title_bold = QCheckBox("B")
        self._title_bold.setChecked(True)
        self._title_bold.stateChanged.connect(self._apply_all_labels)
        self._title_italic = QCheckBox("I")
        self._title_italic.stateChanged.connect(self._apply_all_labels)
        self._title_edit.editingFinished.connect(self._apply_all_labels)
        self._title_fs.valueChanged.connect(self._apply_all_labels)
        self._title_tex.stateChanged.connect(self._apply_all_labels)
        layout.addWidget(self._make_label_group(
            "Title", self._title_edit, self._title_fs,
            self._title_color_btn, self._title_tex,
            self._title_bold, self._title_italic))

        layout.addSpacing(6)

        # X Label
        self._xlabel_edit = QLineEdit()
        self._xlabel_fs = _make_fs_spin(16)
        self._xlabel_color_btn = _make_color_btn("#000000")
        self._xlabel_color_hex = "#000000"
        self._xlabel_color_btn.clicked.connect(lambda: _pick_color(
            self, self._xlabel_color_hex, self._set_xlabel_color))
        self._xlabel_tex = QCheckBox("TeX")
        self._xlabel_bold = QCheckBox("B")
        self._xlabel_bold.stateChanged.connect(self._apply_all_labels)
        self._xlabel_italic = QCheckBox("I")
        self._xlabel_italic.stateChanged.connect(self._apply_all_labels)
        self._xlabel_edit.editingFinished.connect(self._apply_all_labels)
        self._xlabel_fs.valueChanged.connect(self._apply_all_labels)
        self._xlabel_tex.stateChanged.connect(self._apply_all_labels)
        layout.addWidget(self._make_label_group(
            "X Label", self._xlabel_edit, self._xlabel_fs,
            self._xlabel_color_btn, self._xlabel_tex,
            self._xlabel_bold, self._xlabel_italic))

        layout.addSpacing(6)

        # Y Label
        self._ylabel_edit = QLineEdit()
        self._ylabel_fs = _make_fs_spin(16)
        self._ylabel_color_btn = _make_color_btn("#000000")
        self._ylabel_color_hex = "#000000"
        self._ylabel_color_btn.clicked.connect(lambda: _pick_color(
            self, self._ylabel_color_hex, self._set_ylabel_color))
        self._ylabel_tex = QCheckBox("TeX")
        self._ylabel_bold = QCheckBox("B")
        self._ylabel_bold.stateChanged.connect(self._apply_all_labels)
        self._ylabel_italic = QCheckBox("I")
        self._ylabel_italic.stateChanged.connect(self._apply_all_labels)
        self._ylabel_edit.editingFinished.connect(self._apply_all_labels)
        self._ylabel_fs.valueChanged.connect(self._apply_all_labels)
        self._ylabel_tex.stateChanged.connect(self._apply_all_labels)
        layout.addWidget(self._make_label_group(
            "Y Label", self._ylabel_edit, self._ylabel_fs,
            self._ylabel_color_btn, self._ylabel_tex,
            self._ylabel_bold, self._ylabel_italic))

        layout.addSpacing(6)

        # Shared tick color (drives both axes); font family follows the top-of-tab combo.
        self._tick_color_btn = _make_color_btn("#000000")
        self._tick_color_hex = "#000000"
        self._tick_color_btn.clicked.connect(lambda: _pick_color(
            self, self._tick_color_hex, self._set_tick_color))
        tick_color_row = QHBoxLayout()
        tick_color_row.addWidget(QLabel("Tick Color:"))
        tick_color_row.addWidget(self._tick_color_btn)
        tick_color_row.addStretch()
        layout.addLayout(tick_color_row)

        # Per-axis tick groups: Major/Minor toggles, counts, size.
        (xg, self._xtick_major, self._xtick_minor,
         self._xtick_nmajor, self._xtick_nminor,
         self._xtick_fs) = self._make_axis_ticks_group("X", default_size=10)
        layout.addWidget(xg)
        (yg, self._ytick_major, self._ytick_minor,
         self._ytick_nmajor, self._ytick_nminor,
         self._ytick_fs) = self._make_axis_ticks_group("Y", default_size=10)
        layout.addWidget(yg)

        layout.addStretch()
        return page

    def _build_tab_style(self):
        """Data Series & Plot Options tab."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Data Series group ---
        series_group = QGroupBox("Data Series")
        series_layout = QVBoxLayout(series_group)

        self._series_combo = QComboBox()
        self._series_combo.currentIndexChanged.connect(self._on_series_changed)
        series_layout.addWidget(self._series_combo)

        form1 = QFormLayout()

        self._line_label = QLineEdit()
        self._line_label.setPlaceholderText("(legend label)")
        self._line_label.editingFinished.connect(self._apply_line_label)

        self._line_color_btn = _make_color_btn("#1f77b4")
        self._line_color_btn.clicked.connect(self._pick_line_color)

        self._line_width = QDoubleSpinBox()
        self._line_width.setRange(0.1, 10.0)
        self._line_width.setSingleStep(0.25)
        self._line_width.setValue(0.5)
        self._line_width.valueChanged.connect(self._apply_line_style)

        self._line_type = QComboBox()
        for key, label in _LINESTYLES:
            self._line_type.addItem(label, key)
        self._line_type.currentIndexChanged.connect(self._apply_line_style)

        form1.addRow("Label:", self._line_label)
        color_row = QHBoxLayout()
        color_row.addWidget(self._line_color_btn)
        color_row.addStretch()
        form1.addRow("Color:", color_row)
        form1.addRow("Width:", self._line_width)
        form1.addRow("Type:", self._line_type)
        series_layout.addLayout(form1)
        layout.addWidget(series_group)

        layout.addSpacing(12)

        # --- Plot Options group ---
        options_group = QGroupBox("Plot Options")
        options_layout = QVBoxLayout(options_group)

        self._grid_cb = QCheckBox("Show Grid")
        self._grid_cb.stateChanged.connect(self._apply_display)
        options_layout.addWidget(self._grid_cb)
        self._legend_cb = QCheckBox("Show Legend")
        self._legend_cb.stateChanged.connect(self._apply_display)
        options_layout.addWidget(self._legend_cb)

        form2 = QFormLayout()
        self._legend_pos = QComboBox()
        self._legend_pos.addItems([
            "best", "upper right", "upper left", "lower left",
            "lower right", "center left", "center right",
            "lower center", "upper center", "center",
        ])
        self._legend_pos.currentTextChanged.connect(self._apply_display)
        form2.addRow("Legend Pos:", self._legend_pos)

        self._legend_font = _make_font_combo()
        self._legend_font.currentIndexChanged.connect(self._apply_display)
        form2.addRow("Legend Font:", self._legend_font)

        self._legend_fs = _make_fs_spin(12, lo=6, hi=36)
        self._legend_fs.setFixedWidth(50)
        self._legend_fs.valueChanged.connect(self._apply_display)
        form2.addRow("Legend Size:", self._legend_fs)

        options_layout.addLayout(form2)
        layout.addWidget(options_group)

        layout.addStretch()
        return page

    def _build_tab_annotate(self):
        """Annotation tab with style controls."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Content / position group
        content_group = QGroupBox("Content")
        form = QFormLayout(content_group)
        form.setContentsMargins(8, 14, 8, 8)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._annot_text = QLineEdit()
        self._annot_text.setPlaceholderText("Text to add...")
        self._annot_x = QDoubleSpinBox()
        self._annot_x.setDecimals(4)
        self._annot_x.setRange(-1e12, 1e12)
        self._annot_y = QDoubleSpinBox()
        self._annot_y.setDecimals(4)
        self._annot_y.setRange(-1e12, 1e12)
        form.addRow("Text:", self._annot_text)
        form.addRow("X:", self._annot_x)
        form.addRow("Y:", self._annot_y)
        layout.addWidget(content_group)

        # Style group
        style_group = QGroupBox("Style")
        style_form = QFormLayout(style_group)
        style_form.setContentsMargins(8, 14, 8, 8)
        style_form.setHorizontalSpacing(6)
        style_form.setVerticalSpacing(4)
        style_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._annot_font = _make_font_combo()
        style_form.addRow("Font:", self._annot_font)

        self._annot_fs = _make_fs_spin(12, lo=6, hi=36)
        self._annot_color_btn = _make_color_btn("#000000")
        self._annot_color_hex = "#000000"
        self._annot_color_btn.clicked.connect(lambda: _pick_color(
            self, self._annot_color_hex, self._set_annot_color))
        self._annot_fs.setFixedWidth(50)
        annot_sc = QHBoxLayout()
        annot_sc.addWidget(QLabel("Size:"))
        annot_sc.addWidget(self._annot_fs)
        annot_sc.addWidget(QLabel("pt"))
        annot_sc.addStretch()
        annot_sc.addWidget(QLabel("Color:"))
        annot_sc.addWidget(self._annot_color_btn)
        style_form.addRow(annot_sc)

        self._annot_tex = QCheckBox("TeX")
        style_form.addRow("", self._annot_tex)

        self._annot_bg_btn = _make_color_btn("#faffa0")
        self._annot_bg_hex = "#faffa0"
        self._annot_bg_btn.clicked.connect(lambda: _pick_color(
            self, self._annot_bg_hex, self._set_annot_bg))
        self._annot_border_btn = _make_color_btn("#808080")
        self._annot_border_hex = "#808080"
        self._annot_border_btn.clicked.connect(lambda: _pick_color(
            self, self._annot_border_hex, self._set_annot_border))
        bg_border = QHBoxLayout()
        bg_border.addWidget(self._annot_bg_btn)
        bg_border.addWidget(QLabel("bg"))
        bg_border.addStretch()
        bg_border.addWidget(self._annot_border_btn)
        bg_border.addWidget(QLabel("border"))
        style_form.addRow("Box:", bg_border)

        layout.addWidget(style_group)

        layout.addSpacing(8)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Annotation")
        btn_add.clicked.connect(self._add_annotation)
        btn_row.addWidget(btn_add, stretch=1)
        btn_clear = QPushButton("Clear Annotations")
        btn_clear.clicked.connect(self._clear_annotations)
        btn_row.addWidget(btn_clear, stretch=1)
        layout.addLayout(btn_row)

        layout.addStretch()
        return page

    def _build_tab_figure(self):
        """Figure size tab with unit selection."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl = QLabel("Figure Size")
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)

        form = QFormLayout()

        self._fig_unit = QComboBox()
        self._fig_unit.addItems(["inches", "mm", "pixels"])
        self._fig_unit.currentIndexChanged.connect(self._on_unit_changed)
        form.addRow("Unit:", self._fig_unit)

        layout.addLayout(form)

        # Width/Height grid with a connecting bracket + lock button on the right.
        self._fig_width = QDoubleSpinBox()
        self._fig_width.setDecimals(1)
        self._fig_width.setSingleStep(0.5)
        self._fig_height = QDoubleSpinBox()
        self._fig_height.setDecimals(1)
        self._fig_height.setSingleStep(0.5)

        self._fig_lock_ratio = QPushButton()
        self._fig_lock_open = _lock_icon(open_state=True)
        self._fig_lock_closed = _lock_icon(open_state=False)
        self._fig_lock_ratio.setIcon(self._fig_lock_open)
        self._fig_lock_ratio.setIconSize(QSize(14, 14))
        self._fig_lock_ratio.setCheckable(True)
        self._fig_lock_ratio.setFixedSize(28, 24)
        self._fig_lock_ratio.setToolTip("Lock aspect ratio")

        # `]`-shaped bracket: borders top/right/bottom drawn, left open.
        bracket = QFrame()
        bracket.setFixedWidth(6)
        bracket.setStyleSheet(
            "border-top: 1px solid #888;"
            " border-right: 1px solid #888;"
            " border-bottom: 1px solid #888;"
            " background: transparent;")

        wh_grid = QGridLayout()
        wh_grid.setHorizontalSpacing(6)
        wh_grid.setVerticalSpacing(form.verticalSpacing() if form.verticalSpacing() >= 0 else 6)
        wh_grid.addWidget(QLabel("Width:"),  0, 0)
        wh_grid.addWidget(self._fig_width,   0, 1)
        wh_grid.addWidget(QLabel("Height:"), 1, 0)
        wh_grid.addWidget(self._fig_height,  1, 1)
        wh_grid.addWidget(bracket,           0, 2, 2, 1)
        wh_grid.addWidget(self._fig_lock_ratio, 0, 3, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        wh_grid.setColumnStretch(1, 1)
        layout.addLayout(wh_grid)

        # Initialize ranges and values for inches
        self._fig_width.setRange(2.0, 30.0)
        self._fig_width.setValue(self.fig.get_figwidth())
        self._fig_height.setRange(2.0, 30.0)
        self._fig_height.setValue(self.fig.get_figheight())
        self._fig_locked_ratio = (self._fig_width.value() /
                                  max(self._fig_height.value(), 1e-6))

        self._fig_width.valueChanged.connect(self._on_fig_width_changed)
        self._fig_height.valueChanged.connect(self._on_fig_height_changed)
        self._fig_lock_ratio.toggled.connect(self._on_lock_ratio_toggled)

        layout.addSpacing(8)
        btn_apply = QPushButton("Apply Size")
        btn_apply.clicked.connect(self._apply_fig_size)
        layout.addWidget(btn_apply)

        layout.addStretch()
        return page

    def _on_lock_ratio_toggled(self, checked):
        if checked:
            h = max(self._fig_height.value(), 1e-6)
            self._fig_locked_ratio = self._fig_width.value() / h
        self._fig_lock_ratio.setIcon(
            self._fig_lock_closed if checked else self._fig_lock_open)

    def _on_fig_width_changed(self, w):
        if self._fig_lock_ratio.isChecked():
            new_h = w / max(self._fig_locked_ratio, 1e-6)
            self._fig_height.blockSignals(True)
            self._fig_height.setValue(new_h)
            self._fig_height.blockSignals(False)
        self._update_export_in()

    def _on_fig_height_changed(self, h):
        if self._fig_lock_ratio.isChecked():
            new_w = h * self._fig_locked_ratio
            self._fig_width.blockSignals(True)
            self._fig_width.setValue(new_w)
            self._fig_width.blockSignals(False)
        self._update_export_in()

    def _spin_to_inches(self, unit, w, h):
        """Convert width/height in *unit* to inches using the figure DPI."""
        if unit == "mm":
            return w * _INCH_PER_MM, h * _INCH_PER_MM
        if unit == "pixels":
            dpi = self.fig.get_dpi()
            return w / dpi, h / dpi
        return w, h

    def _update_export_in(self):
        """Refresh the canonical export size from the current spinbox values."""
        self._export_in = self._spin_to_inches(
            self._fig_unit.currentText(),
            self._fig_width.value(), self._fig_height.value())

    # --------------------------------------------------------- axes switching
    def _on_axes_changed(self, idx):
        """Switch to a different source axes."""
        if 0 <= idx < len(self._data_axes):
            self._copy_axes(self._data_axes[idx])
            self.canvas.draw()
            self._sync_from_axes()

    # -------------------------------------------------------- sync from axes
    def _sync_from_axes(self):
        ax = self.ax

        # Title
        self._title_edit.setText(ax.get_title())
        self._title_fs.setValue(int(ax.title.get_fontsize()))
        tc = matplotlib.colors.to_hex(ax.title.get_color())
        self._title_color_hex = tc
        self._title_color_btn.setStyleSheet(
            _swatch_css(tc))
        self._title_bold.setChecked(
            str(ax.title.get_fontweight()) in ('bold', 'heavy'))
        self._title_italic.setChecked(
            ax.title.get_fontstyle() == 'italic')

        # X label
        self._xlabel_edit.setText(ax.get_xlabel())
        self._xlabel_fs.setValue(int(ax.xaxis.label.get_fontsize()))
        xc = matplotlib.colors.to_hex(ax.xaxis.label.get_color())
        self._xlabel_color_hex = xc
        self._xlabel_color_btn.setStyleSheet(
            _swatch_css(xc))
        self._xlabel_bold.setChecked(
            str(ax.xaxis.label.get_fontweight()) in ('bold', 'heavy'))
        self._xlabel_italic.setChecked(
            ax.xaxis.label.get_fontstyle() == 'italic')

        # Y label
        self._ylabel_edit.setText(ax.get_ylabel())
        self._ylabel_fs.setValue(int(ax.yaxis.label.get_fontsize()))
        yc = matplotlib.colors.to_hex(ax.yaxis.label.get_color())
        self._ylabel_color_hex = yc
        self._ylabel_color_btn.setStyleSheet(
            _swatch_css(yc))
        self._ylabel_bold.setChecked(
            str(ax.yaxis.label.get_fontweight()) in ('bold', 'heavy'))
        self._ylabel_italic.setChecked(
            ax.yaxis.label.get_fontstyle() == 'italic')

        # Tick labels — per-axis font size, shared color.
        def _read_tick_fs(axis):
            try:
                return int(axis.get_ticklabels()[0].get_fontsize())
            except (IndexError, AttributeError):
                return 10
        self._xtick_fs.blockSignals(True)
        self._ytick_fs.blockSignals(True)
        self._xtick_fs.setValue(_read_tick_fs(ax.xaxis))
        self._ytick_fs.setValue(_read_tick_fs(ax.yaxis))
        self._xtick_fs.blockSignals(False)
        self._ytick_fs.blockSignals(False)

        try:
            tkc = matplotlib.colors.to_hex(
                ax.xaxis.get_ticklabels()[0].get_color())
        except (IndexError, AttributeError):
            tkc = "#000000"
        self._tick_color_hex = tkc
        self._tick_color_btn.setStyleSheet(_swatch_css(tkc))

        # Populate series combo and sync first series
        self._populate_series_combo()

        # Display
        grid_on = (ax.xaxis.get_gridlines()[0].get_visible()
                   if ax.xaxis.get_gridlines() else False)
        self._grid_cb.setChecked(grid_on)
        self._legend_cb.setChecked(ax.get_legend() is not None)

        # Legend font size
        leg = ax.get_legend()
        if leg:
            texts = leg.get_texts()
            if texts:
                self._legend_fs.setValue(int(texts[0].get_fontsize()))

        # Annotation defaults
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self._annot_x.setValue((xlim[0] + xlim[1]) / 2)
        self._annot_y.setValue((ylim[0] + ylim[1]) / 2)

    # ---------------------------------------------------------- TeX helpers
    def _tex_wrap(self, text, use_tex):
        """Wrap text for TeX rendering if enabled."""
        if not use_tex or not text:
            return text
        # If already wrapped in $, leave as-is
        if text.startswith("$") and text.endswith("$"):
            return text
        return f"${text}$"

    # ------------------------------------------------------------- callbacks
    def _set_title_color(self, hex_c):
        self._title_color_hex = hex_c
        self._title_color_btn.setStyleSheet(
            _swatch_css(hex_c))
        self._apply_all_labels()

    def _set_xlabel_color(self, hex_c):
        self._xlabel_color_hex = hex_c
        self._xlabel_color_btn.setStyleSheet(
            _swatch_css(hex_c))
        self._apply_all_labels()

    def _set_ylabel_color(self, hex_c):
        self._ylabel_color_hex = hex_c
        self._ylabel_color_btn.setStyleSheet(
            _swatch_css(hex_c))
        self._apply_all_labels()

    def _set_tick_color(self, hex_c):
        self._tick_color_hex = hex_c
        self._tick_color_btn.setStyleSheet(
            _swatch_css(hex_c))
        self._apply_all_labels()

    def _set_annot_color(self, hex_c):
        self._annot_color_hex = hex_c
        self._annot_color_btn.setStyleSheet(
            _swatch_css(hex_c))

    def _set_annot_bg(self, hex_c):
        self._annot_bg_hex = hex_c
        self._annot_bg_btn.setStyleSheet(
            _swatch_css(hex_c))

    def _set_annot_border(self, hex_c):
        self._annot_border_hex = hex_c
        self._annot_border_btn.setStyleSheet(
            _swatch_css(hex_c))

    def _apply_all_labels(self):
        ax = self.ax
        # Enable/disable TeX rendering
        matplotlib.rcParams['text.usetex'] = False  # Use mathtext, not full LaTeX

        fam = self._labels_font.currentText()

        title_text = self._tex_wrap(
            self._title_edit.text(), self._title_tex.isChecked())
        ax.set_title(title_text,
                     fontsize=self._title_fs.value(),
                     fontweight="bold" if self._title_bold.isChecked() else "normal",
                     fontstyle="italic" if self._title_italic.isChecked() else "normal",
                     color=self._title_color_hex,
                     fontfamily=fam)

        xlabel_text = self._tex_wrap(
            self._xlabel_edit.text(), self._xlabel_tex.isChecked())
        ax.set_xlabel(xlabel_text,
                      fontsize=self._xlabel_fs.value(),
                      fontweight="bold" if self._xlabel_bold.isChecked() else "normal",
                      fontstyle="italic" if self._xlabel_italic.isChecked() else "normal",
                      color=self._xlabel_color_hex,
                      fontfamily=fam)

        ylabel_text = self._tex_wrap(
            self._ylabel_edit.text(), self._ylabel_tex.isChecked())
        ax.set_ylabel(ylabel_text,
                      fontsize=self._ylabel_fs.value(),
                      fontweight="bold" if self._ylabel_bold.isChecked() else "normal",
                      fontstyle="italic" if self._ylabel_italic.isChecked() else "normal",
                      color=self._ylabel_color_hex,
                      fontfamily=fam)

        # Per-axis tick controls: show major/minor, locator counts, label size.
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator, NullLocator

        def _apply_axis_ticks(mpl_axis, ax_str, show_major_cb, show_minor_cb,
                              n_major, n_minor, fs_spin):
            show_maj = show_major_cb.isChecked()
            show_min = show_minor_cb.isChecked()
            ax.tick_params(axis=ax_str, which='major',
                           bottom=show_maj and ax_str == 'x',
                           top=False,
                           left=show_maj and ax_str == 'y',
                           right=False,
                           labelbottom=show_maj and ax_str == 'x',
                           labelleft=show_maj and ax_str == 'y',
                           labelsize=fs_spin.value(),
                           labelcolor=self._tick_color_hex)
            ax.tick_params(axis=ax_str, which='minor',
                           bottom=show_min and ax_str == 'x',
                           top=False,
                           left=show_min and ax_str == 'y',
                           right=False,
                           labelbottom=False, labelleft=False)
            # Locators: 0 = auto / off.
            n_maj_val = n_major.value()
            if n_maj_val > 0:
                mpl_axis.set_major_locator(MaxNLocator(nbins=n_maj_val))
            n_min_val = n_minor.value()
            if n_min_val > 0:
                mpl_axis.set_minor_locator(AutoMinorLocator(n_min_val + 1))
            else:
                mpl_axis.set_minor_locator(NullLocator())

        _apply_axis_ticks(ax.xaxis, 'x',
                          self._xtick_major, self._xtick_minor,
                          self._xtick_nmajor, self._xtick_nminor,
                          self._xtick_fs)
        _apply_axis_ticks(ax.yaxis, 'y',
                          self._ytick_major, self._ytick_minor,
                          self._ytick_nmajor, self._ytick_nminor,
                          self._ytick_fs)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(fam)

        leg = ax.get_legend()
        if leg:
            leg_fs = self._legend_fs.value()
            leg_fam = self._legend_font.currentText()
            for t in leg.get_texts():
                t.set_fontsize(leg_fs)
                t.set_fontfamily(leg_fam)

        self.fig.tight_layout()
        self.canvas.draw()

    def _populate_series_combo(self):
        """Fill the series combo with labels from all copied lines."""
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        for i, ln in enumerate(self._lines):
            label = ln.get_label()
            if label.startswith("_"):
                label = f"Series {i + 1}"
            self._series_combo.addItem(label)
        self._series_combo.blockSignals(False)
        if self._lines:
            self._series_combo.setCurrentIndex(0)
            self._on_series_changed(0)

    def _on_series_changed(self, idx):
        """Sync style controls from the selected series."""
        if idx < 0 or idx >= len(self._lines):
            return
        ln = self._lines[idx]
        self._line_width.blockSignals(True)
        self._line_type.blockSignals(True)
        self._line_label.blockSignals(True)

        cur_label = ln.get_label() or ""
        self._line_label.setText("" if cur_label.startswith("_") else cur_label)
        self._line_label.blockSignals(False)

        c = matplotlib.colors.to_hex(ln.get_color())
        self._line_color_btn.setStyleSheet(
            _swatch_css(c))
        self._line_width.setValue(ln.get_linewidth())
        ls = ln.get_linestyle()
        matched = False
        for i, (key, _) in enumerate(_LINESTYLES):
            if key == ls or (key.lower() == 'none' and ls.lower() == 'none'):
                self._line_type.setCurrentIndex(i)
                matched = True
                break
        if not matched:
            # Default to None for marker-only lines
            for i, (key, _) in enumerate(_LINESTYLES):
                if key == "None":
                    self._line_type.setCurrentIndex(i)
                    break

        self._line_width.blockSignals(False)
        self._line_type.blockSignals(False)

    def _pick_line_color(self):
        idx = self._series_combo.currentIndex()
        if idx < 0 or idx >= len(self._lines):
            return
        ln = self._lines[idx]
        cur = matplotlib.colors.to_hex(ln.get_color())
        color = QColorDialog.getColor(QColor(cur), self, "Line Color")
        if color.isValid():
            hex_c = color.name()
            ln.set_color(hex_c)
            self._line_color_btn.setStyleSheet(
                _swatch_css(hex_c))
            self.canvas.draw()

    def _apply_line_style(self):
        idx = self._series_combo.currentIndex()
        if idx < 0 or idx >= len(self._lines):
            return
        ln = self._lines[idx]
        ln.set_linewidth(self._line_width.value())
        ls = self._line_type.currentData()
        if ls is not None:
            ln.set_linestyle(ls)
        self.canvas.draw()

    def _apply_line_label(self):
        idx = self._series_combo.currentIndex()
        if idx < 0 or idx >= len(self._lines):
            return
        new = self._line_label.text().strip()
        ln = self._lines[idx]
        # Empty label → hide from legend by prefixing with underscore.
        ln.set_label(new if new else f"_Series {idx + 1}")
        # Refresh the series-combo entry so the dropdown reflects the new name.
        self._series_combo.blockSignals(True)
        self._series_combo.setItemText(idx, new if new else f"Series {idx + 1}")
        self._series_combo.blockSignals(False)
        # If the legend is on, regenerate it from the updated handles.
        if self._legend_cb.isChecked():
            self._apply_display()
        self.canvas.draw()

    def _apply_display(self):
        self.ax.grid(self._grid_cb.isChecked())
        if self._legend_cb.isChecked():
            handles, labels = self.ax.get_legend_handles_labels()
            filtered = [(h, l) for h, l in zip(handles, labels)
                        if not l.startswith("_")]
            leg_fs = self._legend_fs.value()
            leg_fam = self._legend_font.currentText()
            loc = self._legend_pos.currentText()
            if filtered:
                leg = self.ax.legend(*zip(*filtered), loc=loc,
                                     prop={'family': leg_fam, 'size': leg_fs})
            else:
                leg = self.ax.legend([], [], loc=loc,
                                     prop={'family': leg_fam, 'size': leg_fs})
        else:
            leg = self.ax.get_legend()
            if leg:
                leg.remove()
        self.canvas.draw()

    def _add_annotation(self):
        text = self._annot_text.text().strip()
        if not text:
            return
        display_text = self._tex_wrap(text, self._annot_tex.isChecked())
        self.ax.annotate(
            display_text,
            xy=(self._annot_x.value(), self._annot_y.value()),
            fontsize=self._annot_fs.value(),
            fontfamily=self._annot_font.currentText(),
            color=self._annot_color_hex,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=self._annot_bg_hex,
                      ec=self._annot_border_hex,
                      alpha=0.9),
        )
        self.canvas.draw()

    def _clear_annotations(self):
        for child in list(self.ax.texts):
            child.remove()
        self.canvas.draw()

    def _on_unit_changed(self):
        """Update width/height spinboxes when unit changes."""
        # Convert the canonical export size, not the widget-slaved figure size.
        w_in, h_in = self._export_in
        dpi = self.fig.get_dpi()
        unit = self._fig_unit.currentText()

        # Block signals to avoid triggering while we reconfigure
        self._fig_width.blockSignals(True)
        self._fig_height.blockSignals(True)

        if unit == "inches":
            self._fig_width.setDecimals(1)
            self._fig_width.setRange(2.0, 30.0)
            self._fig_width.setSingleStep(0.5)
            self._fig_width.setValue(w_in)
            self._fig_height.setDecimals(1)
            self._fig_height.setRange(2.0, 30.0)
            self._fig_height.setSingleStep(0.5)
            self._fig_height.setValue(h_in)
        elif unit == "mm":
            self._fig_width.setDecimals(0)
            self._fig_width.setRange(50, 800)
            self._fig_width.setSingleStep(10)
            self._fig_width.setValue(round(w_in / _INCH_PER_MM))
            self._fig_height.setDecimals(0)
            self._fig_height.setRange(50, 800)
            self._fig_height.setSingleStep(10)
            self._fig_height.setValue(round(h_in / _INCH_PER_MM))
        else:  # pixels
            self._fig_width.setDecimals(0)
            self._fig_width.setRange(200, 6000)
            self._fig_width.setSingleStep(50)
            self._fig_width.setValue(round(w_in * dpi))
            self._fig_height.setDecimals(0)
            self._fig_height.setRange(200, 6000)
            self._fig_height.setSingleStep(50)
            self._fig_height.setValue(round(h_in * dpi))

        self._fig_width.blockSignals(False)
        self._fig_height.blockSignals(False)

    def _apply_fig_size(self):
        # The preview can't carry an arbitrary inch size (the backend resizes
        # the figure to the widget), so mirror the requested aspect instead and
        # letterbox the canvas. The exact size is applied at Save/Copy time.
        self._update_export_in()
        w_in, h_in = self._export_in
        self._canvas_holder.set_aspect(w_in / max(h_in, 1e-6))

    def _on_canvas_resized(self, _event):
        """Re-tighten the layout whenever the holder refits the canvas."""
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def _save_with_export_size(self, fn):
        """Run *fn()* with the figure temporarily at the requested export size.

        Restores the preview size afterwards so the on-screen canvas stays clean.
        """
        self._update_export_in()
        w_in, h_in = self._export_in
        old = tuple(self.fig.get_size_inches())
        try:
            self.fig.set_size_inches(w_in, h_in)
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            return fn()
        finally:
            self.fig.set_size_inches(old)
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            self.canvas.draw_idle()

    def _save_figure(self):
        persisted = load_settings()
        folder = persisted.get("folder") or os.path.join(project_root(), "Output")
        os.makedirs(folder, exist_ok=True)
        fmt = self._fmt_combo.currentData() or "png"
        base = _default_filename(self.fig, self._info)
        suggested = os.path.join(folder, f"{base}.{fmt}")

        path, _flt = QFileDialog.getSaveFileName(
            self, "Save Figure", suggested, _build_filter(fmt))
        if not path:
            return

        dpi = self._dpi_spin.value()
        ext = self._save_with_export_size(
            lambda: _save_to_path(self.fig, path, dpi)) or fmt
        save_settings({"folder": os.path.dirname(path), "format": ext, "dpi": dpi})
