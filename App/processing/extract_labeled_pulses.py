"""Grid viewer for Extract Labeled Pulses.

Shows the extracted label regions in a paginated 3x3 grid, one pulse per
cell with every zone overlaid and the labeled span shaded. A File selector
restricts the grid to one loaded file (regions never plot across file
seams); a Class selector restricts to one label class.
"""

import warnings
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.dialog_style import apply_dialog_style, style_mpl_figure
from utils.zones import zone_columns


# Standard Data Labeling / Detection Review class ids → name + color.
_CLASS_INFO = {
    1: ('single', '#009900'),
    2: ('coincident', '#dd7700'),
    3: ('noise', '#cc0000'),
    4: ('uncertain', '#8e44ad'),
}


class _ExtractedPulsesDialog(QDialog):
    def __init__(self, zones, regions, boundaries, file_names,
                 sample_rate=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extract Labeled Pulses")
        self.resize(1500, 900)
        apply_dialog_style(self)

        self._zones = zones
        self._n_len = len(zones[0]) if zones else 0
        self._regions = regions              # [(start, end, class_id)]
        self._boundaries = np.asarray(boundaries, dtype=np.int64)
        self._file_names = file_names
        self._fs = sample_rate if sample_rate and sample_rate > 1 else 1
        self._use_time = self._fs > 1
        self._filtered = regions
        self._page = 0
        self._page_size = 9
        self._total_pages = 1

        self._build_ui()
        self._apply_filter()

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("File:"))
        self._file_combo = QComboBox()
        self._file_combo.addItem(f"All files ({len(self._file_names)})")
        self._file_combo.addItems(list(self._file_names))
        self._file_combo.setMinimumWidth(260)
        self._file_combo.currentIndexChanged.connect(self._apply_filter)
        top.addWidget(self._file_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.addItem("All classes")
        self._class_combo.addItems([n for _, (n, _) in sorted(_CLASS_INFO.items())])
        self._class_combo.setFixedWidth(120)
        self._class_combo.currentIndexChanged.connect(self._apply_filter)
        top.addWidget(self._class_combo)
        top.addStretch()
        self._lbl_head = QLabel("")
        self._lbl_head.setTextFormat(Qt.TextFormat.RichText)
        top.addWidget(self._lbl_head)
        main.addLayout(top)

        self.fig = Figure(figsize=(14, 9), tight_layout=False)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        main.addWidget(self.canvas, 1)

        nav = QHBoxLayout()
        nav.setSpacing(4)
        self._lbl_page = QLabel("")
        self._lbl_page.setStyleSheet("font-weight: bold;")
        nav.addWidget(self._lbl_page)
        nav.addStretch()
        for text, slot in (("<<", self._first), ("<", self._prev),
                           (">", self._next), (">>", self._last)):
            b = QPushButton(text)
            b.setFixedWidth(52)
            b.clicked.connect(slot)
            nav.addWidget(b)
        nav.addSpacing(16)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        nav.addWidget(btn_close)
        main.addLayout(nav)

    def _file_span(self, start):
        """[lo, hi) sample range of the file containing *start*."""
        k = int(np.searchsorted(self._boundaries, start, side="right")) - 1
        k = max(0, min(k, len(self._boundaries) - 2))
        return int(self._boundaries[k]), int(self._boundaries[k + 1])

    def _apply_filter(self, *_):
        fi = self._file_combo.currentIndex() - 1        # -1 = all files
        ci = self._class_combo.currentIndex()           # 0 = all classes
        cls = sorted(_CLASS_INFO)[ci - 1] if ci > 0 else None
        regs = self._regions
        if fi >= 0:
            lo, hi = self._boundaries[fi], self._boundaries[fi + 1]
            regs = [r for r in regs if lo <= r[0] < hi]
        if cls is not None:
            regs = [r for r in regs if r[2] == cls]
        self._filtered = regs
        self._page = 0
        self._total_pages = max(1, int(np.ceil(len(regs) / self._page_size)))

        from collections import Counter
        counts = Counter(c for _, _, c in regs)
        parts = [f"Pulses: {len(regs)}"]
        for cid in sorted(counts):
            name, color = _CLASS_INFO.get(cid, (f"class {cid}", "#444"))
            parts.append(f"<span style='color:{color}'>{name}: "
                         f"{counts[cid]}</span>")
        self._lbl_head.setText(" &nbsp;|&nbsp; ".join(parts))
        self._render_page()

    def _render_page(self):
        self.fig.clear()
        p = self._page
        start = p * self._page_size
        end = min(start + self._page_size, len(self._filtered))
        self._lbl_page.setText(f"Page {p + 1} / {self._total_pages}")

        for i in range(start, end):
            s, e, cls = self._filtered[i]
            ax = self.fig.add_subplot(3, 3, i - start + 1)
            name, color = _CLASS_INFO.get(cls, (f"class {cls}", "#444"))

            # Context margin clamped to the pulse's own file.
            lo, hi = self._file_span(s)
            width = e - s + 1
            margin = max(int(0.6 * width), 20)
            a = max(lo, s - margin)
            b = min(hi, e + margin + 1)
            if b <= a:
                continue
            x = (np.arange(a, b) / self._fs) if self._use_time else np.arange(a, b)

            # Center each zone in-window and stack at a common scale.
            segs = [np.asarray(z[a:b], dtype=float) for z in self._zones]
            centered = [sg - np.median(sg) for sg in segs]
            scale = max((np.max(np.abs(c)) for c in centered if c.size), default=1.0)
            if scale <= 0:
                scale = 1.0
            step = 2.4
            n_zones = len(centered)
            for zi, c in enumerate(centered):
                offset = (n_zones - 1 - zi) * step
                ax.plot(x, c / scale + offset, linewidth=0.6, zorder=3)

            xs = (s / self._fs) if self._use_time else s
            xe = (e / self._fs) if self._use_time else e
            ax.axvspan(xs, xe, color=color, alpha=0.22, zorder=0, linewidth=0)

            ax.set_title(f"#{i + 1}: {name}", fontsize=8, color=color,
                         fontweight='bold')
            ax.tick_params(labelsize=6)
            ax.set_yticks([])
            ax.grid(True, axis='x', alpha=0.2)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.8)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*tight_layout.*")
            self.fig.subplots_adjust(left=0.03, right=0.99, top=0.93,
                                     bottom=0.06, hspace=0.35, wspace=0.12)
            self.canvas.draw()

    def _first(self):
        if self._page != 0:
            self._page = 0
            self._render_page()

    def _prev(self):
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def _next(self):
        if self._page < self._total_pages - 1:
            self._page += 1
            self._render_page()

    def _last(self):
        if self._page != self._total_pages - 1:
            self._page = self._total_pages - 1
            self._render_page()


def extracted_pulses_view(data, regions, boundaries, file_names,
                          sample_rate=1):
    """Open the paginated 3x3 grid of extracted labeled pulses."""
    zones = zone_columns(np.asarray(data))
    dlg = _ExtractedPulsesDialog(zones, regions, boundaries, file_names,
                                 sample_rate=sample_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()
