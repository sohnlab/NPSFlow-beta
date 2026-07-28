"""Debug viewer for Data Labeling output.

Extracts the labeled regions (contiguous non-zero runs of the per-sample label
array) and shows them in a paginated 3x3 grid, one region per cell, with every
zone overlaid and the labeled span shaded. A quick way to eyeball whether the
labels actually land on the events in the signal.
"""

import warnings
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
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


def _label_regions(labels):
    """Return [(start, end, class_id), ...] for contiguous non-zero runs."""
    labels = np.asarray(labels).astype(int).ravel()
    n = len(labels)
    regions = []
    i = 0
    while i < n:
        c = labels[i]
        if c == 0:
            i += 1
            continue
        j = i
        while j < n and labels[j] == c:
            j += 1
        regions.append((i, j - 1, int(c)))
        i = j
    return regions


class _DebugLabelingDialog(QDialog):
    def __init__(self, zones, regions, sample_rate=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Debug - Labeling")
        self.resize(1500, 900)
        apply_dialog_style(self)

        self._zones = zones
        self._n_len = len(zones[0]) if zones else 0
        self._regions = regions
        self._fs = sample_rate if sample_rate and sample_rate > 1 else 1
        self._use_time = self._fs > 1
        self._page = 0
        self._page_size = 9
        self._total_pages = max(1, int(np.ceil(len(regions) / self._page_size)))

        self._build_ui()
        self._render_page()

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        # Header: totals + per-class counts.
        from collections import Counter
        counts = Counter(c for _, _, c in self._regions)
        parts = [f"Regions: {len(self._regions)}"]
        for cid in sorted(counts):
            name, color = _CLASS_INFO.get(cid, (f"class {cid}", "#444"))
            parts.append(f"<span style='color:{color}'>{name} (id={cid}): "
                         f"{counts[cid]}</span>")
        self._lbl_head = QLabel(" &nbsp;|&nbsp; ".join(parts))
        self._lbl_head.setTextFormat(Qt.TextFormat.RichText)
        main.addWidget(self._lbl_head)

        self.fig = Figure(figsize=(14, 9), tight_layout=False)
        style_mpl_figure(self.fig)
        self.canvas = FigureCanvas(self.fig)
        main.addWidget(self.canvas, 1)

        # Navigation bar.
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

    def _render_page(self):
        self.fig.clear()
        p = self._page
        start = p * self._page_size
        end = min(start + self._page_size, len(self._regions))
        self._lbl_page.setText(f"Page {p + 1} / {self._total_pages}")

        for i in range(start, end):
            s, e, cls = self._regions[i]
            ax = self.fig.add_subplot(3, 3, i - start + 1)
            name, color = _CLASS_INFO.get(cls, (f"class {cls}", "#444"))

            width = e - s + 1
            margin = max(int(0.6 * width), 20)
            a = max(0, s - margin)
            b = min(self._n_len, e + margin + 1)
            if b <= a:
                continue
            x = (np.arange(a, b) / self._fs) if self._use_time else np.arange(a, b)

            # Center each zone in-window and stack at a common scale.
            segs = [np.asarray(z[a:b], dtype=float) for z in self._zones]
            centered = [sg - np.median(sg) for sg in segs]
            scale = max((np.max(np.abs(c)) for c in centered if c.size), default=1.0)
            if scale <= 0:
                scale = 1.0
            # Stack with Zone 1 on top, last zone at the bottom.
            step = 2.4
            n_zones = len(centered)
            for zi, c in enumerate(centered):
                offset = (n_zones - 1 - zi) * step
                ax.plot(x, c / scale + offset, linewidth=0.6, zorder=3)

            # Shade the labeled span.
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


def debug_labeling_ui(data, labels, sample_rate=1):
    """Open the paginated 3x3 debug grid of labeled regions."""
    data = np.asarray(data)
    zones = zone_columns(data)
    regions = _label_regions(labels)

    dlg = _DebugLabelingDialog(zones, regions, sample_rate=sample_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()
    return {"regionCount": len(regions)}
