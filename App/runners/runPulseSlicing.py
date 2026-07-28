"""Runner for PulseSlicing block - slices pulses into segments.

Maps segment boundaries from the template onto each pulse using peak
positions (when available) or proportional scaling, then visualises
the result in a paginated 3x3 grid.

Inputs (individual ports):
  pulseSegments, pulsePeakLocations, rectangularizedPulses,
  acceptanceID, pulseStartIndices
"""

import logging
import re

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.dialog_style import style_mpl_figure
from utils.zones import zone_list

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTabBar,
)

logger = logging.getLogger(__name__)


def _make_valid_name(name):
    """Convert a label to a valid Python identifier."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    if name and name[0].isdigit():
        name = "_" + name
    return name or "Segment"


def _map_template_value_to_pulse(value, template_bounds, pulse_bounds):
    """Map a 0-indexed template position to a 0-indexed pulse position.

    Uses piecewise-linear interpolation between corresponding boundary
    points (peaks + endpoints).  Both *template_bounds* and *pulse_bounds*
    are sorted arrays starting at 0 and ending at length-1.
    """
    # Clamp to template range
    value = max(template_bounds[0], min(template_bounds[-1], value))

    # Find interval in template_bounds
    for i in range(len(template_bounds) - 1):
        if template_bounds[i] <= value <= template_bounds[i + 1]:
            span = template_bounds[i + 1] - template_bounds[i]
            if span == 0:
                frac = 0.0
            else:
                frac = (value - template_bounds[i]) / span
            # Map to corresponding interval in pulse
            j = min(i, len(pulse_bounds) - 2)
            mapped = pulse_bounds[j] + frac * (pulse_bounds[j + 1] - pulse_bounds[j])
            return int(round(mapped))

    # Fallback: proportional
    return int(round(value * (pulse_bounds[-1] / max(template_bounds[-1], 1))))


def _slice_zone(pf, rectangularized_pulses, pulse_peak_locations,
                acceptance_id, pulse_start_indices):
    """Slice one zone's pulses into segments.

    Verbatim single-zone slicing pipeline. Returns
    ``(segment_data, pulse_vis, labels)`` where *segment_data* maps each
    field name to the legacy single-zone segment dict.
    """
    # Extract segment definitions (0-indexed bounds from PulseFeatureExtraction)
    bounds = np.asarray(pf.get("bounds", []))
    if bounds.size == 0:
        raise ValueError("pulseSegments.bounds is empty.")
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, -1)
    num_segments = bounds.shape[0]

    labels = pf.get("labels", [f"Segment {i+1}" for i in range(num_segments)])
    methods = pf.get("methods", ["mean"] * num_segments)
    if len(methods) < num_segments:
        methods = list(methods) + ["mean"] * (num_segments - len(methods))

    # Template signal and length
    template_signal = pf.get("signal", None)
    if template_signal is not None and hasattr(template_signal, "__len__"):
        template_length = len(template_signal)
    else:
        template_length = int(np.max(bounds)) + 1

    # Template peak locations (0-indexed)
    template_peaks = pf.get("peak_locations", pf.get("peakLocations", None))
    if template_peaks is not None:
        if isinstance(template_peaks, list) and len(template_peaks) > 0:
            if isinstance(template_peaks[0], (list, np.ndarray)):
                template_peaks = template_peaks[0]
        template_peaks = np.unique(np.round(
            np.asarray(template_peaks, dtype=float)).astype(int))
        template_peaks = template_peaks[
            (template_peaks >= 0) & (template_peaks < template_length)]
    else:
        template_peaks = np.array([], dtype=int)

    # Build sorted template boundary list: [0, peak1, peak2, ..., template_length-1]
    template_bounds = np.unique(np.concatenate(
        [[0], template_peaks, [template_length - 1]]).astype(int))
    num_template_peaks = len(template_peaks)

    # Slice all pulses
    n_pulses = len(rectangularized_pulses)
    pulse_vis = [{"seg": None, "segments": []} for _ in range(n_pulses)]

    # Per-segment accumulators
    segment_data = {}
    for k in range(num_segments):
        field_name = _make_valid_name(labels[k])
        segment_data[field_name] = {
            "label": str(labels[k]),
            "method": str(methods[k]),
            "segmentId": k + 1,
            "pulseIndex": [],
            "startLocal": [],
            "endLocal": [],
            "startGlobal": [],
            "endGlobal": [],
            "values": [],
        }

    if pulse_start_indices is not None:
        pulse_start_indices = np.asarray(pulse_start_indices).ravel()

    for k in range(num_segments):
        field_name = _make_valid_name(labels[k])
        seg_start_tpl = int(bounds[k, 0])  # 0-indexed
        seg_end_tpl = int(bounds[k, 1])    # 0-indexed
        if seg_end_tpl < seg_start_tpl:
            seg_start_tpl, seg_end_tpl = seg_end_tpl, seg_start_tpl

        for p in range(n_pulses):
            # Skip rejected pulses
            if acceptance_id is not None:
                if hasattr(acceptance_id, "__len__") and p < len(acceptance_id):
                    if not acceptance_id[p]:
                        continue

            seg_struct = rectangularized_pulses[p]
            if seg_struct is None:
                continue

            # Extract signal from struct or raw array
            if isinstance(seg_struct, dict):
                seg = seg_struct.get("signal",
                                     seg_struct.get("segment", None))
                start_idx_global = seg_struct.get(
                    "startIndex", seg_struct.get("startIdx", 0))
            else:
                seg = np.asarray(seg_struct, dtype=float).ravel()
                start_idx_global = 0

            if seg is None or len(seg) == 0:
                continue

            seg = np.asarray(seg, dtype=float).ravel()

            # Global start index override
            if (pulse_start_indices is not None and
                    len(pulse_start_indices) > 0 and
                    p < len(pulse_start_indices)):
                start_idx_global = int(pulse_start_indices[p])

            lp = len(seg)

            # Store signal for visualization (once per pulse)
            if pulse_vis[p]["seg"] is None:
                pulse_vis[p]["seg"] = seg

            # Get local peak locations for this pulse (convert to 0-indexed)
            pulse_peaks_local = np.array([], dtype=int)
            if (pulse_peak_locations is not None and
                    len(pulse_peak_locations) > p and
                    pulse_peak_locations[p] is not None):
                ppg = np.asarray(pulse_peak_locations[p], dtype=float).ravel()
                if len(ppg) > 0:
                    ppl = (ppg - start_idx_global).astype(int)
                    pulse_peaks_local = ppl[(ppl >= 0) & (ppl < lp)]

            # Build pulse boundary list: [0, peak1, peak2, ..., lp-1]
            pulse_bounds = np.unique(np.concatenate(
                [[0], np.sort(pulse_peaks_local), [lp - 1]]).astype(int))

            # Map template segment bounds to pulse using peak correspondence
            if (len(pulse_peaks_local) > 0 and num_template_peaks > 0 and
                    len(pulse_bounds) == len(template_bounds)):
                # Direct piecewise-linear mapping
                s_idx = _map_template_value_to_pulse(
                    seg_start_tpl, template_bounds, pulse_bounds)
                e_idx = _map_template_value_to_pulse(
                    seg_end_tpl, template_bounds, pulse_bounds)
            else:
                # Proportional scaling fallback
                scale = (lp - 1) / max(template_length - 1, 1)
                s_idx = int(round(seg_start_tpl * scale))
                e_idx = int(round(seg_end_tpl * scale))

            # Clamp to valid range
            s_idx = max(0, min(lp - 1, s_idx))
            e_idx = max(0, min(lp - 1, e_idx))

            if e_idx < s_idx:
                continue

            segment_data[field_name]["pulseIndex"].append(p + 1)
            segment_data[field_name]["startLocal"].append(s_idx)
            segment_data[field_name]["endLocal"].append(e_idx)
            segment_data[field_name]["startGlobal"].append(
                start_idx_global + s_idx)
            segment_data[field_name]["endGlobal"].append(
                start_idx_global + e_idx)
            segment_data[field_name]["values"].append(seg[s_idx:e_idx + 1])

            # Visualization info (0-indexed)
            pulse_vis[p]["segments"].append({
                "bounds": (s_idx, e_idx),
                "segmentIdx": k + 1,
                "label": str(labels[k]),
            })

    return segment_data, pulse_vis, labels


def run(inputs, params, block):
    pulse_segments = inputs.get("pulseSegments")
    rectangularized_pulses = inputs.get("rectangularizedPulses")
    pulse_peak_locations = inputs.get("pulsePeakLocations")
    acceptance_id = inputs.get("acceptanceID")
    pulse_start_indices = inputs.get("pulseStartIndices")

    if pulse_segments is None:
        raise ValueError("No pulseSegments provided to PulseSlicing.")
    if not rectangularized_pulses:
        raise ValueError("No rectangularizedPulses provided to PulseSlicing.")

    # Split each input into per-zone payloads. pulseSegments defines the zone
    # count; batch arrays broadcast a single bare value across all zones.
    seg_zones = zone_list(pulse_segments)
    nz = len(seg_zones)
    if nz == 0:
        raise ValueError("pulseSegments is empty.")

    def _expand(obj, name, required):
        if obj is None:
            return [None] * nz
        zl = zone_list(obj)
        if len(zl) == nz:
            return zl
        if len(zl) == 1:
            return [zl[0]] * nz
        raise ValueError(
            f"PulseSlicing: {name} has {len(zl)} zones but pulseSegments "
            f"has {nz}.")

    rect_zones = _expand(rectangularized_pulses, "rectangularizedPulses", True)
    peak_zones = _expand(pulse_peak_locations, "pulsePeakLocations", False)
    acc_zones = _expand(acceptance_id, "acceptanceID", False)
    start_zones = _expand(pulse_start_indices, "pulseStartIndices", False)

    per_zone_segment_data = []
    per_zone_vis = []
    per_zone_labels = []
    for z in range(nz):
        sd, vis, labels = _slice_zone(
            seg_zones[z], rect_zones[z], peak_zones[z],
            acc_zones[z], start_zones[z])
        per_zone_segment_data.append(sd)
        per_zone_vis.append(vis)
        per_zone_labels.append(labels)

    # Output structure, by zone count:
    #   1 zone  -> legacy per-segment ports (one output per segment, bare
    #              values). Byte-for-byte identical to the single-signal path.
    #   N zones -> one output port per zone, each carrying that zone's full
    #              segment struct ``{segLabel: segdict, ...}``. Each zone keeps
    #              its OWN segments (different counts/labels are fine — nothing
    #              is forced into a shared field set). Downstream unpacks each
    #              zone port into its segments.
    if nz == 1:
        segment_data = per_zone_segment_data[0]
    else:
        segment_data = {f"Zone {z + 1}": per_zone_segment_data[z]
                        for z in range(nz)}

    # Visualization: one tab per zone (single zone → no tab bar).
    _plot_pulse_segments(per_zone_vis,
                         [len(lbls) for lbls in per_zone_labels],
                         per_zone_labels)

    return segment_data


# Hand-picked distinct segment colors (no two adjacent segments share a color)
_SEGMENT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
]


class _SlicedPulsesDialog(QDialog):
    """Paginated 3x3 grid of sliced pulses using QDialog + FigureCanvas.

    Multi-zone: one tab per zone, each with its own pagination. Single zone:
    no tab bar (identical to the legacy view).
    """

    PULSES_PER_PAGE = 9

    def __init__(self, per_zone_vis, per_zone_num_segments, per_zone_labels,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sliced Pulses")
        self.resize(1200, 850)

        from utils.dialog_style import apply_dialog_style
        apply_dialog_style(self)

        self._zones_vis = list(per_zone_vis)
        self._zones_num_segments = list(per_zone_num_segments)
        self._zones_labels = list(per_zone_labels)
        self._n_zones = len(self._zones_vis)

        # Per-zone derived state.
        self._zone_valid_idx = []
        self._zone_total_pages = []
        for vis in self._zones_vis:
            valid = [i for i, pv in enumerate(vis) if pv["seg"] is not None]
            self._zone_valid_idx.append(valid)
            self._zone_total_pages.append(max(
                1, (len(valid) + self.PULSES_PER_PAGE - 1)
                // self.PULSES_PER_PAGE))
        self._zone_page = [1] * self._n_zones   # per-tab current page
        self._active_zone = -1

        self._build_ui()
        self._load_zone(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Zone tab bar (multi-zone only).
        self._zone_bar = None
        if self._n_zones > 1:
            self._zone_bar = QTabBar()
            self._zone_bar.setExpanding(False)
            for z in range(self._n_zones):
                self._zone_bar.addTab(f"Zone {z + 1}")
            self._zone_bar.currentChanged.connect(self._on_zone_changed)
            layout.addWidget(self._zone_bar)

        # Matplotlib canvas
        self.fig = Figure(figsize=(12, 9))
        style_mpl_figure(self.fig)
        self.fig.set_facecolor("#f5f5f5")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Button row: page label on the left; Previous + Next + Confirm
        # grouped in the bottom-right corner. Previous/Next share one width.
        btn_layout = QHBoxLayout()

        self._page_label = QLabel()
        btn_layout.addWidget(self._page_label)

        btn_layout.addStretch()

        self._btn_prev = QPushButton("Previous")
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next = QPushButton("Next")
        self._btn_next.clicked.connect(self._on_next)
        nav_w = max(self._btn_prev.sizeHint().width(),
                    self._btn_next.sizeHint().width())
        self._btn_prev.setFixedWidth(nav_w)
        self._btn_next.setFixedWidth(nav_w)
        btn_layout.addWidget(self._btn_prev)
        btn_layout.addWidget(self._btn_next)

        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setObjectName("confirmBtn")
        self._btn_confirm.clicked.connect(self.accept)
        btn_layout.addWidget(self._btn_confirm)

        layout.addLayout(btn_layout)

    def _load_zone(self, z):
        """Point the live drawing state at zone *z* and redraw its page."""
        if not (0 <= z < self._n_zones) or z == self._active_zone:
            return
        self._active_zone = z
        self._pulse_vis = self._zones_vis[z]
        self._labels = self._zones_labels[z]
        self._valid_idx = self._zone_valid_idx[z]
        self._n_valid = len(self._valid_idx)
        self._total_pages = self._zone_total_pages[z]
        self._current_page = self._zone_page[z]
        ns = self._zones_num_segments[z]
        self._seg_colors = [_SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]
                            for i in range(ns)]
        self._draw_page()

    def _on_zone_changed(self, idx):
        self._zone_page[self._active_zone] = self._current_page  # remember page
        self._load_zone(idx)

    def _update_nav(self):
        self._btn_prev.setEnabled(self._current_page > 1)
        self._btn_next.setEnabled(self._current_page < self._total_pages)
        self._page_label.setText(
            f"Page {self._current_page} / {self._total_pages}"
            f"  ({self._n_valid} pulses)")

    def _draw_page(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                                   bottom=0.04, top=0.91, left=0.06, right=0.97)
        start_i = (self._current_page - 1) * self.PULSES_PER_PAGE

        for j in range(self.PULSES_PER_PAGE):
            row, col = divmod(j, 3)
            ax = self.fig.add_subplot(gs[row, col])
            vi = start_i + j
            if vi >= self._n_valid:
                ax.set_visible(False)
                continue

            p_idx = self._valid_idx[vi]
            seg = self._pulse_vis[p_idx]["seg"]
            regs = self._pulse_vis[p_idx]["segments"]
            lp = len(seg)
            x = np.arange(lp)

            y_min, y_max = np.min(seg), np.max(seg)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = max(abs(y_max), 1)
            y_pad = 0.08 * y_range

            for rr in regs:
                rb = rr["bounds"]
                s0 = max(0, min(lp - 1, rb[0]))
                e0 = max(0, min(lp - 1, rb[1]))
                c_idx = (rr["segmentIdx"] - 1) % len(self._seg_colors)
                ax.axvspan(s0, e0, alpha=0.25, color=self._seg_colors[c_idx])

            ax.plot(x, seg, linewidth=1.2, color="#1f77b4")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Pulse {p_idx + 1}", fontsize=9, fontweight="bold")
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlim(0, lp - 1)
            ax.tick_params(labelsize=7)

        legend_str = " | ".join(str(lb) for lb in self._labels)
        self.fig.suptitle(
            f"Sliced Pulses  --  {legend_str}",
            fontsize=11, fontweight="bold")

        self._update_nav()
        self.canvas.draw()

    def _on_prev(self):
        if self._current_page > 1:
            self._current_page -= 1
            self._zone_page[self._active_zone] = self._current_page
            self._draw_page()

    def _on_next(self):
        if self._current_page < self._total_pages:
            self._current_page += 1
            self._zone_page[self._active_zone] = self._current_page
            self._draw_page()


def _plot_pulse_segments(per_zone_vis, per_zone_num_segments, per_zone_labels):
    """Show the paginated sliced-pulse dialog. Multi-zone → one tab per zone.

    ``per_zone_vis`` / ``per_zone_labels`` are lists with one entry per zone;
    ``per_zone_num_segments`` a list of each zone's segment count.
    """
    valid = any(pv["seg"] is not None
                for vis in per_zone_vis for pv in vis)
    if not valid:
        return
    dlg = _SlicedPulsesDialog(per_zone_vis, per_zone_num_segments,
                              per_zone_labels)
    dlg.exec()
