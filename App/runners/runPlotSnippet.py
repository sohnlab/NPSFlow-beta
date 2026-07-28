"""Runner for PlotSnippet block – plots 50 data points from a given index."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QDialog, QVBoxLayout

from utils.dialog_style import apply_dialog_style, style_mpl_figure


def run(inputs, params, block):
    data = inputs.get("dataIn")
    idx = inputs.get("index", 0)
    n_points = inputs.get("nPoints", 50)

    if data is None:
        raise ValueError("No data connected to PlotSnippet.")

    data = np.asarray(data, dtype=float).ravel()
    idx = int(idx)
    n_points = int(n_points)

    if idx < 0 or idx >= len(data):
        raise ValueError(f"Index {idx} out of range [0, {len(data) - 1}].")

    end = min(idx + n_points, len(data))
    snippet = data[idx:end]
    x = np.arange(idx, end)

    # --- dialog ---
    dlg = QDialog()
    dlg.setWindowTitle(f"Plot Snippet  [{idx} : {end}]")
    dlg.resize(600, 400)
    apply_dialog_style(dlg)

    fig, ax = plt.subplots()
    ax.plot(x, snippet, "-o", markersize=3)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.set_title(f"Data [{idx} : {end}]  ({len(snippet)} pts)")
    style_mpl_figure(fig)
    fig.tight_layout()

    canvas = FigureCanvasQTAgg(fig)
    layout = QVBoxLayout(dlg)
    layout.addWidget(canvas)

    dlg.exec()
    plt.close(fig)

    return {}
