"""Min-max decimation for plotting long signals without losing peaks."""

import numpy as np


def minmax_decimate(x, y, max_points=4000):
    """Decimate (x, y) for plotting, keeping each bin's min and max sample.

    Returns the inputs unchanged when len(x) <= max_points. Output holds
    two points per bin (min and max, in temporal order) plus any tail
    samples that did not fill a whole bin, so peaks/spikes survive.
    """
    n = len(x)
    if n <= max_points:
        return x, y
    n_bins = max_points // 2
    bin_size = n // n_bins
    usable = n_bins * bin_size
    y_bins = y[:usable].reshape(n_bins, bin_size)
    x_bins = x[:usable].reshape(n_bins, bin_size)
    i_min = np.argmin(y_bins, axis=1)
    i_max = np.argmax(y_bins, axis=1)
    rows = np.arange(n_bins)
    lo = np.minimum(i_min, i_max)
    hi = np.maximum(i_min, i_max)
    out_x = np.empty(n_bins * 2)
    out_y = np.empty(n_bins * 2)
    out_x[0::2] = x_bins[rows, lo]
    out_y[0::2] = y_bins[rows, lo]
    out_x[1::2] = x_bins[rows, hi]
    out_y[1::2] = y_bins[rows, hi]
    if usable < n:
        out_x = np.concatenate([out_x, x[usable:]])
        out_y = np.concatenate([out_y, y[usable:]])
    return out_x, out_y
