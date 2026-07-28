"""Runner for Cluster block — K-Means clustering on X/Y data.

Inputs: x, y arrays of equal length.
Outputs:
  - labels: cluster assignment (1-based) per point
  - x_1, y_1, x_2, y_2, ... : per-cluster X and Y arrays
"""

import numpy as np
from scipy.cluster.vq import kmeans2, whiten


def run(inputs, params, block):
    x = inputs.get("x")
    y = inputs.get("y")
    if x is None or y is None:
        raise ValueError("Both x and y inputs are required.")

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"x and y must have equal length ({len(x)} vs {len(y)}).")

    n_clusters = int(params.get("n_clusters", 2))
    n_clusters = max(1, min(n_clusters, len(x)))

    # Standardize and cluster
    data = np.column_stack([x, y])
    data_w = whiten(data)
    centroids, labels = kmeans2(data_w, n_clusters, minit="++", iter=50)

    # Sort clusters by centroid x value for consistent ordering
    # Compute centroids in original space
    orig_centroids_x = [x[labels == k].mean() if np.any(labels == k) else 0
                        for k in range(n_clusters)]
    order = np.argsort(orig_centroids_x)
    remap = np.zeros(n_clusters, dtype=int)
    for new_k, old_k in enumerate(order):
        remap[old_k] = new_k
    labels = remap[labels]

    # Build outputs: per-cluster x and y (1-based naming)
    result = {"labels": labels + 1}
    for k in range(n_clusters):
        mask = labels == k
        result[f"x_{k + 1}"] = x[mask]
        result[f"y_{k + 1}"] = y[mask]

    # Update display text
    counts = [int(np.sum(labels == k)) for k in range(n_clusters)]
    block.parameters["displayText"] = f"K={n_clusters}\n" + ", ".join(str(c) for c in counts)

    return result
