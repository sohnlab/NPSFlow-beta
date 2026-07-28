"""Multizone variant of Detection Review — requires multi-zone (mzNPS) data.

Reuses the paginated review/classification dialog from
``pulse_region_review``; the only difference is that this entry point rejects
single-zone (1-D or single-column) input, since the block exists specifically
for reviewing multi-zone signals across all zones at once.
"""

import warnings
import numpy as np

from utils.zones import is_multizone
from processing.pulse_region_review import _PulseReviewDialog


def multizone_detection_review(data, detected_pulses, sample_rate=1,
                               init_labels=None, init_regions=None,
                               init_cutoff=None, init_zone_roles=None,
                               init_manual=None):
    """Interactive UI to review/classify detected pulses across zones.

    Identical to :func:`pulse_region_review.pulse_region_review` but requires
    *data* to be multi-zone (2-D N×Z with Z ≥ 2). Returns a dict keyed by
    class label ('single', 'coincident', 'noise', 'uncertain') → Nx2 indices.
    """
    data = np.asarray(data) if data is not None else np.array([])
    if not is_multizone(data):
        raise ValueError(
            "Detection Review (mz) requires multi-zone (mzNPS) data with 2 or "
            "more zones. Use Detection Review for single-zone data.")

    pulse_regions = detected_pulses if detected_pulses is not None else np.empty((0, 2), dtype=int)
    pulse_regions = np.atleast_2d(pulse_regions)
    n_total = len(pulse_regions)

    if n_total == 0:
        return {
            'single': np.empty((0, 2), dtype=int),
            'coincident': np.empty((0, 2), dtype=int),
            'noise': np.empty((0, 2), dtype=int),
            'uncertain': np.empty((0, 2), dtype=int),
        }

    dlg = _PulseReviewDialog(data, pulse_regions, sample_rate,
                             init_labels=init_labels, init_regions=init_regions,
                             init_cutoff=init_cutoff, zone_roles=True,
                             init_zone_roles=init_zone_roles,
                             init_manual=init_manual)
    dlg.setWindowTitle("Detection Review (mz)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*tight_layout.*")
        dlg.exec()

    if not dlg.confirmed:
        raise ValueError("User closed Detection Review (mz) without confirming.")

    regions = np.array(dlg._pulse_regions, dtype=int)
    labels = dlg._labels

    result = {}
    for cls in ('single', 'coincident', 'noise', 'uncertain'):
        mask = [i for i, lbl in enumerate(labels) if lbl == cls]
        result[cls] = regions[mask] if mask else np.empty((0, 2), dtype=int)

    # Chosen Start/Measurement/End zone assignments (0-based zone indices).
    result['zoneRoles'] = dlg._zone_roles()

    return result
