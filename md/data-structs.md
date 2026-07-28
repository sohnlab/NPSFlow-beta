# Key Data Structures

## TPsettings (Template Settings Output)

Zone-grouped: one entry in `zones` per template column (a 1-D template
yields a single zone). `filter_config`/`filter_padding` are shared across
zones and kept at the top level; every other field is per-zone.

```python
{
    'num_zones': int,
    'filter_config': dict,                       # shared filter chain
    'filter_padding': {'pad_length': int, 'method': str},
    'zones': [
        {                                        # one self-contained per-zone dict
            'template_original': np.ndarray,
            'pulse_template_rec': np.ndarray,
            'threshold': {'upper': float, 'lower': float},
            'edge_margin': {'first_peak_prop': float, 'last_peak_prop': float},
            'peak_counts': {'positive': int, 'negative': int},
            'peak_locations': np.ndarray,         # segment boundaries (FWHM-refined when edge_method='fwhm')
            'peak_locations_apex': np.ndarray,    # derivative-peak apexes (pre-refinement); key-peak matching anchor
            'peak_sequence': np.ndarray,          # e.g. [0,1,0,1,0,1,0,1]
            'key_peaks': np.ndarray,              # Template sample indices
            'key_peak_properties': list[str],     # 'regular'/'max'/'min' per key peak
            'key_peak_seq_indices': list[int],    # Sequence index per key peak
            'exclusion_zones': [(start, end), ...],
            'exclusion_zone_seq_bounds': [(s_seq, e_seq), ...],  # key-peak-relative: -1=before first KP, N=after last KP
            'end_zones': {'enabled': bool, 'percent': int, 'zones': [...]},
            'methods': list[str],                 # Per-segment method
            'edge_method': str,                   # 'peaks' | 'fwhm' slicing indices
        },
        ...
    ],
}
```

**Accessing zones:** downstream consumers that process a single zone use
`utils.tpsettings_io.get_zone(tp, k=0)`, which returns zone *k* as a flat
dict with `filter_config`/`filter_padding` merged back in — the legacy
pre-grouping shape. `normalize_tpsettings(tp)` upgrades a legacy flat dict
(no `zones` key) to the canonical shape, so old saved `.json` settings and
old workflows keep loading.

## Segment Data Structure

Output of PulseSlicing / SegmentProcessing:

```python
{
    "startLocal": [...],      # Local indices (start of segment within pulse)
    "endLocal": [...],        # Local indices (end of segment)
    "startGlobal": [...],     # Absolute position in data
    "endGlobal": [...],
    "values": [...],          # List of arrays per pulse
    "width": [...],           # endLocal - startLocal + 1
    "startValue": [...],      # Signal value at segment start
    "endValue": [...],        # Signal value at segment end
}
```

## Peak Detection Algorithm (Batch Processing)

Purely sequence-based, no proportions or distances:

1. **Phase 1**: Find min/max anchors — strongest positive (max) and negative (min) derivative peaks in entire pulse
2. **Phase 2**: Find regular key peaks by counting from anchors — for each ★ key peak, count N peaks from the nearest anchor in the template sequence direction, using only threshold-exceeding peaks
3. **Phase 3**: Build exclusion zones — map zone boundaries from template sequence indices to found peak positions in pulse
4. **Phase 4**: Find remaining peaks — region-aware thresholding with exclusion zones

Key peak markers: ▲ max, ▼ min, ★ regular, ● non-key

## Workflow JSON Format

```json
{
  "blocks": [
    {
      "id": "blk_<uuid>",
      "definition": "BlockDefinitionName",
      "displayName": "User Name",
      "position": [x, y],
      "size": [w, h],
      "color": [r, g, b],
      "parameters": {}
    }
  ],
  "wires": [
    {
      "id": "wire_<uuid>",
      "source": "blk_id:portName",
      "destination": "blk_id:portName",
      "waypoints": [[x1, y1], ...]
    }
  ]
}
```
