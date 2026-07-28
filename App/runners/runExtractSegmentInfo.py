"""Runner for ExtractSegmentInfo block - flattens segment slices into per-segment structs."""


def run(inputs, params, block):
    segment_slices = inputs.get("segmentSlices")

    if segment_slices is None or len(segment_slices) == 0:
        return {"segmentInfo": []}

    segments = []

    for rs in segment_slices:
        rect_segments = rs.get("rectSegments", [])
        n_segs = len(rect_segments)

        for p in range(n_segs):
            if rect_segments[p] is None or (hasattr(rect_segments[p], "__len__")
                                            and len(rect_segments[p]) == 0):
                continue

            # Pulse index
            pulse_indices = rs.get("pulseIndices", [])
            if isinstance(pulse_indices, list) and p < len(pulse_indices):
                pulse_index = pulse_indices[p]
            else:
                pulse_index = p

            # Local bounds
            segment_bounds = rs.get("segmentBounds", [])
            lb = segment_bounds[p] if p < len(segment_bounds) else (0, 0)

            # Global bounds
            global_bounds = rs.get("globalBounds", [])
            gb = global_bounds[p] if p < len(global_bounds) else (0, 0)

            segments.append({
                "pulseIndex": pulse_index,
                "segmentId": rs.get("segmentId", 0),
                "segmentLabel": str(rs.get("label", "")),
                "startIdxLocal": lb[0] if hasattr(lb, "__getitem__") else lb,
                "endIdxLocal": lb[1] if hasattr(lb, "__getitem__") else lb,
                "startIdxGlobal": gb[0] if hasattr(gb, "__getitem__") else gb,
                "endIdxGlobal": gb[1] if hasattr(gb, "__getitem__") else gb,
                "segmentValue": rect_segments[p],
            })

    return {"segmentInfo": segments}
