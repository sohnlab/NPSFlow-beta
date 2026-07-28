"""Runner for RegionSelectionUI block - calls utils function."""


def run(inputs, params, block):
    from utils.region_selection_ui import region_selection_ui

    template_data = inputs.get("templateData")

    if template_data is None:
        raise ValueError("No template data provided to RegionSelectionUI.")

    selected_segments, segment_slices = region_selection_ui(
        template_data, None)
    return {
        "selectedSegments": selected_segments,
        "segmentSlices": segment_slices,
    }
