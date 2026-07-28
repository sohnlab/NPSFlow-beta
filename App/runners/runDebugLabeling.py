"""Runner for DebugLabeling block - paginated 3x3 grid of labeled regions."""


def run(inputs, params, block):
    from processing.debug_labeling import debug_labeling_ui

    labeled = inputs.get("labeledData")
    if labeled is None:
        raise ValueError("No labeledData connected to Debug - Labeling.")

    # Data Labeling emits {"data": ..., "labels": ...}. Also accept the
    # dot-notation keys that Save File / Load Labeled Data use.
    if isinstance(labeled, dict):
        from processing.ml_training_core import entry_data_labels
        data, labels = entry_data_labels(labeled)
    else:
        data, labels = labeled, None

    if data is None or labels is None:
        raise ValueError(
            "labeledData must be a struct with 'data' and 'labels' "
            "(the output of the Data Labeling block).")

    sample_rate = inputs.get("sampleRate", 1)
    debug_labeling_ui(data, labels, sample_rate=sample_rate)
    return {}
