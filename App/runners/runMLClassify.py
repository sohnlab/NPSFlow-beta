"""Runner for MLClassify block — streams a signal through a trained event
classifier and reviews predictions against true labels."""

import numpy as np


def _extract(data_in, file_index=0):
    """(data, labels, file_name) from the accepted dataIn shapes."""
    from processing.ml_training_core import entry_data_labels
    from utils.app_logger import logger
    if isinstance(data_in, dict):
        entries = data_in.get("files")
        if isinstance(entries, (list, tuple)) and entries:
            idx = max(0, min(int(file_index), len(entries) - 1))
            if len(entries) > 1:
                logger.warning(
                    f"[MLClassify] {len(entries)} files on dataIn — "
                    f"classifying [{idx}] (File index parameter); use "
                    f"Combine Data upstream to stream them all as one.")
            data_in = entries[idx]
        data, labels = entry_data_labels(data_in)
        return data, labels, str(data_in.get("fileName", ""))
    return data_in, None, ""


def run(inputs, params, block):
    from processing.ml_classify import ml_classify_ui
    from processing.ml_training_core import resolve_path

    model_file = inputs.get("modelFile")
    if not model_file or not isinstance(model_file, str):
        raise ValueError("ML Classify needs a modelFile path from the "
                         "ML Training block.")

    data, labels, file_name = _extract(inputs.get("dataIn"),
                                       params.get("fileIndex", 0))
    if data is None:
        raise ValueError("No signal found on dataIn.")
    true_labels = inputs.get("trueLabels")
    if true_labels is not None:
        labels = true_labels
    if labels is not None:
        labels = np.asarray(labels).astype(np.int32).ravel()
        if len(labels) != len(np.asarray(data)):
            from utils.app_logger import logger
            logger.warning(
                "[MLClassify] labels length mismatch — ignoring labels.")
            labels = None

    result = ml_classify_ui(resolve_path(model_file.strip()), data,
                            labels=labels, file_name=file_name)

    n_events = sum(1 for e in result["events"] if e["label"] > 0)
    block.parameters["displayText"] = f"{n_events} events"
    return result
