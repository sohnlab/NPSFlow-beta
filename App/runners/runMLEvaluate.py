"""Runner for MLEvaluate block — scores a trained event classifier on a
labeled file set (window-level per-horizon metrics, optional stream sim)."""


def run(inputs, params, block):
    from processing.ml_training_core import (
        evaluate_on_entries, extract_entries, horizon_sort_key,
    )

    model_file = inputs.get("modelFile")
    if not model_file or not isinstance(model_file, str):
        raise ValueError("ML Evaluate needs a modelFile path.")
    entries = extract_entries(inputs.get("dataSet"))
    if not entries:
        raise ValueError("ML Evaluate: no file entries on dataSet.")

    from utils.app_logger import logger
    result = evaluate_on_entries(
        model_file.strip(), entries,
        stream_eval=str(params.get("streamEval", "no")) == "yes",
        progress=logger.info)

    metrics = result["metrics"]
    if metrics:
        last = max(metrics, key=horizon_sort_key)
        m = metrics[last]
        block.parameters["displayText"] = (
            f"acc {m['accuracy']:.3f}, mF1 {m['macro_f1']:.3f} @ {last}")
    return {"metrics": metrics, "streamScores": result["stream"]}
