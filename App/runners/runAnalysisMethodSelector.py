"""Runner for AnalysisMethodSelector block - calls utils function."""


def run(inputs, params, block):
    from utils.analysis_method_selector import analysis_method_selector

    segment_data = inputs.get("pulseFeatures")
    sample_rate = inputs.get("sampleRate", 1)

    if segment_data is None:
        raise ValueError(
            "No segment data provided to AnalysisMethodSelector.")

    result = analysis_method_selector(segment_data, sample_rate)
    return {"analysisResult": result}
