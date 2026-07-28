"""Human-readable block descriptions for the right-click "Description" panel.

Keyed by block definition name (``defn["name"]``). Each value states the
block's general purpose; the panel lists inputs/outputs automatically from
the block definition, so these strings don't need to restate every port.

Sourced/adapted from ``md/blocks-*.md``. Edit here to refine wording.
"""

BLOCK_DESCRIPTIONS = {
    # --- Data I/O ---------------------------------------------------------
    "LoadData":
        "Unified loader for .mat / .csv / .npz / .json files (single file or "
        "whole folder). Emits one struct of per-file entries meant to feed "
        "an Unpack block (or Combine Data to concatenate all files). "
        "Remembers the last path.",
    "LoadNPSData":
        "Loads a .mat NPS signal file via a file browser (defaults to "
        "SignalData/). Outputs the filename and the loaded signal data.",
    "LoadCSVData":
        "Legacy loader for CSV data files. Outputs the loaded data and the "
        "list of file names.",
    "LoadLabeledData":
        "Loads a folder of labeled data files (signal + class labels) for "
        "training or evaluation.",
    "SaveFile":
        "Saves the connected inputs to a CSV file, one column per input. "
        "A terminal sink with no outputs (default folder: Output/).",
    "ExportAcceptedPulses":
        "Exports the accepted pulses to a file. A terminal sink with no "
        "outputs.",
    "TextBlock":
        "Emits a constant text string for use downstream. Double-click to "
        "edit the text.",
    "ConstantBlock":
        "Emits a fixed numeric constant for use downstream.",
    "DisplayBlock":
        "Opens a popup showing the input data. A terminal viewer with no "
        "outputs.",
    "DataSummary":
        "Reports summary statistics for the input signal and passes the data "
        "through, along with filename/platform info.",
    "UnpackStruct":
        "Unpacks selected fields of a (possibly nested) struct, list, or "
        "array into individual output ports. One block can replace a chain "
        "of cascaded unpackers.",
    "CombineData":
        "Concatenates the per-file entries of a Load File struct into single "
        "data/labels arrays, with per-file boundary indices so downstream "
        "blocks can mask the seams between files.",
    "CompileTable":
        "Packs multiple connected inputs into a single table/struct, one "
        "column per input.",
    "DeviceGeometry":
        "Interactively defines the NPS device geometry (segments, channel "
        "height, dimensions) and emits it as a geometry struct.",

    # --- Variables --------------------------------------------------------
    "DataBus":
        "Named bus that carries data (and the Run signal) between "
        "disconnected parts of the pipeline. Outputs mirror the connected "
        "inputs under a named pool.",
    "LoadGeometry":
        "Loads a previously saved device geometry from the Reference store.",
    "GlobalSampleRate":
        "Sets the global sample-rate variable that every block reads.",
    "DefineParameters":
        "Defines global parameters injected into all blocks, and passes the "
        "Run signal through.",

    # --- Flow Control -----------------------------------------------------
    "StartBlock":
        "Pipeline entry point — emits the trigger that starts execution. "
        "One per pipeline; double-click to run the whole pipeline.",
    "Priority":
        "Defines execution order for downstream branches: blocks on output 1 "
        "run before output 2, and so on.",

    # --- Preprocessing ----------------------------------------------------
    "TrimData":
        "Interactive trimming of the signal by range (and mzNPS zone) "
        "selection. Outputs the trimmed data and tracks cumulative trim "
        "indices.",
    "PlatformDetection":
        "Auto-detects the acquisition platform (mzNPS vs mechanoNPS) and "
        "passes the data through with the detected platform.",
    "ZoneSelection":
        "Interactively defines signal zones (for multi-zone mzNPS) and emits "
        "the selected zone boundaries.",
    "InvertData":
        "Inverts the signal polarity (multiplies the data by −1).",
    "TrainingData":
        "Assembles labeled training data by tagging the signal with the "
        "single / coincident / noise / uncertain index sets from Detection "
        "Review.",

    # --- Filters ----------------------------------------------------------
    "FilterUI":
        "Interactive filter-design tool with live preview. Applies the chosen "
        "filter to the data and also emits the resulting filterConfig.",
    "FilterConfig":
        "Assembles one or more filter stages into a single filter-chain "
        "configuration (no data processing).",
    "FilterLowpass":
        "Defines a lowpass Butterworth filter stage and emits it as a "
        "filterConfig for the Filter / Filter Config blocks.",
    "FilterHighpass":
        "Defines a highpass Butterworth filter stage and emits it as a "
        "filterConfig.",
    "FilterBandpass":
        "Defines a bandpass Butterworth filter stage and emits it as a "
        "filterConfig.",
    "FilterNotch":
        "Defines a notch (band-stop) filter stage for removing a specific "
        "frequency and emits it as a filterConfig.",
    "FilterRows":
        "Filters rows of the connected tabular data, keeping only those that "
        "match the configured condition.",

    # --- Template ---------------------------------------------------------
    "PulseTemplateExtractor":
        "Interactively selects a region of the signal to use as a single "
        "pulse template.",
    "PulseTemplateDetrend":
        "Removes a linear baseline drift from a pulse template.",
    "PulseTemplateProcessing":
        "Four-step interactive template setup (thresholds, key peaks, "
        "exclusion zones, per-segment methods). Emits the TPsettings struct "
        "used by detection and feature extraction.",
    "WaveletEditor":
        "Interactive editor for the rectangularized template that produces a "
        "1-D mother wavelet (for matched-filter / CWT detection) plus a "
        "segment description.",

    # --- Detection --------------------------------------------------------
    "PulseDetectionBC":
        "Interactive pulse detection with baseline correction (derivative "
        "thresholding + baseline interpolation). Outputs the baseline "
        "trendline and the detected pulses.",
    "PulseRegionReview":
        "Paginated 3×3 review of detected pulses, classifying each as single, "
        "coincident, noise, or uncertain. Outputs index arrays per class.",
    "CwtEvaluator":
        "Compares pulse-detection performance across several wavelet/scoring "
        "configurations on the same signal, and forwards the winning config's "
        "regions and score.",

    # --- Sorting (real-time-style detection) ------------------------------
    "RealtimeDetectionCWT":
        "Real-time-style matched-filter detector: convolves the signal with a "
        "mother wavelet across width scales, scores against a robust MAD "
        "threshold, and splits coincident events. Best for low-SNR data.",
    "ThresholdDetection":
        "Real-time-style causal detector on the running first-difference of "
        "the signal: slow drift cancels out while sharp pulse edges cluster "
        "into events. Best for high-SNR multi-plateau pulses.",
    "CombinedDetection":
        "Real-time-style detector running a causal matched-step plus an "
        "optional Ricker (Mexican-hat) CWT in parallel; either firing opens a "
        "region, which is then clustered, padded, and merged.",
    "MLSplit":
        "Splits labeled recordings into train/validation/test — by whole "
        "file (leak-free default) or by event, where per-event segments "
        "from all files are pooled and mixed across splits (optimistic "
        "scores: splits share recording conditions). Emits three "
        "Load-File-style structs plus the assignment map.",
    "MLTrain":
        "Trains the baseline/single/coincident event classifier on the "
        "wired training set and saves the model bundle. Non-interactive; "
        "configure model/strategy/oversampling via block parameters.",
    "MLEvaluate":
        "Scores a trained classifier bundle on a labeled file set: "
        "per-horizon accuracy, macro-F1, per-class recalls, and confusion "
        "(optionally a per-file real-time stream simulation).",
    "MLClassify":
        "Streams a signal through a trained event-classifier bundle (from "
        "ML Training) and plots predicted events against the true labels "
        "when available. Outputs per-sample predicted labels, event dicts, "
        "and an event-level score summary.",
    "MLTraining":
        "Trains a machine-learning event classifier (baseline / single / "
        "coincident; noise counts as baseline) from a folder of labeled "
        "recordings, split train/validation/test by file. The saved model "
        "classifies live streams causally at multiple latency horizons, and "
        "the dialog reports metrics plus a simulated real-time stream "
        "evaluation on the held-out test files.",
    "ExtractLabeledPulses":
        "Converts a per-sample labeled signal (Load File struct or "
        "{data, labels} entry) into Detection-Review-style Nx2 pulse "
        "regions, one output per class (single / coincident / noise / "
        "uncertain), plus the pass-through signal — ready to wire into "
        "Pulse Processing. Opens a paginated 3x3 grid viewer of the "
        "extracted pulses, filterable by file and class.",

    # --- Feature Extraction ----------------------------------------------
    "ExtractSinglePulseFeatures":
        "Interactive per-pulse review: detects peaks, matches them to the "
        "expected sequence, and lets you accept/reject each pulse. Outputs "
        "peak locations, rectangularized pulses, acceptance IDs, start "
        "indices, and features.",
    "ExtractCoincidentPulseFeatures":
        "Interactive decomposition of coincident (overlapping) pulse regions "
        "into individual single-particle pulses using start/end indicator-zone "
        "timing (template-tiling fallback). Outputs the same schema as Pulse "
        "Processing plus a coincidence map, as a standalone branch.",
    "PulseFeatureExtraction":
        "Interactive UI to define per-pulse feature segments from the "
        "template settings. Outputs the feature definitions.",
    "PulseSlicing":
        "Slices each pulse into segments according to the feature "
        "definitions. Emits one segment-data output per feature.",
    "SegmentProcessing":
        "Processes connected segment data (e.g. width, start/end values), "
        "mapping each input to a corresponding output.",
    "ExtractSegmentWidth":
        "Reads the width from each connected segment and stacks them into an "
        "N_pulses × N_segments matrix.",
    "TotalSegmentWidth":
        "Computes the total span from the start of one segment to the end of "
        "another (end[segment 2] − start[segment 1]).",
    "ExtractSegmentInfo":
        "Extracts per-segment info (start/end values and width) from segment "
        "slices.",
    "RegionSelectionUI":
        "Interactive selection of analysis regions on the template data. "
        "Outputs the selected segments and their slices.",

    # --- Analysis ---------------------------------------------------------
    "MeanBlock":
        "Computes the mean of the connected inputs (column- or row-wise for "
        "matrices).",
    "WeightedMean":
        "Computes a weighted mean from connected segment data.",
    "DROverR":
        "Computes the relative resistance change dR/R from a baseline R and a "
        "change dR.",
    "DifferenceBlock":
        "Computes the difference of two inputs (in 1 − in 2), element-wise "
        "for vectors.",
    "SizeCalculation":
        "Calculates particle diameter from dR/R using the device geometry "
        "(effective diameter and channel length).",
    "DeformedDiameter":
        "Calculates the deformed diameter from a contact diameter and the "
        "channel width.",
    "Strain1D":
        "Computes 1-D strain from diameter and channel width.",
    "PoreDiameter":
        "Computes the pore diameter from dR/R and the channel dimensions "
        "(d, L).",
    "NodePoreVelocity":
        "Computes transit velocity through the node-pore segments from segment "
        "time and channel length. Input units (time, length) and output unit "
        "(m/mm/µm per s/ms) are selectable; per-segment.",
    "SegmentWidthToTime":
        "Converts a segment width matrix (samples) to transit time using the "
        "global sample rate. Output unit selectable (ms/s).",
    "WCDI":
        "Computes the weighted Cell Deformability Index from cell / node-pore "
        "velocities, diameter, and channel height.",
    "RecoveryAnalysis":
        "Analyzes post-pore recovery from reference/recovery deltas and "
        "segment times. Outputs recovery time, category, rate, and fit "
        "quality (R²).",
    "RecoveryInfo":
        "Extracts recovery information (R′ and segment times) from connected "
        "segment data.",
    "StatisticsBlock":
        "Computes summary statistics of the input data: mean, std, median, "
        "min, max, and count.",
    "ClusterBlock":
        "Clusters 2-D points (x, y) into groups and emits a per-point cluster "
        "label.",
    "AnalysisMethodSelector":
        "Interactive picker that selects and runs an analysis method on the "
        "extracted pulse features, using the template definition for context.",
    "DetectionAgreement":
        "Symmetrically compares two detectors' pulse lists on the same trace "
        "(neither treated as truth). Outputs an agreement score and a "
        "per-pulse match table.",

    # --- Utility ----------------------------------------------------------
    "PlotData":
        "Full-featured interactive 2-D plot with per-series styling, curve "
        "fits, annotations, and export. A terminal viewer.",
    "PlotVector":
        "Simple non-interactive plot of a 1-D vector.",
    "PlotSnippet":
        "Plots a short window of the signal around a chosen index (length n). "
        "A terminal viewer.",
    "CustomCode":
        "Runs arbitrary user Python code over its inputs and exposes the "
        "result(s) as outputs.",
    "GainBlock":
        "Scales its input(s) by a constant gain factor.",
    "ReciprocalBlock":
        "Computes the reciprocal (1 / x) of its input.",
    "UserChoiceBlock":
        "Prompts the user with a question and choices at run time, emitting "
        "the selected choice.",
    "RerouteNode":
        "Pass-through wire-routing helper for tidying connections; the output "
        "equals the input.",
}


def get_description(name):
    """Return the purpose text for a block definition name, or '' if none."""
    return BLOCK_DESCRIPTIONS.get(name, "")
