"""ExportAcceptedPulses - Export pulse data to file.

Translates MATLAB ExportAcceptedPulses.m: structured export of accepted
pulse data with metadata to .mat, .npz, or .csv format, using a file
dialog for destination selection.
"""

import os
import numpy as np


def export_accepted_pulses(accepted_pulses, sample_rate=None,
                            pulse_features=None, segment_info=None,
                            template_data=None):
    """Export accepted pulse data to file with metadata.

    Parameters
    ----------
    accepted_pulses : list or np.ndarray or dict
        Accepted pulse arrays.  If dict, expects key 'accepted_pulses'.
    sample_rate : float, optional
        Sampling rate in Hz.
    pulse_features : dict, optional
        Per-pulse feature arrays (mean, std, area, etc.).
    segment_info : dict, optional
        Segment boundary/label information.
    template_data : dict, optional
        Template-related data (mean template, rectangularized, peaks).

    Returns
    -------
    filepath : str or None
        Path to the saved file, or None if cancelled.
    """
    # Try PySide6 dialog first, fall back to tkinter, then to terminal input
    filepath = _get_save_path()
    if not filepath:
        from utils.app_logger import logger
        logger.info("Export cancelled.")
        return None

    # Normalize input
    if isinstance(accepted_pulses, dict):
        pulse_dict = accepted_pulses
        pulses = pulse_dict.get('accepted_pulses', pulse_dict.get('pulses', []))
        if sample_rate is None:
            sample_rate = pulse_dict.get('sample_rate', None)
        if pulse_features is None:
            pulse_features = pulse_dict.get('features', None)
        if segment_info is None:
            segment_info = pulse_dict.get('segment_info', None)
        if template_data is None:
            template_data = pulse_dict.get('template_data', None)
    elif isinstance(accepted_pulses, np.ndarray):
        if accepted_pulses.ndim == 1:
            pulses = [accepted_pulses]
        else:
            pulses = [accepted_pulses[i] for i in range(len(accepted_pulses))]
    elif isinstance(accepted_pulses, list):
        pulses = accepted_pulses
    else:
        pulses = [np.atleast_1d(accepted_pulses)]

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.mat':
        _export_mat(filepath, pulses, sample_rate, pulse_features,
                     segment_info, template_data)
    elif ext == '.npz':
        _export_npz(filepath, pulses, sample_rate, pulse_features,
                     segment_info, template_data)
    else:
        _export_csv(filepath, pulses, sample_rate)

    n_pulses = len(pulses) if pulses else 0
    from utils.app_logger import logger
    logger.info(f"Exported {n_pulses} pulses to {filepath}")
    return filepath


def _get_save_path():
    """Open a file-save dialog. Try PySide6, then tkinter, then stdin."""
    # Try PySide6
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        filepath, _ = QFileDialog.getSaveFileName(
            None, "Export Accepted Pulses", "",
            "MAT Files (*.mat);;NPZ Files (*.npz);;CSV Files (*.csv);;All Files (*)")
        return filepath if filepath else None
    except ImportError:
        pass

    # Try tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.asksaveasfilename(
            title="Export Accepted Pulses",
            filetypes=[("MAT Files", "*.mat"), ("NPZ Files", "*.npz"),
                       ("CSV Files", "*.csv"), ("All Files", "*.*")])
        root.destroy()
        return filepath if filepath else None
    except Exception:
        pass

    # Fallback to terminal
    filepath = input("Enter export file path (or empty to cancel): ").strip()
    return filepath if filepath else None


def _export_mat(filepath, pulses, sample_rate, pulse_features,
                 segment_info, template_data):
    """Export to MATLAB .mat format."""
    from scipy.io import savemat

    save_dict = {}

    # Pulse data - pad to uniform length
    if pulses:
        max_len = max(len(np.atleast_1d(p)) for p in pulses)
        padded = np.full((len(pulses), max_len), np.nan)
        for i, p in enumerate(pulses):
            p = np.atleast_1d(p).ravel()
            padded[i, :len(p)] = p
        save_dict['accepted_pulses'] = padded
        save_dict['n_pulses'] = len(pulses)
    else:
        save_dict['accepted_pulses'] = np.array([])
        save_dict['n_pulses'] = 0

    if sample_rate is not None:
        save_dict['sample_rate'] = float(sample_rate)

    if pulse_features is not None and isinstance(pulse_features, dict):
        for key, val in pulse_features.items():
            save_dict[f'feature_{key}'] = np.atleast_1d(val)

    if segment_info is not None and isinstance(segment_info, dict):
        for key, val in segment_info.items():
            if isinstance(val, np.ndarray):
                save_dict[f'segment_{key}'] = val
            elif isinstance(val, list):
                # String lists need special handling for MATLAB
                if val and isinstance(val[0], str):
                    save_dict[f'segment_{key}'] = np.array(val, dtype=object)
                else:
                    save_dict[f'segment_{key}'] = np.array(val)

    if template_data is not None and isinstance(template_data, dict):
        for key in ['template', 'pulse_template_rec', 'peak_locations']:
            if key in template_data:
                save_dict[f'template_{key}'] = np.atleast_1d(template_data[key])

    savemat(filepath, save_dict, do_compression=True)


def _export_npz(filepath, pulses, sample_rate, pulse_features,
                 segment_info, template_data):
    """Export to compressed NumPy .npz format."""
    save_dict = {}

    if pulses:
        max_len = max(len(np.atleast_1d(p)) for p in pulses)
        padded = np.full((len(pulses), max_len), np.nan)
        for i, p in enumerate(pulses):
            p = np.atleast_1d(p).ravel()
            padded[i, :len(p)] = p
        save_dict['accepted_pulses'] = padded

    if sample_rate is not None:
        save_dict['sample_rate'] = np.array([float(sample_rate)])

    if pulse_features is not None and isinstance(pulse_features, dict):
        for key, val in pulse_features.items():
            save_dict[f'feature_{key}'] = np.atleast_1d(val)

    if segment_info is not None and isinstance(segment_info, dict):
        for key, val in segment_info.items():
            if isinstance(val, (np.ndarray, list)):
                save_dict[f'segment_{key}'] = np.atleast_1d(val)

    if template_data is not None and isinstance(template_data, dict):
        for key in ['template', 'pulse_template_rec', 'peak_locations']:
            if key in template_data:
                save_dict[f'template_{key}'] = np.atleast_1d(template_data[key])

    np.savez_compressed(filepath, **save_dict)


def _export_csv(filepath, pulses, sample_rate):
    """Export to CSV format (each row = one pulse)."""
    if not pulses:
        np.savetxt(filepath, np.array([]), delimiter=',')
        return

    max_len = max(len(np.atleast_1d(p)) for p in pulses)
    padded = np.full((len(pulses), max_len), np.nan)
    for i, p in enumerate(pulses):
        p = np.atleast_1d(p).ravel()
        padded[i, :len(p)] = p

    # Header
    if sample_rate is not None:
        time_cols = [f'{j / sample_rate * 1000:.4f}ms' for j in range(max_len)]
        header = ','.join(time_cols)
    else:
        header = ','.join([f'sample_{j}' for j in range(max_len)])

    np.savetxt(filepath, padded, delimiter=',', header=header, comments='')
