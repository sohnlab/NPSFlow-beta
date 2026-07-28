"""Block definition for Wavelet Editor.

See docs/superpowers/specs/2026-04-24-wavelet-editor-design.md
"""


def get_definition():
    return {
        "name": "WaveletEditor",
        "displayName": "Wavelet Editor",
        "category": "Template",
        "color": [0.45, 0.35, 0.75],
        "inputs": [
            {"name": "TPsettings", "type": "struct", "required": True},
            {"name": "loadFile",   "type": "string", "required": False,
             "description":
               "Path to a saved wavelet-editor settings JSON. Relative paths "
               "resolve from project root; bare names resolve to "
               "SavedTemplates/Wavelet/<name>.json."},
        ],
        "outputs": [
            {"name": "wavelet",  "type": "numeric"},
            {"name": "segments", "type": "struct"},
        ],
        "isInteractive": True,
        "runner": "runWaveletEditor",
    }
