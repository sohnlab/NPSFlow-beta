"""Runner for DeviceGeometry block — opens a UI to define device geometry."""

import json
import os
from utils.paths import project_root
from utils.device_geometry_dialog import DeviceGeometryDialog


def run(inputs, params, block):
    # If geometry is provided via input port, use it as starting point
    loaded = inputs.get("geometry")
    if isinstance(loaded, str):
        # Treat as file path to a saved geometry JSON
        path = loaded.strip()
        if os.path.isfile(path):
            with open(path, "r") as f:
                loaded = json.load(f)
    if isinstance(loaded, dict) and "components" in loaded:
        # Geometry provided via input port — skip UI, pass through directly
        result = dict(loaded)
        block.parameters["components"] = result["components"]
        block.parameters["ch_height"] = result["ch_height"]
        block.parameters["De_np"] = result["De_np"]
        block.parameters["electrode_pairs"] = result.get("electrode_pairs", [])
        return {"geometry": result}

    init_params = dict(params)

    dialog = DeviceGeometryDialog(init_params)
    if dialog.exec():
        result = dialog.get_result()
        block.parameters["components"] = result["components"]
        block.parameters["ch_height"] = result["ch_height"]
        block.parameters["De_np"] = result["De_np"]
        block.parameters["electrode_pairs"] = result.get("electrode_pairs", [])
        # Auto-save to SavedTemplates/Geometry/temp.json
        try:
            root = project_root()
            folder = os.path.join(root, "SavedTemplates", "Geometry")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "temp.json"), "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        return {"geometry": result}
    else:
        return {"geometry": dict(params)}
