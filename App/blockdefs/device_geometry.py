def get_definition():
    return {
        "name": "DeviceGeometry",
        "displayName": "Device Geometry",
        "category": "DataIO",
        "color": [0.28, 0.62, 0.58],
        "inputs": [
            {"name": "geometry", "type": "any", "required": False,
             "description": "Load saved geometry struct"},
        ],
        "outputs": [
            {"name": "geometry", "type": "struct", "required": True,
             "description": "Device geometry struct"},
        ],
        "isInteractive": True,
        "noEditor": True,  # double-click / Run opens the DeviceGeometryDialog;
                           # the generic parameter editor isn't useful here.
        "isDeviceGeometry": True,  # adds an "Open Geometry UI" context-menu action
        "runner": "runDeviceGeometry",
        "defaultParameters": {
            "components": [
                {"type": "Node", "width": 85, "length": 120},
                {"type": "Pore", "width": 25, "length": 300, "property": "sizing"},
                {"type": "Node", "width": 85, "length": 50},
                {"type": "Pore", "width": 25, "length": 300, "property": "sizing"},
                {"type": "Node", "width": 85, "length": 50},
                {"type": "Pore", "width": 25, "length": 300, "property": "sizing"},
                {"type": "Node", "width": 85, "length": 50},
                {"type": "Pore", "width": 25, "length": 300, "property": "contraction"},
                {"type": "Node", "width": 85, "length": 50},
                {"type": "Pore", "width": 25, "length": 300, "property": "recovery"},
                {"type": "Node", "width": 85, "length": 50},
                {"type": "Pore", "width": 25, "length": 300, "property": "recovery"},
                {"type": "Node", "width": 85, "length": 50},
                {"type": "Pore", "width": 25, "length": 300, "property": "recovery"},
                {"type": "Node", "width": 85, "length": 120},
            ],
            "ch_height": 30,
            "De_np": 20,
        },
    }
