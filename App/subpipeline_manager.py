"""Sub-pipeline manager — create, load, and execute sub-pipeline templates."""

import json
import os
import uuid

import numpy as np
from block_node import BlockNode
from port import Port
from wire_connection import WireConnection
from utils.app_logger import logger


def _generate_sub_color(name):
    """Generate a unique color for a sub-pipeline based on its name."""
    import colorsys
    h = hash(name) % 360 / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.48, 0.65)
    return [round(r, 2), round(g, 2), round(b, 2)]


def _sub_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "subpipelines")


def list_templates():
    """Return list of (name, filepath) for all sub-pipeline templates."""
    d = _sub_dir()
    if not os.path.isdir(d):
        return []
    result = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json") and not fn.startswith("_"):
            path = os.path.join(d, fn)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                result.append((data.get("name", fn[:-5]), path))
            except Exception:
                pass
    return result


def load_template(path):
    """Load a sub-pipeline template from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_definition(template, filepath):
    """Build a synthetic block definition dict from a template."""
    name = f"SubPipeline_{template['name']}"
    inputs = [{"name": ei["name"], "type": ei.get("type", "any"),
               "required": ei.get("required", False)}
              for ei in template.get("exposedInputs", [])]
    outputs = [{"name": eo["name"], "type": eo.get("type", "any")}
               for eo in template.get("exposedOutputs", [])]
    return {
        "name": name,
        "displayName": template.get("displayName", template["name"]),
        "category": "SubPipeline",
        "color": template.get("color", _generate_sub_color(name)),
        "inputs": inputs,
        "outputs": outputs,
        "isSubPipeline": True,
        "subPipelineFile": filepath,
        "runner": None,
    }


def create_from_selection(blocks, wires, name, display_name=None):
    """Create a sub-pipeline template from selected blocks and their wires.

    Parameters
    ----------
    blocks : list[BlockNode]
        The selected blocks.
    wires : list[WireConnection]
        ALL wires in the workflow (not just internal ones).
    name : str
        Template name (used as filename).
    display_name : str, optional
        Display name shown on the block.

    Returns
    -------
    filepath : str
        Path to the saved template JSON file.
    """
    block_ids = {b.id for b in blocks}

    # Separate internal wires (both ends inside selection) from boundary wires
    internal_wires = []
    for w in wires:
        src_blk = w.source_port.parent_block if w.source_port else None
        dst_blk = w.dest_port.parent_block if w.dest_port else None
        if src_blk and dst_blk and src_blk.id in block_ids and dst_blk.id in block_ids:
            internal_wires.append(w)

    # Detect exposed inputs: input ports on inner blocks with no internal wire
    internal_dest_set = set()
    for w in internal_wires:
        if w.dest_port:
            internal_dest_set.add((w.dest_port.parent_block.id, w.dest_port.name))

    # Also track which ports have external wires (source outside selection)
    external_input_set = set()
    for w in wires:
        src_blk = w.source_port.parent_block if w.source_port else None
        dst_blk = w.dest_port.parent_block if w.dest_port else None
        if (src_blk and dst_blk
                and src_blk.id not in block_ids
                and dst_blk.id in block_ids):
            external_input_set.add((dst_blk.id, w.dest_port.name))

    exposed_inputs = []
    seen_input_names = set()
    for b in blocks:
        for p in b.input_ports:
            if p.name in ("addInput",):
                continue
            key = (b.id, p.name)
            # Expose if: has external wire, OR has no internal wire and is connected
            if key in external_input_set or (key not in internal_dest_set and p.is_connected):
                ename = p.display_name or p.name
                # Ensure unique name
                base = ename
                i = 1
                while ename in seen_input_names:
                    ename = f"{base}_{i}"
                    i += 1
                seen_input_names.add(ename)
                exposed_inputs.append({
                    "name": ename,
                    "innerBlock": b.id,
                    "innerPort": p.name,
                    "type": p.type or "any",
                })

    # Detect exposed outputs: output ports on inner blocks with external wires
    internal_src_set = set()
    for w in internal_wires:
        if w.source_port:
            internal_src_set.add((w.source_port.parent_block.id, w.source_port.name))

    exposed_outputs = []
    seen_output_names = set()
    for b in blocks:
        for p in b.output_ports:
            # Check if this port has any wire going outside the selection
            has_external = False
            for w in wires:
                if (w.source_port and w.source_port.parent_block is b
                        and w.source_port.name == p.name
                        and w.dest_port and w.dest_port.parent_block
                        and w.dest_port.parent_block.id not in block_ids):
                    has_external = True
                    break
            if has_external:
                ename = p.display_name or p.name
                base = ename
                i = 1
                while ename in seen_output_names:
                    ename = f"{base}_{i}"
                    i += 1
                seen_output_names.add(ename)
                exposed_outputs.append({
                    "name": ename,
                    "innerBlock": b.id,
                    "innerPort": p.name,
                    "type": p.type or "any",
                })

    # Serialize inner blocks and wires
    inner_blocks_data = [b.to_struct() for b in blocks]
    inner_wires_data = [w.to_struct() for w in internal_wires]

    template = {
        "name": name,
        "displayName": display_name or name,
        "color": [0.4, 0.6, 0.8],
        "exposedInputs": exposed_inputs,
        "exposedOutputs": exposed_outputs,
        "innerBlocks": inner_blocks_data,
        "innerWires": inner_wires_data,
    }

    d = _sub_dir()
    os.makedirs(d, exist_ok=True)
    filepath = os.path.join(d, f"{name}.json")
    with open(filepath, "w") as f:
        json.dump(template, f, indent=2)

    logger.info(f"Sub-pipeline saved: {filepath}")
    return filepath


def instantiate_inner_graph(template, registry):
    """Create temporary BlockNode and WireConnection objects from a template.

    Returns
    -------
    inner_blocks : list[BlockNode]
    inner_wires : list[WireConnection]
    id_map : dict
        Mapping from template block IDs to new instance IDs.
    block_map : dict
        Mapping from new block ID to BlockNode.
    """
    id_map = {}  # old_id -> new_id
    block_map = {}  # new_id -> BlockNode

    for bd in template["innerBlocks"]:
        old_id = bd["id"]
        def_name = bd["definition"]

        block = BlockNode(def_name, tuple(bd.get("position", [0, 0])))
        if registry.has(def_name):
            block.init_from_definition(registry.get(def_name))
        # Restore parameters
        block.parameters.update(bd.get("parameters", {}))
        block.display_name = bd.get("displayName", def_name)
        block.size = tuple(bd.get("size", [140, 60]))
        block.color = tuple(bd.get("color", [0.5, 0.5, 0.5]))

        # Restore port definitions if saved
        if "inputPortDefs" in block.parameters:
            block.input_ports = []
            for i, pd in enumerate(block.parameters["inputPortDefs"]):
                p = Port(
                    name=pd.get("name", ""),
                    direction="input",
                    type_=pd.get("type", "any"),
                    required=pd.get("required", False),
                    description=pd.get("description", ""),
                )
                p.display_name = pd.get("displayName", p.name)
                p.parent_block = block
                p.index = i + 1
                block.input_ports.append(p)

        if "outputPortDefs" in block.parameters:
            block.output_ports = []
            for i, pd in enumerate(block.parameters["outputPortDefs"]):
                p = Port(
                    name=pd.get("name", ""),
                    direction="output",
                    type_=pd.get("type", "any"),
                    required=True,
                    description=pd.get("description", ""),
                )
                p.display_name = pd.get("displayName", p.name)
                p.parent_block = block
                p.index = i + 1
                block.output_ports.append(p)

        new_id = f"blk_{uuid.uuid4()}"
        block.id = new_id
        id_map[old_id] = new_id
        block_map[new_id] = block

    # Build wires using remapped IDs
    inner_blocks = list(block_map.values())
    inner_wires = []
    for wd in template["innerWires"]:
        src_id = id_map.get(wd.get("sourceBlock"))
        dst_id = id_map.get(wd.get("destBlock"))
        if not src_id or not dst_id:
            continue
        src_block = block_map[src_id]
        dst_block = block_map[dst_id]
        src_port_name = wd.get("sourcePort")
        dst_port_name = wd.get("destPort")
        src_port = next((p for p in src_block.output_ports if p.name == src_port_name), None)
        dst_port = next((p for p in dst_block.input_ports if p.name == dst_port_name), None)
        if src_port and dst_port:
            wire = WireConnection(src_port, dst_port)
            src_port.connections.append(wire)
            dst_port.connections.append(wire)
            inner_wires.append(wire)

    return inner_blocks, inner_wires, id_map, block_map
