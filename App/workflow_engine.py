"""WorkflowEngine - Executes the block pipeline in topological order."""

import os
import uuid
import traceback
import importlib

# Max columns of a 2-D array exposed as individual Unpack output ports.
_MAX_UNPACK_COLS = 64


def _flatten_struct(d, max_depth, separator=".", prefix="", current_depth=0):
    """Flatten a (possibly nested) dict into [(flat_key, value), ...] up
    to `max_depth` levels. Stops descending at `max_depth` even if the
    value at that depth is still a dict.

    max_depth=1: top-level keys only (no flattening — current behavior).
    max_depth=2: also flatten one level of nested dicts.
    Etc.
    """
    if not isinstance(d, dict):
        return [(prefix, d)] if prefix else [("value", d)]
    if current_depth >= max_depth:
        return [(prefix, d)] if prefix else [("value", d)]
    out = []
    for k, v in d.items():
        flat = f"{prefix}{separator}{k}" if prefix else str(k)
        if isinstance(v, dict) and current_depth + 1 < max_depth:
            nested = _flatten_struct(v, max_depth, separator, flat,
                                     current_depth + 1)
            if nested:
                out.extend(nested)
            else:
                out.append((flat, v))
        else:
            out.append((flat, v))
    return out


def _enumerate_keys_by_depth(d, max_depth=10, separator="."):
    """Return {depth: [flat_keys]} for every depth in [1, max_depth].

    Used by the edit-Unpack dialog so the user can preview which fields
    appear at each depth without rerunning the pipeline.
    """
    by_depth = {}
    for depth in range(1, max_depth + 1):
        keys = [k for k, _ in _flatten_struct(d, depth, separator)]
        by_depth[depth] = keys
        if depth > 1 and by_depth.get(depth) == by_depth.get(depth - 1):
            break
    return by_depth


def _shape_node(val, max_depth=10, current_depth=0):
    """Return a JSON-friendly description of `val`'s shape (type, size,
    and child shapes if applicable). Used by the Unpack edit dialog to
    render a structure tree without storing the data itself.

    Descends into:
      - dicts (each key becomes a child)
      - lists/tuples whose first element is a dict or ndarray
        (each index becomes a child labeled "[i]")
    Stops at depth limit or at scalar/atomic types.

    Each node is a dict with keys:
      type:     str (e.g. "dict", "list", "ndarray (float64)", "int")
      size:     str (e.g. "3", "1024x2", "1x1")
      children: dict[str, node] | None    (key for dict children;
                                           "[i]" for list children)
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(val, dict):
        node = {"type": "dict", "size": str(len(val)), "children": None}
        if current_depth < max_depth:
            node["children"] = {
                str(k): _shape_node(v, max_depth, current_depth + 1)
                for k, v in val.items()
            }
        return node
    if _np is not None and isinstance(val, _np.ndarray):
        node = {"type": f"ndarray ({val.dtype})",
                "size": "x".join(str(s) for s in val.shape),
                "children": None}
        # 2-D arrays: expose each column as a selectable child so the Unpack
        # editor can split e.g. multi-zone data into per-column output ports.
        if val.ndim == 2 and current_depth < max_depth:
            n_cols = int(val.shape[1])
            if 0 < n_cols <= _MAX_UNPACK_COLS:
                node["children"] = {
                    f"col{j}": {"type": f"ndarray ({val.dtype})",
                                "size": str(int(val.shape[0])),
                                "children": None}
                    for j in range(n_cols)
                }
        return node
    if isinstance(val, (list, tuple)):
        node = {"type": type(val).__name__,
                "size": str(len(val)),
                "children": None}
        # Expand lists whose first element has nested structure (mirrors
        # Data Inspector behavior — avoids expanding [1, 2, 3]).
        nestable = (dict, _np.ndarray) if _np is not None else (dict,)
        if val and current_depth < max_depth and isinstance(val[0], nestable):
            node["children"] = {
                f"[{i}]": _shape_node(sub, max_depth, current_depth + 1)
                for i, sub in enumerate(val)
            }
        return node
    return {"type": type(val).__name__,
            "size": "1x1",
            "children": None}


def _resolve_path(d, path, separator="."):
    """Walk a dotted path through a (possibly nested) dict / list and
    return the value at that path, or raise KeyError.

    Path segments:
      "name"   → dict key lookup (parent must be a dict)
      "[i]"    → list index lookup (parent must be a list/tuple)
      "colN"   → column N of a 2-D ndarray (parent must be an ndarray)

    Examples:
      "pulse"               → d["pulse"]
      "value.[0]"           → d["value"][0]
      "value.[0].data"      → d["value"][0]["data"]
      "value.col2"          → d["value"][:, 2]
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    cur = d
    for part in str(path).split(separator):
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            try:
                idx = int(part[1:-1])
            except ValueError:
                raise KeyError(path)
            if isinstance(cur, (list, tuple)) and 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                raise KeyError(path)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif (_np is not None and isinstance(cur, _np.ndarray)
              and part.startswith("col") and part[3:].isdigit()):
            idx = int(part[3:])
            if cur.ndim == 2 and 0 <= idx < cur.shape[1]:
                cur = cur[:, idx]
            else:
                raise KeyError(path)
        else:
            raise KeyError(path)
    return cur


class WorkflowEngine:
    """Manages the execution of blocks: gathers inputs from upstream
    outputs, calls the wrapped function, and stores results.
    """

    def __init__(self, registry=None):
        self.registry = registry
        self.status_callback = lambda block: None
        self.log_callback = lambda msg: None
        self.debug_log_callback = lambda msg: None
        self.wire_remove_callback = lambda wire: None
        self.stop_requested = False
        self.reference_store = {}
        self.data_bus_store = {}
        self.global_vars = {}
        self.all_blocks = []
        self._running_blocks = set()  # guard against re-entrant run_block
        self._in_pipeline_run = False  # True during run_all (strict sequential mode)
        self._reset_ids = set()  # blocks reset in current pipeline run
        # True during the smart run_to_block: reuse `done` blocks as-is instead
        # of refreshing them as stale side-chains (which would cascade upstream).
        self._reuse_done_blocks = False
        # Debug stepping
        self.debug_mode = False
        self.debug_pause_callback = None  # called before each block; should block until user acts
        self._debug_continue = False  # set True to skip pausing for remaining blocks

    def topological_sort(self, blocks, wires):
        """DFS-based topological sort with Sequence-aware ordering."""
        n = len(blocks)
        if n == 0:
            return []

        block_map = {blk.id: blk for blk in blocks}

        # Build adjacency: for each block, which blocks depend on it?
        # adj[block_id] = [(dest_block_id, source_port_index)]
        adj = {blk.id: [] for blk in blocks}
        for wire in wires:
            if (wire.source_port and wire.dest_port and
                    wire.source_port.parent_block and wire.dest_port.parent_block):
                src_id = wire.source_port.parent_block.id
                dst_id = wire.dest_port.parent_block.id
                if src_id in adj:
                    port_idx = wire.source_port.index
                    adj[src_id].append((dst_id, port_idx))

        # Implicit ordering edges: a Data Bus that WRITES a pool variable must
        # run before a Data Bus that READS it (they share a pool, not a wire,
        # so the data flow is invisible to the sort otherwise).
        self._add_databus_pool_edges(adj, blocks, wires)

        # Identify Sequence/Priority blocks
        seq_blocks = set()
        for blk in blocks:
            if blk.definition_name and self.registry and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isPriority") or defn.get("isSequence"):
                    seq_blocks.add(blk.id)

        visited = set()
        order = []

        def dfs(block_id):
            if block_id in visited:
                return
            visited.add(block_id)

            neighbors = adj.get(block_id, [])

            if block_id in seq_blocks:
                # Group neighbors by source port index, process each group fully
                port_groups = {}
                for dst_id, port_idx in neighbors:
                    if port_idx not in port_groups:
                        port_groups[port_idx] = []
                    port_groups[port_idx].append(dst_id)

                for port_idx in sorted(port_groups.keys(), reverse=True):
                    for dst_id in port_groups[port_idx]:
                        dfs(dst_id)
            else:
                for dst_id, _ in neighbors:
                    dfs(dst_id)

            order.append(block_id)

        # Find root blocks (no incoming edges). Derive from adj so the implicit
        # Data Bus pool edges count too — a pool reader must not be treated as a
        # root when a writer precedes it.
        has_incoming = set()
        for src_id, neighbors in adj.items():
            for dst_id, _ in neighbors:
                has_incoming.add(dst_id)

        roots = [blk.id for blk in blocks if blk.id not in has_incoming]
        if not roots:
            roots = [blocks[0].id]

        for root_id in roots:
            dfs(root_id)

        order.reverse()
        return [block_map[bid] for bid in order if bid in block_map]

    def _add_databus_pool_edges(self, adj, blocks, wires):
        """Add implicit edges so a Data Bus that WRITES a pool variable is
        ordered before any Data Bus that READS the same variable in the same
        pool. They communicate through the pool, not a wire, so without this
        the reader (and its downstream consumers) can run before the writer and
        find an empty pool.
        """
        if not self.registry:
            return
        # Input ports that are actually wired (i.e. genuinely written).
        connected_in = set()
        for w in wires:
            if w.dest_port and w.dest_port.parent_block:
                connected_in.add((w.dest_port.parent_block.id, w.dest_port.name))

        writers = {}   # pool -> {label: [block_id, ...]}
        readers = {}   # pool -> {label: [block_id, ...]}
        for blk in blocks:
            if not blk.definition_name or not self.registry.has(blk.definition_name):
                continue
            if not self.registry.get(blk.definition_name).get("isDataBus"):
                continue
            pool = blk.parameters.get("pool", "Default")
            for p in blk.input_ports:
                if p.name in ("Run", "addInput"):
                    continue
                if (blk.id, p.name) in connected_in:
                    writers.setdefault(pool, {}).setdefault(
                        p.display_name, []).append(blk.id)
            for p in blk.output_ports:
                if p.name == "Run":
                    continue
                label = p.description if p.description else p.display_name
                readers.setdefault(pool, {}).setdefault(label, []).append(blk.id)

        for pool, label_writers in writers.items():
            pool_readers = readers.get(pool, {})
            for label, w_ids in label_writers.items():
                for r_id in pool_readers.get(label, []):
                    for w_id in w_ids:
                        if w_id != r_id and w_id in adj:
                            adj[w_id].append((r_id, 0))

    def _warn_duplicate_pool_writers(self, blocks, wires):
        """Warn when one pool variable has multiple wired Data Bus writers —
        their values silently overwrite each other, so readers get whichever
        writer's source was applied last."""
        if not self.registry:
            return
        connected = set()
        for w in wires:
            if w.dest_port and w.dest_port.parent_block:
                connected.add((w.dest_port.parent_block.id, w.dest_port.name))
        writers = {}
        for blk in blocks:
            if not blk.definition_name or not self.registry.has(blk.definition_name):
                continue
            if not self.registry.get(blk.definition_name).get("isDataBus"):
                continue
            pool = blk.parameters.get("pool", "Default")
            for p in blk.input_ports:
                if p.name in ("Run", "addInput"):
                    continue
                if (blk.id, p.name) in connected:
                    writers.setdefault((pool, p.display_name), []).append(blk)
        for (pool, label), blks in writers.items():
            if len(blks) > 1:
                names = ", ".join(b.display_name for b in blks)
                self.log_callback(
                    f"Warning: Data Bus variable '{label}' in pool '{pool}' "
                    f"has {len(blks)} writers ({names}) — they overwrite "
                    f"each other; rename one to keep the values distinct.")

    def _find_reachable_from_start(self, blocks, wires):
        """Return the set of block IDs reachable from the Start block via wires."""
        # Build adjacency from wires (source -> dest)
        adj = {blk.id: [] for blk in blocks}
        for wire in wires:
            if (wire.source_port and wire.dest_port and
                    wire.source_port.parent_block and wire.dest_port.parent_block):
                src_id = wire.source_port.parent_block.id
                dst_id = wire.dest_port.parent_block.id
                if src_id in adj:
                    adj[src_id].append(dst_id)

        # Find start block
        start_id = None
        for blk in blocks:
            if blk.definition_name and self.registry and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isStart"):
                    start_id = blk.id
                    break

        if start_id is None:
            # No start block found — fall back to all blocks
            return {blk.id for blk in blocks}

        # BFS from start block
        reachable = set()
        queue = [start_id]
        while queue:
            bid = queue.pop(0)
            if bid in reachable:
                continue
            reachable.add(bid)
            for neighbor in adj.get(bid, []):
                if neighbor not in reachable:
                    queue.append(neighbor)

        return reachable

    def _global_setter_ids(self, blocks):
        """IDs of blocks that publish an engine-wide global (isGlobalSampleRate).
        They must run in Phase 1 regardless of wiring — the whole pipeline reads
        the global they set, so dropping to Phase 2 would publish it too late."""
        ids = set()
        for blk in blocks:
            if blk.definition_name and self.registry and \
               self.registry.has(blk.definition_name) and \
               self.registry.get(blk.definition_name).get("isGlobalSampleRate"):
                ids.add(blk.id)
        return ids

    def _promote_global_setters(self, sorted_blocks, wires, gsr_ids):
        """Pull each GlobalSampleRate block forward to run just after its OWN
        last upstream dependency (or first if it has none). Done per-GSR — not
        lumped together — so an early file-fed base-rate GSR publishes before
        early consumers (e.g. Data Summary) while a Preprocess-fed effective-rate
        GSR still runs after Preprocess. Safe: a GSR's dependents are downstream,
        so moving it no earlier than its own deps never breaks an edge. Returns a
        reordered list (does not mutate the input)."""
        if not gsr_ids:
            return sorted_blocks
        dep_map = {}  # block_id -> set of direct upstream block ids
        for w in wires:
            if (w.source_port and w.dest_port and
                    w.source_port.parent_block and w.dest_port.parent_block):
                dep_map.setdefault(w.dest_port.parent_block.id, set()).add(
                    w.source_port.parent_block.id)

        def upstream_of(bid):
            seen, stack = set(), [bid]
            while stack:
                for p in dep_map.get(stack.pop(), ()):
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
            return seen

        order = list(sorted_blocks)
        for gid in [b.id for b in order if b.id in gsr_ids]:
            ids = [b.id for b in order]
            cur = ids.index(gid)
            up = upstream_of(gid)
            last_up = max((ids.index(u) for u in up if u in ids), default=-1)
            target = last_up + 1
            if target < cur:
                order.insert(target, order.pop(cur))
        return order

    @staticmethod
    def _as_scalar(v):
        """Coerce a possibly array-valued input to a plain float, or None if
        empty/non-numeric. File sample rates load as (1,1)/0-d arrays, which
        would otherwise break `f"{rate:g}"` formatting and rate arithmetic."""
        if v is None:
            return None
        import numpy as np
        try:
            arr = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
        if arr.size == 0 or not np.isfinite(arr[0]):
            return None
        return float(arr[0])

    def run_all(self, blocks, wires, keep_interactive=False):
        """Execute the main pipeline, then disconnected chains.

        Phase 1: Run blocks reachable (downstream) from the Start block in
        topological order.  Side-chain blocks (wired to the main component
        but upstream-only, e.g. Text constants) are auto-run on demand by
        ``_gather_inputs`` when a downstream block needs their data.

        Phase 2: Find truly disconnected chains — connected components that
        share NO wire with the Start block's component — and run each in
        left-to-right, top-to-bottom spatial order, finishing one chain
        before starting the next.  An error stops that chain, then moves on.

        Parameters
        ----------
        keep_interactive : bool
            If True, interactive blocks that already completed successfully
            (status "done" with output data) are NOT reset — their cached
            results are reused.
        """
        self.stop_requested = False
        self.reference_store = {}
        self.data_bus_store = {}
        self.global_vars = {}
        self._running_blocks = set()
        self._in_pipeline_run = True
        self._reset_ids = set()  # track blocks reset in this run
        self.all_blocks = list(blocks)

        # Reuse-settings mode: when the Load File path port is supplied, every
        # block reuses its last-saved settings; otherwise all run fresh.
        self._apply_reuse_mode(blocks, wires)

        # Clear auto-saved temp states so interactive blocks start fresh
        self._clear_saved_states()

        # Reset port data flags
        self.reset_port_data(blocks)
        self.sync_toggle_outputs(blocks)
        self._warn_duplicate_pool_writers(blocks, wires)

        # Find the Start block's connected component (undirected wire graph).
        # This includes ALL blocks wired to it — both downstream (reachable)
        # and upstream-only (side-chains like Text constants).
        main_component = self._find_start_component(blocks, wires)

        # Find blocks reachable downstream from Start (directed)
        reachable = self._find_reachable_from_start(blocks, wires)

        # GlobalSampleRate blocks publish a global the whole pipeline reads, so
        # force them into Phase 1 even when their inputs are unwired or dangling
        # (otherwise they drop to Phase 2 and publish after their consumers have
        # already run with a missing rate). The reorder below still runs them
        # after any block that feeds them.
        gsr_ids = self._global_setter_ids(blocks)
        reachable |= gsr_ids

        # Reset only the main component blocks; side-chains are reset too so
        # _gather_inputs can auto-run them fresh.  But only reachable blocks
        # go into _reset_ids (which controls the "don't auto-run pending
        # blocks that are scheduled later in topo order" guard).
        for blk in blocks:
            if blk.id not in main_component and blk.id not in gsr_ids:
                continue
            if blk.status == "skipped":
                continue
            if keep_interactive and blk.status == "done" and blk.output_data:
                if blk.definition_name and self.registry and \
                   self.registry.has(blk.definition_name):
                    defn = self.registry.get(blk.definition_name)
                    if defn.get("isInteractive"):
                        for p in blk.output_ports:
                            if p.name in blk.output_data:
                                p.has_data = True
                        continue
            blk.status = "pending"
            blk.output_data = {}
            if blk.id in reachable:
                self._reset_ids.add(blk.id)

        try:
            # --- Phase 1: run Start-connected pipeline ---
            sorted_blocks = self.topological_sort(blocks, wires)

            # Pull each GlobalSampleRate block forward to run just after its own
            # upstream dependencies (see _promote_global_setters). Done per-GSR,
            # not lumped, so an early file-fed base-rate GSR publishes before
            # early consumers (e.g. Data Summary) while a Preprocess-fed
            # effective-rate GSR still runs after Preprocess.
            sorted_blocks = self._promote_global_setters(
                sorted_blocks, wires, gsr_ids)

            for block in sorted_blocks:
                if self.stop_requested:
                    self.log_callback("Execution stopped by user.")
                    return
                if block.id not in reachable:
                    continue
                if block.status == "done":
                    continue
                self.run_block(block, wires)
                if block.status == "error":
                    self.log_callback(
                        f"Pipeline stopped: error in '{block.display_name}'.")
                    return

            # --- Phase 2: run truly disconnected chains ---
            # GlobalSampleRate blocks were forced into Phase 1 above; exclude
            # them here so they don't run a second time.
            disconnected = [b for b in blocks
                            if b.id not in main_component and b.id not in gsr_ids]
            if not disconnected:
                return

            chains = self._find_disconnected_chains(disconnected, wires)

            # Sort chains spatially: left-to-right, then top-to-bottom
            def chain_sort_key(chain):
                min_x = min(b.position[0] for b in chain)
                max_y = max(b.position[1] for b in chain)
                return (min_x, -max_y)

            chains.sort(key=chain_sort_key)

            for chain in chains:
                if self.stop_requested:
                    self.log_callback("Execution stopped by user.")
                    return

                # Reset chain blocks now (not earlier, to keep them
                # isolated from Phase 1's _reset_ids logic)
                for blk in chain:
                    if blk.status == "skipped":
                        continue
                    if keep_interactive and blk.status == "done" and blk.output_data:
                        if blk.definition_name and self.registry and \
                           self.registry.has(blk.definition_name):
                            defn = self.registry.get(blk.definition_name)
                            if defn.get("isInteractive"):
                                for p in blk.output_ports:
                                    if p.name in blk.output_data:
                                        p.has_data = True
                                continue
                    blk.status = "pending"
                    blk.output_data = {}

                chain_sorted = self.topological_sort(chain, wires)
                for block in chain_sorted:
                    if self.stop_requested:
                        self.log_callback("Execution stopped by user.")
                        return
                    if block.status in ("done", "skipped"):
                        continue
                    self.run_block(block, wires)
                    if block.status == "error":
                        self.log_callback(
                            f"Pipeline stopped: error in '{block.display_name}'.")
                        return
        finally:
            self._in_pipeline_run = False

    def _clear_saved_states(self):
        """Remove auto-saved temp.json files so interactive blocks start fresh."""
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for subfolder in ("Detection", "Review", "Processing"):
            path = os.path.join(root, "SavedTemplates", subfolder, "temp.json")
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

    def _reuse_settings_mode(self, blocks):
        """True when the Load File block's ``path`` port is supplied a
        non-empty value (a wired path) — the global "reuse last settings"
        switch.  False when the port is blank (data is picked interactively),
        so every block runs fresh.  Only the LoadData block drives this.
        """
        for blk in blocks:
            if blk.definition_name != "LoadData":
                continue
            port = next((p for p in blk.input_ports if p.name == "path"), None)
            if port and port.is_connected:
                src_port = port.connections[0].source_port
                src = src_port.parent_block if src_port else None
                if src is not None:
                    # A Text constant must carry a non-empty string; any other
                    # source (a computed path) counts as supplied.
                    if src.definition_name == "TextBlock":
                        val = src.parameters.get("valueString", "")
                        if isinstance(val, str) and val.strip():
                            return True
                    else:
                        return True
                continue
            # Unwired path port: the Load File picker drives the mode. Its last
            # recent-vs-new choice is persisted on the block, so reuse mode is
            # known at run start even when the picker doesn't re-open this run.
            if bool(getattr(blk, "parameters", {}).get("reuseSettings")):
                return True
        return False

    def _apply_reuse_mode(self, blocks, wires):
        """Compute reuse-settings mode for this run and publish it to the
        shared settings-input resolver."""
        try:
            from utils.settings_input import set_reuse_when_empty
            set_reuse_when_empty(self._reuse_settings_mode(blocks))
        except Exception:
            pass

    def _find_start_component(self, blocks, wires):
        """Return the set of block IDs in the Start block's undirected connected component."""
        block_ids = {b.id for b in blocks}

        # Build undirected adjacency
        adj = {bid: set() for bid in block_ids}
        for wire in wires:
            if (wire.source_port and wire.dest_port and
                    wire.source_port.parent_block and wire.dest_port.parent_block):
                src_id = wire.source_port.parent_block.id
                dst_id = wire.dest_port.parent_block.id
                if src_id in adj and dst_id in adj:
                    adj[src_id].add(dst_id)
                    adj[dst_id].add(src_id)

        # Find start block
        start_id = None
        for blk in blocks:
            if blk.definition_name and self.registry and self.registry.has(blk.definition_name):
                defn = self.registry.get(blk.definition_name)
                if defn.get("isStart"):
                    start_id = blk.id
                    break

        if start_id is None:
            return block_ids  # no Start block — treat everything as main

        # BFS undirected from start
        visited = set()
        queue = [start_id]
        while queue:
            bid = queue.pop(0)
            if bid in visited:
                continue
            visited.add(bid)
            for neighbor in adj[bid]:
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited

    def _find_disconnected_chains(self, blocks, wires):
        """Group blocks into connected components via wires."""
        block_ids = {b.id for b in blocks}
        block_map = {b.id: b for b in blocks}

        # Build undirected adjacency among these blocks
        adj = {bid: set() for bid in block_ids}
        for wire in wires:
            if (wire.source_port and wire.dest_port and
                    wire.source_port.parent_block and wire.dest_port.parent_block):
                src_id = wire.source_port.parent_block.id
                dst_id = wire.dest_port.parent_block.id
                if src_id in block_ids and dst_id in block_ids:
                    adj[src_id].add(dst_id)
                    adj[dst_id].add(src_id)

        # BFS to find components
        visited = set()
        chains = []
        for bid in block_ids:
            if bid in visited:
                continue
            component = []
            queue = [bid]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(block_map[cur])
                for neighbor in adj[cur]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            chains.append(component)

        return chains

    def run_block(self, block, wires):
        """Execute a single block."""
        # Skip disabled or already-completed blocks
        if block.status in ("skipped", "done"):
            return

        # Guard against re-entrant calls (prevents infinite recursion
        # when auto-run-upstream encounters cycles via Reference blocks)
        if block.id in self._running_blocks:
            return
        self._running_blocks.add(block.id)

        # Check if block definition exists
        if not block.definition_name or not self.registry or not self.registry.has(block.definition_name):
            block.status = "error"
            block.error_message = f"Unknown block type: {block.definition_name}"
            self.status_callback(block)
            self._running_blocks.discard(block.id)
            return

        defn = self.registry.get(block.definition_name)

        # Skip visualization-only blocks during pipeline run
        if defn.get("isVisualizationOnly"):
            block.status = "skipped"
            self._running_blocks.discard(block.id)
            self.status_callback(block)
            return

        # Debug stepping: pause before executing this block.
        # Only pause for blocks in the main pipeline (reachable from Start).
        # Side-chain blocks auto-run by _gather_inputs run silently.
        if self.debug_mode and not self._debug_continue \
                and self._in_pipeline_run and self.debug_pause_callback \
                and block.id in self._reset_ids:
            if not defn.get("isStart") and not defn.get("isToggle"):
                self.debug_pause_callback(block)
                if self.stop_requested:
                    self._running_blocks.discard(block.id)
                    return

        # Set status to running
        block.status = "running"
        bid = getattr(block, 'block_id', None)
        bid_tag = f"B{bid}" if bid is not None else block.id[:12]
        self.debug_log_callback(
            f"[{bid_tag}] Running: {block.display_name} ({block.definition_name})")
        self.status_callback(block)

        try:
            # Gather inputs from upstream wires
            inputs = self._gather_inputs(block, wires)

            # Inject global variables. Global-setter blocks are exempt: a GSR
            # must never read back the effective rate it published as if it
            # were the base (a missed wire delivery would then divide base/N
            # again on every run: 50 kHz -> 2.5 kHz -> 125 Hz -> ...).
            if not defn.get("isGlobalSampleRate"):
                for key, val in self.global_vars.items():
                    if key not in inputs:
                        inputs[key] = val

            # Check required inputs
            for port in block.input_ports:
                if port.required and port.name not in inputs:
                    if port.is_connected:
                        raise ValueError(
                            f"Required input '{port.name}' has no data "
                            f"(upstream block may not have run yet)")
                    else:
                        raise ValueError(f"Required input '{port.name}' not connected")

            # Execute the block
            outputs = self._execute_block(block, defn, inputs, wires)

            # Store outputs
            if outputs is not None:
                block.output_data = outputs
                for port in block.output_ports:
                    port.has_data = True

            # Create dynamic output ports for Cluster blocks
            if defn.get("isCluster") and outputs:
                self._update_cluster_ports(block, outputs)

            # Mark all connected input ports as having data
            for port in block.input_ports:
                if port.is_connected:
                    port.has_data = True

            # Handle global variable blocks
            self._apply_global_sample_rate(block, defn)

            # Handle DefineParameters blocks
            if defn.get("isDefineParameters"):
                dp_vars = block.parameters.get("variables", [])
                for var in dp_vars:
                    name = var.get("name", "")
                    if name:
                        self.reference_store[name] = var.get("value")
                dp_sr = block.parameters.get("globalSampleRate")
                if dp_sr is not None:
                    self.global_vars["sampleRate"] = dp_sr

            # Handle AddToReference blocks
            if defn.get("isAddToReference"):
                for port in block.input_ports:
                    if port.name == "addInput":
                        continue
                    if port.name in inputs:
                        label = port.display_name
                        self.reference_store[label] = inputs[port.name]

            # Handle Reference blocks (output)
            if defn.get("isReference"):
                for port in block.output_ports:
                    var_label = port.description if port.description else port.display_name
                    if var_label in self.reference_store:
                        block.output_data[port.name] = self.reference_store[var_label]
                        port.has_data = True

            block.status = "done"
            self.log_callback(f"  [{block.display_name}] done.")
            self.debug_log_callback(f"[{bid_tag}] Done: {block.display_name}")

        except Exception as e:
            # "Save for later" is a deliberate pause, not an error; the
            # saved state will be picked up by a subsequent Continue run.
            if type(e).__name__ == "SavedForLater":
                block.status = "done"
                block.error_message = ""
                self.log_callback(f"  [{block.display_name}] {e}")
                self.debug_log_callback(
                    f"[{bid_tag}] Saved: {block.display_name} — {e}")
            else:
                block.status = "error"
                block.error_message = str(e)
                self.log_callback(f"  [{block.display_name}] ERROR: {e}")
                self.debug_log_callback(
                    f"[{bid_tag}] ERROR: {block.display_name} — {e}")
            traceback.print_exc()

        self._running_blocks.discard(block.id)
        self.status_callback(block)

    def run_from_block(self, start_block, all_blocks, wires):
        """Run from a specific block and all downstream blocks."""
        self._running_blocks = set()
        self.all_blocks = list(all_blocks)
        self._apply_reuse_mode(all_blocks, wires)
        sorted_blocks = self.topological_sort(all_blocks, wires)

        # Find start index
        start_idx = None
        for i, blk in enumerate(sorted_blocks):
            if blk.id == start_block.id:
                start_idx = i
                break

        if start_idx is None:
            self.run_block(start_block, wires)
            return

        # GlobalSampleRate blocks publish a global downstream blocks read; run
        # any that sit before the start point first so this partial run still
        # sees the rate.
        gsr_ids = self._global_setter_ids(all_blocks)
        pre = [b for b in sorted_blocks[:start_idx] if b.id in gsr_ids]
        run_list = pre + sorted_blocks[start_idx:]

        # Reset status and output for blocks that will be re-run
        for blk in run_list:
            blk.status = "pending"
            blk.output_data = {}

        for blk in run_list:
            if self.stop_requested:
                break
            self.run_block(blk, wires)
            if blk.status == "error":
                self.log_callback(
                    f"Pipeline stopped: error in '{blk.display_name}'.")
                break

    def run_to_block(self, end_block, all_blocks, wires):
        """Smart "run to here": run only what's needed to produce ``end_block``.

        Searches backward from the target toward Start and runs forward from the
        earliest un-run block in the cone (the target plus its ancestors),
        reusing blocks that are already ``done``/``skipped``.  Errored (and
        pending) ancestors count as un-run and are re-executed.  Stops at the
        first error; nothing downstream of the target runs.

        Reference / Data Bus / global-var stores are NOT cleared: reused
        ``done`` blocks are not re-run, so their contributions to those stores
        must survive from the prior run (carried over in _create_engine).
        Shared-state blocks themselves (Data Bus / Reference readers) are the
        exception — they are always re-run so they serve the pool's CURRENT
        values instead of a stale cached copy.
        """
        self.stop_requested = False
        self._running_blocks = set()
        self.all_blocks = list(all_blocks)
        self._in_pipeline_run = True
        # Reuse `done` ancestors as-is: without this, _gather_inputs treats a
        # done source that isn't in _reset_ids as a stale side-chain, resets it,
        # and re-runs it — cascading the whole cone back to Start.
        self._reuse_done_blocks = True
        self._apply_reuse_mode(all_blocks, wires)
        self.sync_toggle_outputs(all_blocks)

        # Backward cone: the target plus every block it (transitively) depends
        # on, via wires and Data Bus pool edges.
        cone = self._ancestors_of(end_block, all_blocks, wires)
        cone.add(end_block.id)
        # GlobalSampleRate blocks publish a global the target reads implicitly
        # (through the engine, not a wire), so they never appear as ancestors —
        # include them explicitly or the target runs with a missing rate.
        cone |= self._global_setter_ids(all_blocks)

        # A click always re-runs the target; errored cone blocks are retried.
        end_block.status = "pending"
        end_block.output_data = {}
        for blk in all_blocks:
            if blk.id in cone and blk.status == "error":
                blk.status = "pending"
                blk.output_data = {}
            elif blk.id in cone and blk.status == "done" \
                    and self._is_shared_state_block(blk):
                # Shared-state outputs (Data Bus / Reference) mirror a pool
                # that a re-run peer may refresh this pass — a reused cached
                # value would serve the PREVIOUS run's data. Re-running them
                # is a store lookup, so always refresh.
                blk.status = "pending"
                blk.output_data = {}

        # Blocks that will actually run this pass (pending in the cone) — drives
        # the _gather_inputs "don't auto-run a block scheduled later" guard.
        self._reset_ids = {
            blk.id for blk in all_blocks
            if blk.id in cone and blk.status not in ("done", "skipped")}

        try:
            for blk in self.topological_sort(all_blocks, wires):
                if self.stop_requested:
                    self.log_callback("Execution stopped by user.")
                    return
                if blk.id not in cone:
                    continue
                if blk.status in ("done", "skipped"):
                    # Reuse cached output; re-assert has_data for wire colouring.
                    for p in blk.output_ports:
                        if p.name in blk.output_data:
                            p.has_data = True
                    if blk.id == end_block.id:
                        break
                    continue
                self.run_block(blk, wires)
                if blk.status == "error":
                    self.log_callback(
                        f"Pipeline stopped: error in '{blk.display_name}'.")
                    break
                if blk.id == end_block.id:
                    break
        finally:
            self._in_pipeline_run = False
            self._reuse_done_blocks = False

    def _ancestors_of(self, target, blocks, wires):
        """Return the set of block IDs that ``target`` (transitively) depends
        on, excluding ``target`` itself.  Mirrors topological_sort's adjacency
        (wires + Data Bus pool edges) so pool-only dependencies are included.
        """
        adj = {blk.id: [] for blk in blocks}
        for wire in wires:
            if (wire.source_port and wire.dest_port and
                    wire.source_port.parent_block and wire.dest_port.parent_block):
                src_id = wire.source_port.parent_block.id
                dst_id = wire.dest_port.parent_block.id
                if src_id in adj:
                    adj[src_id].append((dst_id, wire.source_port.index))
        self._add_databus_pool_edges(adj, blocks, wires)

        # Invert to reverse adjacency: dst -> [src, ...]
        rev = {blk.id: [] for blk in blocks}
        for src_id, neighbors in adj.items():
            for dst_id, _ in neighbors:
                if dst_id in rev:
                    rev[dst_id].append(src_id)

        ancestors = set()
        stack = [target.id]
        while stack:
            cur = stack.pop()
            for parent in rev.get(cur, []):
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
        return ancestors

    def stop(self):
        """Request stop of execution."""
        self.stop_requested = True

    def reset_port_data(self, blocks):
        """Reset HasData flag on all ports."""
        for blk in blocks:
            for p in blk.input_ports:
                p.has_data = False
            for p in blk.output_ports:
                p.has_data = False

    def sync_toggle_outputs(self, blocks):
        """Sync toggle block outputs with their parameter values."""
        for blk in blocks:
            if not blk.definition_name or not self.registry:
                continue
            if not self.registry.has(blk.definition_name):
                continue
            defn = self.registry.get(blk.definition_name)
            if defn.get("isToggle"):
                val = blk.parameters.get("value", False)
                blk.output_data = {"value": val}
                for p in blk.output_ports:
                    p.has_data = True

    def _is_shared_state_block(self, block):
        """True for blocks whose output depends on a shared store populated
        by peers (Data Bus, Reference, Add-to-Reference, Load Geometry).
        Such blocks legitimately benefit from a re-run after a peer fills the
        pool; ordinary deterministic blocks do not.
        """
        if not block.definition_name or not self.registry or \
           not self.registry.has(block.definition_name):
            return False
        defn = self.registry.get(block.definition_name)
        return bool(defn.get("isDataBus") or defn.get("isReference")
                    or defn.get("isAddToReference")
                    or defn.get("isLoadGeometry"))

    def _gather_inputs(self, block, wires):
        """Gather input values from upstream block outputs via wires.

        If an upstream block has no output data yet (missing key or None
        value), auto-run it first so that the current block can pull from
        connected sources.  During a pipeline run, side-chain blocks
        (not reachable from Start) that still have stale "done" status
        are reset and re-run to ensure fresh data.
        """
        inputs = {}
        for wire in wires:
            if wire.dest_port and wire.dest_port.parent_block and \
               wire.dest_port.parent_block.id == block.id:
                dest_port_name = wire.dest_port.name
                if wire.source_port and wire.source_port.parent_block:
                    src_block = wire.source_port.parent_block
                    src_port_name = wire.source_port.name

                    need_run = False
                    if src_block.status == "error":
                        # A wired source that failed must stop its consumer
                        # during a pipeline run, so the pipeline halts at the
                        # first error.  Outside a pipeline run (lazy single-
                        # block / inspector paths) stay tolerant as before.
                        if self._in_pipeline_run:
                            raise ValueError(
                                f"Upstream block '{src_block.display_name}' "
                                f"failed: {src_block.error_message}")
                        pass  # don't retry errored blocks
                    elif src_port_name not in src_block.output_data or \
                            src_block.output_data[src_port_name] is None:
                        # During pipeline run, don't auto-run blocks that
                        # are pending in the current run — they'll execute
                        # in topological order later. This prevents pulling
                        # blocks from later Sequence outputs out of order.
                        if self._in_pipeline_run and \
                                src_block.id in self._reset_ids and \
                                src_block.status == "pending":
                            pass
                        elif (self._in_pipeline_run
                              and src_block.id in self._reset_ids
                              and src_block.status == "done"
                              and not self._is_shared_state_block(src_block)):
                            # The source already executed in this pipeline pass
                            # and still doesn't expose this port. Re-running it
                            # with identical inputs cannot conjure a missing
                            # output, so skip the reset+rerun to avoid thrashing.
                            # (Shared-state blocks — Data Bus / Reference — are
                            # excepted: their output can change once peers
                            # populate the shared pool.)
                            self.debug_log_callback(
                                f"[{src_block.display_name}] port "
                                f"'{src_port_name}' not produced this run "
                                f"(has: {list(src_block.output_data.keys())}); "
                                f"'{block.display_name}.{dest_port_name}' "
                                f"left unconnected.")
                        else:
                            # If block is "done" but missing this port's data
                            # (e.g. Data Bus ran before pool was populated),
                            # reset it so run_block will actually re-execute.
                            if src_block.status == "done":
                                src_block.status = "pending"
                                src_block.output_data = {}
                            need_run = True
                    elif self._in_pipeline_run and not self._reuse_done_blocks \
                            and src_block.status == "done" \
                            and src_block.id not in self._reset_ids:
                        # Side-chain block with stale data from a previous run;
                        # reset and re-run so it picks up fresh upstream data.
                        # Skipped during a smart run_to_block (_reuse_done_blocks)
                        # where done ancestors are intentionally reused as-is.
                        src_block.status = "pending"
                        src_block.output_data = {}
                        need_run = True

                    if need_run:
                        self.run_block(src_block, wires)
                        # Mark as run in this pipeline so the stale-data
                        # check doesn't re-run it a second time.
                        self._reset_ids.add(src_block.id)
                        # If the auto-run source failed, stop the consumer too
                        # (pipeline run halts at the first error).
                        if self._in_pipeline_run and src_block.status == "error":
                            raise ValueError(
                                f"Upstream block '{src_block.display_name}' "
                                f"failed: {src_block.error_message}")

                    if src_port_name in src_block.output_data and \
                       src_block.output_data[src_port_name] is not None:
                        inputs[dest_port_name] = src_block.output_data[src_port_name]
                        wire.dest_port.has_data = True
        return inputs

    def _apply_global_sample_rate(self, block, defn):
        """Publish sample-rate globals from a block that declares it —
        GlobalSampleRate (isGlobalSampleRate) or any block flagged
        updatesGlobalSampleRate. Sets global_vars['sampleRate'] and, when
        present, global_vars['downsampleFactor']. Only isGlobalSampleRate blocks
        get run-first reordering (see the reorder logic in run()); an
        updatesGlobalSampleRate block runs in plain DAG order."""
        # Presence-based (not truthiness) to preserve the prior inline
        # isGlobalSampleRate behavior exactly — that block always emits the key.
        if defn.get("isGlobalSampleRate") or defn.get("updatesGlobalSampleRate"):
            if "sampleRate" in block.output_data:
                self.global_vars["sampleRate"] = block.output_data["sampleRate"]
            if "downsampleFactor" in block.output_data:
                self.global_vars["downsampleFactor"] = \
                    block.output_data["downsampleFactor"]

    def _execute_block(self, block, defn, inputs, wires):
        """Execute a block's function and return outputs."""
        # Handle special block types
        if defn.get("isStart"):
            return {"trigger": True}

        if defn.get("isGlobalSampleRate"):
            # Effective rate = base / N. The base is immutable: wire-delivered
            # values are latched into block params so a run where the wire
            # doesn't deliver (partial/smart run) reuses the same base — it is
            # never re-derived from the published effective global. Chain:
            # wired input -> latched/legacy param -> current global (first-run
            # seed) -> 10 kHz. Inputs are coerced to plain floats — file
            # sample rates load as (1,1)/0-d arrays — and a real rate is
            # always published, never None (a stale None param must not shadow
            # the fallback). N (downsample factor) defaults to 1.
            base = self._as_scalar(inputs.get("sampleRate"))
            if base is not None:
                block.parameters["sampleRate"] = base
            else:
                base = self._as_scalar(block.parameters.get("sampleRate"))
            if base is None:
                base = self._as_scalar(self.global_vars.get("sampleRate"))
            if base is None:
                base = 10000.0
            N = self._as_scalar(inputs.get("downsampleFactor"))
            if N is not None:
                block.parameters["downsampleFactor"] = N
            else:
                N = self._as_scalar(block.parameters.get("downsampleFactor"))
            try:
                N = max(1, int(N)) if N is not None else 1
            except (TypeError, ValueError):
                N = 1
            eff = base / N if N != 1 else base
            return {"sampleRate": eff, "downsampleFactor": N}

        if defn.get("isToggle"):
            return {"value": block.parameters.get("value", False)}

        if defn.get("isPause"):
            return self._execute_pause(block, defn, inputs)

        if defn.get("isCustomCode"):
            return self._execute_compute(block, inputs)

        if defn.get("isPlotDisplay"):
            return self._execute_plot_display(block, inputs)

        if defn.get("isReference"):
            self._ensure_reference_store(wires)
            # Reference block outputs are populated post-execution in run_block
            # from the reference_store. Return empty here.
            return {}

        if defn.get("isAddToReference"):
            # AddToReference is handled post-execution in run_block;
            # just pass through inputs as outputs.
            return {}

        if defn.get("isDataBus"):
            return self._execute_data_bus(block, inputs, wires)

        if defn.get("isPriority"):
            # Pass the Run signal to all numbered output ports
            run_signal = inputs.get("Run", True)
            return {p.name: run_signal for p in block.output_ports
                    if p.name != "addOutput"}

        if defn.get("isLoadGeometry"):
            return self._execute_load_geometry(block, wires)

        if defn.get("isUnpacker"):
            return self._execute_unpack(block, inputs, wires)

        if defn.get("isPulseSlicing"):
            return self._execute_pulse_slicing(block, inputs, wires)

        if defn.get("isSegmentProcessing"):
            return self._execute_segment_processing(block, inputs, wires)

        if defn.get("isSubPipeline"):
            return self._execute_sub_pipeline(block, defn, inputs)

        # Standard block execution via runner
        runner_name = defn.get("runner")
        if not runner_name:
            # Try to find runner by convention
            runner_name = f"run_{block.definition_name}"

        try:
            runner_module = importlib.import_module(f"runners.{runner_name}")
            run_func = getattr(runner_module, "run")
        except (ImportError, AttributeError):
            # Try function handle from definition
            func = defn.get("functionHandle")
            if func is not None:
                return self._call_function(func, defn, inputs, block.parameters)
            raise RuntimeError(f"No runner found for block '{block.definition_name}'")

        # Surface run mode to the runner: True during a multi-block pipeline
        # run (Run All / Run to Here), False for an individual block run.
        # Interactive blocks use this to decide whether a remembered
        # "continue vs. restart" choice may suppress their prompt.
        block._auto_run = self._in_pipeline_run

        return run_func(inputs, block.parameters, block)

    def _update_cluster_ports(self, block, outputs):
        """Create dynamic output ports for Cluster block results."""
        from port import Port

        # Expected output keys: labels, x_1, y_1, x_2, y_2, ...
        new_names = [k for k in outputs if k != "labels"]
        new_names.sort()  # x_1, x_2, ..., y_1, y_2, ...
        # Reorder to: x_1, y_1, x_2, y_2, ...
        n = len(new_names) // 2
        ordered = []
        for i in range(1, n + 1):
            ordered.append(f"x_{i}")
            ordered.append(f"y_{i}")
        new_names = ["labels"] + ordered

        old_port_map = {p.name: p for p in block.output_ports}
        new_ports = []
        for i, name in enumerate(new_names):
            if name in old_port_map:
                p = old_port_map[name]
                p.index = i + 1
            else:
                p = Port(name=name, direction="output", type_="numeric",
                         required=True, description="")
                p.display_name = name
                p.parent_block = block
                p.index = i + 1
            new_ports.append(p)

        # Remove wires for ports that no longer exist
        kept = {p.name for p in new_ports}
        for name, port in old_port_map.items():
            if name not in kept:
                for wire in list(port.connections):
                    if self.wire_remove_callback:
                        self.wire_remove_callback(wire)

        block.output_ports = new_ports
        block.parameters["outputPortDefs"] = block.port_defs("output")
        block.resize_to_fit_ports()
        self.status_callback(block)

    def _execute_unpack(self, block, inputs, wires):
        """Execute an UnpackStruct block: create dynamic output ports from
        struct fields, optionally flattening nested dicts.

        `unpackDepth` parameter controls how many levels to descend into
        nested dicts. Default 1 = unwrap top-level only (legacy behavior).
        depth=2 flattens one nested level (`parent.child`); larger values
        flatten further. This avoids cascading unpack blocks for deeply
        nested structures.

        Mirrors MATLAB's evalUnpackBlock: preserves existing ports/wires
        that match new fields, removes wires for ports that no longer
        exist, and resizes the block to fit.
        """
        from port import Port

        struct_in = inputs.get("structIn", {})
        if struct_in is None:
            raise ValueError(
                f'Unpack Variable "{block.display_name}" requires a non-empty input')
        # If input is not a dict, pass it through as a single output named "value"
        if not isinstance(struct_in, dict):
            struct_in = {"value": struct_in}
        if not struct_in:
            raise ValueError(
                f'Unpack Variable "{block.display_name}" requires a non-empty input')

        # Two output-selection modes:
        # - selectedPaths (new): explicit list of dotted paths chosen by
        #   the user via the structure tree in the edit dialog. Each
        #   path becomes one output port carrying the value at that path.
        # - unpackDepth + visibleOutputs (legacy): uniform depth-based
        #   flattening. Used when no selectedPaths is set.
        selected_paths = block.parameters.get("selectedPaths")
        if selected_paths:
            new_fields = []
            flat_map = {}
            for path in selected_paths:
                try:
                    val = _resolve_path(struct_in, path)
                except KeyError:
                    # Path no longer exists in this struct; skip silently.
                    continue
                new_fields.append(path)
                flat_map[path] = val
            all_fields = list(new_fields)
            depth = None
        else:
            depth = int(block.parameters.get("unpackDepth", 1) or 1)
            depth = max(1, depth)
            flat_pairs = _flatten_struct(struct_in, depth)
            all_fields = [k for k, _ in flat_pairs]
            flat_map = dict(flat_pairs)
            visible = block.parameters.get("visibleOutputs")
            if visible:
                new_fields = [f for f in all_fields if f in visible]
            else:
                new_fields = list(all_fields)

        # Respect any user-customized output port order: keep fields that
        # already have a port in their current (possibly reordered) order,
        # then append newly-appearing fields in their natural order. Without
        # this, a run rebuilds ports in selectedPaths/flatten order and resets
        # an order the user set via Rearrange Ports.
        new_set = set(new_fields)
        ordered = [p.name for p in block.output_ports if p.name in new_set]
        ordered += [f for f in new_fields if f not in ordered]
        new_fields = ordered

        # Map existing output port names
        old_port_map = {p.name: p for p in block.output_ports}

        # Remove wires for ports whose fields no longer exist
        removed_names = set(old_port_map.keys()) - set(new_fields)
        for name in removed_names:
            port = old_port_map[name]
            for wire in list(port.connections):
                self.wire_remove_callback(wire)

        # Build new output ports: reuse existing where names match
        new_ports = []
        for i, field_name in enumerate(new_fields):
            if field_name in old_port_map and field_name not in removed_names:
                # Reuse existing port (preserves wires)
                p = old_port_map[field_name]
                p.index = i + 1
            else:
                # Create new port
                p = Port(
                    name=field_name,
                    direction="output",
                    type_="any",
                    required=True,
                    description=f"Field: {field_name}",
                )
                p.parent_block = block
                p.index = i + 1
            new_ports.append(p)

        block.output_ports = new_ports

        # Resize block for new port count
        block.resize_to_fit_ports()

        # Build output data using the flattened map
        output_data = {f: flat_map[f] for f in new_fields}

        # Store port definitions and available fields for serialization / UI
        block.parameters["outputPortDefs"] = [
            {"name": f, "type": "any", "description": f"Field: {f}"}
            for f in new_fields
        ]
        block.parameters["availableFields"] = all_fields
        # Cache the deepest possible flattening so the edit dialog can let
        # the user preview fields at any depth without re-running the
        # pipeline. Only key shapes are stored, not values.
        block.parameters["availableFieldsByDepth"] = _enumerate_keys_by_depth(
            struct_in, max_depth=10)
        # Cache the full structural shape of the input (types + sizes +
        # nested dict children, no actual data). Used by the edit-Unpack
        # dialog to render a structure tree similar to the Data Inspector.
        block.parameters["structShape"] = _shape_node(struct_in, max_depth=10)

        return output_data

    def _ensure_reference_store(self, wires):
        """Ensure all AddToReference blocks have populated the reference_store."""
        for blk in self.all_blocks:
            if blk.definition_name and self.registry and \
               self.registry.has(blk.definition_name):
                blk_defn = self.registry.get(blk.definition_name)
                if blk_defn.get("isAddToReference"):
                    if blk.status in ("error", "skipped"):
                        continue
                    if blk.status == "done":
                        for port in blk.input_ports:
                            if port.name == "addInput":
                                continue
                            label = port.display_name
                            for w in wires:
                                if w.dest_port is port and w.source_port:
                                    src = w.source_port.parent_block
                                    sp = w.source_port.name
                                    if src and sp in src.output_data and \
                                            src.output_data[sp] is not None:
                                        self.reference_store[label] = src.output_data[sp]
                        continue

    def _ensure_data_bus_store(self, wires):
        """Ensure all Data Bus blocks have populated their pools."""
        for blk in self.all_blocks:
            if not blk.definition_name or not self.registry or \
               not self.registry.has(blk.definition_name):
                continue
            blk_defn = self.registry.get(blk.definition_name)
            if not blk_defn.get("isDataBus"):
                continue
            if blk.status in ("error", "skipped"):
                continue
            pool = blk.parameters.get("pool", "Default")
            if pool not in self.data_bus_store:
                self.data_bus_store[pool] = {}
            if blk.status == "done":
                for port in blk.input_ports:
                    if port.name in ("Run", "addInput"):
                        continue
                    label = port.display_name
                    for w in wires:
                        if w.dest_port is port and w.source_port:
                            src = w.source_port.parent_block
                            sp = w.source_port.name
                            if src and sp in src.output_data and \
                                    src.output_data[sp] is not None:
                                self.data_bus_store[pool][label] = src.output_data[sp]
                continue

    def _execute_load_geometry(self, block, wires):
        """Load geometry dict from reference store and unpack fields as output ports."""
        from port import Port

        self._ensure_reference_store(wires)
        self._ensure_data_bus_store(wires)

        ref_var = block.parameters.get("refVariable", "geometry")
        struct_in = self.reference_store.get(ref_var)

        # Fall back to data_bus_store (search all pools)
        if not isinstance(struct_in, dict) or not struct_in:
            for pool_store in self.data_bus_store.values():
                if ref_var in pool_store and isinstance(pool_store[ref_var], dict):
                    struct_in = pool_store[ref_var]
                    break

        if not isinstance(struct_in, dict) or not struct_in:
            raise ValueError(
                f'LoadGeometry "{block.display_name}" could not find '
                f'"{ref_var}" in the reference store or data bus (or it is empty)')

        new_fields = list(struct_in.keys())

        # Map existing output port names
        old_port_map = {p.name: p for p in block.output_ports}

        # Remove wires for ports whose fields no longer exist
        removed_names = set(old_port_map.keys()) - set(new_fields)
        for name in removed_names:
            port = old_port_map[name]
            for wire in list(port.connections):
                self.wire_remove_callback(wire)

        # Build new output ports: reuse existing where names match
        new_ports = []
        for i, field_name in enumerate(new_fields):
            if field_name in old_port_map and field_name not in removed_names:
                p = old_port_map[field_name]
                p.index = i + 1
            else:
                p = Port(
                    name=field_name,
                    direction="output",
                    type_="any",
                    required=True,
                    description=f"Field: {field_name}",
                )
                p.parent_block = block
                p.index = i + 1
            new_ports.append(p)

        block.output_ports = new_ports

        # Resize block for new port count
        block.resize_to_fit_ports()

        # Build output data
        output_data = {f: struct_in[f] for f in new_fields}

        # Store port definitions for serialization
        block.parameters["outputPortDefs"] = [
            {"name": f, "type": "any", "description": f"Field: {f}"}
            for f in new_fields
        ]

        return output_data

    def _execute_pulse_slicing(self, block, inputs, wires):
        """Execute PulseSlicing: run the runner, then create dynamic output ports.

        The runner returns a dict of segment structs keyed by segment name.
        Dynamic output ports are created for each key, similar to UnpackStruct.
        """
        from port import Port
        from utils.zones import zone_list

        runner_module = importlib.import_module("runners.runPulseSlicing")
        segment_data = runner_module.run(inputs, block.parameters, block)

        new_fields = list(segment_data.keys())

        def _seg_label(field):
            """Display label for a segment or per-zone output value.

            Single-zone: the value is a segment dict with a string ``label``,
            so use it. Multi-zone: the value is a per-zone struct
            ``{segLabel: segdict, ...}`` with no top-level string label, so fall
            back to the field name (the runner names these ports "Zone N").
            """
            value = segment_data[field]
            zones = zone_list(value)
            first = zones[0] if zones else {}
            if isinstance(first, dict):
                lbl = first.get("label")
                if isinstance(lbl, str) and lbl:
                    return lbl
            return field

        # Map existing output port names
        old_port_map = {p.name: p for p in block.output_ports}

        # Remove wires for ports whose fields no longer exist
        removed_names = set(old_port_map.keys()) - set(new_fields)
        for name in removed_names:
            port = old_port_map[name]
            for wire in list(port.connections):
                self.wire_remove_callback(wire)

        # Build new output ports: reuse existing where names match
        new_ports = []
        for i, field_name in enumerate(new_fields):
            if field_name in old_port_map and field_name not in removed_names:
                p = old_port_map[field_name]
                p.index = i + 1
            else:
                p = Port(
                    name=field_name,
                    direction="output",
                    type_="any",
                    required=True,
                    description=f"Segment: {field_name}",
                )
                p.display_name = _seg_label(field_name)
                p.parent_block = block
                p.index = i + 1
            new_ports.append(p)

        block.output_ports = new_ports

        # Resize block
        block.resize_to_fit_ports()

        # Store port definitions for serialization
        block.parameters["outputPortDefs"] = [
            {"name": f, "type": "any",
             "description": f"Segment: {f}",
             "displayName": _seg_label(f)}
            for f in new_fields
        ]

        return segment_data

    def _execute_segment_processing(self, block, inputs, wires):
        """Execute SegmentProcessing: run the runner, then create output ports.

        Single-zone: one output per input (``inK -> outK``).
        Multi-zone: each input is a per-zone struct of segments, UNPACKED so
        every segment gets its own output port (named by the runner). Either
        way the output ports are derived from the runner's returned keys.
        """
        from port import Port

        runner_module = importlib.import_module("runners.runSegmentProcessing")
        result = runner_module.run(inputs, block.parameters, block)

        # Destructive-rebuild guard: only (re)build ports when EVERY connected
        # input delivered data this run. When the block is auto-run as a
        # downstream dependency before its upstream is ready, the inputs are
        # incomplete — keep the existing ports/wires intact instead of tearing
        # them down (the structural-mirror robustness the old code provided).
        connected_ins = [p.name for p in block.input_ports
                         if p.name != "addInput" and p.is_connected]
        have_all_inputs = all(n in inputs for n in connected_ins)
        if not result or not have_all_inputs:
            return getattr(block, "output_data", None) or {}

        # Display names: keep the input's label for mirrored ``outK`` ports;
        # unpacked segment ports use their own key.
        in_display = {inp.name.replace("in", "out", 1): inp.display_name
                      for inp in block.input_ports if inp.name != "addInput"}

        new_fields = list(result.keys())
        old_port_map = {p.name: p for p in block.output_ports}

        removed_names = set(old_port_map.keys()) - set(new_fields)
        for name in removed_names:
            for wire in list(old_port_map[name].connections):
                self.wire_remove_callback(wire)

        new_ports = []
        for i, fname in enumerate(new_fields):
            if fname in old_port_map and fname not in removed_names:
                p = old_port_map[fname]
                p.index = i + 1
            else:
                p = Port(name=fname, direction="output", type_="any",
                         required=True, description="Processed segment")
                p.parent_block = block
                p.index = i + 1
            p.display_name = in_display.get(fname, fname)
            new_ports.append(p)

        output_data = dict(result)
        block.output_ports = new_ports

        # Resize block. Inline (not resize_to_fit_ports/port_defs): this path
        # is driven by duck-typed stub blocks in test_segment_processing_ports.
        max_ports = max(len(block.input_ports), len(block.output_ports))
        header_h = 22
        new_height = max(header_h + 35, header_h + 10 + max_ports * 20)
        block.size = (block.size[0], new_height)

        # Store port definitions for serialization
        block.parameters["outputPortDefs"] = [
            {"name": p.name, "type": "any",
             "description": p.description,
             "displayName": p.display_name}
            for p in new_ports
        ]

        return output_data

    def _execute_sub_pipeline(self, block, defn, inputs):
        """Execute a sub-pipeline block by running its inner blocks."""
        from subpipeline_manager import load_template, instantiate_inner_graph
        from block_node import BlockNode
        from port import Port
        from wire_connection import WireConnection

        filepath = defn.get("subPipelineFile", "")
        if not filepath or not os.path.isfile(filepath):
            raise FileNotFoundError(f"Sub-pipeline template not found: {filepath}")

        template = load_template(filepath)
        inner_blocks, inner_wires, id_map, block_map = \
            instantiate_inner_graph(template, self.registry)

        # Inject external inputs into boundary inner blocks via stub blocks
        for exp_in in template.get("exposedInputs", []):
            if exp_in["name"] not in inputs:
                continue
            target_id = id_map.get(exp_in["innerBlock"])
            if not target_id or target_id not in block_map:
                continue
            target_block = block_map[target_id]
            target_port = next(
                (p for p in target_block.input_ports if p.name == exp_in["innerPort"]),
                None)
            if not target_port:
                continue

            # Create a stub source block with the input data
            stub = BlockNode("_stub")
            stub.id = f"blk_{uuid.uuid4()}"
            stub.status = "done"
            stub.output_data = {"out": inputs[exp_in["name"]]}
            stub_port = Port(name="out", direction="output", type_="any")
            stub_port.parent_block = stub
            stub_port.has_data = True
            stub.output_ports = [stub_port]

            wire = WireConnection(stub_port, target_port)
            inner_blocks.append(stub)
            inner_wires.append(wire)

        # Inject global variables into inner block execution
        # by making them available through a shared engine context
        saved_all_blocks = self.all_blocks
        saved_in_pipeline_run = self._in_pipeline_run
        saved_reset_ids = self._reset_ids
        self.all_blocks = inner_blocks
        self._in_pipeline_run = True
        self._reset_ids = {b.id for b in inner_blocks}

        # Run inner pipeline in topological order
        try:
            sorted_inner = self.topological_sort(inner_blocks, inner_wires)
            for inner_block in sorted_inner:
                if inner_block.status == "done":
                    continue
                if inner_block.definition_name == "_stub":
                    continue
                self.run_block(inner_block, inner_wires)
                if inner_block.status == "error":
                    raise RuntimeError(
                        f"Sub-pipeline error in '{inner_block.display_name}': "
                        f"{inner_block.error_message}")
        finally:
            self.all_blocks = saved_all_blocks
            self._in_pipeline_run = saved_in_pipeline_run
            self._reset_ids = saved_reset_ids

        # Collect outputs from exposed output ports.
        # Match template exposed names to the block's actual output ports
        # by name or display_name (handles renamed/rearranged ports).
        outputs = {}
        # Build lookup: name -> port, display_name -> port
        port_by_name = {p.name: p for p in block.output_ports}
        port_by_display = {p.display_name: p for p in block.output_ports}
        exposed_outputs = template.get("exposedOutputs", [])
        for exp_out in exposed_outputs:
            source_id = id_map.get(exp_out["innerBlock"])
            if not source_id or source_id not in block_map:
                continue
            source_block = block_map[source_id]
            port_name = exp_out["innerPort"]
            if port_name not in source_block.output_data:
                continue
            data = source_block.output_data[port_name]
            exp_name = exp_out["name"]
            # Find matching output port: by name first, then display_name
            port = port_by_name.get(exp_name) or port_by_display.get(exp_name)
            if port:
                outputs[port.name] = data
            else:
                outputs[exp_name] = data

        return outputs

    def _execute_data_bus(self, block, inputs, wires):
        """Execute a Data Bus block: store inputs into data_bus_store, read outputs from it.

        Data Bus combines AddToRef + Ref behaviour with a Run signal.
        Each block belongs to a named pool (parameter "pool", default "Default").
        - All data inputs (not Run, addInput) are stored into data_bus_store[pool]
          keyed by the port's display_name.
        - All data outputs (not Run) are populated from data_bus_store[pool].
        - The Run output simply passes a True signal to trigger downstream blocks.
        """
        pool = block.parameters.get("pool", "Default")

        # Ensure pool dict exists
        if pool not in self.data_bus_store:
            self.data_bus_store[pool] = {}
        store = self.data_bus_store[pool]

        # --- Ensure other Data Bus blocks in the same pool have populated the store ---
        for blk in self.all_blocks:
            if blk.id == block.id:
                continue
            if not blk.definition_name or not self.registry or \
               not self.registry.has(blk.definition_name):
                continue
            blk_defn = self.registry.get(blk.definition_name)
            if not blk_defn.get("isDataBus"):
                continue
            if blk.parameters.get("pool", "Default") != pool:
                continue
            if blk.status in ("error", "skipped"):
                continue
            if blk.status == "done":
                # Re-populate store from this block's input ports
                for port in blk.input_ports:
                    if port.name in ("Run", "addInput"):
                        continue
                    label = port.display_name
                    for w in wires:
                        if w.dest_port is port and w.source_port:
                            src = w.source_port.parent_block
                            sp = w.source_port.name
                            if src and sp in src.output_data and \
                                    src.output_data[sp] is not None:
                                store[label] = src.output_data[sp]
                continue
        # --- Store phase: write all connected data inputs into the pool ---
        for port in block.input_ports:
            if port.name in ("Run", "addInput"):
                continue
            if port.name in inputs:
                label = port.display_name
                store[label] = inputs[port.name]

        # --- Retrieve phase: populate data outputs from pool + reference store ---
        outputs = {}
        for port in block.output_ports:
            if port.name == "Run":
                continue
            var_label = port.description if port.description else port.display_name
            if var_label in store:
                outputs[port.name] = store[var_label]
                port.has_data = True
            elif var_label in self.reference_store:
                outputs[port.name] = self.reference_store[var_label]
                port.has_data = True

        # Run pass-through: if Run input received, propagate a signal
        if "Run" in inputs:
            outputs["Run"] = True

        return outputs

    def _call_function(self, func, defn, inputs, params):
        """Call a function with mapped inputs and return mapped outputs."""
        input_mapping = defn.get("inputMapping", [])
        output_mapping = defn.get("outputMapping", [])

        # Build argument list
        args = []
        for mapping in input_mapping:
            port_name = mapping
            if port_name in inputs:
                args.append(inputs[port_name])
            else:
                args.append(None)

        # Add params if the function expects it
        result = func(*args, params=params)

        # Map outputs
        if isinstance(result, dict):
            return result
        elif isinstance(result, (list, tuple)):
            outputs = {}
            for i, mapping in enumerate(output_mapping):
                if i < len(result):
                    outputs[mapping] = result[i]
            return outputs
        else:
            if output_mapping:
                return {output_mapping[0]: result}
            return {"output": result}

    def _execute_pause(self, block, defn, inputs):
        """Execute a pause/junction block - shows dialog."""
        from PySide6.QtWidgets import QMessageBox
        msg = block.parameters.get("message", "Continue?")
        reply = QMessageBox.question(None, block.display_name, msg)
        choice = "Yes" if reply == QMessageBox.StandardButton.Yes else "No"

        outputs = {"Choice": choice}
        # Pass through all data inputs
        for port in block.input_ports:
            if port.name == "Pause" or port.name == "addInput":
                continue
            port_idx = port.name.replace("in", "")
            out_name = f"out{port_idx}"
            if port.name in inputs:
                outputs[out_name] = inputs[port.name]

        return outputs

    def _execute_plot_display(self, block, inputs):
        """Store numeric vector data for in-canvas plot rendering."""
        import numpy as np
        val = next(iter(inputs.values()), None) if inputs else None
        if val is not None and hasattr(val, '__len__'):
            arr = np.atleast_1d(val).ravel().astype(float)
            block.parameters['plotData'] = arr
        else:
            block.parameters['plotData'] = None

        # Auto-size to give enough room for the plot
        header_h = 22
        max_ports = max(len(block.input_ports), len(block.output_ports))
        port_area = 10 + max(1, max_ports) * 20
        plot_area = 60
        new_h = header_h + max(port_area, plot_area)
        new_w = max(block.size[0], 160)
        block.size = (new_w, max(block.size[1], new_h))
        return {}

    def _execute_compute(self, block, inputs):
        """Execute a ComputeBlock with user-defined expression."""
        expression = block.parameters.get("expression", "")
        if not expression:
            return {}

        # Create local namespace with inputs
        local_ns = dict(inputs)
        # Add common imports
        import numpy as np
        import scipy
        local_ns["np"] = np
        local_ns["numpy"] = np
        local_ns["scipy"] = scipy

        exec(expression, {"__builtins__": __builtins__}, local_ns)

        # Extract outputs based on output port definitions
        outputs = {}
        for port in block.output_ports:
            if port.name in local_ns:
                outputs[port.name] = local_ns[port.name]

        return outputs
