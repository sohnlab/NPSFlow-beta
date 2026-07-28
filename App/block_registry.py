"""BlockRegistry - Registry of available block types."""

import colorsys
import importlib
import os
from block_node import BlockNode
from utils.app_logger import logger


# Categories are ordered so that File I/O ("DataIO") gets the first hue
# (red) and the rainbow progresses through the rest. Categories not in this
# list are appended in alphabetical order.
_CATEGORY_ORDER = [
    "DataIO", "Variables", "FlowControl", "Preprocessing", "Filters",
    "Template", "Detection", "Sorting", "FeatureExtraction", "Analysis",
    "Calculation", "Math", "Plot", "Utility", "SubPipeline",
]


def assign_dynamic_colors(definitions):
    """Mutate each block definition's `color` so that:
      • each category gets a distinct base hue (red → rainbow), and
      • blocks within a category get a small saturation/value variation
        based on their *position* (insertion order in the registry), not
        on their name.

    Saturation/value are clamped to a band that keeps white tile text
    readable across the spectrum.
    """
    # Group by category, preserving insertion order (dict preserves it in
    # Python 3.7+). That order matches the registry scan order (filename),
    # so adding a new block file slots its color into the next position.
    # Super-pipeline blocks are excluded — they retain their per-template
    # color (set in subpipeline_manager.build_definition) so the user can
    # customize each one individually via the palette's Color picker.
    cat_blocks = {}
    for defn in definitions.values():
        if defn.get("isSubPipeline") or defn.get("category") == "SubPipeline":
            continue
        cat = defn.get("category", "Other")
        cat_blocks.setdefault(cat, []).append(defn)

    known = [c for c in _CATEGORY_ORDER if c in cat_blocks]
    extras = sorted(c for c in cat_blocks if c not in known)
    ordered_cats = known + extras
    n_cats = max(1, len(ordered_cats))

    for ci, cat in enumerate(ordered_cats):
        hue = (ci / n_cats) % 1.0  # 0=red, then rainbow
        blocks = cat_blocks[cat]
        n = max(1, len(blocks))
        for bi, defn in enumerate(blocks):
            # Position 0 → richer/darker; position n-1 → lighter.
            # Both bands stay dark enough for legible white text.
            t = bi / max(1, n - 1) if n > 1 else 0.0
            sat = 0.70 - 0.15 * t   # 0.70 → 0.55
            val = 0.55 + 0.20 * t   # 0.55 → 0.75
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            defn["color"] = [r, g, b]


class BlockRegistry:
    """Scans the blockdefs package and maintains a catalog of all
    block definitions that can be instantiated on the canvas.
    """

    def __init__(self):
        self.definitions = {}  # name -> definition dict

    def scan_block_defs(self):
        """Scan the blockdefs package for block definition functions."""
        blockdefs_dir = os.path.join(os.path.dirname(__file__), "blockdefs")
        if not os.path.isdir(blockdefs_dir):
            logger.error(f"BlockRegistry: blockdefs directory not found: {blockdefs_dir}")
            return

        for filename in os.listdir(blockdefs_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            module_name = filename[:-3]
            try:
                mod = importlib.import_module(f"blockdefs.{module_name}")
                if not hasattr(mod, "get_definition"):
                    logger.warning(f"BlockRegistry: '{module_name}' has no get_definition()")
                    continue
                defn = mod.get_definition()

                # Validate required fields
                required = {"name", "displayName", "category", "color", "inputs", "outputs"}
                missing = required - set(defn.keys())
                if missing:
                    logger.warning(f"BlockRegistry: '{module_name}' missing fields: {missing}")
                    continue

                # Ensure inputs/outputs are lists
                if defn["inputs"] is None:
                    defn["inputs"] = []
                if defn["outputs"] is None:
                    defn["outputs"] = []

                self.definitions[defn["name"]] = defn

            except Exception as e:
                logger.error(f"BlockRegistry: Error loading '{module_name}': {e}")

        # Scan sub-pipeline templates
        self._scan_sub_pipelines()

        # Seed each defn with a fallback dynamic color in case it's used
        # before the palette has a chance to apply section-based coloring.
        assign_dynamic_colors(self.definitions)

        logger.info(f"BlockRegistry: loaded {len(self.definitions)} block definitions")

    def _scan_sub_pipelines(self):
        """Scan App/subpipelines/ for sub-pipeline templates."""
        try:
            from subpipeline_manager import list_templates, load_template, build_definition
        except ImportError:
            return
        for name, filepath in list_templates():
            try:
                template = load_template(filepath)
                defn = build_definition(template, filepath)
                self.definitions[defn["name"]] = defn
            except Exception as e:
                logger.error(f"BlockRegistry: Error loading sub-pipeline '{name}': {e}")

    def get(self, name):
        """Get a block definition by name."""
        name = name.strip()
        if name in self.definitions:
            return self.definitions[name]
        raise KeyError(f'Block definition "{name}" not found')

    def has(self, name):
        name = name.strip()
        return name in self.definitions

    def list_names(self):
        """List all registered block definition names."""
        return list(self.definitions.keys())

    def list_by_category(self):
        """Group block definitions by category."""
        groups = {}
        for name, defn in self.definitions.items():
            cat = defn["category"]
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(defn)
        return groups

    def create_block(self, def_name, position):
        """Create a new BlockNode from a registered definition."""
        defn = self.get(def_name)
        block = BlockNode(def_name, position)
        block.init_from_definition(defn)
        return block
