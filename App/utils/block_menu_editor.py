"""BlockMenuEditorDialog - Editor for customising the block palette menu.

Allows users to:
- Reorder categories and blocks via drag-and-drop
- Rename category labels and block display names
- Move blocks between categories (including sub-categories)
- Hide blocks by moving them to the "Unused" category
- Reset to default layout
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget,
    QTreeWidgetItem, QAbstractItemView, QDialogButtonBox, QInputDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QDrag, QFont
import os
import sys as _sys

from utils.dialog_style import apply_dialog_style

_SANS = "Helvetica Neue" if _sys.platform == "darwin" else "Segoe UI"

# Internal key used for the "Unused" bucket
_UNUSED_KEY = "__unused__"

# Keywords used to classify DataIO blocks as Input vs Output
_IO_INPUT_KEYWORDS = {"load", "import", "read", "constant", "toggle",
                      "unpack", "filepath", "text"}


def _is_io_input(def_name):
    """Check if a DataIO block name belongs to the Input sub-category."""
    return any(kw in def_name.lower() for kw in _IO_INPUT_KEYWORDS)


class _MenuTree(QTreeWidget):
    """Tree widget with internal drag-and-drop for blocks between categories."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(20)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setExpandsOnDoubleClick(False)
        self._drag_source_item = None  # stashed by startDrag

    def _item_role(self, item):
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def _find_drop_container(self, target):
        """Walk up from *target* to find the nearest category or subcategory."""
        while target is not None:
            role = self._item_role(target)
            if role in ("category", "subcategory"):
                return target
            target = target.parent()
        return None

    _MENU_DRAG_MIME = "application/x-npsview-menu-drag"

    def startDrag(self, supportedActions):
        """Create the QDrag manually so Qt's default startDrag never runs
        its post-drop clearOrRemove cleanup on the selected items."""
        item = self.currentItem()
        if item is None:
            return
        self._drag_source_item = item
        mime = QMimeData()
        mime.setData(self._MENU_DRAG_MIME, b"1")
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction | Qt.DropAction.CopyAction)
        self._drag_source_item = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self._MENU_DRAG_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self._MENU_DRAG_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        target = self.itemAt(event.position().toPoint())
        dragged = self._drag_source_item
        if dragged is None:
            event.ignore()
            return

        d_role = self._item_role(dragged)

        if d_role == "block":
            container = self._find_drop_container(target)
            if container is None:
                event.ignore()
                return
            def_name = dragged.data(0, Qt.ItemDataRole.UserRole + 1)
            display_name = dragged.text(0)
            # Remove from old parent
            old = dragged.parent()
            if old:
                old.removeChild(dragged)
            else:
                idx = self.indexOfTopLevelItem(dragged)
                if idx >= 0:
                    self.takeTopLevelItem(idx)
            # Insert a fresh item into the target container
            new_item = QTreeWidgetItem([display_name])
            new_item.setData(0, Qt.ItemDataRole.UserRole, "block")
            new_item.setData(0, Qt.ItemDataRole.UserRole + 1, def_name)
            new_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            container.addChild(new_item)
            container.setExpanded(True)
            self.setCurrentItem(new_item)
            event.accept()

        elif d_role == "category":
            cat_key = dragged.data(0, Qt.ItemDataRole.UserRole + 1)
            if cat_key == _UNUSED_KEY:
                event.ignore()
                return
            # Manual category reorder
            target_top = target
            while target_top and target_top.parent():
                target_top = target_top.parent()
            drop_idx = self.indexOfTopLevelItem(target_top) if target_top else self.topLevelItemCount()
            if drop_idx < 0:
                drop_idx = self.topLevelItemCount()
            old_idx = self.indexOfTopLevelItem(dragged)
            if old_idx >= 0:
                self.takeTopLevelItem(old_idx)
                if drop_idx > old_idx:
                    drop_idx -= 1
                self.insertTopLevelItem(drop_idx, dragged)
            self._ensure_unused_last()
            event.accept()

        else:
            event.ignore()

    def _ensure_unused_last(self):
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole + 1) == _UNUSED_KEY:
                if i != self.topLevelItemCount() - 1:
                    self.takeTopLevelItem(i)
                    self.addTopLevelItem(item)
                break


class BlockMenuEditorDialog(QDialog):
    """Dialog for editing the block palette menu structure."""

    def __init__(self, menu_config, registry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Block Menu")
        self.setMinimumSize(420, 520)
        self.resize(420, 560)
        apply_dialog_style(self)

        self._registry = registry
        self._result_config = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)

        # Instructions
        hint = QPushButton("Drag blocks between categories. Double-click to rename.")
        hint.setFlat(True)
        hint.setEnabled(False)
        hint.setStyleSheet("QPushButton { color: #888; font-size: 11px; border: none; }")
        layout.addWidget(hint)

        # Tree
        self._tree = _MenuTree()
        self._tree.setStyleSheet("""
            QTreeWidget {
                background: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 12px;
            }
            QTreeWidget::item { padding: 3px 0; }
            QTreeWidget::item:selected { background: #d0e0f0; color: black; }
        """)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._tree)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_add_cat = QPushButton("Add Category")
        btn_add_cat.clicked.connect(self._add_category)
        btn_row.addWidget(btn_add_cat)

        btn_remove_cat = QPushButton("Remove Category")
        btn_remove_cat.clicked.connect(self._remove_category)
        btn_row.addWidget(btn_remove_cat)

        btn_row.addStretch()

        btn_reset = QPushButton("Reset Default")
        btn_reset.clicked.connect(self._reset_default)
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate(menu_config)

    # ------------------------------------------------------------------
    # Tree population helpers
    # ------------------------------------------------------------------

    def _make_cat_item(self, label, cat_key):
        cat_font = QFont(_SANS, 12)
        cat_font.setBold(True)
        item = QTreeWidgetItem([label])
        item.setData(0, Qt.ItemDataRole.UserRole, "category")
        item.setData(0, Qt.ItemDataRole.UserRole + 1, cat_key)
        item.setFont(0, cat_font)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        )
        return item

    def _make_subcat_item(self, label):
        sub_font = QFont(_SANS, 11)
        sub_font.setBold(True)
        item = QTreeWidgetItem([label])
        item.setData(0, Qt.ItemDataRole.UserRole, "subcategory")
        item.setFont(0, sub_font)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        return item

    @staticmethod
    def _make_block_item(display_name, def_name):
        item = QTreeWidgetItem([display_name])
        item.setData(0, Qt.ItemDataRole.UserRole, "block")
        item.setData(0, Qt.ItemDataRole.UserRole + 1, def_name)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        return item

    def _populate(self, config):
        self._tree.clear()
        for entry in config:
            cat_item = self._make_cat_item(entry["label"], entry["category"])
            self._tree.addTopLevelItem(cat_item)

            subcats = entry.get("subcategories")
            if subcats:
                for sc in subcats:
                    sc_item = self._make_subcat_item(sc["label"])
                    cat_item.addChild(sc_item)
                    for blk in sc["blocks"]:
                        sc_item.addChild(
                            self._make_block_item(blk["displayName"], blk["defName"]))
                    sc_item.setExpanded(True)
            else:
                for blk in entry.get("blocks", []):
                    cat_item.addChild(
                        self._make_block_item(blk["displayName"], blk["defName"]))

            cat_item.setExpanded(entry["category"] != _UNUSED_KEY)

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def _on_double_click(self, item, column):
        role = item.data(0, Qt.ItemDataRole.UserRole)
        if role == "category":
            cat_key = item.data(0, Qt.ItemDataRole.UserRole + 1)
            if cat_key == _UNUSED_KEY:
                return
            text, ok = QInputDialog.getText(
                self, "Rename Category", "Category name:", text=item.text(0))
            if ok and text.strip():
                item.setText(0, text.strip())
        elif role == "subcategory":
            text, ok = QInputDialog.getText(
                self, "Rename Sub-Category", "Sub-category name:",
                text=item.text(0))
            if ok and text.strip():
                item.setText(0, text.strip())
        elif role == "block":
            text, ok = QInputDialog.getText(
                self, "Rename Block", "Display name:", text=item.text(0))
            if ok and text.strip():
                item.setText(0, text.strip())

    def _add_category(self):
        text, ok = QInputDialog.getText(self, "New Category", "Category name:")
        if not ok or not text.strip():
            return
        label = text.strip()
        key = label.replace(" ", "")
        cat_item = self._make_cat_item(label, key)
        # Insert before Unused
        unused_idx = None
        for i in range(self._tree.topLevelItemCount()):
            if self._tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole + 1) == _UNUSED_KEY:
                unused_idx = i
                break
        if unused_idx is not None:
            self._tree.insertTopLevelItem(unused_idx, cat_item)
        else:
            self._tree.addTopLevelItem(cat_item)
        cat_item.setExpanded(True)
        self._tree.setCurrentItem(cat_item)

    def _remove_category(self):
        item = self._tree.currentItem()
        if item is None:
            return
        role = item.data(0, Qt.ItemDataRole.UserRole)
        if role not in ("category", "subcategory"):
            QMessageBox.information(self, "Remove Category",
                                    "Select a category to remove.")
            return
        if role == "category":
            cat_key = item.data(0, Qt.ItemDataRole.UserRole + 1)
            if cat_key == _UNUSED_KEY:
                QMessageBox.information(self, "Remove Category",
                                        "The Unused category cannot be removed.")
                return

        # Collect all block children (recursively)
        blocks = self._collect_block_children(item)

        # Find Unused
        unused_item = None
        for i in range(self._tree.topLevelItemCount()):
            ti = self._tree.topLevelItem(i)
            if ti and ti.data(0, Qt.ItemDataRole.UserRole + 1) == _UNUSED_KEY:
                unused_item = ti
                break

        # Move blocks to Unused
        if unused_item:
            for b in blocks:
                unused_item.addChild(b)

        # Remove the category/subcategory item
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            idx = self._tree.indexOfTopLevelItem(item)
            self._tree.takeTopLevelItem(idx)

    def _collect_block_children(self, item):
        """Recursively collect and detach all block-role children."""
        blocks = []
        # Process in reverse so removal doesn't shift indices
        for i in range(item.childCount() - 1, -1, -1):
            child = item.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == "block":
                item.takeChild(i)
                blocks.append(child)
            elif child.data(0, Qt.ItemDataRole.UserRole) == "subcategory":
                blocks.extend(self._collect_block_children(child))
                item.takeChild(i)
        blocks.reverse()
        return blocks

    def _reset_default(self):
        config = build_default_menu_config(self._registry)
        self._populate(config)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _on_apply(self):
        self._result_config = self._read_config()
        self.accept()

    def _read_config(self):
        """Read current tree state into a config list."""
        config = []
        for i in range(self._tree.topLevelItemCount()):
            cat_item = self._tree.topLevelItem(i)
            entry = {
                "category": cat_item.data(0, Qt.ItemDataRole.UserRole + 1),
                "label": cat_item.text(0),
            }

            # Check if any children are subcategories
            has_subcats = any(
                cat_item.child(j).data(0, Qt.ItemDataRole.UserRole) == "subcategory"
                for j in range(cat_item.childCount())
            )

            if has_subcats:
                subcats = []
                direct_blocks = []
                for j in range(cat_item.childCount()):
                    child = cat_item.child(j)
                    if child.data(0, Qt.ItemDataRole.UserRole) == "subcategory":
                        sc_blocks = []
                        for k in range(child.childCount()):
                            blk = child.child(k)
                            sc_blocks.append({
                                "defName": blk.data(0, Qt.ItemDataRole.UserRole + 1),
                                "displayName": blk.text(0),
                            })
                        subcats.append({
                            "label": child.text(0),
                            "blocks": sc_blocks,
                        })
                    elif child.data(0, Qt.ItemDataRole.UserRole) == "block":
                        direct_blocks.append({
                            "defName": child.data(0, Qt.ItemDataRole.UserRole + 1),
                            "displayName": child.text(0),
                        })
                entry["subcategories"] = subcats
                entry["blocks"] = direct_blocks
            else:
                blocks = []
                for j in range(cat_item.childCount()):
                    child = cat_item.child(j)
                    blocks.append({
                        "defName": child.data(0, Qt.ItemDataRole.UserRole + 1),
                        "displayName": child.text(0),
                    })
                entry["blocks"] = blocks

            config.append(entry)
        return config

    def result_config(self):
        return self._result_config


# ------------------------------------------------------------------
# Default config builder
# ------------------------------------------------------------------

_DEFAULT_CATEGORY_ORDER = [
    ("DataIO", "IO"),
    ("Variables", "Variables"),
    ("FlowControl", "Flow Control"),
    ("Preprocessing", "Preprocessing"),
    ("Filters", "Filters"),
    ("Template", "Template"),
    ("Detection", "Detection"),
    ("FeatureExtraction", "Feature Extraction"),
    ("Analysis", "Analysis"),
    ("Utility", "Utility"),
    ("SubPipeline", "Super Blocks"),
]


def _is_hidden(defn):
    """Return True for blocks that should never appear in the menu."""
    return defn.get("isReroute") or not defn.get("displayName")


# Block-definition categories whose natural menu section uses a different
# `category` key (so a new block lands beside its siblings, not in Unused).
_CATEGORY_SECTION_ALIASES = {
    "DataIO": "FileI/O",
    "Template": "Preprocessing",
    "FeatureExtraction": "Analysis",
}

# Friendly labels for sections created on demand for an unmapped category.
_CATEGORY_LABELS = {
    "DataIO": "File I/O", "FlowControl": "Flow Control",
    "FeatureExtraction": "Feature Extraction", "SubPipeline": "Super Blocks",
}


def _find_section(config, category):
    """Return the menu section a block of *category* belongs in, or None."""
    target = _CATEGORY_SECTION_ALIASES.get(category, category)
    for entry in config:
        if entry.get("category") == target:
            return entry
    return None


def add_missing_blocks(config, registry):
    """Append registry blocks absent from *config* into the section matching
    each block's category, so newly-added block definitions show up in the
    palette automatically. A visible section is created on demand rather than
    falling back to the hidden Unused bucket. Mutates and returns *config*.
    """
    known = set()
    for entry in config:
        for blk in entry.get("blocks", []):
            known.add(blk.get("defName"))
        for sc in entry.get("subcategories", []):
            for blk in sc.get("blocks", []):
                known.add(blk.get("defName"))

    for name in sorted(n for n in registry.list_names() if n not in known):
        if not registry.has(name):
            continue
        defn = registry.get(name)
        if _is_hidden(defn):
            continue
        category = defn.get("category") or "Utility"
        section = _find_section(config, category)
        if section is None:
            # Insert a fresh, visible section just before Unused.
            section = {"category": category,
                       "label": _CATEGORY_LABELS.get(category, category),
                       "blocks": []}
            insert_at = next((i for i, e in enumerate(config)
                              if e.get("category") == _UNUSED_KEY), len(config))
            config.insert(insert_at, section)
        section.setdefault("blocks", []).append({
            "defName": name, "displayName": defn["displayName"],
        })
    return config


def build_default_menu_config(registry):
    """Build the default menu config.

    Loads from the bundled default_block_menu.json if available,
    otherwise falls back to building from the registry by category.
    Hidden blocks (reroute nodes, empty displayName) are excluded.
    """
    import json as _json

    # Try bundled default first
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "default_block_menu.json")
    if os.path.isfile(default_path):
        try:
            with open(default_path, "r") as f:
                config = _json.load(f)
            # Reconcile: strip removed blocks, then add new ones by category.
            for entry in config:
                entry["blocks"] = [
                    b for b in entry.get("blocks", [])
                    if registry.has(b["defName"])
                ]
                for sc in entry.get("subcategories", []):
                    sc["blocks"] = [
                        b for b in sc["blocks"]
                        if registry.has(b["defName"])
                    ]

            # New blocks go to their category section (visible), not Unused.
            return add_missing_blocks(config, registry)
        except Exception:
            pass  # fall through to registry-based builder

    # Fallback: build from registry categories
    groups = registry.list_by_category()
    used_names = set()
    config = []

    for cat_key, cat_label in _DEFAULT_CATEGORY_ORDER:
        if cat_key not in groups:
            continue
        defs = sorted(groups[cat_key], key=lambda d: d["displayName"].lower())

        if cat_key == "DataIO":
            input_blocks = []
            output_blocks = []
            for d in defs:
                if _is_hidden(d):
                    used_names.add(d["name"])
                    continue
                blk = {"defName": d["name"], "displayName": d["displayName"]}
                if _is_io_input(d["name"]):
                    input_blocks.append(blk)
                else:
                    output_blocks.append(blk)
                used_names.add(d["name"])
            config.append({
                "category": cat_key,
                "label": cat_label,
                "blocks": [],
                "subcategories": [
                    {"label": "Input", "blocks": input_blocks},
                    {"label": "Output", "blocks": output_blocks},
                ],
            })
        else:
            blocks = []
            for d in defs:
                if _is_hidden(d):
                    used_names.add(d["name"])
                    continue
                blocks.append({"defName": d["name"], "displayName": d["displayName"]})
                used_names.add(d["name"])
            config.append({"category": cat_key, "label": cat_label, "blocks": blocks})

    for cat_key in sorted(groups.keys()):
        if cat_key in {c for c, _ in _DEFAULT_CATEGORY_ORDER}:
            continue
        defs = sorted(groups[cat_key], key=lambda d: d["displayName"].lower())
        blocks = []
        for d in defs:
            if _is_hidden(d):
                used_names.add(d["name"])
                continue
            blocks.append({"defName": d["name"], "displayName": d["displayName"]})
            used_names.add(d["name"])
        config.append({
            "category": cat_key,
            "label": cat_key.replace("_", " "),
            "blocks": blocks,
        })

    config.append({"category": _UNUSED_KEY, "label": "Unused", "blocks": []})
    return config
