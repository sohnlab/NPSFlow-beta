"""Centralized design tokens for NPSflow.

Exposes a module-level ``theme`` proxy whose attribute lookups forward to the
currently active theme. Consumers can keep a single ``from theme import theme``
import and still see updates after ``set_theme(...)``.

Usage:
    from theme import theme, set_theme, build_qpalette

    rgb = theme.status.get(block.status)        # 0..1 RGB tuple
    w, h = theme.sizes.block_default
    button.setStyleSheet(theme.qss.small_button)

    set_theme("dark")
    QApplication.instance().setPalette(build_qpalette())
"""

import sys
from dataclasses import dataclass, field
from typing import Tuple

RGB = Tuple[float, float, float]

_SANS = "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"
_TILE_PT = 9 if sys.platform == "darwin" else 7


@dataclass(frozen=True)
class StatusPalette:
    pending:   RGB = (0.7, 0.7, 0.7)
    running:   RGB = (1.0, 0.8, 0.0)
    done:      RGB = (0.2, 0.8, 0.2)
    error:     RGB = (0.9, 0.2, 0.2)
    cancelled: RGB = (0.9, 0.2, 0.2)
    skipped:   RGB = (0.5, 0.5, 0.5)

    def get(self, status: str) -> RGB:
        return getattr(self, status, self.pending)


@dataclass(frozen=True)
class Sizes:
    block_default:      Tuple[int, int] = (160, 80)
    annotation_default: Tuple[int, int] = (200, 150)
    grid:               int = 20
    block_header:       int = 22  # title-bar height for regular (non-compact) blocks


@dataclass(frozen=True)
class Fonts:
    sans:    str = _SANS
    tile_pt: int = _TILE_PT


@dataclass(frozen=True)
class Palette:
    """Application chrome colors as hex strings."""

    window_bg:       str  # main window background
    panel_bg:        str  # docks, log panel
    base_bg:         str  # input field background
    alt_base_bg:     str  # alternating list rows
    text:            str  # primary text
    text_muted:      str  # captions, status label
    border:          str  # subtle borders
    button_bg:       str
    button_bg_hover: str
    button_bg_press: str
    button_text:     str
    accent:          str  # selection / highlight
    accent_text:     str  # text on accent
    section_open_bg:    str  # expanded collapsible-section header (palette)
    section_open_text:  str  # text/icon on an expanded section header
    section_open_panel: str  # block area behind an expanded section's tiles
    check_bg:        str  # checked-button background
    check_border:    str
    check_text:      str
    canvas_bg:       str  # workflow canvas scene background
    canvas_grid:     str  # canvas grid lines / dots


@dataclass(frozen=True)
class QSS:
    """QSS snippets derived from a Palette. Built once per theme."""

    status_label: str
    small_button: str
    small_button_checkable: str
    scrollbar: str

    @classmethod
    def from_palette(cls, p: Palette) -> "QSS":
        small_button = (
            f"QPushButton {{ background: {p.button_bg}; border: 1px solid {p.border};"
            f" border-radius: 3px; font-size: 9px; color: {p.text_muted}; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover}; }}"
            f"QPushButton:pressed {{ background: {p.button_bg_press}; }}"
        )
        small_button_checkable = (
            f"QPushButton {{ background: {p.button_bg}; border: 1px solid {p.border};"
            f" border-radius: 3px; font-size: 9px; color: {p.text_muted}; }}"
            f"QPushButton:hover {{ background: {p.button_bg_hover}; }}"
            f"QPushButton:checked {{ background: {p.check_bg};"
            f" border-color: {p.check_border}; color: {p.check_text}; }}"
        )
        # Slim, rounded, themed scrollbar — no arrow buttons, transparent
        # track. Handle uses the border tone, darkening/lightening to the
        # muted-text tone on hover (works in both light and dark).
        scrollbar = (
            "QScrollBar:vertical { background: transparent; width: 11px;"
            " margin: 0; }"
            f"QScrollBar::handle:vertical {{ background: {p.border};"
            " border-radius: 4px; min-height: 28px; margin: 2px; }"
            f"QScrollBar::handle:vertical:hover {{ background: {p.text_muted}; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            " height: 0; background: none; border: none; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
            " background: none; }"
            "QScrollBar:horizontal { background: transparent; height: 11px;"
            " margin: 0; }"
            f"QScrollBar::handle:horizontal {{ background: {p.border};"
            " border-radius: 4px; min-width: 28px; margin: 2px; }"
            f"QScrollBar::handle:horizontal:hover {{ background: {p.text_muted}; }}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {"
            " width: 0; background: none; border: none; }"
            "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {"
            " background: none; }"
        )
        return cls(
            status_label=f"color: {p.text_muted}; font-size: 10px;",
            small_button=small_button,
            small_button_checkable=small_button_checkable,
            scrollbar=scrollbar,
        )


@dataclass(frozen=True)
class Theme:
    name:    str
    status:  StatusPalette
    sizes:   Sizes
    fonts:   Fonts
    palette: Palette
    qss:     QSS


# -----------------------------------------------------------------------------
# Concrete themes
# -----------------------------------------------------------------------------

_LIGHT_PALETTE = Palette(
    window_bg       = "#f0f0f0",
    panel_bg        = "#f5f5f5",
    base_bg         = "#ffffff",
    alt_base_bg     = "#f5f5f5",
    text            = "#000000",
    text_muted      = "#555555",
    border          = "#bbbbbb",
    button_bg       = "#e0e0e0",
    button_bg_hover = "#d6d6d6",
    button_bg_press = "#cccccc",
    button_text     = "#000000",
    accent          = "#2a82da",
    accent_text     = "#ffffff",
    section_open_bg    = "#4d4d4d",   # dark grey, clear on the light panel
    section_open_text  = "#ffffff",
    section_open_panel = "#e4e4e4",   # darker than panel_bg behind open tiles
    check_bg        = "#dbeafe",
    check_border    = "#2a82da",
    check_text      = "#1f4e8e",
    canvas_bg       = "#f7f7f7",
    canvas_grid     = "#dcdcdc",
)

_DARK_PALETTE = Palette(
    window_bg       = "#2b2b2b",
    panel_bg        = "#313335",
    base_bg         = "#3c3f41",
    alt_base_bg     = "#353739",
    text            = "#dcdcdc",
    text_muted      = "#9aa0a6",
    border          = "#555555",
    button_bg       = "#444444",
    button_bg_hover = "#525252",
    button_bg_press = "#333333",
    button_text     = "#dcdcdc",
    accent          = "#4a86e8",
    accent_text     = "#ffffff",
    section_open_bg    = "#4a4a4a",   # dark grey, a lift above the dark panel
    section_open_text  = "#ffffff",
    section_open_panel = "#2a2c2e",   # darker than panel_bg behind open tiles
    check_bg        = "#1e3a5f",
    check_border    = "#4a86e8",
    check_text      = "#a9c7ff",
    canvas_bg       = "#1f2123",
    canvas_grid     = "#33363a",
)


def _build(name: str, palette: Palette) -> Theme:
    return Theme(
        name=name,
        status=StatusPalette(),
        sizes=Sizes(),
        fonts=Fonts(),
        palette=palette,
        qss=QSS.from_palette(palette),
    )


LIGHT = _build("light", _LIGHT_PALETTE)
DARK  = _build("dark",  _DARK_PALETTE)

_THEMES = {"light": LIGHT, "dark": DARK}


# -----------------------------------------------------------------------------
# Active-theme proxy
# -----------------------------------------------------------------------------

class _ThemeProxy:
    """Attribute-access proxy to the currently active Theme."""

    __slots__ = ()
    _active: Theme = LIGHT  # class-level so __getattr__ stays simple

    def __getattr__(self, name):
        return getattr(_ThemeProxy._active, name)


theme = _ThemeProxy()


def set_theme(name: str) -> Theme:
    """Activate the named theme (``"light"`` or ``"dark"``). Returns the Theme."""
    t = _THEMES.get(name)
    if t is None:
        raise ValueError(f"Unknown theme: {name!r}. Known: {sorted(_THEMES)}")
    _ThemeProxy._active = t
    return t


def get_theme_name() -> str:
    return _ThemeProxy._active.name


# -----------------------------------------------------------------------------
# Qt integration (lazy import so theme.py stays importable without Qt)
# -----------------------------------------------------------------------------

def build_qpalette(theme_obj: Theme = None):
    """Return a QPalette populated from the active theme's Palette tokens."""
    from PySide6.QtGui import QPalette, QColor
    t = theme_obj or _ThemeProxy._active
    p = t.palette
    qp = QPalette()
    qp.setColor(QPalette.ColorRole.Window,          QColor(p.window_bg))
    qp.setColor(QPalette.ColorRole.WindowText,      QColor(p.text))
    qp.setColor(QPalette.ColorRole.Base,            QColor(p.base_bg))
    qp.setColor(QPalette.ColorRole.AlternateBase,   QColor(p.alt_base_bg))
    qp.setColor(QPalette.ColorRole.ToolTipBase,     QColor(p.base_bg))
    qp.setColor(QPalette.ColorRole.ToolTipText,     QColor(p.text))
    qp.setColor(QPalette.ColorRole.Text,            QColor(p.text))
    qp.setColor(QPalette.ColorRole.Button,          QColor(p.button_bg))
    qp.setColor(QPalette.ColorRole.ButtonText,      QColor(p.button_text))
    qp.setColor(QPalette.ColorRole.BrightText,      QColor("#ff0000"))
    qp.setColor(QPalette.ColorRole.Link,            QColor(p.accent))
    qp.setColor(QPalette.ColorRole.Highlight,       QColor(p.accent))
    qp.setColor(QPalette.ColorRole.HighlightedText, QColor(p.accent_text))
    return qp
