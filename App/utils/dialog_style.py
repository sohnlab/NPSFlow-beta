"""Shared dialog stylesheet, built from the active theme's tokens.

Dialogs call ``apply_dialog_style(dlg)``; the stylesheet is rebuilt each call
so dialogs opened after a theme switch use the current palette.
"""

import sys
from theme import theme

_SANS = "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"
_FONT_SIZE = 12


def _build_stylesheet() -> str:
    p = theme.palette
    return f"""
        * {{ font-family: "{_SANS}"; font-size: {_FONT_SIZE}px; color: {p.text}; }}
        QDialog {{ background: {p.window_bg}; }}
        QLabel {{ background: transparent; color: {p.text}; }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {p.border};
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 14px;
            color: {p.text};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }}
        QRadioButton {{ spacing: 4px; color: {p.text}; }}
        QCheckBox {{ spacing: 4px; color: {p.text}; }}
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
            border: 1px solid {p.border};
            border-radius: 3px;
            padding: 2px 6px;
            background: {p.base_bg};
            color: {p.text};
        }}
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {p.accent};
        }}
        QLineEdit:disabled, QTextEdit:disabled,
        QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background: {p.alt_base_bg};
            color: {p.text_muted};
        }}
        /* Hide spin up/down arrows app-wide for a cleaner, flatter look. */
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 0px;
            border: none;
        }}
        QComboBox {{
            border: 1px solid {p.border};
            border-radius: 3px;
            padding: 2px 6px;
            background: {p.base_bg};
            color: {p.text};
        }}
        QComboBox:focus {{ border-color: {p.accent}; }}
        QComboBox QAbstractItemView {{
            background: {p.base_bg};
            color: {p.text};
            selection-background-color: {p.accent};
            selection-color: {p.accent_text};
        }}
        QListWidget {{
            border: 1px solid {p.border};
            border-radius: 3px;
            background: {p.base_bg};
            color: {p.text};
        }}
        QListWidget::item:selected {{
            background: {p.accent};
            color: {p.accent_text};
        }}
        QPushButton {{
            border: 1px solid {p.border};
            border-radius: 4px;
            padding: 4px 12px;
            background: {p.button_bg};
            color: {p.button_text};
        }}
        QPushButton:hover {{ background: {p.button_bg_hover}; }}
        QPushButton:pressed {{ background: {p.button_bg_press}; }}
        QPushButton:default {{ border-color: {p.accent}; }}
        QPushButton:disabled {{
            background: {p.alt_base_bg};
            color: {p.text_muted};
            border-color: {p.border};
        }}
        QPushButton:checked {{
            background: {p.check_bg};
            border-color: {p.check_border};
            color: {p.check_text};
            font-weight: bold;
        }}
        QPushButton#confirmBtn {{
            border: 2px solid #44AA44;
            color: {p.text};
            font-weight: bold;
        }}
        QPushButton#confirmBtn:hover {{ border-color: #339933; }}
        QPushButton#confirmBtn:disabled {{
            background: {p.alt_base_bg};
            color: {p.text_muted};
            border-color: {p.border};
            font-weight: normal;
        }}
        QPushButton#primaryBtn {{
            border: 2px solid {p.accent};
            color: {p.text};
            font-weight: bold;
        }}
        QPushButton#primaryBtn:hover {{ border-color: {p.accent}; background: {p.button_bg_hover}; }}
        QPushButton#primaryBtn:disabled {{
            background: {p.alt_base_bg};
            color: {p.text_muted};
            border-color: {p.border};
            font-weight: normal;
        }}
        QDialogButtonBox QPushButton {{ min-width: 70px; }}
        QToolButton {{
            border: 1px solid {p.border};
            border-radius: 4px;
            padding: 2px 10px;
            background: {p.button_bg};
            color: {p.button_text};
        }}
        QToolButton:hover {{ background: {p.button_bg_hover}; }}
        QToolButton:pressed {{ background: {p.button_bg_press}; }}
        QToolButton:disabled {{
            background: {p.alt_base_bg};
            color: {p.text_muted};
            border-color: {p.border};
        }}
        QToolButton[popupMode="1"] {{ padding-right: 22px; }}
        QToolButton::menu-button {{
            border-left: 1px solid {p.border};
            width: 16px;
        }}
        QToolButton::menu-button:hover {{ background: {p.button_bg_hover}; }}
        QTabWidget::pane {{
            border: 1px solid {p.border};
            border-radius: 4px;
            background: {p.window_bg};
        }}
        QTabBar::tab {{
            border: 1px solid {p.border};
            border-radius: 0;
            padding: 5px 16px 5px 12px;
            background: {p.alt_base_bg};
            color: {p.text};
            margin-right: 2px;
        }}
        QTabBar::tab:hover {{ background: {p.button_bg_hover}; }}
        QTabBar::tab:selected {{ background: {p.window_bg}; }}
        QTabBar::tab:selected:hover {{ background: {p.window_bg}; }}
        QTabBar::close-button {{ margin-left: 8px; margin-right: 6px; }}
        QScrollArea {{ border: none; background: transparent; }}
        QToolBox::tab {{
            background: transparent;
            color: {p.text};
            border: none;
            border-top: 1px solid {p.border};
            border-radius: 0;
            padding: 4px;
            min-height: 28px;
            text-align: left;
            font-weight: bold;
        }}
        QToolBox::tab:hover {{ color: {p.accent}; }}
        QToolBox::tab:selected {{ color: {p.accent}; }}
        QToolBox {{ background: {p.window_bg}; }}
        QToolBox > QWidget,
        QToolBox QScrollArea,
        QToolBox QScrollArea > QWidget,
        QToolBox QScrollArea > QWidget > QWidget {{
            background: {p.window_bg};
        }}
    """


def apply_dialog_style(dialog):
    """Apply the theme-derived stylesheet to a QDialog."""
    dialog.setStyleSheet(_build_stylesheet())


_CONFIRM_GREEN        = '#44AA44'  # matches QSS #confirmBtn outline
_CONFIRM_TEXT_LIGHT   = '#2a7a2a'
_CONFIRM_TEXT_DARK    = '#7adb7a'


def _confirm_text_color() -> str:
    return _CONFIRM_TEXT_DARK if theme.name == 'dark' else _CONFIRM_TEXT_LIGHT


def style_mpl_button(btn, is_confirm=False):
    """Style a matplotlib.widgets.Button using the active theme palette."""
    p = theme.palette
    ax = btn.ax

    btn.color = p.button_bg
    btn.hovercolor = p.button_bg_hover
    ax.set_facecolor(p.button_bg)

    for spine in ax.spines.values():
        spine.set_visible(True)
        if is_confirm:
            spine.set_edgecolor(_CONFIRM_GREEN)
            spine.set_linewidth(2)
        else:
            spine.set_edgecolor(p.border)
            spine.set_linewidth(1)

    if btn.label:
        btn.label.set_fontfamily(_SANS)
        btn.label.set_fontsize(12)
        if is_confirm:
            btn.label.set_color(_confirm_text_color())
            btn.label.set_fontweight('bold')
        else:
            btn.label.set_color(p.text)
            btn.label.set_fontweight('normal')


def style_mpl_figure(fig):
    """Style a matplotlib Figure (and its existing axes) to match the active theme.

    Sets figure/axes face, spine/tick/label colors, and grid color from
    ``theme.palette``. Safe to call after axes are populated; re-call after a
    theme switch to refresh colors.
    """
    import matplotlib
    p = theme.palette

    fig.set_facecolor(p.panel_bg)

    for ax in fig.axes:
        ax.set_facecolor(p.base_bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(p.border)
        ax.tick_params(colors=p.text, which='both')
        ax.xaxis.label.set_color(p.text)
        ax.yaxis.label.set_color(p.text)
        if ax.get_title():
            ax.title.set_color(p.text)
        ax.grid(color=p.canvas_grid, which='major')
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(p.panel_bg)
            frame.set_edgecolor(p.border)
            for text in legend.get_texts():
                text.set_color(p.text)

    matplotlib.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  [_SANS, 'DejaVu Sans'],
        'font.size':        11,
        'axes.titlesize':   13,
        'axes.labelsize':   11,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':  10,
        'figure.facecolor': p.panel_bg,
        'axes.facecolor':   p.base_bg,
        'axes.edgecolor':   p.border,
        'axes.labelcolor':  p.text,
        'axes.titlecolor':  p.text,
        'xtick.color':      p.text,
        'ytick.color':      p.text,
        'text.color':       p.text,
        'grid.color':       p.canvas_grid,
    })
