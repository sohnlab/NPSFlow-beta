# Utility Blocks

## 2D Plot (`PlotData`) — interactive, dynamic
Full-featured interactive 2D plotting with collapsible accordion settings panel.

- Dynamic inputs: paired x/y ports per series (`x1:`, `y1:`, `Add series`)
- Only one of x or y required per series; missing axis uses indices
- Port labels show `xN: varName` when connected

**Settings panel** (collapsible sections, QSplitter-resizable):

| Section | Controls |
|---------|----------|
| Figure | Subplot count/arrange/share-X, figure size, background colors, gridlines (major/minor with color/alpha/style/width) |
| Series & Type | Plot type (Line/Scatter/Bar/Histogram/Stem/Step/Area), color, visibility |
| Axes | X/Y limits (auto/manual), scale (linear/log), invert, aspect ratio |
| Ticks | Font/size/color, rotation, major tick length/width/direction, minor ticks toggle/length/width/direction |
| Labels | Title, X/Y labels with font/size/color/bold/italic/TeX |
| Line & Marker | Width, style, marker type/size, fill color, edge color, edge width, line/marker opacity |
| Data Transform | X/Y offset, normalize (0-1), secondary Y-axis |
| Curve Fit | Polynomial/Exponential/Power/Logarithmic, degree, equation/R2 display |
| Legend | Show/hide (draggable), position, font size |
| Annotate | Text annotations (position, font, color, bg box), reference lines (H/V, color, style) |
| Export | Colorbar (show/label/orientation), DPI, Save Figure, Export CSV |

**Features:**
- All combo boxes and spin boxes ignore scroll wheel (click only)
- Subplot mode: manual count, per-subplot series assignment via checkboxes
- "Apply to" dropdown: All Subplots or individual subplot
- Figure size Apply/Reset with centered scroll area
- Draggable legend

## Plot (`PlotVector`)
Simple vector plot (non-interactive).
- Inputs: `vector`

## Custom Code (`CustomCode`) — dynamic
Execute arbitrary Python code on inputs.
- Dynamic inputs
- Outputs: `result`

## Ask User (`UserChoiceBlock`) — interactive
Prompts user with a question and options.
- Outputs: `choice`

## Reroute (`RerouteNode`)
Wire routing helper (pass-through).
