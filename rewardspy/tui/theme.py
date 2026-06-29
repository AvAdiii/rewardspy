"""Shared colors and status glyphs for the dashboard.

A cool "instrument panel" palette: deep slate background, a teal primary for
structure, a violet line for the reward curve, and vivid traffic-light status
colors. One palette, used by both the render functions and the stylesheet.
"""

from __future__ import annotations

# Structure.
ACCENT = "#2dd4bf"        # teal: borders, brand, bars
ACCENT_SOFT = "#9af0e6"
CURVE = "#a78bfa"         # violet: the reward line edge
CURVE_FILL = "#5a4a8f"    # dim violet: the area under the line
TEXT = "#cdd6e4"
MUTED = "#69768c"
DIM = "#2f3a48"
SCREEN_BG = "#0a0e14"
PANEL_BG = "#0f141c"

# Status (traffic lights).
OK = "#3fcf8e"
WARN = "#e6b54a"
ALERT = "#ff5c7a"

# Block ramp for bars, empty to full.
BLOCKS = " ▁▂▃▄▅▆▇█"
ASCII_BLOCKS = " .:-=+*#@"

_STATUS = {
    "OK": ("✓", OK),
    "WARNING": ("▲", WARN),
    "ALERT": ("✕", ALERT),
    "INSUFFICIENT_DATA": ("·", DIM),
    "NOT_APPLICABLE": ("–", DIM),
}


def status_style(status: str) -> tuple[str, str]:
    """Return ``(glyph, color)`` for a status string. Unknown maps to dim."""
    return _STATUS.get(status, ("·", DIM))
