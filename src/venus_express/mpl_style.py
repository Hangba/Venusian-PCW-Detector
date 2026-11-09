"""Shared matplotlib style helpers for Venus Express plots."""

from __future__ import annotations

import matplotlib


LIBERTINE_SERIF_FALLBACKS = [
    "Linux Libertine O",
    "Linux Libertine",
    "Libertinus Serif",
    "Times New Roman",
    "DejaVu Serif",
]


def apply_matplotlib_style(font_size: float = 12) -> None:
    """Apply the project-wide matplotlib font and math style."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": LIBERTINE_SERIF_FALLBACKS,
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.titlesize": font_size + 4,
            "axes.labelsize": font_size + 1,
            "legend.fontsize": max(font_size - 1, 8),
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.formatter.use_mathtext": True,
        }
    )
