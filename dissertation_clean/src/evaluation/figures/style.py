"""Unified figure styling — the ONE place for fonts, sizes, colours and saving.

Call ``apply_style()`` once per notebook. Every figure then inherits it. Colours come
from the Okabe-Ito colourblind-safe palette. This module is pure matplotlib: it never
imports torch, never computes a metric, never reads a model.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# --- Okabe-Ito colourblind-safe palette -------------------------------------
OKABE_ITO = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "bluish_green": "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "purple":       "#CC79A7",
}

# Semantic assignments — use these names in plotting code, never raw hexes.
ARCH_COLORS = {
    "dual": OKABE_ITO["blue"],
    "mono": OKABE_ITO["vermillion"],
}
# A stable qualitative order for multi-condition plots (ablation channels, ladder rungs…).
QUALITATIVE = [OKABE_ITO[k] for k in
               ("blue", "vermillion", "bluish_green", "orange", "purple", "sky_blue", "black")]

# --- Physical sizing --------------------------------------------------------
# Thesis text block width, in inches. Measured by putting ``\the\textwidth`` in
# the document, compiling, and dividing the printed pt value by 72.27 (TeX
# points per inch). ~5.9 in is typical for infthesis on A4.
TEXT_WIDTH_IN = 5.9


def figure_size(fraction: float = 1.0, aspect: float = 0.618) -> tuple[float, float]:
    """Figsize for a figure meant to occupy ``fraction`` of the text width.

    Author at the display width, then include with
    ``\\includegraphics[width=<fraction>\\textwidth]`` (or ``width=\\linewidth``).
    With a 1:1 match the LaTeX scale factor is 1, so the point sizes below are
    the point sizes on the page. ``aspect`` is height/width (default: golden ratio).
    """
    width = TEXT_WIDTH_IN * fraction
    return (width, width * aspect)


def arch_color(architecture: str) -> str:
    """Colour for 'dual' / 'mono'; falls back to the first qualitative colour."""
    return ARCH_COLORS.get(architecture, QUALITATIVE[0])


def apply_style() -> None:
    """Set global rcParams for publication-quality figures. Idempotent."""
    mpl.rcParams.update({
        # fonts — sized to stay legible after LaTeX resizing. When authored at
        # the display width via figure_size(), these are the on-page point sizes.
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        # lines & markers — bumped so thin strokes don't vanish when scaled down
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.prop_cycle": cycler(color=QUALITATIVE),
        # axes: clean, no top/right spine, light y-grid only
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        # legend
        "legend.frameon": False,
        # figure & saving. PDF is the LaTeX deliverable (vector, crisp at any
        # scale); PNG is kept for quick preview. 300 dpi print-quality for PNG.
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": True,
    })


def save_figure(fig: plt.Figure, path: str | Path, *,
                formats: Sequence[str] = ("pdf", "png"),
                close: bool = True) -> Path:
    """Save ``fig`` to ``path`` (its stem) in each requested format.

    Writes a vector ``.pdf`` (include this one in LaTeX) and a ``.png`` preview by
    default. Returns the PNG path when PNG is requested — preserving the previous
    return contract for callers that display the preview — otherwise the first
    written path. The parent directory is created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path.with_suffix(f".{fmt}") for fmt in formats]
    for out in written:
        fig.savefig(out)
    if close:
        plt.close(fig)
    return path.with_suffix(".png") if "png" in formats else written[0]