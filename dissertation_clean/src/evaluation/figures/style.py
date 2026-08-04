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


def arch_color(architecture: str) -> str:
    """Colour for 'dual' / 'mono'; falls back to the first qualitative colour."""
    return ARCH_COLORS.get(architecture, QUALITATIVE[0])


def apply_style() -> None:
    """Set global rcParams for publication-quality figures. Idempotent."""
    mpl.rcParams.update({
        # fonts
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        # lines & markers
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
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
        # figure & saving (PNG is the deliverable; 300 dpi is print-quality)
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": True,
    })


def save_figure(fig: plt.Figure, path: str | Path, *, formats: Sequence[str] = ("png",),
                close: bool = True) -> Path:
    """Save ``fig`` to ``path`` (its stem) in each requested format; returns the PNG path.

    ``formats`` defaults to PNG only (the dissertation deliverable); pass ('png', 'pdf')
    for a vector copy. The parent directory is created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_png = path.with_suffix(".png")
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"))
    if close:
        plt.close(fig)
    return out_png