"""Delay-plane heatmap (R4 / R2 extension, Layer 5) — PURE matplotlib.

"Which channel tolerates more injected feedback delay?" A grid over injected
proprioceptive delay (y) × injected visual delay (x), coloured by mean terminal error
at the trained horizon (t = 1 s), averaged across the architecture's seeds. Each cell
is annotated with its value. Runnable for dual and mono to ask whether the tolerance
asymmetry is architecture-specific or a shared property of the plant/task.

The hold-instability contour is OFF by default (it conflated the message): this figure
is a pure accuracy landscape. To restore the stability boundary, pass a holdrms grid
and draw_contour=True.

Selected cells (``marks``) are ringed so a heat cell ties to its trajectory / phase-plane
composite — rendered by the driver, which reuses hold_dynamics.plot_reach_2d_by_time and
phaseplane_composite.plot_phaseplane_composite (NOT reimplemented here).

Pure: numpy + matplotlib + .style only. Metrics computed upstream, handed in as a DelayPlane.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from ..analyzers.oscillation import OSC_RMS_THRESHOLD_CMS
from .style import apply_style, save_figure

__all__ = ["DelayPlane", "delay_plane_figure"]


@dataclass(frozen=True)
class DelayPlane:
    """Terminal-error landscape over the injected-delay grid.

    ``prop_ms`` / ``vision_ms``   : grid axes (added ms on each channel), lengths P, V.
    ``terminal_cm``               : (P, V) across-seed MEAN terminal error, cm [prop=rows].
    ``terminal_cm_per_seed``      : (P, V, n_seeds) raw per-seed values, kept so panel cells
                                    can be checked for representativeness. Optional.
    ``holdrms_cms``               : (P, V) hold-RMS, cm/s — only needed to re-enable the
                                    instability contour. Optional (default off).
    ``n_seeds``                   : number of seeds averaged (for the title).
    ``marks``                     : [(prop_ms, vision_ms, label), ...] cells with panels.
    """
    prop_ms: np.ndarray
    vision_ms: np.ndarray
    terminal_cm: np.ndarray
    terminal_cm_per_seed: Optional[np.ndarray] = None
    holdrms_cms: Optional[np.ndarray] = None
    n_seeds: int = 0
    marks: Sequence[tuple] = field(default_factory=tuple)


def _edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges from (possibly non-uniform) centres, for pcolormesh."""
    c = np.asarray(centres, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def delay_plane_figure(plane: DelayPlane, *,
                       arch: str = "",
                       vmin: Optional[float] = None, vmax: Optional[float] = None,
                       cmap: str = "magma_r",
                       annotate: bool = True, annotate_fmt: str = "{:.1f}",
                       draw_contour: bool = False,
                       contour_cms: float = OSC_RMS_THRESHOLD_CMS,
                       paths=None, out_dir=None,
                       save: bool = True, show: bool = True):
    """Heatmap of mean terminal error over the (prop, vision) injected-delay grid, each
    cell labelled with its value. Pass ``vmin``/``vmax`` to fix the colour scale across
    two figures (essential for a fair dual-vs-mono comparison). ``arch`` names the file
    and title so dual/mono don't overwrite."""
    apply_style()
    Z = np.asarray(plane.terminal_cm, float)
    xe, ye = _edges(plane.vision_ms), _edges(plane.prop_ms)   # x = vision, y = prop

    fig, ax = plt.subplots(figsize=(8.8, 6.9))
    fig.set_layout_engine("none")                            # defeat apply_style autolayout vs colourbar
    norm = Normalize(vmin if vmin is not None else np.nanmin(Z),
                     vmax if vmax is not None else np.nanmax(Z))
    im = ax.pcolormesh(xe, ye, Z, cmap=cmap, norm=norm, shading="flat")

    # per-cell numeric labels, contrast-aware (white on dark cells, black on light)
    if annotate:
        cmap_obj = plt.get_cmap(cmap)
        fs = 8.0 if max(Z.shape) <= 7 else 6.5
        for i, p in enumerate(plane.prop_ms):
            for j, v in enumerate(plane.vision_ms):
                val = Z[i, j]
                if not np.isfinite(val):
                    continue
                r, g, b, _ = cmap_obj(norm(val))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(v, p, annotate_fmt.format(val), ha="center", va="center",
                        fontsize=fs, color="white" if lum < 0.5 else "black")

    # optional instability contour (default OFF; needs a holdrms grid)
    contour_handle = None
    if draw_contour:
        if plane.holdrms_cms is None:
            raise ValueError("draw_contour=True but plane.holdrms_cms is None — "
                             "populate holdrms_cms in the driver to use the contour.")
        Xc, Yc = np.meshgrid(plane.vision_ms, plane.prop_ms)
        ax.contour(Xc, Yc, plane.holdrms_cms, levels=[contour_cms],
                   colors="cyan", linewidths=2.0)
        contour_handle = Line2D([0], [0], color="cyan", lw=2.0,
                                label=f"hold-RMS = {contour_cms:g} cm/s ($\\Delta_c$)")

    # ring the cells that have trajectory / composite panels
    for m in plane.marks:
        pm, vm, lbl = m[0], m[1], (m[2] if len(m) > 2 else "")
        ax.plot(vm, pm, marker="o", ms=12, mfc="none", mec="#39ff14", mew=2.2, zorder=6)
        if lbl:
            ax.annotate(lbl, (vm, pm), textcoords="offset points", xytext=(8, 7),
                        color="#39ff14", fontsize=8, fontweight="bold", zorder=7)

    ax.set_xlabel("injected visual delay (ms)")
    ax.set_ylabel("injected proprioceptive delay (ms)")
    ttl = "Delay-tolerance plane: proprioceptive vs visual injected delay"
    if arch:
        ttl += f"  ({arch}" + (f", n={plane.n_seeds}" if plane.n_seeds else "") + ")"
    ax.set_title(ttl + "\nmean terminal error at $t=1$ s (cm)")

    fig.subplots_adjust(left=0.11, right=0.86, top=0.88, bottom=0.10)
    cax = fig.add_axes([0.885, 0.10, 0.03, 0.78])
    fig.colorbar(im, cax=cax, label="terminal error (cm)")
    if contour_handle is not None:
        ax.legend(handles=[contour_handle], loc="upper right", fontsize=8, framealpha=0.9)

    if save:
        d = out_dir if out_dir is not None else paths.aggregated_dir("R4", create=True)
        name = "fig_R4_delay_plane" + (f"_{arch}" if arch else "")
        save_figure(fig, d / name, formats=("png",), close=False)
    if show:
        plt.show()
    return {f"fig_R4_delay_plane_{arch or 'dual'}": fig}