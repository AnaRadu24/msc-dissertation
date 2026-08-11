"""R1 ablation figures (Layer 5, Figure Factory) — PURE matplotlib.

No torch, no rollout, no store. This module receives already-computed per-seed
scalars (terminal error, hold-phase speed RMS) in small dataclasses and turns them
into the three R1 ablation figures, matching the house style (``style.apply_style``)
and the seed-colour convention. The notebook never writes a rollout/render/save
loop by hand — it builds the dataclasses once (in the orchestration layer) and calls
``ablation_figures(...)`` with a single line.

Figures produced
----------------
1. ``fig_R1_ablation_bars``          — the feedback-ablation cascade (dual vs mono;
   intact / vision / prop / both freeze), per-seed dots coloured by seed, mean±SD
   printed above each bar.
2. ``fig_R1_loopspeed``              — the 2×2 loop-speed dissection: accuracy and
   stability, each shown ROBUST (dual n=5, per-seed dots) and CAUSAL (single-seed
   baseline / matched / swapped delay assignments).
3. ``fig_R1_config_ablation_grid``  — (optional / appendix) the full 4-condition ×
   3-config grid for terminal error and hold-RMS.

Layering rule: this file imports only numpy, matplotlib and ``.style``. Anything that
touches a model or the environment lives upstream (engine + analysers); the metrics
arrive here as plain arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .style import OKABE_ITO, QUALITATIVE, apply_style, arch_color, save_figure

__all__ = [
    "AblationCascade",
    "AblationSeries",
    "ConfigGrid",
    "LoopSpeed",
    "LoopSpeedCausal",
    "LoopSpeedRobust",
    "ablation_cascade_figure",
    "ablation_figures",
    "config_grid_figure",
    "loopspeed_figure",
]

# ---------------------------------------------------------------------------
# Semantic colours. Defaults chosen to match the attached figures; point any of
# these at a style.QUALITATIVE entry if you prefer the house categorical palette.
# The freeze-channel roles keep proprioception "warm" and vision "cool/green"
# consistently across every panel, so the reader learns the mapping once.
# ---------------------------------------------------------------------------
_C = {
    "dual":     arch_color("dual"),   # architecture bars (cascade)
    "mono":     arch_color("mono"), 
    "prop":     OKABE_ITO["orange"],   # frozen-channel roles (loop-speed panels)
    "vision":   OKABE_ITO["sky_blue"],
    "baseline": OKABE_ITO["bluish_green"],   # delay-assignment configs (causal / config grid)
    "matched":  OKABE_ITO["yellow"],
    "swapped":  OKABE_ITO["purple"],
}

# Reproducible horizontal jitter for per-seed dots (so overlapping seeds separate).
_RNG = np.random.default_rng(0)

# Canonical ablation conditions and their display labels.
_CONDITIONS = ("intact", "vision", "prop", "both")
_COND_LABEL = {"intact": "intact", "vision": "vision freeze",
               "prop": "prop freeze", "both": "both freeze"}


# ===========================================================================
# Data contracts — what the caller assembles upstream and passes in.
# Every array is per-seed (or per-config for the single-seed causal cut), in the
# stated units. Keeping these tiny and explicit is what lets the figure stay pure.
# ===========================================================================
@dataclass(frozen=True)
class AblationSeries:
    """Per-seed terminal error (cm) for ONE architecture across the four conditions.

    ``values[cond]`` is a ``(n_seeds,)`` array aligned to ``seeds``; ``cond`` is one
    of ``_CONDITIONS``. ``seeds`` is e.g. ``(42, 43, 44, 45, 46)``.
    """
    seeds: Sequence[int]
    values: Mapping[str, np.ndarray]   # condition -> (n_seeds,) terminal error, cm

    def arr(self, cond: str) -> np.ndarray:
        return np.asarray(self.values[cond], float)


@dataclass(frozen=True)
class AblationCascade:
    """The cascade figure's data: the two architectures side by side."""
    dual: AblationSeries
    mono: AblationSeries


@dataclass(frozen=True)
class LoopSpeedRobust:
    """Dual n=5 freezing of the two channels, both metrics (per-seed arrays)."""
    seeds: Sequence[int]
    terminal_prop:   np.ndarray   # (n_seeds,) cm   — freeze proprioception (the 50 ms loop)
    terminal_vision: np.ndarray   # (n_seeds,) cm   — freeze vision (the 100 ms loop)
    holdrms_prop:    np.ndarray   # (n_seeds,) cm/s
    holdrms_vision:  np.ndarray   # (n_seeds,) cm/s


@dataclass(frozen=True)
class LoopSpeedCausal:
    """Single-seed freezing across the three delay assignments (the causal test)."""
    configs: Sequence[str]        # ("baseline", "matched", "swapped")
    terminal_prop:   np.ndarray   # (n_configs,) cm
    terminal_vision: np.ndarray   # (n_configs,) cm
    holdrms_prop:    np.ndarray   # (n_configs,) cm/s
    holdrms_vision:  np.ndarray   # (n_configs,) cm/s


@dataclass(frozen=True)
class LoopSpeed:
    robust: LoopSpeedRobust
    causal: LoopSpeedCausal


@dataclass(frozen=True)
class ConfigGrid:
    """Optional appendix cut: every condition × every config, single-seed."""
    configs: Sequence[str]                    # ("baseline", "matched", "swapped")
    conditions: Sequence[str]                 # subset/order of _CONDITIONS
    terminal: Mapping[str, np.ndarray]        # config -> (n_conditions,) cm
    holdrms:  Mapping[str, np.ndarray]        # config -> (n_conditions,) cm/s


# ===========================================================================
# Small drawing primitives (reused across panels).
# ===========================================================================
def _seed_colour_map(*seed_groups: Sequence[int]) -> dict[int, object]:
    """Deterministic seed -> colour, shared across every panel so a seed keeps its
    colour everywhere. Cycles the house QUALITATIVE palette in ascending seed order."""
    seeds = sorted({int(s) for grp in seed_groups for s in grp})
    return {s: QUALITATIVE[i % len(QUALITATIVE)] for i, s in enumerate(seeds)}


def _seeded_bar(ax, x: float, values: np.ndarray, seeds: Sequence[int],
                colour, seed_cmap: dict[int, object], *, width: float = 0.8,
                jitter: float = 0.09) -> tuple[float, float]:
    """One bar = across-seed mean; SD whisker (ddof=1); per-seed dots coloured by
    seed; ``mean±SD`` printed above. Returns (mean, sd)."""
    v = np.asarray(values, float)
    m = float(np.nanmean(v))
    s = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
    ax.bar(x, m, width=width, color=colour, alpha=0.85, zorder=2,
           yerr=(s if v.size > 1 else None), capsize=4,
           error_kw=dict(zorder=4, lw=1.2, ecolor="0.15"))
    xj = x + _RNG.uniform(-jitter, jitter, v.size)
    for xi, vi, sd in zip(xj, v, seeds):
        ax.scatter(xi, vi, s=26, color=seed_cmap[int(sd)],
                   edgecolor="k", linewidth=0.4, zorder=5)
    ax.annotate(f"{m:.1f}±{s:.1f}", (x, m + s), textcoords="offset points",
                xytext=(0, 5), ha="center", va="bottom",
                fontsize=8, fontweight="bold", zorder=6)
    return m, s


def _grouped_bars(ax, groups: Sequence[str], series: Mapping[str, np.ndarray],
                  colours: Mapping[str, object], *, annotate: bool = True) -> None:
    """Single-seed grouped bars: one cluster per group, one bar per series key."""
    names = list(series)
    n = len(names)
    x = np.arange(len(groups))
    width = 0.8 / n
    for i, name in enumerate(names):
        arr = np.asarray(series[name], float)
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, arr, width=width * 0.95, color=colours[name],
               alpha=0.85, label=name, zorder=2)
        if annotate:
            for xi, vi in zip(x + off, arr):
                ax.annotate(f"{vi:.1f}", (xi, vi), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def _seed_legend(fig_or_ax, seed_cmap: dict[int, object], **kw):
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=c, markeredgecolor="k", markeredgewidth=0.4,
                      label=f"seed {s}") for s, c in seed_cmap.items()]
    return fig_or_ax.legend(handles=handles, title="network seed",
                            fontsize=8, title_fontsize=8, **kw)


# ===========================================================================
# Figure 1 — the ablation cascade (dual vs mono).
# ===========================================================================
def ablation_cascade_figure(cascade: AblationCascade,
                            seed_cmap: Optional[dict[int, object]] = None):
    """Reproduce the feedback-ablation cascade: four conditions, two architectures,
    per-seed dots coloured by seed, mean±SD above each bar."""
    seed_cmap = seed_cmap or _seed_colour_map(cascade.dual.seeds, cascade.mono.seeds)
    x = np.arange(len(_CONDITIONS))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, cond in enumerate(_CONDITIONS):
        _seeded_bar(ax, x[i] - w / 2, cascade.dual.arr(cond), cascade.dual.seeds,
                    _C["dual"], seed_cmap, width=w * 0.95)
        _seeded_bar(ax, x[i] + w / 2, cascade.mono.arr(cond), cascade.mono.seeds,
                    _C["mono"], seed_cmap, width=w * 0.95)

    ax.set_xticks(x)
    ax.set_xticklabels([_COND_LABEL[c] for c in _CONDITIONS])
    ax.set_ylabel("terminal error at $t=1$ s (cm)")
    ax.set_title("Feedback ablation — terminal error across seeds (dots = per-seed means)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    arch = [mpatches.Patch(facecolor=_C["dual"], alpha=0.85,
                           label=f"dual (mean ± SD, n={len(cascade.dual.seeds)})"),
            mpatches.Patch(facecolor=_C["mono"], alpha=0.85,
                           label=f"mono (mean ± SD, n={len(cascade.mono.seeds)})")]
    leg1 = ax.legend(handles=arch, loc="upper left", fontsize=9)
    ax.add_artist(leg1)
    _seed_legend(ax, seed_cmap, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    return fig


# ===========================================================================
# Figure 2 — the 2×2 loop-speed dissection.
# ===========================================================================
def loopspeed_figure(loop: LoopSpeed,
                     seed_cmap: Optional[dict[int, object]] = None):
    """Reproduce the accuracy/stability × robust/causal 2×2. Robust panels carry the
    per-seed dots and the prop/vision fold; causal panels are single-seed and show
    the catastrophic-channel FLIP across delay assignments."""
    r, c = loop.robust, loop.causal
    seed_cmap = seed_cmap or _seed_colour_map(r.seeds)
    fold = lambda p, v: float(np.nanmean(p)) / float(np.nanmean(v))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_ar, ax_sr), (ax_ac, ax_sc) = axes

    # ---- top-left: ACCURACY, robust ------------------------------------------------
    _seeded_bar(ax_ar, 0, r.terminal_prop,   r.seeds, _C["prop"],   seed_cmap)
    _seeded_bar(ax_ar, 1, r.terminal_vision, r.seeds, _C["vision"], seed_cmap)
    ax_ar.set_xticks([0, 1])
    ax_ar.set_xticklabels(["prop freeze\n(50 ms loop)", "vision freeze\n(100 ms loop)"])
    ax_ar.set_ylabel("terminal error at 1 s (cm)")
    ax_ar.set_title("ACCURACY — robust (dual n={}): freezing the FASTER (prop) loop is worse\n"
                    "prop/vision terminal fold = {:.1f}×"
                    .format(len(r.seeds), fold(r.terminal_prop, r.terminal_vision)),
                    fontsize=9)

    # ---- top-right: STABILITY, robust ----------------------------------------------
    _seeded_bar(ax_sr, 0, r.holdrms_prop,   r.seeds, _C["prop"],   seed_cmap)
    _seeded_bar(ax_sr, 1, r.holdrms_vision, r.seeds, _C["vision"], seed_cmap)
    ax_sr.set_xticks([0, 1])
    ax_sr.set_xticklabels(["prop freeze", "vision freeze"])
    ax_sr.set_ylabel("hold-phase speed RMS (cm/s)")
    ax_sr.set_title("STABILITY — robust (dual n={}): prop-freeze more destabilising\n"
                    "prop/vision hold-RMS fold = {:.1f}× (large across-seed SD)"
                    .format(len(r.seeds), fold(r.holdrms_prop, r.holdrms_vision)),
                    fontsize=9)

    # ---- bottom-left: ACCURACY, causal (single-seed) -------------------------------
    _grouped_bars(ax_ac, list(c.configs),
                  {"vision freeze": c.terminal_vision, "prop freeze": c.terminal_prop},
                  {"vision freeze": _C["vision"], "prop freeze": _C["prop"]})
    ax_ac.set_ylabel("terminal error @1 s (cm)")
    ax_ac.set_title("ACCURACY — causal (single-seed): catastrophic channel FLIPS with the\n"
                    "faster loop (baseline→prop, swapped→vision)", fontsize=9)
    ax_ac.legend(fontsize=8, loc="upper left")

    # ---- bottom-right: STABILITY, causal (single-seed) -----------------------------
    _grouped_bars(ax_sc, list(c.configs),
                  {"vision freeze": c.holdrms_vision, "prop freeze": c.holdrms_prop},
                  {"vision freeze": _C["vision"], "prop freeze": _C["prop"]})
    ax_sc.set_ylabel("hold-phase speed RMS (cm/s)")
    ax_sc.set_title("STABILITY — causal (single-seed): hold-RMS does NOT flip to vision when\n"
                    "swapped (vision-freeze wrecks accuracy there, not hold-stability)", fontsize=9)
    ax_sc.legend(fontsize=8, loc="upper right")

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    _seed_legend(fig, seed_cmap, loc="lower center", ncol=len(seed_cmap),
                 bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ===========================================================================
# Figure 3 — optional full condition × config grid (appendix).
# ===========================================================================
def config_grid_figure(grid: ConfigGrid):
    """The exploratory full grid: every condition frozen, across all three delay
    assignments, for both metrics. Single-seed per config (an appendix robustness
    view; the distilled claim lives in ``loopspeed_figure``)."""
    labels = [_COND_LABEL.get(c, c) for c in grid.conditions]
    colours = {cfg: _C[cfg] for cfg in grid.configs}

    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(13, 4.8))
    _grouped_bars(ax_t, labels, {cfg: grid.terminal[cfg] for cfg in grid.configs},
                  colours, annotate=False)
    ax_t.set_ylabel("terminal error at 1 s (cm)")
    ax_t.set_title("terminal error (cm)", fontsize=10)
    ax_t.legend(fontsize=8, loc="upper left")

    _grouped_bars(ax_h, labels, {cfg: grid.holdrms[cfg] for cfg in grid.configs},
                  colours, annotate=False)
    ax_h.set_ylabel("hold-phase speed RMS (cm/s)")
    ax_h.set_title("hold-phase speed RMS (cm/s)", fontsize=10)
    ax_h.legend(fontsize=8, loc="upper left")

    for ax in (ax_t, ax_h):
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    return fig


# ===========================================================================
# Public entry — the notebook one-liner.
# ===========================================================================
def ablation_figures(*, cascade: Optional[AblationCascade] = None,
                     loopspeed: Optional[LoopSpeed] = None,
                     config_grid: Optional[ConfigGrid] = None,
                     paths=None, out_dir=None, save: bool = True,
                     show: bool = True) -> dict:
    """Render whichever ablation figures the supplied data support, save them under a
    stable name, and return ``{name: fig}``. Seed colours are shared across figures.

    Example (notebook, one line)::

        figs = ablation_figures(cascade=cascade, loopspeed=loop, paths=paths)
    """
    apply_style()

    # one shared seed→colour map across every figure that shows seeds
    seed_groups = []
    if cascade is not None:
        seed_groups += [cascade.dual.seeds, cascade.mono.seeds]
    if loopspeed is not None:
        seed_groups.append(loopspeed.robust.seeds)
    seed_cmap = _seed_colour_map(*seed_groups) if seed_groups else {}

    figs: dict = {}
    if cascade is not None:
        figs["fig_R1_ablation_bars"] = ablation_cascade_figure(cascade, seed_cmap)
    if loopspeed is not None:
        figs["fig_R1_loopspeed"] = loopspeed_figure(loopspeed, seed_cmap)
    if config_grid is not None:
        figs["fig_R1_config_ablation_grid"] = config_grid_figure(config_grid)

    if save and figs:
        d = out_dir if out_dir is not None else paths.aggregated_dir("R1", create=True)
        for name, fig in figs.items():
            save_figure(fig, d / name, formats=("png",), close=False)
    if show:
        plt.show()
    return figs

# orchestrate.py — deviation-from-intact time course. Reuses the intact rollout per model.
import numpy as np

from ..engine import obs_layout, rollout
from ..engine.types import IDENTITY
from .ablation_deviation import DeviationSeries
from ..manipulators.ablation import ablate

_KW = dict(task="reach", duration_s=1.0, batch_size=256, seed=42)   # same targets across intact/frozen
_M_TO_CM = 100.0

def _deviation_series(group, paths):
    per_seed = {c: [] for c in ("vision", "prop", "both")}
    t = None
    for rec in group.records:
        layout = obs_layout(rec)
        intact = rollout(rec, paths, intervention=IDENTITY, **_KW)          # (T+1, B, 2), metres
        t = np.arange(intact.trajectories.shape[0]) * intact.dt
        for c in per_seed:
            frozen = rollout(rec, paths, intervention=ablate(c, layout), **_KW)
            # per-trial Euclidean deviation from the SAME seed's intact path, then mean over trials
            dev = np.linalg.norm(frozen.trajectories - intact.trajectories, axis=-1) * _M_TO_CM  # (T+1, B)
            per_seed[c].append(np.nanmean(dev, axis=1))                     # -> (T+1,)
    return DeviationSeries(t, tuple(group.seeds),
                           {c: np.array(v) for c, v in per_seed.items()})

def build_deviation(cat, paths):
    return (_deviation_series(cat.canonical("dual"), paths),
            _deviation_series(cat.canonical("mono"), paths))