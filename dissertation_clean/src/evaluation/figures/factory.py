"""Figure factory (Layer 4 orchestration) — the ONE place that turns a (model, VariantSpec) into a
rendered, correctly-saved figure. The notebook calls this with a one-liner; it never writes a
rollout/render/save loop by hand.

Mirrors orchestrate.run_variant: same VariantSpec (so a figure and its metric atom describe the
IDENTICAL rollout) and same Paths contract (figures land at
results/figures/<model>/<regime>/<run_id>/<variant>/<name>.png, beside the atom). The pure figure
functions stay pure — they take a RolloutResult and return a Figure; only this factory rolls out
and saves. A figure spec maps a name -> (function, which RolloutResult field(s) it needs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..catalogue import ModelGroup, ModelRecord
from ..engine import rollout
from ..engine.types import VariantSpec
from ..paths import Paths
from ..settings import DEFAULT, Settings
from .style import apply_style, save_figure


@dataclass(frozen=True)
class FigureSpec:
    """A named figure = a pure plot function + how to build its title. ``fn`` takes a RolloutResult
    (+ kwargs) and returns a Figure. ``needs_dual`` skips it for mono models with a logged reason."""
    name: str
    fn: Callable[..., Figure]
    kwargs: dict
    needs_dual: bool = False


def render_figure(
    record: ModelRecord,
    paths: Paths,
    variant: VariantSpec,
    figure: FigureSpec,
    settings: Settings = DEFAULT,
    *,
    save: bool = True,
    show: bool = True,
    formats: Sequence[str] = ("png",),
) -> Optional[Figure]:
    """Roll ``record`` out under ``variant`` once, render ``figure``, save it to the per-model
    figure dir. Returns the Figure (or None if skipped). One rollout per call — the factory owns it."""
    if figure.needs_dual and record.architecture != "dual":
        print(f"[skip] {figure.name}: needs dual, {record.label} is {record.architecture}")
        return None
    apply_style()
    profile = settings.profile(variant.profile)
    res = rollout(record, paths, task=variant.task, duration_s=profile.duration_s,
                  batch_size=profile.batch_size, intervention=variant.intervention,
                  seed=settings.rollout_seed, device=settings.device)
    title = f"{record.label} · seed {record.seed} · {variant.name}"
    fig = figure.fn(res, title=title, **figure.kwargs)
    if save:
        out = paths.figure_dir(record.key, variant.name, create=True) / figure.name
        save_figure(fig, out, formats=formats, close=not show)
        print(f"[fig] {record.label} s{record.seed} · {variant.name}/{figure.name}.png -> {out}")
    if show:
        plt.show()
    return fig


def render_group(
    group: ModelGroup,
    paths: Paths,
    variant: VariantSpec,
    figure: FigureSpec,
    settings: Settings = DEFAULT,
    *,
    seeds: Optional[Sequence[int]] = None,
    save: bool = True,
    show: bool = False,
    formats: Sequence[str] = ("png",),
) -> list:
    """Render ``figure`` for every seed of a group (or a subset) under one variant. Each seed saves
    to ITS OWN run_id folder, so nothing overwrites — the fix for the flat-scratch-path collision."""
    recs = group.records if seeds is None else [group.record(s) for s in seeds]
    return [render_figure(r, paths, variant, figure, settings, save=save, show=show, formats=formats)
            for r in recs]
    
def delay_sweep_figure(
    sweeps_by_arch: dict,
    paths: Paths,
    *,
    name: str = "F5_money_plot",
    save: bool = True,
    show: bool = True,
    formats=("png",),
    **plot_kwargs,
):
    """Render + save the money plot from pre-computed sweeps. Saves under
    results/aggregated/delay/ because it is a CROSS-MODEL, CROSS-SEED figure (not per-model)."""
    from .money_plot import plot_money_plot
    apply_style()
    fig = plot_money_plot(sweeps_by_arch, **plot_kwargs)
    if save:
        out = paths.aggregated_dir("delay", create=True) / name
        save_figure(fig, out, formats=formats, close=not show)
        print(f"[fig] money plot -> {out}.png")
    if show:
        plt.show()
    return fig

def render_delay_series(
    group: ModelGroup,
    paths: Paths,
    figures: Sequence[FigureSpec],
    *,
    deltas_ms: Sequence[float] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150),
    mode: str = "additive_both",
    seeds: Optional[Sequence[int]] = None,
    settings: Settings = DEFAULT,
    save: bool = True,
    show: bool = False,
    formats: Sequence[str] = ("png",),
) -> int:
    """Render each FigureSpec for every (seed × delta) of a group — the batch loop, abstracted.

    One rollout per (seed, delta) is REUSED across all `figures` (so hold_dynamics and
    hold_trajectory_3d for the same condition don't roll out twice). Saves each to its per-model
    per-variant folder. `show=False` by default (batch mode: write, don't display). Returns the
    figure count. Pass a SMALL deltas_ms + one seed for the illustrative set; the full grid for the
    archive.
    """
    from ..engine import rollout
    from ..engine.types import VariantSpec
    from ..manipulators.delay import delay
    apply_style()
    recs = group.records if seeds is None else [group.record(s) for s in seeds]
    n = 0
    for rec in recs:
        bp, bv = rec.config.proprioception_delay, rec.config.vision_delay
        for d in deltas_ms:
            if d == 0:
                spec = VariantSpec(name="baseline_hold", task="reach", profile="tremor", metrics="hold")
            else:
                iv = delay(d, mode=mode, base_prop_ms=bp, base_vision_ms=bv)
                spec = VariantSpec(name=iv.label, intervention=iv, task="reach",
                                   profile="tremor", metrics="hold")
            profile = settings.profile(spec.profile)
            res = rollout(rec, paths, task=spec.task, duration_s=profile.duration_s,
                          batch_size=profile.batch_size, intervention=spec.intervention,
                          seed=settings.rollout_seed, device=settings.device)   # ONE rollout, reused
            for figspec in figures:
                if figspec.needs_dual and rec.architecture != "dual":
                    continue
                title = f"{rec.label} · seed {rec.seed} · {spec.name}"
                fig = figspec.fn(res, title=title, **figspec.kwargs)
                if save:
                    out = paths.figure_dir(rec.key, spec.name, create=True) / figspec.name
                    save_figure(fig, out, formats=formats, close=not show)
                    n += 1
                if show:
                    import matplotlib.pyplot as plt
                    plt.show()
    print(f"[batch] rendered {n} figures")
    return n

def crescendo_figure(
    record: ModelRecord,
    paths: Paths,
    *,
    deltas_ms: Sequence[float] = (0, 50, 100),
    mode: str = "additive_both",
    name: str = "R4_crescendo",
    settings: Settings = DEFAULT,
    save: bool = True,
    show: bool = True,
    formats=("png",),
) -> Figure:
    """R4 crescendo: roll ``record`` out at each delay, compute the intention-tremor profile, overlay.
    Saves to results/aggregated/tremor/ (a cross-condition figure for one model). Uses the REACH task
    (crescendo is a reach-phase measure) under the tremor profile duration."""
    from ..analyzers.tremor import intention_tremor_profile
    from ..engine import rollout
    from ..engine.types import IDENTITY
    from ..manipulators.delay import delay
    from .crescendo import plot_crescendo
    apply_style()
    bp, bv = record.config.proprioception_delay, record.config.vision_delay
    profile = settings.profile("tremor")
    profiles = {}
    for d in deltas_ms:
        iv = IDENTITY if d == 0 else delay(d, mode=mode, base_prop_ms=bp, base_vision_ms=bv)
        res = rollout(record, paths, task="reach", duration_s=profile.duration_s,
                      batch_size=profile.batch_size, intervention=iv,
                      seed=settings.rollout_seed, device=settings.device)
        profiles[f"+{d:g} ms" if d else "intact (Δ=0)"] = intention_tremor_profile(res)
    fig = plot_crescendo(profiles, title=f"Intention-tremor crescendo — {record.label} seed {record.seed}")
    if save:
        out = paths.aggregated_dir("tremor", create=True) / f"{name}_{record.label}_s{record.seed}"
        save_figure(fig, out, formats=formats, close=not show)
        print(f"[fig] crescendo -> {out}.png")
    if show:
        import matplotlib.pyplot as plt; plt.show()
    return fig