"""Delay-sweep driver — sweep a DelayMode for one model-seed and extract the critical value.

Generalised over DelayMode: additive (x in ms) or multiplicative (x = severity factor k). For each
sweep value it runs the delay intervention under the 'tremor' profile, records the ORDER PARAMETER
(hold-phase speed RMS by default), writes a per-value atom (so the whole curve persists — the money
plot / heatmap data), and extracts the critical value where the order parameter first crosses the
threshold. Δc is reported as BOTH a linear-interpolated point AND the bracket [last sub-threshold,
first supra-threshold] value, so the point never overstates the grid resolution.

This is orchestration (drives rollouts), so it lives beside orchestrate.py; analyzers/ stay pure.
The order-parameter metric is a parameter, so the same driver serves whatever metric a later claim
rests on (hold-RMS now; a crescendo/intention metric later) without changing the sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .analyzers.oscillation import OSC_RMS_THRESHOLD_CMS
from .catalogue import ModelGroup, ModelRecord
from .engine.types import IDENTITY, VariantSpec
from .manipulators.delay import DELAY_MODES, delay
from .orchestrate import run_variant
from .paths import Paths
from .settings import DEFAULT, Settings

# default order parameter: the metric key the sweep thresholds on, and its threshold.
DEFAULT_ORDER_PARAM = "hold_speed_rms_cms"
DEFAULT_METRIC_SET = "hold"


@dataclass
class DelaySweepResult:
    """One model-seed's order-parameter-vs-sweep curve and its critical value."""
    model: str
    seed: int
    mode: str
    unit: str                       # sweep-axis unit ('ms' or '×'), from the DelayMode
    order_param: str                # which metric was thresholded
    x: np.ndarray                   # [G] sweep-parameter grid
    y: np.ndarray                   # [G] order parameter at each x
    threshold: float
    delta_c: float                  # linear-interpolated critical value (NaN if never crosses)
    bracket: tuple                  # (last sub-threshold x, first supra-threshold x); NaNs if N/A

    @property
    def crossed(self) -> bool:
        return np.isfinite(self.delta_c)


def _crossing(x: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, tuple]:
    """Linear-interpolated first upward crossing of ``thr`` + the bracketing grid values.
    Returns (delta_c, (last_below, first_above)). NaN point if never crosses; if already above at
    x[0], point = x[0] and bracket = (nan, x[0])."""
    above = y >= thr
    if not above.any():
        return float("nan"), (float("nan"), float("nan"))
    first = int(np.argmax(above))
    if first == 0:
        return float(x[0]), (float("nan"), float(x[0]))
    x0, x1, y0, y1 = x[first - 1], x[first], y[first - 1], y[first]
    dc = float(x1) if y1 == y0 else float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))
    return dc, (float(x0), float(x1))


def delay_sweep(
    record: ModelRecord,
    paths: Paths,
    *,
    mode: str = "additive_both",
    grid: Sequence[float] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150),
    order_param: str = DEFAULT_ORDER_PARAM,
    threshold: float = OSC_RMS_THRESHOLD_CMS,
    metric_set: str = DEFAULT_METRIC_SET,
    settings: Settings = DEFAULT,
    save: bool = True,
) -> DelaySweepResult:
    """Sweep ``grid`` under ``mode`` for one model-seed; threshold ``order_param`` to get Δc.

    ``grid`` is in ms for additive modes, or severity factors k (>=1) for multiplicative. The order
    parameter and threshold are parameters, so the same driver serves any metric a claim rests on.
    x=0 (additive) / k=1 (multiplicative) is the intact baseline (IDENTITY), reusing the baseline
    atom."""
    m = DELAY_MODES[mode]
    bp, bv = float(record.config.proprioception_delay), float(record.config.vision_delay)
    baseline_x = 0.0 if m.x_is_ms else 1.0

    xs, ys = [], []
    for x in grid:
        is_baseline = (x == baseline_x)
        spec = VariantSpec(
            name="baseline_hold" if is_baseline else delay(x, mode=mode, base_prop_ms=bp,
                                                            base_vision_ms=bv).label,
            intervention=IDENTITY if is_baseline else delay(x, mode=mode, base_prop_ms=bp,
                                                            base_vision_ms=bv),
            task="reach", profile="tremor", metrics=metric_set,
        )
        atom = run_variant(record, paths, spec, settings, save=save)
        xs.append(float(x))
        ys.append(atom.metrics[order_param]["mean"])

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dc, bracket = _crossing(xs, ys, threshold)
    return DelaySweepResult(model=record.label, seed=record.seed, mode=mode, unit=m.unit,
                            order_param=order_param, x=xs, y=ys, threshold=threshold,
                            delta_c=dc, bracket=bracket)


def delay_sweep_group(group: ModelGroup, paths: Paths, *, mode: str = "additive_both",
                      grid: Sequence[float] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150),
                      order_param: str = DEFAULT_ORDER_PARAM, threshold: float = OSC_RMS_THRESHOLD_CMS,
                      metric_set: str = DEFAULT_METRIC_SET, settings: Settings = DEFAULT,
                      save: bool = True) -> List[DelaySweepResult]:
    """Run the sweep across every seed of a group — Δc + bracket per seed, ready for mean ± SD."""
    return [delay_sweep(r, paths, mode=mode, grid=grid, order_param=order_param,
                        threshold=threshold, metric_set=metric_set, settings=settings, save=save)
            for r in group.records]

def load_delay_sweep(record: ModelRecord, paths: Paths, *, mode: str = "additive_both",
                     channel_key: str = "prop",  # which channel's added delay indexes the x-axis
                     order_param: str = DEFAULT_ORDER_PARAM,
                     threshold: float = OSC_RMS_THRESHOLD_CMS) -> DelaySweepResult:
    """Rebuild a DelaySweepResult for one model-seed from its ALREADY-SAVED atoms — no rollout.

    Reads the seed's baseline_hold atom (the Δ=0 point) and every delay_<mode>_* atom, recovering
    the added delay from each atom's config_overrides and the order parameter from its metrics, then
    re-derives Δc + bracket. This is the store-driven replot path: once a sweep has run, F5 can be
    rebuilt after any kernel restart with no recomputation.

    ``channel_key`` picks which channel's increment labels the x-axis (both are equal for
    additive_both; use 'prop' by convention). For multiplicative sweeps the x is the factor, stored
    below as the max increment / base — kept simple: additive modes are the common case.
    """
    import json
    import os

    from .store import MetricAtom

    m = DELAY_MODES[mode]
    metrics_root = paths.results / "metrics" / record.key.model / record.key.regime / record.key.run_id
    if not metrics_root.is_dir():
        raise FileNotFoundError(f"no saved atoms for {record.label} seed {record.seed} at {metrics_root}")

    xs, ys = [], []
    for variant_dir in sorted(metrics_root.iterdir()):
        atom_path = variant_dir / "metrics.json"
        if not atom_path.is_file():
            continue
        atom = MetricAtom.load(atom_path)
        if atom.variant == "baseline_hold":
            x = 0.0 if m.x_is_ms else 1.0
        elif atom.variant.startswith(f"delay_{mode}_"):
            ov = atom.config_overrides
            add = ov.get(f"_delay_add_{channel_key}_ms")
            if add is None:                                  # this channel wasn't delayed (e.g. vision-only)
                add = ov.get("_delay_add_vision_ms", ov.get("_delay_add_prop_ms"))
            x = float(add)
        else:
            continue                                          # not part of this sweep
        if order_param not in atom.metrics:
            continue
        xs.append(x); ys.append(atom.metrics[order_param]["mean"])

    if not xs:
        raise FileNotFoundError(f"no delay/baseline atoms for {record.label} seed {record.seed} "
                                f"under mode '{mode}' at {metrics_root}")
    order = np.argsort(xs)
    xs = np.asarray(xs, float)[order]
    ys = np.asarray(ys, float)[order]
    dc, bracket = _crossing(xs, ys, threshold)
    return DelaySweepResult(model=record.label, seed=record.seed, mode=mode, unit=m.unit,
                            order_param=order_param, x=xs, y=ys, threshold=threshold,
                            delta_c=dc, bracket=bracket)


def load_delay_sweep_group(group: ModelGroup, paths: Paths, *, mode: str = "additive_both",
                           order_param: str = DEFAULT_ORDER_PARAM,
                           threshold: float = OSC_RMS_THRESHOLD_CMS) -> List[DelaySweepResult]:
    """Rebuild every seed's sweep from saved atoms — the replot-after-restart path for F5."""
    return [load_delay_sweep(r, paths, mode=mode, order_param=order_param, threshold=threshold)
            for r in group.records]