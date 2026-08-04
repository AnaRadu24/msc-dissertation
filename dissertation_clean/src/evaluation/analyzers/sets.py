"""Named metric SETS — which analysers run for a given variant, declared in one place.

A VariantSpec names a set (e.g. 'competence'); the orchestrator looks it up here and runs
exactly those analysers on the rollout. Adding a set is one entry, so the contents of an
atom are always declared, never buried in orchestration logic.

Each analyser is a callable RolloutResult -> {metric_name: per-trial [B] array}. A set is
just an ordered list of them; the orchestrator merges their outputs into one atom.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from ..engine.types import RolloutResult
from .oscillation import oscillation_metrics
from .reach import reach_metrics
from .tremor import tremor_metrics

Analyser = Callable[[RolloutResult], Dict[str, np.ndarray]]

# name -> ordered analysers. Keys must not collide across analysers in one set (the
# orchestrator asserts this), so every metric in an atom traces to exactly one analyser.
METRIC_SETS: Dict[str, List[Analyser]] = {
    # R1 competence: accuracy at the trained horizon + reach quality. Roll under 'competence'.
    "competence": [reach_metrics],
    # R2 stability: hold-phase amplitude + signed-coordinate frequency. Roll under 'tremor'.
    "hold":       [oscillation_metrics],
    # R4 intention tremor: the distance-binned crescendo index. Roll under 'tremor'.
    "tremor":     [oscillation_metrics, tremor_metrics],
}


def analysers_for(metric_set: str) -> List[Analyser]:
    if metric_set not in METRIC_SETS:
        raise KeyError(f"unknown metric set {metric_set!r}; choose from {sorted(METRIC_SETS)}")
    return METRIC_SETS[metric_set]