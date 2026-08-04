"""Orchestrator — the join between the engine, the analysers and the store.

run_variant  : one trained model-seed × one VariantSpec -> one MetricAtom on disk.
run_group    : every seed of a ModelGroup under one spec -> a list of atoms.
run_specs    : one model × several specs (e.g. the baseline's competence + hold).

The ONLY orchestration logic lives here: resolve the profile to (duration, batch), roll out
once (option a — one profile per spec), run the declared metric set, summarise ACROSS TRIALS,
and persist. It records the full provenance (intervention, task, profile, metric set) into the
atom so a stored number always traces to exactly how it was produced. No maths, no plotting.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .analyzers.sets import analysers_for
from .catalogue import ModelGroup, ModelRecord
from .engine import rollout
from .engine.types import VariantSpec
from .paths import Paths
from .settings import DEFAULT, Settings
from .store import MetricAtom


def _summarise_across_trials(per_trial: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Collapse each metric's per-trial [B] array to {mean, std, n} across trials. NaNs are
    ignored (nanmean/nanstd) so a metric undefined on some trials (e.g. an unresolvable
    frequency) does not poison the summary; n counts the finite trials."""
    out = {}
    for name, values in per_trial.items():
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        n = int(finite.sum())
        out[name] = {
            "mean": float(np.nanmean(v)) if n else float("nan"),
            "std":  float(np.nanstd(v)) if n else float("nan"),
            "n":    n,
        }
    return out


def run_variant(record: ModelRecord, paths: Paths, spec: VariantSpec,
                settings: Settings = DEFAULT, *, save: bool = True) -> MetricAtom:
    """Roll ``record`` out under ``spec`` once, compute its metric set, write the atom."""
    profile = settings.profile(spec.profile)

    res = rollout(
        record, paths,
        task=spec.task,
        duration_s=profile.duration_s,          # None => trained horizon
        batch_size=profile.batch_size,
        intervention=spec.intervention,
        seed=settings.rollout_seed,
        device=settings.device,
    )

    # run the declared analysers; assert no metric-name collisions across them
    per_trial: Dict[str, np.ndarray] = {}
    for analyser in analysers_for(spec.metrics):
        produced = analyser(res)
        clash = set(produced) & set(per_trial)
        assert not clash, f"metric-set '{spec.metrics}' has colliding keys {clash}"
        per_trial.update(produced)

    atom = MetricAtom(
        model=record.label, regime=record.regime, run_id=record.key.run_id,
        seed=record.seed, architecture=record.architecture,
        hidden_units=record.hidden_units,
        variant=spec.name, description=spec.intervention.description,
        config_overrides=dict(spec.intervention.config_overrides),
        task=spec.task, profile=spec.profile,
        duration_s=res.duration_s, batch_size=profile.batch_size,
        rollout_seed=settings.rollout_seed, n_trials=res.n_trials,
        metrics=_summarise_across_trials(per_trial),
    )
    if save:
        path = paths.metrics_atom(record.key, spec.name, create=True)
        atom.save(path)
        print(f"[atom] {record.label} seed{record.seed} · {spec.name} -> {path}")
    return atom


def run_group(group: ModelGroup, paths: Paths, spec: VariantSpec,
              settings: Settings = DEFAULT, *, save: bool = True) -> List[MetricAtom]:
    """Run one spec across every seed of a model group (what you aggregate over)."""
    return [run_variant(g, paths, spec, settings, save=save) for g in group.records]


def run_specs(record: ModelRecord, paths: Paths, specs: List[VariantSpec],
              settings: Settings = DEFAULT, *, save: bool = True) -> List[MetricAtom]:
    """Run several specs on one model (e.g. the baseline's competence + hold measurements)."""
    return [run_variant(record, paths, spec, settings, save=save) for spec in specs]