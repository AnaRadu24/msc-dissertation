"""Metric store — write per-(model, seed, variant) atoms, read them back as one tidy
DataFrame, aggregate across seeds. Pure I/O + pandas. No torch, no matplotlib.

The metric ATOM (results/metrics/<model>/<regime>/<run_id>/<variant>/metrics.json) is the
single source of truth for a number. Aggregation READS atoms; it never recomputes. Because
the seed is the replication unit, cross-seed SD is the sample SD (ddof=1) of the per-seed
means, and a single-seed metric therefore reports SD = NaN — which is honest, not a bug.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class MetricAtom:
    """One rollout's worth of metrics for one model-seed under one variant.

    `metrics` maps a metric name -> {"mean": float, "std": float, "n": int}, where mean/std
    are ACROSS TRIALS within this one rollout. Cross-SEED aggregation happens later, in
    aggregate_seeds, from the per-seed means these atoms hold.
    """
    model: str
    regime: str
    run_id: str
    seed: int
    architecture: str
    hidden_units: int
    variant: str
    description: str
    config_overrides: Dict[str, Any]
    task: str
    profile: str
    duration_s: float
    batch_size: int
    rollout_seed: int
    n_trials: int
    metrics: Dict[str, Dict[str, float]]
    schema: int = 1

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MetricAtom":
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


# identity columns carried into every DataFrame row (everything except the metric value)
_ID_COLS = ["model", "regime", "run_id", "seed", "architecture", "hidden_units",
            "variant", "task", "profile"]


def load_atoms(metrics_root: str | Path) -> pd.DataFrame:
    """Glob every metrics.json under `metrics_root` into ONE tidy (long-form) DataFrame:
    one row per (atom, metric), columns = identity + [metric, mean, std, n_trials].
    Long form is what seaborn and groupby want; pivot to wide only for a printed table."""
    rows = []
    for path in sorted(Path(metrics_root).rglob("metrics.json")):
        atom = MetricAtom.load(path)
        base = {c: getattr(atom, c) for c in _ID_COLS}
        base["n_trials"] = atom.n_trials
        for metric, stat in atom.metrics.items():
            rows.append({**base, "metric": metric,
                         "mean": stat.get("mean"), "std": stat.get("std")})
    if not rows:
        raise FileNotFoundError(f"no metrics.json found under {metrics_root}")
    return pd.DataFrame(rows)


def aggregate_seeds(df: pd.DataFrame, *, ddof: int = 1) -> pd.DataFrame:
    """Collapse across seeds: for each (model, variant, task, metric), the across-seed mean
    and sample SD (ddof) of the per-seed means, plus n_seeds. This is the dissertation
    number. n_seeds = 1 -> seed_sd = NaN (a sample SD needs >= 2), surfaced not hidden."""
    keys = ["model", "regime", "architecture", "hidden_units", "variant", "task", "profile", "metric"]
    g = df.groupby(keys, dropna=False)["mean"]
    out = g.agg(seed_mean="mean",
                seed_sd=lambda s: s.std(ddof=ddof),
                n_seeds="count").reset_index()
    return out