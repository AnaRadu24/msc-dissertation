"""The directory contract — the ONE place that knows where anything lives.

Outputs mirror the ``trained_models/`` input tree, with a ``<variant>`` leaf:

    trained_models/<model>/<regime>/<run_id>/            {config.json, *.pth, *_loss.npy}
    results/metrics/<model>/<regime>/<run_id>/<variant>/metrics.json
    results/figures/<model>/<regime>/<run_id>/<variant>/kinematics.png
    results/aggregated/<analysis>/<name>.png

The seed is carried inside ``<run_id>`` (e.g. '20260708_0006_seed42'), so seeds never
collide. Grouping runs into a model lives in the catalogue, never in these paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunKey:
    """Identifies one trained run by its position in the input tree."""
    model: str      # "dual_64+64", "mono_64u", ...
    regime: str     # "interleaved_training", "unperturbed_training"
    run_id: str     # "20260708_0006_seed42" — carries the seed

    def __str__(self) -> str:
        return f"{self.model}/{self.regime}/{self.run_id}"


class Paths:
    """Builds every input and output path from the project root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.trained_models = self.root / "results" / "trained_models"
        self.results = self.root / "results"

    # ---- inputs (read-only) ----
    def run_dir(self, key: RunKey) -> Path:
        return self.trained_models / key.model / key.regime / key.run_id

    def config_json(self, key: RunKey) -> Path:
        return self.run_dir(key) / "config.json"

    # ---- per-variant outputs ----
    def metrics_atom(self, key: RunKey, variant: str, *, create: bool = False) -> Path:
        d = self.results / "metrics" / key.model / key.regime / key.run_id / variant
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d / "metrics.json"

    def figure_dir(self, key: RunKey, variant: str, *, create: bool = False) -> Path:
        d = self.results / "figures" / key.model / key.regime / key.run_id / variant
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- aggregated (cross-model / cross-seed) outputs ----
    def aggregated_dir(self, analysis: str, *, create: bool = False) -> Path:
        d = self.results / "aggregated" / analysis
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d