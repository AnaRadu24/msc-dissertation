"""Model catalogue — discover, validate and GROUP the trained models.

Single source of truth for "what models exist and which runs are seeds of the same
model". Replaces the old hard-coded registry of timestamped paths (which went stale the
moment anything was retrained or moved) with DISCOVERY from disk + VALIDATION against
each run's own config.json.

Two bug classes this exists to kill:
  1. The mislabelled run. A run's identity comes from its config.json (written at
     training time); the folder it sits in is cross-checked against it. A mono config in
     a dual folder is a hard error, not a silently corrupted average.
  2. Seeds vs variants confused. Within one model folder sit both extra SEEDS of the
     canonical model AND separate training VARIANTS (different delays, penalties, ...).
     Grouping by full config-identity-minus-seed separates them automatically.

Pure discovery + dataclasses. No torch, no matplotlib, no rollout.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Optional

from config import ExperimentConfig

from .paths import Paths, RunKey

_SEED_RE = re.compile(r"_seed(\d+)")
_REGIME_MAP = {"interleaved_training": "interleaved", "unperturbed_training": "unperturbed"}

# Config fields that do NOT define "the same model": two runs identical except these are
# SEEDS of one model. Everything else (architecture, units, delays, penalties, curl, ...)
# is part of the identity, so a genuine training variant forms its own group. If groups
# ever over-split, widen this set deliberately — never silently.
_IDENTITY_EXCLUDE = frozenset({"seed", "config_version", "max_epochs"})


class CatalogueError(ValueError):
    """A discovered run is internally inconsistent (folder disagrees with its config)."""
    
    def __init__(self, message):
        super().__init__(message) 


@dataclass(frozen=True)
class ModelRecord:
    """One trained run: where it lives + the config it was trained under."""
    key: RunKey
    config: ExperimentConfig
    
    _SEED_SUFFIX_RE = re.compile(r"^\d{8}_\d{4}_seed\d+")   # class-level: strip 'DATE_TIME_seedN'

    @property
    def architecture(self) -> str:
        return self.config.architecture

    @property
    def hidden_units(self) -> int:
        return self.config.hidden_units

    @property
    def split(self) -> float:
        return self.config.predictive_split_ratio

    @property
    def regime(self) -> str:
        return _REGIME_MAP.get(self.key.regime, self.key.regime)

    @property
    def seed(self) -> int:
        return int(self.config.seed)
    
    @property
    def variant_suffix(self) -> str:
        """The descriptive tail of the run_id after 'DATE_TIME_seedN', e.g.
        '20260719_1708_seed42_equal_delays' -> 'equal_delays'. Empty for a canonical run."""
        tail = self._SEED_SUFFIX_RE.sub("", self.key.run_id).lstrip("_")
        return tail

    @property
    def label(self) -> str:
        """Human display name = model folder + any trained-variant suffix, so a separately
        TRAINED variant (different weights) is a distinct model, e.g. 'dual_64+64u_equal_delays'.
        This is Axis 1 (training condition); test-time interventions are Axis 2, applied on top."""
        base = self.key.model
        return f"{base}_{self.variant_suffix}" if self.variant_suffix else base

    def identity(self) -> tuple:
        """Hashable key equal for two runs iff they are SEEDS of the same trained model.
        Includes the trained-variant suffix so naming and grouping never disagree."""
        d = dataclasses.asdict(self.config)
        for f in _IDENTITY_EXCLUDE:
            d.pop(f, None)
        return (self.variant_suffix,) + tuple(sorted(d.items()))


@dataclass(frozen=True)
class ModelGroup:
    """A set of seeds of ONE trained configuration — the unit that gets aggregated over."""
    label: str
    architecture: str
    hidden_units: int
    split: float
    regime: str
    records: tuple[ModelRecord, ...]

    @property
    def seeds(self) -> list[int]:
        return sorted(r.seed for r in self.records)

    @property
    def n_seeds(self) -> int:
        return len(self.records)

    def record(self, seed: int) -> ModelRecord:
        for r in self.records:
            if r.seed == seed:
                return r
        raise KeyError(f"{self.label}: no seed {seed} (have {self.seeds})")


class Catalogue:
    """Scans trained_models/, validates every run, clusters seeds into groups."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.records: list[ModelRecord] = []
        self._scan()

    def _scan(self) -> None:
        root = self.paths.trained_models
        if not root.is_dir():
            raise CatalogueError(
                f"trained_models directory not found at:\n  {root}\n"
                f"Paths root resolved to {self.paths.root}. When running from a "
                f"notebooks/ kernel, point Paths at the project root, e.g. "
                f"Paths(Path.cwd().parent).")

        found = sorted(root.rglob("config.json"))
        if not found:
            raise CatalogueError(
                f"No config.json anywhere under {root}. Did the copy include the JSONs, "
                f"not just the .pth weights?")

        expected_depth = 3            # <model>/<regime>/<run_id>/config.json
        good, wrong_depth = [], []
        for cfg_path in found:
            rel = cfg_path.parent.relative_to(root).parts
            (good if len(rel) == expected_depth else wrong_depth).append((rel, cfg_path))
        
        # Populate the catalogue records
        for parts, cfg_path in good:
            config = ExperimentConfig.load_from_json(cfg_path)
            key = RunKey(model=parts[0], regime=parts[1], run_id=parts[2])
            record = ModelRecord(key=key, config=config)
            self._validate(record)
            self.records.append(record)

    @staticmethod
    def _validate(rec: ModelRecord) -> None:
        """Folder-declared identity must match the config written at training time."""
        problems = []
        folder_arch = rec.key.model.split("_", 1)[0]             # 'dual_64+64' -> 'dual'
        if folder_arch not in ("dual", "mono"):
            problems.append(f"folder {rec.key.model!r} does not start with dual/mono")
        elif rec.config.architecture != folder_arch:
            problems.append(f"folder says '{folder_arch}' but config.architecture="
                            f"'{rec.config.architecture}'")
        if (rec.key.regime in _REGIME_MAP
                and rec.config.training_regime != _REGIME_MAP[rec.key.regime]):
            problems.append(f"folder regime '{rec.key.regime}' but config.training_regime="
                            f"'{rec.config.training_regime}'")
        m = _SEED_RE.search(rec.key.run_id)
        if m and int(m.group(1)) != int(rec.config.seed):
            problems.append(f"run_id says seed {m.group(1)} but config.seed={rec.config.seed}")
        if problems:
            raise CatalogueError(f"[{rec.key}] " + "; ".join(problems))

    def groups(self) -> list[ModelGroup]:
        """Cluster runs so one group == the seeds of one trained configuration."""
        buckets: dict[tuple, list[ModelRecord]] = {}
        for r in self.records:
            buckets.setdefault(r.identity(), []).append(r)
        out = []
        for recs in buckets.values():
            recs.sort(key=lambda x: x.seed)
            r0 = recs[0]
            out.append(ModelGroup(r0.label, r0.architecture, r0.hidden_units, r0.split,
                                  r0.regime, tuple(recs)))
        return sorted(out, key=lambda g: (g.label, g.regime, -g.n_seeds))

    def canonical(self, architecture: str, *, regime: str = "interleaved") -> ModelGroup:
        """The primary multi-seed group for architecture+regime = the one with the MOST
        seeds. Raises on a tie (real ambiguity to resolve with select())."""
        cands = sorted((g for g in self.groups()
                        if g.architecture == architecture and g.regime == regime),
                       key=lambda g: g.n_seeds, reverse=True)
        if not cands:
            raise KeyError(f"no {architecture}/{regime} group found")
        if len(cands) > 1 and cands[0].n_seeds == cands[1].n_seeds:
            raise CatalogueError(
                f"ambiguous canonical {architecture}/{regime}: "
                + ", ".join(f"{g.label}({g.n_seeds})" for g in cands)
                + " — disambiguate with select(**config_filters).")
        return cands[0]

    def select(self, **config_filters) -> list[ModelRecord]:
        """Every run whose config matches all field==value filters, e.g.
        select(architecture='dual', proprioception_delay=70) -> the 70-ms-prop variant."""
        return [r for r in self.records
                if all(getattr(r.config, k) == v for k, v in config_filters.items())]

    def report(self) -> str:
        """Human-readable listing of every group + its seeds — run once and eyeball it."""
        lines = [f"Catalogue: {len(self.records)} runs in {len(self.groups())} groups"]
        for g in self.groups():
            lines.append(f"  {g.label:<34}{g.regime:<13}"
                         f"{g.architecture} {g.hidden_units}u split{g.split:g}  "
                         f"seeds={g.seeds} (n={g.n_seeds})")
        return "\n".join(lines)