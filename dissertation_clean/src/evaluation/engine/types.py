"""The engine's data contracts — pure data, no torch, no matplotlib.

Two objects flow through the whole pipeline:

* RolloutResult : what the engine PRODUCES (analysers and figures consume it).
* Intervention  : the single seam for test-time changes. A manipulator is a pure
                  factory that returns one of these; the engine applies it. This one
                  object replaces the old scatter of six keyword hooks.

Keeping these torch-free means analysers/figures can import them without pulling in
torch, and this file alone is enough to understand what moves through the system.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# observation layout (what manipulators need to slice obs by channel)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObsLayout:
    """Channel sizes of the observation vector [goal | vision | prop]. Plain ints, so a
    manipulator can build a channel-aware hook without ever touching the env or torch."""
    goal: int
    vision: int
    prop: int

    @property
    def total(self) -> int:
        return self.goal + self.vision + self.prop

    @property
    def goal_slice(self) -> slice:
        return slice(0, self.goal)

    @property
    def vision_slice(self) -> slice:
        return slice(self.goal, self.goal + self.vision)

    @property
    def prop_slice(self) -> slice:
        return slice(self.goal + self.vision, self.total)


# --------------------------------------------------------------------------- #
# the single test-time intervention seam
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Intervention:
    """One reproducible test-time modification of a trained model (weights always frozen).

    Two flavours, unified:
      * config_overrides : fields set on the config BEFORE the env is built. This is how
        a DELAY intervention works (proprioception_delay / vision_delay are plain config
        fields read only at env-construction).
      * runtime hooks : callables applied INSIDE the rollout loop, for perturbations that
        are not expressible as config (ablation, noise, substitution, force pulse).

    A manipulator (layer 3) is a pure function returning one of these. The engine takes
    exactly one `intervention=` argument and applies whatever it carries. `label` and
    `description` are provenance — they are written into metrics.json so an artefact
    always records what was done to produce it.
    """
    label: str = "baseline"
    description: str = "intact model, no intervention"
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    # runtime hooks (None = leave untouched). Each maps a tensor -> tensor, except
    # endpoint_load_fn (step_index -> [B,2] force or None) and estimate_source (a str).
    obs_transform: Optional[Callable] = None          # sensory feedback ablation
    estimate_transform: Optional[Callable] = None     # predictive-pathway lesion (dual)
    command_transform: Optional[Callable] = None       # motor / descending-command noise
    pred_input_transform: Optional[Callable] = None    # afferent noise into the forward model
    endpoint_load_fn: Optional[Callable] = None        # physical force pulse
    estimate_source: str = "intact"                    # substitution ladder (dual): intact |
                                                       # zero | delayed_feedback | instant_feedback


IDENTITY = Intervention()   # the intact baseline; the engine's default

# --------------------------------------------------------------------------- #
# a pure data contract the orchestrator consumes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VariantSpec:
    """A complete, self-describing experiment on ONE trained model: the perturbation, the
    task it's measured on, the rollout profile, and which metrics to compute.

    This is the unit a notebook cell passes to the orchestrator. Because it names everything
    (intervention + task + profile + metric set), the spec IS the human-readable description
    of the experiment, and all four facts are written into the atom as provenance.

    One profile per spec => one rollout => metrics valid by construction (option a). The
    baseline is therefore two specs — competence (reach accuracy at the horizon) and hold
    (oscillation at 6 s) — which §4 wants reported as separate measurements anyway.
    """
    name: str                              # the variant leaf on disk, e.g. 'baseline_competence'
    intervention: "Intervention" = IDENTITY
    task: str = "reach"                    # key into engine.TASKS
    profile: str = "competence"            # key into settings.PROFILES (sets duration + batch)
    metrics: str = "competence"            # key into analyzers.METRIC_SETS


# convenient canonical specs (the baseline's two measurements)
BASELINE_COMPETENCE = VariantSpec(name="baseline_competence", task="reach",
                                  profile="competence", metrics="competence")
BASELINE_HOLD = VariantSpec(name="baseline_hold", task="reach",
                            profile="tremor", metrics="hold")


# --------------------------------------------------------------------------- #
# the one data object the engine produces
# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    """Everything one batched rollout produces, time on the first axis. Pure numpy.

    Shapes (T = action steps, B = trials):
        trajectories      [T+1, B, 2]  fingertip position, true reset .. final
        motor_commands    [T,   B, 6]  muscle activations issued each step (pre-noise)
        hidden_states     [T,   B, H]  task-network GRU state that produced each command
        targets           [B, 2]       target coordinates
        target_idx        [B]          reach-direction index (0..n_targets-1)
        pred_estimates    [T, B, P]    (dual only) predictor state estimate x̂
        pred_hidden_states[T, B, Hp]   (dual only) predictive GRU state
        prop_obs          [T, B, P]    (dual only) delayed proprioception x̂ is scored on
        curl_coeff        [B]          (curl env only) signed field coefficient per trial

    Context needed to slice correctly:
        dt                integration timestep (s)
        trained_horizon_s the model's TRAINED episode length. hold_start defaults to this,
                          so the out-of-distribution hold begins where training ended even
                          when the rollout itself is longer (e.g. a 6 s tremor rollout).
        architecture      'dual' | 'mono'
        n_targets         number of reach directions (for per-direction binning)
    """
    trajectories: np.ndarray
    motor_commands: np.ndarray
    hidden_states: np.ndarray
    targets: np.ndarray
    target_idx: np.ndarray
    dt: float
    trained_horizon_s: float
    architecture: str
    n_targets: int
    pred_estimates: Optional[np.ndarray] = None
    pred_hidden_states: Optional[np.ndarray] = None
    prop_obs: Optional[np.ndarray] = None
    curl_coeff: Optional[np.ndarray] = None

    # ---- convenience ----
    @property
    def is_dual(self) -> bool:
        return self.pred_estimates is not None

    @property
    def n_trials(self) -> int:
        return int(self.trajectories.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.motor_commands.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.trajectories.shape[0] - 1) * self.dt

    @property
    def traj_time_s(self) -> np.ndarray:
        """Timestamps for the [T+1] trajectory samples."""
        return np.arange(self.trajectories.shape[0]) * self.dt

    def hold_start_index(self, hold_start_s: Optional[float] = None) -> int:
        """First trajectory index of the hold window. None => the trained horizon, so the
        OOD hold boundary is set by training, not fitted to behaviour."""
        hs = self.trained_horizon_s if hold_start_s is None else hold_start_s
        return int(round(hs / self.dt))

    # ---- caching (re-plot without re-running inference) ----
    _SCALARS = ("dt", "trained_horizon_s", "architecture", "n_targets")

    def save(self, path: str | Path) -> Path:
        """Persist to compressed .npz. Absent (mono) fields are simply not written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays, scalars = {}, {}
        for name, val in vars(self).items():
            if isinstance(val, np.ndarray):
                arrays[name] = val
            elif val is not None:
                scalars[name] = np.array(val)
        np.savez_compressed(path, __scalars__=np.array(list(scalars)), **arrays, **scalars)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RolloutResult":
        with np.load(path, allow_pickle=False) as data:
            scalar_names = set(data["__scalars__"].tolist())
            kwargs = {}
            for k in data.files:
                if k == "__scalars__":
                    continue
                kwargs[k] = data[k].item() if k in scalar_names else data[k]
        return cls(**kwargs)