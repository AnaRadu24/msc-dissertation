"""Global evaluation settings — the single place every knob lives.

Pure configuration: nothing here computes anything or has side effects. The engine,
analysers and figures all read from here. Override per-notebook by deriving a copy
(``DEFAULT.with_(...)``), never by editing this file mid-session.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass(frozen=True)
class RolloutProfile:
    """A (duration, batch) pair bound to a scientific PURPOSE.

    Rollout length and trial count are not free-floating magic numbers — each pair is
    dictated by the question being asked. Selecting a profile by name in a notebook
    makes that intent explicit and legible to someone with no context.

        duration_s : rollout length (s). None => use the model's TRAINED horizon
                     (``config.max_ep_duration``): evaluate on exactly the task it was
                     trained for. A concrete value (e.g. 6.0) deliberately rolls LONGER
                     than training — an out-of-distribution probe, required for tremor
                     (a limit cycle takes seconds of hold to express and to resolve in
                     frequency, Δf ≈ 1 / T_hold).
        batch_size : number of trials. LARGE for averaged metrics (tight estimates);
                     SMALL for illustrative example plots (a readable handful of reaches).
    """
    duration_s: Optional[float]
    batch_size: int
    purpose: str = ""


# The canonical profiles. Add one here rather than passing loose numbers around a notebook.
PROFILES = {
    # R1 competence: the TRAINED task (reach, then hold, for the trained ~1 s); many trials.
    "competence": RolloutProfile(duration_s=None, batch_size=256,
                                 purpose="R1 reach accuracy on the trained horizon"),
    # R2/R4 stability: a long, deliberately out-of-distribution hold so a Hopf limit cycle
    # can express and its frequency resolve.
    "tremor":     RolloutProfile(duration_s=6.0, batch_size=256,
                                 purpose="delay/noise-driven hold-phase oscillation"),
    # Illustrative trajectory panels: a few clean reaches at the trained horizon.
    "illustrate": RolloutProfile(duration_s=None, batch_size=32,
                                 purpose="behavioural example figures (trajectories)"),
}


@dataclass(frozen=True)
class TimeWindows:
    """Named windows over a rollout. Every metric slices a rollout by one of these; no
    analysis file hard-codes a timestep.

    REACH ACCURACY (R1) = distance-to-target at the FINAL timestep of the COMPETENCE
    rollout (the trained horizon, config.max_ep_duration). Zero free parameters. It is
    simply traj[-1] of the competence rollout, so there is no field for it here.

    HOLD / TREMOR (R2/R4) = analysed over [hold_start_s, end] of the longer TREMOR
    rollout, a deliberately out-of-distribution window.

    Reach-vs-drift is shown, not quantified: a signed radial-error trace over the long
    rollout (individual trials + mean, target-crossing visible as a sign change) makes
    the point directly, so no drift metric or movement-offset detection lives here.
    """
    # None => resolve to the model's trained horizon (config.max_ep_duration), so the OOD
    # hold BEGINS exactly where training ended — a principled boundary, not one fitted to
    # the model's behaviour. Set a float only to override for a specific analysis.
    hold_start_s: Optional[float] = None


@dataclass(frozen=True)
class Settings:
    """Everything the evaluation layer needs that is not model-specific."""
    rollout_seed: int = 42           # fixed => identical target set across every model (like-for-like)
    device: str = "auto"            # 'auto' | 'cpu' | 'cuda'
    n_dirs: int = 8                  # centre-out reach directions
    sd_ddof: int = 1                 # seed is the replication unit => sample SD (ddof=1), never 0
    windows: TimeWindows = field(default_factory=TimeWindows)

    def profile(self, name: str) -> RolloutProfile:
        return PROFILES[name]

    def with_(self, **overrides) -> "Settings":
        """A copy with fields overridden (frozen-safe): DEFAULT.with_(rollout_seed=7)."""
        return replace(self, **overrides)


DEFAULT = Settings()