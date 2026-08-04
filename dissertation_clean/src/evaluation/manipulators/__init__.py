"""Manipulators (Layer 3) — pure factories returning an Intervention. No torch-load, no rollout.
Delay is a config override; ablation/noise/substitution are runtime hooks."""
from .ablation import ablate
from .delay import DELAY_MODES, delay
from .noise import SITES as NOISE_SITES
from .noise import noise
from .substitution import SOURCES as SUBSTITUTION_SOURCES
from .substitution import substitute

__all__ = ["DELAY_MODES", "NOISE_SITES", "SUBSTITUTION_SOURCES", "ablate", "delay", "noise", "substitute"]