"""Test-time intervention hooks used for the dissertation's exploratory (pre-``evaluation/``)
analyses in notebooks/explore.ipynb. Each factory returns a hook function that modifies
observations, motor commands, or the Dual Network's internal state estimate at rollout time,
without altering the trained weights:

    get_sensory_ablation_hook     — freeze or zero a sensory channel (vision / proprioception)
    get_noise_injection_hook      — inject Gaussian motor noise, optionally signal-dependent
    get_predictive_ablation_hook  — lesion the Predictive Network's forward-model estimate

These predate and are superseded by the channel-aware, ``Intervention``-based manipulators in
``evaluation/manipulators/``, which is the API the current evaluation pipeline calls. Kept here
because the notebook cells that used these hooks to produce the dissertation's early figures
were since removed from the notebook.
"""
import torch as th


def get_sensory_ablation_hook(channel: str = "vision", mode: str = "freeze"):
    """Returns an obs_transform hook for ablating vision or proprioception."""
    first_obs = None

    def obs_transform(obs, step):
        nonlocal first_obs
        if step == 0:
            first_obs = obs.clone()

        modified_obs = obs.clone()

        # Observation layout (MotorNet standard: obs = [goal_x, goal_y, vis_x, vis_y, prop_1...prop_n]):
        # vision is indices 2 and 3, proprioception is 4 onwards.
        if channel == "vision" or channel == "both":
            if mode == "freeze":
                modified_obs[:, 2:4] = first_obs[:, 2:4]
            elif mode == "zero":
                modified_obs[:, 2:4] = 0.0

        if channel == "prop" or channel == "both":
            if mode == "freeze":
                modified_obs[:, 4:] = first_obs[:, 4:]
            elif mode == "zero":
                modified_obs[:, 4:] = 0.0

        return modified_obs

    return obs_transform


def get_noise_injection_hook(sigma: float, signal_dependent: bool = False):
    """Returns a command_transform hook for injecting motor noise."""

    def command_transform(u, step):
        noise = th.randn_like(u) * sigma
        if signal_dependent:
            # Harris-Wolpert style: noise scales with command magnitude
            noise = noise * th.abs(u)
        return u + noise

    return command_transform


def get_predictive_ablation_hook(mode: str = "zero", delay_steps: int = 5):
    """Returns an estimate_transform hook for lesioning the Dual Network's forward model."""

    def estimate_transform(x_hat, step, env):
        if x_hat is None:
            return None  # Monolithic network, nothing to ablate

        if mode == "zero":
            return th.zeros_like(x_hat)
        elif mode == "instant_feedback":
            # Perfect estimator ceiling: give it the true current state
            return env.get_proprioception(delay=0)
        elif mode == "delayed_feedback":
            # Give it raw delayed feedback instead of the prediction
            return env.get_proprioception()

        return x_hat

    return estimate_transform