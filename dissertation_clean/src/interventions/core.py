"""
The Interventions API
Generates hook functions to modify observations, commands, or internal estimates at test time.

Example usage:
    from interventions import get_sensory_ablation_hook, get_noise_injection_hook, get_predictive_ablation_hook
    
    # 1. Get the intervention
    lesion_hook = get_predictive_ablation_hook(mode="zero")

    # 2. Run the physics
    raw_data = evaluate_policy(..., estimate_transform=lesion_hook)

    # 3. Do the math
    metrics = calculate_reach_metrics(raw_data["trajectories"], raw_data["targets"])
"""
import numpy as np
import torch as th


def get_sensory_ablation_hook(channel: str = "vision", mode: str = "freeze"):
    """Returns an obs_transform hook for ablating vision or proprioception."""
    first_obs = None
    
    def obs_transform(obs, step):
        nonlocal first_obs
        if step == 0:
            first_obs = obs.clone()
            
        modified_obs = obs.clone()
        
        # Determine indices (MotorNet standard: obs = [goal_x, goal_y, vis_x, vis_y, prop_1...prop_n])
        # Vision is indices 2 and 3. Proprioception is 4 onwards.
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
    """Returns an estimate_transform hook for lesigning the Dual network's forward model."""
    
    def estimate_transform(x_hat, step, env):
        if x_hat is None:
            return None # Monolithic network, nothing to ablate
            
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