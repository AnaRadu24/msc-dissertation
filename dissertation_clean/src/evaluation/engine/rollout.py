"""The engine — the ONLY place that touches torch and the environment.

`rollout` reconstructs a trained model from disk, applies an Intervention, runs one
batched forward pass under no_grad, and returns a RolloutResult. `obs_layout` reports the
observation channel sizes so a manipulator can build a channel-aware hook without ever
building an env itself.

Layering rule this enforces: analysers, figures and manipulators never import torch or
environments — they go through the RolloutResult / ObsLayout this module hands back.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch as th

from config import ExperimentConfig
from environments import (
    CentreOutReachFixedPosture,
    ConstantForceHold,
    ContinuousReachCurlField,
    ContinuousReachFixedPosture,
    set_global_seed,
)
from train import build_pred_input, build_task_input, construct_architecture

from ..paths import Paths, RunKey
from .types import IDENTITY, Intervention, ObsLayout, RolloutResult

# Named tasks -> environment class. A notebook says task="reach"; the env stays an
# implementation detail. Add a task here rather than passing env classes around.
TASKS = {
    "reach":          ContinuousReachFixedPosture,   # canonical unperturbed R1/R2 task
    "reach_discrete": CentreOutReachFixedPosture,     # 8 fixed directions — clean trajectory panels
    "curl":           ContinuousReachCurlField,       # perturbed (velocity-dependent curl)
    "hold":           ConstantForceHold,              # loaded postural hold
}


def _device(spec: str) -> th.device:
    if spec == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(spec)


def _load_weights(net, path, device):
    net.load_state_dict(th.load(path, map_location=device, weights_only=True))
    net.eval()


def _resolved_config(record_config: ExperimentConfig, intervention: Intervention,
                     duration_s: Optional[float]) -> tuple[ExperimentConfig, float]:
    """Apply the intervention's config overrides + requested duration to a COPY of the trained
    config. Additive-delay sentinels (_delay_add_{prop,vision}_ms) are resolved against THIS
    model's trained base delay, so '+100 ms' means +100 on top of whatever it was trained with
    — correct for the swapped/matched variants whose base delays differ. Returns
    (config_for_this_rollout, trained_horizon_s)."""
    trained_horizon_s = float(record_config.max_ep_duration)
    overrides = dict(intervention.config_overrides)

    add_prop = overrides.pop("_delay_add_prop_ms", 0.0)
    add_vision = overrides.pop("_delay_add_vision_ms", 0.0)
    overrides.pop("_estimate_delay_ms", None)    # consumed by the rollout loop, not a config field
    if add_prop:
        overrides["proprioception_delay"] = float(record_config.proprioception_delay) + add_prop
    if add_vision:
        overrides["vision_delay"] = float(record_config.vision_delay) + add_vision

    if duration_s is not None:
        overrides["max_ep_duration"] = float(duration_s)
    config = dataclasses.replace(record_config, **overrides) if overrides else record_config
    return config, trained_horizon_s


def obs_layout(record) -> ObsLayout:
    """Observation channel sizes for a model, as plain ints — what a manipulator needs to
    build a channel-aware hook. Builds the env once from the config; no weights, no rollout."""
    env = TASKS["reach"](config=record.config)
    d = env.obs_dims
    return ObsLayout(goal=d["goal"], vision=d["vision"], prop=d["prop"])


def rollout(
    record,
    paths: Paths,
    *,
    task: str = "reach",
    duration_s: Optional[float] = None,
    batch_size: int = 256,
    intervention: Intervention = IDENTITY,
    seed: int = 42,
    device: str = "auto",
) -> RolloutResult:
    """Roll a trained model out once.

    Args:
        record: a ModelRecord from the catalogue (its .config + .key locate everything).
        paths: the project Paths (to find the .pth weights).
        task: key into TASKS (the environment to evaluate in).
        duration_s: rollout length. None => the model's TRAINED horizon (config.max_ep_duration).
        batch_size: number of trials (set by the caller's RolloutProfile).
        intervention: the single test-time modification (default IDENTITY = intact).
        seed: locks target generation so every model sees an identical target set.
        device: 'auto' | 'cpu' | 'cuda'.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    dev = _device(device)
    env_class = TASKS[task]
    config, trained_horizon_s = _resolved_config(record.config, intervention, duration_s)

    set_global_seed(seed)                         # BEFORE building env + nets -> reproducible targets
    env = env_class(config=config)
    dims = env.obs_dims
    g, v, p = dims["goal"], dims["vision"], dims["prop"]

    task_net, pred_net = construct_architecture(config, env, dev)
    run_dir = paths.run_dir(record.key)
    _load_weights(task_net, run_dir / "task_net.pth", dev)
    if pred_net is not None:
        _load_weights(pred_net, run_dir / "pred_net.pth", dev)

    # intervention hooks (default to no-op)
    obs_tf = intervention.obs_transform or (lambda o: o)
    est_tf = intervention.estimate_transform or (lambda e: e)
    cmd_tf = intervention.command_transform or (lambda c: c)
    pin_tf = intervention.pred_input_transform or (lambda x: x)
    load_fn = intervention.endpoint_load_fn
    est_src = intervention.estimate_source

    # 'delayed_estimate' staling: how many steps to buffer x̂ by (from the intervention's overrides)
    est_delay_ms = intervention.config_overrides.get("_estimate_delay_ms", 0.0)
    est_delay_steps = int(round(est_delay_ms / (env.dt * 1000.0)))   # env.dt is in seconds
    est_buffer = []          # holds past intact x̂ for the 'delayed_estimate' source
    const_estimate = None    # frozen first in-distribution x̂ for 'constant_predictor_output'

    fingertip, hidden, commands = [], [], []
    est_list, pred_hidden, prop_list = [], [], []

    with th.no_grad():
        obs, info = env.reset(options={"batch_size": batch_size})
        obs = obs_tf(obs)
        h_task = task_net.init_hidden(batch_size)
        state_estimate = None
        if pred_net is not None:
            h_pred = pred_net.init_hidden(batch_size)
            state_estimate = th.zeros(batch_size, p, device=dev)

        goal, target_idx = info["goal"], info["target_idx"]
        curl_coeff = getattr(env, "curl_coeff", None)
        task_input = build_task_input(obs, goal, state_estimate, config, dims)
        fingertip.append(info["states"]["fingertip"])         # true reset position (t=0)

        terminated, step = False, 0
        while not terminated:
            command, h_task = task_net(task_input, h_task)
            hidden.append(h_task.squeeze(0))
            commands.append(command)                          # record the INTENDED (pre-noise) command

            plant_command = cmd_tf(command)                   # descending-command noise hits the plant only
            load = load_fn(step) if load_fn is not None else None
            if load is not None:
                obs, _, terminated, _, info = env.step(action=plant_command, endpoint_load=load)
            else:
                obs, _, terminated, _, info = env.step(action=plant_command)
            step += 1
            obs = obs_tf(obs)
            fingertip.append(info["states"]["fingertip"])

            state_estimate = None
            if pred_net is not None:
                pred_in = pin_tf(build_pred_input(obs, command, config, dims))   # efference copy = intended
                state_estimate, h_pred = pred_net(pred_in, h_pred)
                state_estimate = est_tf(state_estimate)                          # predictive-pathway lesion
                if est_src == "zero":
                    state_estimate = th.zeros_like(state_estimate)
                elif est_src == "relayed_feedback":
                    state_estimate = obs[:, g + v:g + v + p]               # raw delayed proprioception
                elif est_src == "instant_feedback":
                    state_estimate = env.get_proprioception()             # true undelayed y(t)
                elif est_src == "delayed_estimate":
                    est_buffer.append(state_estimate)
                    if len(est_buffer) > est_delay_steps:
                        state_estimate = est_buffer[-(est_delay_steps + 1)]  # good x̂ from N steps ago
                    # else: not enough history yet → keep current (start-of-episode transient)
                elif est_src == "constant_predictor_output":
                    if const_estimate is None:
                        const_estimate = state_estimate.detach().clone()  # freeze first in-dist x̂ (rest posture)
                    state_estimate = const_estimate
                est_list.append(state_estimate)
                pred_hidden.append(h_pred.squeeze(0))
                prop_list.append(obs[:, g + v:g + v + p])

            task_input = build_task_input(obs, info["goal"], state_estimate, config, dims)

    def stack(seq):
        return th.stack(seq, dim=0).cpu().numpy()

    return RolloutResult(
        trajectories=stack(fingertip),
        motor_commands=stack(commands),
        hidden_states=stack(hidden),
        targets=goal.cpu().numpy(),
        target_idx=target_idx.detach().cpu().numpy(),
        dt=float(env.dt),
        trained_horizon_s=trained_horizon_s,
        architecture=config.architecture,
        n_targets=int(getattr(config, "n_targets", 8)),
        pred_estimates=stack(est_list) if pred_net is not None else None,
        pred_hidden_states=stack(pred_hidden) if pred_net is not None else None,
        prop_obs=stack(prop_list) if pred_net is not None else None,
        curl_coeff=curl_coeff.detach().cpu().numpy() if curl_coeff is not None else None,
    )