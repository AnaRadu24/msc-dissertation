import math
import os
import random

import motornet as mn
import numpy as np
import torch as th

from config import ExperimentConfig


def set_global_seed(seed: int = 42):
    """
    Enforces deterministic execution across all libraries.

    Args:
        seed (int): The integer value used to lock the pseudo-random number generators.
    """
    random.seed(seed)       # lock the built-in Python stochastic engine
    np.random.seed(seed)    # lock the NumPy stochastic engine
    th.manual_seed(seed)    # lock the PyTorch CPU stochastic engine

    if th.cuda.is_available():          # lock the PyTorch GPU stochastic engines (if CUDA is available)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)   # required for multi-GPU setups

        # enforce deterministic algorithm selection in the cuDNN backend
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    # lock hash-based stochasticity in standard Python data structures
    os.environ["PYTHONHASHSEED"] = str(seed)


class CentreOutReach(mn.environment.Environment):
    """
    A custom MotorNet environment for a 2D centre-out reaching task where the elbow is fixed at 90 degrees, coordinates (0, 0). The fingertip starts at a random position and must reach out to one of 8 targets arranged around the fingertip. The targets are located at a fixed radial distance (e.g., 10 cm) from the center, with equal angular spacing (45 degrees apart).
    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        self.config = config

        if self.config.effector_type == "rigid_arm26":
            # MotorNet default, RigidTendonHillMuscle: Force output is non-linear, dependent on the current length of the muscle (Force-Length relationship) and how fast it is contracting (Force-Velocity relationship). However, the tendon is modelled as infinitely stiff. Force is transmitted to the bone instantaneously.
            muscle = mn.muscle.RigidTendonHillMuscle()
        elif self.config.effector_type == "compliant_arm26":
            # The most biologically accurate and mathematically complex model. It introduces a Series Elastic Element (SEE). The tendon stretches under load. Consequently, neural activation does not produce instantaneous joint torque; the muscle fibre must first contract to stretch the tendon, creating a non-linear temporal delay in force transmission.
            muscle = mn.muscle.CompliantTendonHillMuscle()
        else:
            raise ValueError(f"Unknown effector type: {self.config.effector_type}")
        effector = mn.effector.RigidTendonArm26(
            muscle=muscle, integration_method=config.integration_method
        )

        # Feedback delays. Config delays are in MILLISECONDS; internally they are integer numbers
        # of timesteps (buffer lengths). MotorNet expects the delay in SECONDS and does
        # int(delay/dt) with a float-fragile "integer multiple" check that rejects some exact
        # multiples (e.g. 0.14/0.01 = 14.0000000000000018 > its 1e-15 tolerance). So we compute the
        # step counts ourselves, pass a safe 1-step delay through MotorNet's check, then set the true
        # delays and buffer lengths directly. (Historically the ms values were passed raw as
        # "seconds" -> 500-step delay >> episode -> the buffer never shifted -> feedback FROZEN at t=0.)
        dt = effector.dt
        prop_steps = int(round(self.config.proprioception_delay / (dt * 1000.0)))
        vision_steps = int(round(self.config.vision_delay / (dt * 1000.0)))
        super().__init__(
            effector=effector,
            proprioception_delay=dt,   # safe placeholder (1 step) — overwritten just below
            vision_delay=dt,
            max_ep_duration=self.config.max_ep_duration,
            *args,
            **kwargs,
        )
        self.proprioception_delay = prop_steps
        self.vision_delay = vision_steps
        self.obs_buffer["proprioception"] = [None] * prop_steps
        self.obs_buffer["vision"] = [None] * vision_steps

        # Sensorimotor noise. MotorNet's __init__ zeroes self.action_noise / self.obs_noise after
        # storing the constructor args (so passing them in silently no-ops), while prop/vision noise
        # survive as lists. Set all of them explicitly here (same post-construction pattern as the
        # delays) so config values actually take effect. action_noise is a scalar; prop/vision are
        # lists, matching how MotorNet's get_proprioception/get_vision call apply_noise.
        self.action_noise = [float(self.config.action_noise)]
        self.proprioception_noise = [float(self.config.proprioception_noise)]
        self.vision_noise = [float(self.config.vision_noise)]

        # Descending (efferent) motor delay: buffer the command before it reaches the plant. MotorNet
        # exposes only sensory delays, so we implement this ourselves in step(); off when 0 ms.
        self._action_delay_steps = int(round(float(getattr(self.config, "action_delay", 0.0)) / (dt * 1000.0)))
        self._action_buffer = []

        self.current_goals = None
        self.obs_dims = {"goal": 2, "vision": 2, "prop": self.n_muscles * 2}

    def _apply_action_delay(self, args, kwargs):
        """Replace the incoming muscle command with one from ``action_delay`` steps ago (FIFO buffer),
        modelling slowed descending conduction. No-op when the efferent delay is 0. Returns possibly
        modified ``(args, kwargs)``; before the buffer fills, a zero command (passive muscle) is used."""
        steps = getattr(self, "_action_delay_steps", 0)
        if steps <= 0:
            return args, kwargs
        from_kwarg = "action" in kwargs
        action = kwargs.get("action") if from_kwarg else (args[0] if args else None)
        if action is None:
            return args, kwargs
        self._action_buffer.append(action)
        delayed = self._action_buffer.pop(0) if len(self._action_buffer) > steps else th.zeros_like(action)
        if from_kwarg:
            kwargs = {**kwargs, "action": delayed}
        else:
            args = (delayed,) + tuple(args[1:])
        return args, kwargs

    def _sample_targets(self, batch_size: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample one reach target per trial, returning ``(angles, target_idx)``.

        Base (DISCRETE) policy: pick one of ``config.n_targets`` equally-spaced angles on the circle — the classic centre-out layout used for comparison with primate electrophysiology (e.g. Perich et al. 2018). ``target_idx`` is the integer direction label (0..n_targets-1), used downstream for per-direction colouring and grouping.

        Override this method alone to change where targets are drawn from (e.g. the continuous-angle variant) while keeping the touching posture or perturbation logic.
        """
        target_idx = th.randint(0, self.config.n_targets, (batch_size,))  # discrete direction
        angles = target_idx.float() * (2 * math.pi / self.config.n_targets)
        return angles, target_idx

    def reset(self, *args, **kwargs):
        """Overrides the base reset to generate spatial targets at t=0 which is meant to place the arm in its starting posture and generate a brand-new set of spatial targets for the batch"""
        obs, info = super().reset(*args, **kwargs)
        self._action_buffer = []                       # clear the efferent-delay buffer each episode
        batch_size = kwargs.get("options", {}).get("batch_size", 1)
        device = info["states"]["fingertip"].device

        angles, target_idx = self._sample_targets(batch_size)
        self.current_target_indices = target_idx

        x_offset = self.config.reach_distance * th.cos(angles)
        y_offset = self.config.reach_distance * th.sin(angles)
        offsets = th.stack([x_offset, y_offset], dim=-1).to(device)

        start_pos = info["states"]["fingertip"]
        self.current_goals = start_pos + offsets # apply offset to the initial resting position of the arm

        info["goal"] = self.current_goals
        info["target_idx"] = self.current_target_indices
        return obs, info

    def step(self, *args, **kwargs):
        """Overrides the base step to persist the targets throughout the trial."""

        args, kwargs = self._apply_action_delay(args, kwargs)   # efferent (descending) motor delay
        obs, reward, terminated, truncated, info = super().step(*args, **kwargs)
        info["goal"] = (
            self.current_goals
        )  # inject the constant target coordinates into the info dictionary at every millisecond
        info["target_idx"] = self.current_target_indices
        return obs, reward, terminated, truncated, info

class CentreOutReachFixedPosture(CentreOutReach):
    """
    Contstrained version of the CentreOutReach task where the arm always starts from a fixed central posture (Shoulder 45°, Elbow 90°) rather than a randomised starting position. This mirrors standard primate electrophysiology paradigms where the monkey's arm is typically restrained in a fixed starting posture at the beginning of each trial. The network must learn to reach out to the targets from this consistent starting configuration, which may facilitate learning by reducing variability in the initial conditions.
    """
    
    # __init__ is implicitly inherited from CentreOutReach
    # step() is implicitly inherited from CentreOutReach

    def reset(self, *args, **kwargs):
        # print("\n Executing reset with fixed starting posture: Shoulder 45°, Elbow 90°")
        batch_size = kwargs.get("options", {}).get("batch_size", 1)
        
        shoulder_angle = math.pi / 4
        elbow_angle = math.pi / 2
        fixed_posture = th.tensor([shoulder_angle, elbow_angle, 0.0, 0.0]).repeat(batch_size, 1)
        
        if "options" not in kwargs:
            kwargs["options"] = {}
        kwargs["options"]["joint_state"] = fixed_posture
        self.effector.joint_state = fixed_posture

        # passing the modified kwargs up to the parent class (CentreOutReach) to handle the rest of the reset logic, including target generation
        return super().reset(*args, **kwargs)

class CentreOutReachCurlField(CentreOutReachFixedPosture):
    """
    Similar ``CentreOutReachFixedPosture`` reaching task applying a velocity-dependent curl force field to the fingertip. Mirrors the perturbation paradigms used in primate electrophysiology (e.g., Perich et al., 2018).

    The curl coefficient is sampled PER TRIAL at reset (held constant within the trial) so direction (clockwise / counter-clockwise) and magnitude can be randomised:

    * ``config.curl_randomise_sign``  — flip CW/CCW with p=0.5 per trial for forcing closed-loop control: with both signs present in one batch there is
      no fixed feedforward correction, so the network should READ the feedback to tell which field it is in.
    * ``config.curl_magnitude_jitter`` — fractional spread on |b|: b ~ curl_viscosity * U(1 - jitter, 1 + jitter). Modest jitter removes the magnitude predictability. This turns the ``curl_viscosity`` scalar into the centre of a distribution rather than a fixed value; with sign randomisation the field's expectation is zero, so ``curl_viscosity`` now sets the typical magnitude, not a fixed bias (hopefully).
    """

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        batch_size = kwargs.get("options", {}).get("batch_size", 1)
        device = info["states"]["fingertip"].device
        self.curl_coeff = self._sample_curl_coeff(batch_size, device)   # [B], signed N·s/m
        return obs, info

    def _sample_curl_coeff(self, batch_size: int, device) -> th.Tensor:
        """Per-trial signed viscosity. Draws use the global RNG (like target sampling), so a
        fixed seed reproduces the schedule; default flags reproduce the legacy scalar field.

        Direction: ``curl_randomise_sign`` => per-trial random CW/CCW; otherwise the fixed
        ``curl_direction`` ('ccw' => b > 0, 'cw' => b < 0)."""
        coeff = th.full((batch_size,), float(self.config.curl_viscosity), device=device)
        jitter = self.config.curl_magnitude_jitter
        if jitter > 0.0:
            coeff = coeff * (1.0 + jitter * (2.0 * th.rand(batch_size, device=device) - 1.0))  # |b| ~ b·U(1-j, 1+j)
        if self.config.curl_randomise_sign:
            signs = th.where(th.rand(batch_size, device=device) < 0.5, -1.0, 1.0)
        elif self.config.curl_direction == 'cw':
            signs = -1.0
        elif self.config.curl_direction == 'ccw':
            signs = 1.0
        else:
            raise ValueError(f"Unknown curl_direction: {self.config.curl_direction!r}; expected 'ccw' or 'cw'.")
        return coeff * signs

    def step(self, *args, **kwargs):
        # velocity-dependent curl field, evaluated at the velocity the force acts on: the arm's CURRENT cartesian velocity (the effector's state at the start of this integration step). A counter-clockwise field rotates v by 90 degrees,
        #     [F_x, F_y] = b * [[0, -1], [1, 0]] @ [v_x, v_y] = [-b * v_y, b * v_x]
        # b is now a per-trial [B] coefficient (sampled at reset), so each trial can carry its own direction/magnitude; the broadcast is identical to the scalar case.
        cartesian = self.effector.states["cartesian"]   # [B, 4]: (x, y, v_x, v_y)
        v_x, v_y = cartesian[:, 2], cartesian[:, 3]
        b = self.curl_coeff                              # [B]
        kwargs["endpoint_load"] = th.stack([-b * v_y, b * v_x], dim=-1)
        return super().step(*args, **kwargs)


class _ContinuousTargetMixin:
    """Mixin that draws the reach target from a CONTINUOUS uniform angle on the circle, instead of one of ``n_targets`` discrete directions.

    Rationale (Lazzari et al.; Codol et al.): with only a handful of fixed targets the controller has a trivial open-loop escape route — memorise one ballistic
    motor programme per target and index into it by target identity, never processing feedback during the reach. A continuum of targets destroys that
    lookup table: there is no finite set of programmes to memorise, so the network must learn a state + goal -> action policy, which is inherently feedback-driven.

    It overrides only :meth:`CentreOutReach._sample_targets`, so it composes with the posture (fixed/random) and perturbation (null/curl) classes. The ground-truth goal is the continuous Cartesian target; ``target_idx`` is the *nearest* discrete sector (0..n_targets-1), retained only as a convenience label for per-direction colouring/grouping in plots — it plays no role in the loss.
    """

    def _sample_targets(self, batch_size: int) -> tuple[th.Tensor, th.Tensor]:
        angles = th.rand(batch_size) * (2 * math.pi)               # uniform on [0, 2*pi)
        bin_width = 2 * math.pi / self.config.n_targets
        target_idx = (th.round(angles / bin_width).long() % self.config.n_targets)  # nearest sector (label only)
        return angles, target_idx


class ContinuousReachFixedPosture(_ContinuousTargetMixin, CentreOutReachFixedPosture):
    """Continuous-target reach from the fixed central posture. The main training task for forcing a feedback-driven policy."""
    pass


class ConstantForceHold(CentreOutReachFixedPosture):
    """POSTURAL-HOLD task: initialise the arm at the fixed central posture and set the goal to the
    START fingertip (no reach), then apply a CONSTANT Cartesian force during the hold. This is the
    analogue of holding the arm out against gravity — the condition where postural tremor should appear
    (an unloaded planar hold is nearly effortless, hence quiet). Endpoint stiffness K = F/Δx is read off
    the steady-state displacement; sweeping ``constant_force_mag`` elicits the postural signature
    (sustained, roughly constant-amplitude oscillation with NO distance dependence, unlike intention
    tremor). Weights are frozen — this is a test-time task on a model trained to reach."""

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)      # fixed central posture + a sampled target
        start = info["states"]["fingertip"]             # hold in place: goal = start (zero reach)
        self.current_goals = start
        info["goal"] = self.current_goals
        return obs, info

    def _hold_force(self, batch_size: int, device) -> th.Tensor:
        mag = float(self.config.constant_force_mag)
        ang = math.radians(float(self.config.constant_force_angle))
        return th.tensor([mag * math.cos(ang), mag * math.sin(ang)],
                         dtype=th.float32, device=device).repeat(batch_size, 1)

    def step(self, *args, **kwargs):
        if float(self.config.constant_force_mag) != 0.0:
            cartesian = self.effector.states["cartesian"]
            kwargs["endpoint_load"] = self._hold_force(cartesian.shape[0], cartesian.device)
        return super().step(*args, **kwargs)


class ContinuousReachCurlField(_ContinuousTargetMixin, CentreOutReachCurlField):
    """Continuous-target reach under the velocity-dependent curl field — the perturbed counterpart used in the interleaved training regime."""
    pass