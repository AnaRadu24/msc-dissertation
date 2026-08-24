from dataclasses import dataclass, asdict
import json
import dataclasses
import os

@dataclass 
class ExperimentConfig:
    """Defines the parameters for a specific experimental run."""
    # Architectural flags
    architecture: str                   # 'mono' or 'dual'
    hidden_units: int = 64              # number of hidden units in the network e.g., 32 or 64
    predictive_split_ratio: float = 0.5 # e.g., 0.25 (25% of units go to the Predictive Network in the dual architecture)

    effector_type: str = 'rigid_arm26'   # options: 'rigid_arm26' or 'compliant_arm26'. Attaching RigidTendonHillMuscle or CompliantTendonHillMuscle respectively to the Arm26 skeleton:  2 degrees of freedom (shoulder and elbow) and 6 specific routing paths for the muscles. 
    # Compliant tendon muscles do not compute force-length curves directly. Rather, they make use of tendon forces and assume equilibrium between tendon and muscle forces to obtain output forces for a given muscle. Because the tendon has elasticity, the neural activation does not translate instantaneously to force at the bone. Instead, the force output is governed by an ongoing state of equilibrium between the muscle fibre (CE + PE) and the tendon (SE). This means that the network must learn to account for the additional temporal lag and nonlinearity in the muscle activation to force transformation, which is not present in the rigid tendon muscle model. The compliant setup thus provides a more challenging and realistic testbed for studying how the brain might learn to control muscles with complex dynamics.

    # Task flags
    target_distribution: str = 'discrete'   # 'discrete' (n_targets equally-spaced angles; for Perich-style discrete evaluation) or 'continuous' (uniform random angle on the circle; removes the memorise-one-programme-per-target escape route to force feedback-driven control)
    training_regime: str = 'unperturbed'    # 'unperturbed' or 'interleaved'
    perturb_fraction: float = 0.5   # P(an epoch is a curl-field epoch). 0.5 default makes perturbation non-ignorable so the network is forced into closed-loop control rather than feedforward memorisation
    perturb_seed: int = 1234        # dedicated RNG seed for the per-epoch null/curl SEQUENCE, so dual and mono runs see an identical curriculum

    context_provided: bool = False  # whether the context (task type) is provided to the network or not
    
    # Structural Routing Flags
    task_receives_feedback: bool = False  # overridden at runtime for the monolithic network
    predictive_receives_feedback: bool = True

    # Physiological parameters
    proprioception_delay: float = 50.0  # proprioceptive feedback delay in MILLISECONDS (physiological ~50 ms). Converted to seconds for MotorNet in environments.py and to timesteps via env.proprioception_delay elsewhere. Must be a multiple of the timestep (dt=10 ms) or MotorNet raises.
    vision_delay: float = 100.0         # visual feedback delay in MILLISECONDS (physiological ~100 ms; proprioception leads vision by ~40-50 ms). Must be a multiple of the timestep (10 ms).
    action_delay: float = 0.0           # DESCENDING (efferent) motor delay in MILLISECONDS: the muscle command is buffered this long before reaching the plant. MotorNet exposes only sensory (ascending) delays, so this is implemented as an action buffer in environments.py. Adds to the total loop delay, so like feedback delay it should drive the same delay-induced (Hopf) instability — modelling slowed descending conduction. 0 = off.
    noise_variance: float = 0.0         # variance of the noise added to the network's activations
    action_noise: float = 0.0           # MOTOR (efferent) noise: SD of Gaussian noise added to the muscle command each step (MotorNet action_noise, σ_u). Set post-construction because MotorNet's __init__ zeroes the constructor arg. The disease-progression / feedback-noise-hypothesis knob (demyelination raises noise AND delay). 0 = deterministic.
    proprioception_noise: float = 0.0   # SD of Gaussian noise on the proprioceptive feedback (MotorNet proprioception_noise). Degraded proprioception (MS) => forced reliance on vision. 0 = clean.
    vision_noise: float = 0.0           # SD of Gaussian noise on the visual feedback (MotorNet vision_noise). 0 = clean.
    max_ep_duration: float = 1.0        # maximum episode duration in seconds, e.g., 1.0 for 1-second reaching trials

    # Task Geometry Parameters
    reach_distance: float = 0.10        # 10 centimetres radial distance from the central starting position to the peripheral targets
    n_targets: int = 8                  # 8 targets spaced at 45-degree intervals
    curl_viscosity: float = 2.0         # Viscosity constant in N s/m (the centre/typical magnitude when randomisation is on)
    curl_direction: str = 'ccw'         # Deterministic curl sense when NOT randomising sign: 'ccw' (b > 0, legacy default) or 'cw' (b < 0). Ignored when curl_randomise_sign is True. Mainly an evaluation knob for probing each field direction separately.
    curl_randomise_sign: bool = False   # Per-trial random CW/CCW curl direction (p=0.5). The key lever for forcing closed-loop control: both signs in one batch => no fixed feedforward fix => must read feedback. Default False = single-direction field given by curl_direction (reproducible).
    curl_magnitude_jitter: float = 0.0  # Per-trial fractional spread on |b|: b ~ curl_viscosity * U(1-jitter, 1+jitter). Modest (e.g. 0.25) removes residual magnitude predictability. 0.0 = fixed magnitude.
    constant_force_mag: float = 0.0     # POSTURAL-HOLD task (ConstantForceHold env): magnitude (N) of a constant Cartesian force applied at the fingertip during the hold — the analogue of holding the arm out against gravity (the planar horizontal RigidTendonArm26 has none, so an unloaded hold is nearly effortless and quiet). 0.0 = unloaded. Sweep it to elicit postural tremor / measure endpoint stiffness K=F/Δx.
    constant_force_angle: float = 0.0   # Direction (degrees) of the constant hold force. Combined with constant_force_mag as F = mag·(cosθ, sinθ).
    
    early_stopping: str = 'absolute'    # Patience-based early-stopping mode: 'none' (disable; train to max_epochs), 'absolute' (improve by > min_delta loss units to reset patience), or 'relative' (improve by > min_rel_delta fraction of the current best — scale-free).
    patience_limit: int = 40            # How many evaluation checks with no qualifying improvement to wait before stopping
    evaluation_interval: int = 50       # How many training epochs between each evaluation check
    min_delta: float = 1e-6             # 'absolute' mode: minimum improvement (loss units) to reset patience
    min_rel_delta: float = 0.01         # 'relative' mode: improvement must be at least this fraction of the current best to reset patience (e.g. 0.01 = 1%), otherwise treated as incidental
    window_size: int = 5                # Calculate rolling loss mean over the last 5 evaluations
    
    # Optimisation Hyperparameters
    max_epochs: int = 20000             # maximum number of training epochs after which training will be stopped if convergence has not been reached
    validation_interval: int = 10       # Every N epochs, evaluate on a FIXED held-out (unperturbed) batch and keep ONLY the BEST-so-far model as task_net.pth (0 = disabled => save the final-epoch model instead). Because training loss spikes intermittently, the final-epoch model is often NOT the best; this saves the best one. Each check also writes the loss logs + a resume checkpoint (training_state.pt) so an interrupted run stays fully plottable and resumable. N trades granularity for speed; 10 catches the best while keeping the overhead small.
    validation_seed: int = 999          # RNG seed for the fixed validation batch (distinct from the training seed) so the same held-out targets are scored every check.
    validation_batch_size: int = 64     # Number of held-out trials scored each validation check.
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 45                      # Master training seed (weight init + target/curriculum sampling), applied via set_global_seed before the env and networks are built. Swept by run_multiseed.py to turn the n=1-per-architecture headline into a mean±sd over seeds. Recorded in config.json and appended to the run-directory name (_seed<N>) so seeds never overwrite each other.
    integration_method: str = 'rk4'     # If the the forward pass exhibits numerical overshoot due to Euler integration, the backward pass will exponentially magnify that error, causing catastrophic training collapse. To ensure baseline convergence across random seeds, configure the effector to use RK4 integration over Euler integration. RK4 is a more accurate numerical integration method that significantly reduces overshoot and thus stabilises training, especially in the compliant tendon setup which is more prone to overshoot due to its complex non-linear dynamics.    
    loss_function: str = 'huber'        # Options: 'huber', 'mse', 'mae'
    huber_delta: float = 0.05           # The error threshold at which the Huber loss transitions from quadratic to linear.
    target_radius: float = 0.0          # Dead-zone radius (m) around the target where spatial error is considered zero.
    spatial_loss_temporal_mode: str = 'continuous'   # Options: 'terminal' (endpoint only) or 'continuous' (trajectory-integrated)
    effort_penalty_weight: float = 1e-5     # Weighting factor for Muscle Effort Penalty to ensure spatial accuracy remains the primary objective
    velocity_penalty_weight: float = 0.0    # Weighting factor for the velocity penalty (mean squared fingertip speed over the WHOLE trajectory, uniformly weighted). 0.0 = OFF (baseline reproducible). When > 0 it self-allocates to the hold — the position term dominates during the reach (movement is required), so the speed penalty only becomes the dominant cost once the fingertip is on target — driving residual velocity (which the always-on curl field amplifies into drift) to zero, with no arbitrary hold-window boundary. Calibrate as a single parameter: raise if drift persists, lower if it the reach / peak speed collapses.
    config_version: int = 2             # Schema marker written into every saved config. Its ABSENCE identifies a pre-versioning file, which is the only case where load_from_json applies value-level migrations (notably the timesteps->milliseconds delay rescale). Bump this only when introducing a new migration; backward compatibility for added/removed FIELDS needs no version bump, since every field has a default and unknown keys are dropped.
    feedback_gain_penalty_weight: float = 0.0  # L2 penalty on the Task Net's GRU input weights for the FEEDBACK blocks (vision + proprioception + predictive estimate; the goal is excluded). Shrinks the feedforward feedback pathway => lowers the controller's feedback gain. 0.0 = OFF. The 'matched-gain mono' control: sweep this up until a mono's ablation gain proxy (prop-freeze/intact error) matches the dual's, then compare delay margins — if the low-gain mono is STILL delay-fragile, robustness is architectural, not merely a gain difference. Proxy for behavioural feedback gain; calibrate against the ablation ratio, not in absolute units.

    def save_to_json(self, file_path: str):
        """Saves the experiment configuration to a JSON file for reproducibility."""
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'ExperimentConfig':
        """Load a config saved by ANY previous version of this schema.

        Backward compatibility is by construction: every field has a default, so a config missing newer
        fields simply gets those defaults, and fields that no longer exist are dropped with a warning.
        Value-level migrations (renames, unit changes) are applied only to files written before
        ``config_version`` existed — see :attr:`config_version`."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"No config.json at:\n  {os.path.abspath(filepath)}\n"
                "The run directory is wrong, not the config schema. A common cause in notebooks is a "
                "stale RESULTS/RUN_DIR variable: if RESULTS has been reassigned to a run directory, "
                "os.path.join(RESULTS, 'rigid_arm26/...') nests one path inside the other. Re-run the "
                "cell that defines RESULTS = os.path.abspath(os.path.join('..', 'results'))."
            )
        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        # Files written before `config_version` was introduced need the value-level migrations below;
        # versioned files are trusted as-is (so a legitimate sub-20 ms delay is never rescaled).
        is_legacy = 'config_version' not in raw_data

        # Legacy Mapping Dictionary: map obsolete variable names to their current equivalents
        legacy_mapping = {
            'alpha': 'effort_penalty_weight',
        }

        for old_key, new_key in legacy_mapping.items():
            if old_key in raw_data and new_key not in raw_data:
                raw_data[new_key] = raw_data.pop(old_key)

        # Missing Variables Inference (ie. if no 'loss_function' then the default at the time was MSE etc.)
        if 'loss_function' not in raw_data:
            raw_data['loss_function'] = 'mse'
            raw_data['huber_delta'] = 0.05

        # Task Network was blind bug
        if 'task_receives_feedback' not in raw_data:
            raw_data['task_receives_feedback'] = False

        # Delay-units migration: pre-fix configs stored delays as TIMESTEPS (e.g. 5, 10); the field is now MILLISECONDS (50, 100). Legacy timestep-scale values (< 20) are converted to ms (× dt_ms = × 10). Applied ONLY to unversioned (pre-config_version) files, so a deliberate 10 ms delay in a modern config is never silently rescaled.
        if is_legacy:
            for key in ('proprioception_delay', 'vision_delay'):
                if key in raw_data and 0 < raw_data[key] < 20:
                    print(f"Warning: legacy '{key}'={raw_data[key]} (timesteps) migrated to {raw_data[key] * 10} ms.")
                    raw_data[key] = raw_data[key] * 10

        valid_fields = {field.name for field in dataclasses.fields(cls)}
        filtered_data = {}

        for key, value in raw_data.items():
            if key in valid_fields:
                filtered_data[key] = value
            else:
                print(f"Warning: Obsolete parameter '{key}'={value} ignored from legacy artefact.")

        return cls(**filtered_data)