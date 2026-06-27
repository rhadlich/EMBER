"""TD3 (Twin Delayed DDPG) algorithm configuration, learner, and registration.

Builds on top of SAC since RLlib 2.46.0 has no built-in TD3.  The main
differences are:
  - Deterministic policy (no entropy / alpha).
  - Target-policy smoothing (clipped noise on target actions).
  - Delayed policy updates (actor updated every ``policy_delay`` critic steps).
"""
from typing import Any, Dict, Optional, Type, Union

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.algorithms.sac.sac import SAC, SACConfig
from ray.rllib.algorithms.sac.sac_learner import (
    QF_LOSS_KEY,
    QF_MEAN_KEY,
    QF_MAX_KEY,
    QF_MIN_KEY,
    QF_PREDS,
    QF_TWIN_LOSS_KEY,
    QF_TWIN_PREDS,
    TD_ERROR_MEAN_KEY,
    SACLearner,
)
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.utils import update_target_network
from ray.rllib.core.learner.learner import Learner, POLICY_LOSS_KEY
from ray.rllib.core.rl_module.apis import TargetNetworkAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    LAST_TARGET_UPDATE_TS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_TARGET_UPDATES,
    TD_ERROR_KEY,
)
from ray.rllib.utils.typing import (
    LearningRateOrSchedule,
    ModuleID,
    ParamDict,
    RLModuleSpecType,
    TensorType,
)

torch, nn = try_import_torch()

# =========================================================================
# CLI / config helpers  (same interface as sac_cfg.py)
# =========================================================================

def add_cli_args(parser):
    """Register TD3-specific command-line flags."""
    parser.add_argument(
        "--env-type", type=str, default="continuous",
        help="Action-space type: 'continuous'.",
    )
    parser.add_argument(
        "--target-policy-noise", type=float, default=0.2,
        help="Std of Gaussian noise added to target-policy actions.",
    )
    parser.add_argument(
        "--target-noise-clip", type=float, default=0.5,
        help="Clipping range for target-policy noise.",
    )
    parser.add_argument(
        "--policy-delay", type=int, default=2,
        help="Number of critic updates per actor update.",
    )
    parser.add_argument(
        "--exploration-noise", type=float, default=0.1,
        help="Std of Gaussian noise added during exploration (converged value).",
    )
    parser.add_argument(
        "--initial-steps", type=int, default=10000,
        help="Number of rollouts using pure uniform-random actions.",
    )
    parser.add_argument(
        "--initial-std", type=float, default=0.5,
        help="Starting Gaussian std after the random phase ends.",
    )
    parser.add_argument(
        "--noise-decay-k", type=float, default=1e-5,
        help="Controls the 1/x convergence speed of the noise decay.",
    )
    parser.add_argument(
        "--noise-decay-schedule", type=str, default="linear",
        choices=["linear", "hyperbolic"],
        help="Schedule used to decay actor exploration noise after the random phase.",
    )
    parser.add_argument(
        "--linear-decay-steps", type=int, default=100000,
        help="Number of rollout steps used by the linear exploration-noise decay.",
    )
    return parser


def update_config(cfg, args):
    """Apply TD3 defaults to the RLlib config builder."""
    return (
        cfg.training(
            q_model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,
                "custom_model_config": {},
            },
            policy_model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,
                "custom_model_config": {},
            },
            train_batch_size_per_learner=256,
            gamma=0.1,
            n_step=1,
            grad_clip=10.0,
            actor_lr=2e-5,
            critic_lr=2e-5,
            tau=5e-3,
            num_steps_sampled_before_learning_starts=10000,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": int(5e4),
            },
            # TD3-specific
            target_policy_noise=getattr(args, "target_policy_noise", 0.2),
            target_noise_clip=getattr(args, "target_noise_clip", 0.5),
            policy_delay=getattr(args, "policy_delay", 2),
            exploration_noise=getattr(args, "exploration_noise", 0.1),
            initial_steps=getattr(args, "initial_steps", 10000),
            initial_std=getattr(args, "initial_std", 0.5),
            noise_decay_k=getattr(args, "noise_decay_k", 1e-5),
            noise_decay_schedule=getattr(args, "noise_decay_schedule", "linear"),
            linear_decay_steps=getattr(args, "linear_decay_steps", 100000),
        )
    )


def get_rllib_module_spec(cfg) -> dict:
    """Return architecture-relevant dict for RLlib module compatibility checks."""
    def _to_plain_dict(x):
        if x is None:
            return {}
        if isinstance(x, dict):
            return dict(x)
        try:
            return dict(x)
        except (TypeError, ValueError):
            return {}

    pmc = _to_plain_dict(getattr(cfg, "policy_model_config", None))
    qmc = _to_plain_dict(getattr(cfg, "q_model_config", None))
    return {
        "algorithm": "TD3",
        "policy_model_config": pmc,
        "q_model_config": qmc,
        "twin_q": bool(getattr(cfg, "twin_q", True)),
        "target_policy_noise": getattr(cfg, "target_policy_noise", 0.2),
        "target_noise_clip": getattr(cfg, "target_noise_clip", 0.5),
        "policy_delay": getattr(cfg, "policy_delay", 2),
    }


def _get_buffer_action_dim(action_space, action_adapter) -> int:
    if action_adapter.mode == "continuous":
        return int(action_space.shape[0])
    if action_adapter.mode == "discrete1":
        return 1
    if action_adapter.mode == "multidiscrete":
        return len(action_adapter.nvec)
    raise NotImplementedError(
        f"Unsupported action adapter mode {action_adapter.mode}"
    )


def get_actor_episode_buffer_spec(
    *,
    state_dim: int,
    action_space,
    action_adapter,
    batch_size: int = 32,
    num_slots: int = 8,
    name: str = "episodes",
) -> dict:
    """Return the TD3-specific shared-memory actor rollout schema."""
    bytes_per_float = 4
    action_dim = _get_buffer_action_dim(action_space, action_adapter)
    field_order = ["action", "reward", "next_obs", "action_dist_inputs"]
    field_dims = {
        "action": action_dim,
        "reward": 1,
        "next_obs": state_dim,
        "action_dist_inputs": action_dim,
    }
    elements_per_rollout = sum(field_dims[field] for field in field_order)
    payload_size = elements_per_rollout * batch_size + state_dim
    header_size = 2
    header_slot_size = 1
    slot_size = header_slot_size + payload_size
    total_size = header_size + num_slots * slot_size

    return {
        "algorithm": "TD3",
        "policy_output_kind": "deterministic",
        "BATCH_SIZE": batch_size,
        "NUM_SLOTS": num_slots,
        "ELEMENTS_PER_ROLLOUT": elements_per_rollout,
        "BYTES_PER_ROLLOUT": elements_per_rollout * bytes_per_float,
        "PAYLOAD_SIZE": payload_size,
        "HEADER_SIZE": header_size,
        "HEADER_SLOT_SIZE": header_slot_size,
        "SLOT_SIZE": slot_size,
        "TOTAL_SIZE": total_size,
        "TOTAL_SIZE_BYTES": total_size * bytes_per_float,
        "BYTES_PER_FLOAT": bytes_per_float,
        "name": name,
        "STATE_ACTION_DIMS": {
            "action": action_dim,
            "reward": 1,
            "state": state_dim,
        },
        "ROLLOUT_FIELD_ORDER": field_order,
        "ROLLOUT_FIELD_DIMS": field_dims,
        "HAS_ACTION_LOGP": False,
        "HAS_ACTION_DIST_INPUTS": True,
        "action_dist_size": action_dim,
    }


# =========================================================================
# TD3 Config
# =========================================================================

class TD3Config(SACConfig):
    """Configuration for the TD3 algorithm.

    Extends SACConfig with TD3-specific hyper-parameters and removes
    entropy / alpha settings that do not apply to TD3.
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or TD3)

        # TD3-specific defaults
        self.target_policy_noise = 0.2
        self.target_noise_clip = 0.5
        self.policy_delay = 2
        self.exploration_noise = 0.1
        self.initial_steps = 1000
        self.initial_std = 0.5
        self.noise_decay_k = 0.01
        self.noise_decay_schedule = "linear"
        self.linear_decay_steps = 8000

        # TD3 has no entropy regularisation.
        self.initial_alpha = 1.0
        self.target_entropy = 0.0
        self.alpha_lr = 0.0

    @override(SACConfig)
    def training(
        self,
        *,
        target_policy_noise: Optional[float] = NotProvided,
        target_noise_clip: Optional[float] = NotProvided,
        policy_delay: Optional[int] = NotProvided,
        exploration_noise: Optional[float] = NotProvided,
        initial_steps: Optional[int] = NotProvided,
        initial_std: Optional[float] = NotProvided,
        noise_decay_k: Optional[float] = NotProvided,
        noise_decay_schedule: Optional[str] = NotProvided,
        linear_decay_steps: Optional[int] = NotProvided,
        # Forward everything else to SACConfig.training()
        **kwargs,
    ) -> "TD3Config":
        super().training(**kwargs)

        if target_policy_noise is not NotProvided:
            self.target_policy_noise = target_policy_noise
        if target_noise_clip is not NotProvided:
            self.target_noise_clip = target_noise_clip
        if policy_delay is not NotProvided:
            self.policy_delay = policy_delay
        if exploration_noise is not NotProvided:
            self.exploration_noise = exploration_noise
        if initial_steps is not NotProvided:
            self.initial_steps = initial_steps
        if initial_std is not NotProvided:
            self.initial_std = initial_std
        if noise_decay_k is not NotProvided:
            self.noise_decay_k = noise_decay_k
        if noise_decay_schedule is not NotProvided:
            self.noise_decay_schedule = noise_decay_schedule
        if linear_decay_steps is not NotProvided:
            self.linear_decay_steps = linear_decay_steps

        return self

    @override(SACConfig)
    def validate(self) -> None:
        # TD3 still uses SAC's new-stack replay-buffer and framework validation.
        # Keeping SAC's validation avoids configuration drift from RLlib 2.46.
        super().validate()
        if self.noise_decay_schedule not in {"linear", "hyperbolic"}:
            raise ValueError(
                "noise_decay_schedule must be one of {'linear', 'hyperbolic'}."
            )
        if self.linear_decay_steps <= 0:
            raise ValueError("linear_decay_steps must be > 0.")

    # --- RLModule / Learner defaults ---

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> RLModuleSpecType:
        from core.rl_modules.td3_rl_modules import TD3TorchRLModule
        return RLModuleSpec(module_class=TD3TorchRLModule)

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type[Learner], str]:
        return TD3TorchLearner

    @property
    def _model_config_auto_includes(self):
        return super()._model_config_auto_includes | {
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
        }


# =========================================================================
# TD3 Algorithm
# =========================================================================

class TD3(SAC):
    """TD3 Algorithm – extends SAC with deterministic-policy logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfig:
        return TD3Config()


# Register so that ``get_trainable_cls("TD3")`` works.
from ray.tune.registry import register_trainable  # noqa: E402
register_trainable("TD3", TD3)


# =========================================================================
# TD3 Torch Learner
# =========================================================================

class TD3TorchLearner(SACTorchLearner):
    """PyTorch learner for TD3.

    Key differences from SACTorchLearner:
    - No alpha / entropy optimiser.
    - Delayed policy updates: actor loss is only computed & back-propagated
      every ``policy_delay`` critic updates.
    - Target-policy smoothing: clipped Gaussian noise added to target actions.
    """

    @override(SACTorchLearner)
    def build(self) -> None:
        # DQNLearner.build -> Learner.build (creates target networks, etc.)
        # We deliberately skip SACLearner.build which creates curr_log_alpha
        # and target_entropy that TD3 does not need.
        DQNTorchLearner.build(self)

        self._temp_losses: Dict = {}
        self._critic_update_count: Dict[ModuleID, int] = {}
        self._should_update_targets: Dict[ModuleID, bool] = {}

    # ------------------------------------------------------------------
    # Optimisers – no alpha optimiser
    # ------------------------------------------------------------------

    @override(SACTorchLearner)
    def configure_optimizers_for_module(
        self, module_id: ModuleID, config: AlgorithmConfig = None
    ) -> None:
        module = self._module[module_id]

        # Critic optimiser
        params_critic = self.get_parameters(module.qf_encoder) + self.get_parameters(
            module.qf
        )
        optim_critic = torch.optim.Adam(params_critic, eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="qf",
            optimizer=optim_critic,
            params=params_critic,
            lr_or_lr_schedule=config.critic_lr,
        )

        # Twin-critic optimiser
        if config.twin_q:
            params_twin = self.get_parameters(
                module.qf_twin_encoder
            ) + self.get_parameters(module.qf_twin)
            optim_twin = torch.optim.Adam(params_twin, eps=1e-7)
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="qf_twin",
                optimizer=optim_twin,
                params=params_twin,
                lr_or_lr_schedule=config.critic_lr,
            )

        # Actor optimiser
        params_actor = self.get_parameters(module.pi_encoder) + self.get_parameters(
            module.pi
        )
        optim_actor = torch.optim.Adam(params_actor, eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="policy",
            optimizer=optim_actor,
            params=params_actor,
            lr_or_lr_schedule=config.actor_lr,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @override(SACTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: "TD3Config",
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        module = self._module[module_id]

        # Track critic update count for delayed policy updates.
        self._critic_update_count.setdefault(module_id, 0)
        self._critic_update_count[module_id] += 1
        do_policy_update = (
            self._critic_update_count[module_id] % config.policy_delay == 0
        )
        self._should_update_targets[module_id] = do_policy_update

        # ----- Target-policy smoothing -----
        # Use the target policy for next-state actions.
        q_batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}
        target_action = module.forward_target_policy(q_batch_next).detach()
        noise = torch.clamp(
            torch.randn_like(target_action) * config.target_policy_noise,
            -config.target_noise_clip,
            config.target_noise_clip,
        )
        # tanh-bounded actions live in [-1, 1]
        target_action_smoothed = torch.clamp(target_action + noise, -1.0, 1.0)

        # ----- Q targets: y = r + γ * (1 - d) * min(Q1_t, Q2_t) -----
        q_batch_next_full = {
            Columns.OBS: batch[Columns.NEXT_OBS],
            Columns.ACTIONS: target_action_smoothed,
        }
        q_target_next = module.forward_target(q_batch_next_full).detach()
        q_next_masked = (1.0 - batch[Columns.TERMINATEDS].float()) * q_target_next
        q_selected_target = (
            batch[Columns.REWARDS]
            + (config.gamma ** batch["n_step"]) * q_next_masked
        ).detach()

        # ----- Critic loss (Huber) -----
        q_selected = fwd_out[QF_PREDS]
        critic_loss = torch.mean(
            batch["weights"]
            * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                q_selected, q_selected_target
            )
        )
        if config.twin_q:
            q_twin_selected = fwd_out[QF_TWIN_PREDS]
            critic_twin_loss = torch.mean(
                batch["weights"]
                * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                    q_twin_selected, q_selected_target
                )
            )

        # TD-error for prioritised replay
        td_error = torch.abs(q_selected - q_selected_target)
        if config.twin_q:
            td_error = 0.5 * (
                td_error + torch.abs(q_twin_selected - q_selected_target)
            )

        # ----- Actor loss (delayed) -----
        if do_policy_update:
            actor_loss = -torch.mean(fwd_out["q_curr"])
        else:
            actor_loss = torch.tensor(0.0, device=q_selected.device)

        # ----- Total loss -----
        total_loss = actor_loss + critic_loss
        if config.twin_q:
            total_loss = total_loss + critic_twin_loss

        # ----- Logging -----
        self.metrics.log_value(
            key=(module_id, TD_ERROR_KEY),
            value=td_error,
            reduce=None,
            clear_on_reduce=True,
        )
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: actor_loss,
                QF_LOSS_KEY: critic_loss,
                QF_MEAN_KEY: torch.mean(fwd_out["q_curr"]),
                QF_MAX_KEY: torch.max(fwd_out["q_curr"]),
                QF_MIN_KEY: torch.min(fwd_out["q_curr"]),
                TD_ERROR_MEAN_KEY: torch.mean(td_error),
            },
            key=module_id,
            window=1,
        )
        # Additional TD3 actor diagnostics for CSV analysis in run_algorithm.
        self.metrics.log_value(
            key=(module_id, "last_actor_loss"),
            value=actor_loss,
            window=1,
        )
        self.metrics.log_value(
            key=(module_id, "mean_actor_loss"),
            value=actor_loss,
            reduce="mean",
            clear_on_reduce=True,
        )
        self.metrics.log_value(
            key=(module_id, "actor_update_fraction"),
            value=torch.tensor(
                1.0 if do_policy_update else 0.0, device=q_selected.device
            ),
            reduce="mean",
            clear_on_reduce=True,
        )

        # Store individual losses for compute_gradients.
        if do_policy_update:
            self._temp_losses[(module_id, POLICY_LOSS_KEY)] = actor_loss
        self._temp_losses[(module_id, QF_LOSS_KEY)] = critic_loss
        if config.twin_q:
            self.metrics.log_value(
                key=(module_id, QF_TWIN_LOSS_KEY),
                value=critic_twin_loss,
                window=1,
            )
            self._temp_losses[(module_id, QF_TWIN_LOSS_KEY)] = critic_twin_loss

        return total_loss

    # ------------------------------------------------------------------
    # Gradients – skip actor when not a policy-update step
    # ------------------------------------------------------------------

    @override(SACTorchLearner)
    def compute_gradients(
        self, loss_per_module: Dict[ModuleID, TensorType], **kwargs
    ) -> ParamDict:
        grads = {}
        for module_id in set(loss_per_module.keys()) - {ALL_MODULES}:
            for optim_name, optim in self.get_optimizers_for_module(module_id):
                loss_key = (module_id, optim_name + "_loss")
                if loss_key not in self._temp_losses:
                    # Delayed policy update: no actor loss this step.
                    continue

                optim.zero_grad(set_to_none=True)
                loss_tensor = self._temp_losses.pop(loss_key)
                loss_tensor.backward(retain_graph=True)
                grads.update(
                    {
                        pid: p.grad
                        for pid, p in self.filter_param_dict_for_optimizer(
                            self._params, optim
                        ).items()
                    }
                )

        # Clear any remaining unused entries (e.g. if module list changed).
        self._temp_losses.clear()
        return grads

    @override(Learner)
    def after_gradient_based_update(self, *, timesteps: Dict[str, Any]) -> None:
        """Update TD3 targets only on delayed actor steps."""
        Learner.after_gradient_based_update(self, timesteps=timesteps)

        timestep = timesteps.get(NUM_ENV_STEPS_SAMPLED_LIFETIME, 0)
        for module_id, module in self.module._rl_modules.items():
            if not self._should_update_targets.pop(module_id, False):
                continue
            if not isinstance(module.unwrapped(), TargetNetworkAPI):
                continue

            for main_net, target_net in module.unwrapped().get_target_network_pairs():
                update_target_network(
                    main_net=main_net,
                    target_net=target_net,
                    tau=self.config.get_config_for_module(module_id).tau,
                )

            self.metrics.log_value((module_id, NUM_TARGET_UPDATES), 1, reduce="sum")
            self.metrics.log_value((module_id, LAST_TARGET_UPDATE_TS), timestep, window=1)
