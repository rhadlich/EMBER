from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from typing import Any, Dict

from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.sac.sac_learner import (
    LOGPS_KEY,
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
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import (
    POLICY_LOSS_KEY,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import ALL_MODULES, TD_ERROR_KEY
from ray.rllib.utils.typing import ModuleID, ParamDict, TensorType


torch, nn = try_import_torch()

def add_cli_args(parser):
    """Register only the flags SAC cares about."""
    parser.add_argument("--env-type", type=str, default="continuous",
                        help="whether environment action space is 'continuous' or 'discrete'.")
    return parser


def update_config(cfg, args):
    """Add/override APPO-specific settings on the RLlib config builder."""
    return (
        cfg.training(
            # model={"free_log_std": True},
            q_model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define custom Q-model(s).
                "custom_model_config": {},
            },
            policy_model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define a custom policy model.
                "custom_model_config": {},
            },

            # Generic algorithm hyperparams
            train_batch_size_per_learner=256,
            # training_intensity=256.0,
            gamma=0.95,
            n_step=1,
            grad_clip=10.0,

            # SAC hyperparams
            initial_alpha=0.1,
            target_entropy=-2.0,
            alpha_lr=1e-4,
            actor_lr=2e-5,
            critic_lr=2e-5,
            tau=5e-3,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": int(1e4),
            },
            # learner_class=SACTorchLearnerWithRBS,
        )
        # you can chain more .XYZ() builders here
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
        "policy_model_config": pmc,
        "q_model_config": qmc,
        "twin_q": bool(getattr(cfg, "twin_q", True)),
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


def _build_actor_episode_buffer_spec(
    *,
    algorithm: str,
    state_dim: int,
    action_dim: int,
    action_dist_inputs_dim: int,
    include_action_logp: bool,
    policy_output_kind: str,
    batch_size: int,
    num_slots: int,
    name: str,
) -> dict:
    bytes_per_float = 4
    field_order = ["action", "reward", "next_obs"]
    field_dims = {
        "action": action_dim,
        "reward": 1,
        "next_obs": state_dim,
    }
    if include_action_logp:
        field_order.append("action_logp")
        field_dims["action_logp"] = 1
    if action_dist_inputs_dim > 0:
        field_order.append("action_dist_inputs")
        field_dims["action_dist_inputs"] = action_dist_inputs_dim

    elements_per_rollout = sum(field_dims[field] for field in field_order)
    payload_size = elements_per_rollout * batch_size + state_dim
    header_size = 2
    header_slot_size = 1
    slot_size = header_slot_size + payload_size
    total_size = header_size + num_slots * slot_size

    return {
        "algorithm": algorithm,
        "policy_output_kind": policy_output_kind,
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
        "HAS_ACTION_LOGP": include_action_logp,
        "HAS_ACTION_DIST_INPUTS": action_dist_inputs_dim > 0,
        "action_dist_size": action_dist_inputs_dim,
    }


def get_actor_episode_buffer_spec(
    *,
    state_dim: int,
    action_space,
    action_adapter,
    batch_size: int = 32,
    num_slots: int = 8,
    name: str = "episodes",
) -> dict:
    """Return the SAC-specific shared-memory actor rollout schema."""
    action_dim = _get_buffer_action_dim(action_space, action_adapter)
    if action_adapter.mode == "continuous":
        action_dist_inputs_dim = 2 * int(action_space.shape[0])
        policy_output_kind = "gaussian"
    else:
        action_dist_inputs_dim = int(sum(action_adapter.nvec))
        policy_output_kind = "categorical"

    return _build_actor_episode_buffer_spec(
        algorithm="SAC",
        state_dim=state_dim,
        action_dim=action_dim,
        action_dist_inputs_dim=action_dist_inputs_dim,
        include_action_logp=True,
        policy_output_kind=policy_output_kind,
        batch_size=batch_size,
        num_slots=num_slots,
        name=name,
    )


class SACTorchLearnerWithRBS(SACTorchLearner):
    """
    SACTorchLearner with return-based scaling. This method was copied and pasted from the original SACTorchLearner class
    in ray version 2.46.0, and edits were made to add return-based scaling.
    """

    @override(SACTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: SACConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType]
    ) -> TensorType:
        # Receive the current alpha hyperparameter.
        alpha = torch.exp(self.curr_log_alpha[module_id])

        # Get Q-values for the actually selected actions during rollout.
        # In the critic loss we use these as predictions.
        q_selected = fwd_out[QF_PREDS]
        if config.twin_q:
            q_twin_selected = fwd_out[QF_TWIN_PREDS]

        # Compute value function for next state (see eq. (3) in Haarnoja et al. (2018)).
        # Note, we use here the sampled actions in the log probabilities.
        q_target_next = (
            fwd_out["q_target_next"] - alpha.detach() * fwd_out["logp_next_resampled"]
        )
        # Now mask all Q-values with terminated next states in the targets.
        q_next_masked = (1.0 - batch[Columns.TERMINATEDS].float()) * q_target_next

        # Compute the right hand side of the Bellman equation.
        # Detach this node from the computation graph as we do not want to
        # backpropagate through the target network when optimizing the Q loss.
        q_selected_target = (
            batch[Columns.REWARDS] + (config.gamma ** batch["n_step"]) * q_next_masked
        ).detach()

        # Calculate the TD-error. Note, this is needed for the priority weights in
        # the replay buffer.
        td_error = torch.abs(q_selected - q_selected_target)
        # If a twin Q network should be used, add the TD error of the twin Q network.
        if config.twin_q:
            td_error += torch.abs(q_twin_selected - q_selected_target)
            # Rescale the TD error.
            td_error *= 0.5

        # MSBE loss for the critic(s) (i.e. Q, see eqs. (7-8) Haarnoja et al. (2018)).
        # Note, this needs a sample from the current policy given the next state.
        # Note further, we use here the Huber loss instead of the mean squared error
        # as it improves training performance.
        critic_loss = torch.mean(
            batch["weights"]
            * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                q_selected, q_selected_target
            )
        )
        # If a twin Q network should be used, add the critic loss of the twin Q network.
        if config.twin_q:
            critic_twin_loss = torch.mean(
                batch["weights"]
                * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                    q_twin_selected, q_selected_target
                )
            )

        # For the actor (policy) loss we need sampled actions from the current policy
        # evaluated at the current observations.
        # Note that the `q_curr` tensor below has the q-net's gradients ignored, while
        # having the policy's gradients registered. The policy net was used to rsample
        # actions used to compute `q_curr` (by passing these actions through the q-net).
        # Hence, we can't do `fwd_out[q_curr].detach()`!
        # Note further, we minimize here, while the original equation in Haarnoja et
        # al. (2018) considers maximization.
        # TODO (simon): Rename to `resampled` to `current`.
        actor_loss = torch.mean(
            alpha.detach() * fwd_out["logp_resampled"] - fwd_out["q_curr"]
        )

        # Optimize also the hyperparameter alpha by using the current policy
        # evaluated at the current state (sampled values).
        # TODO (simon): Check, why log(alpha) is used, prob. just better
        # to optimize and monotonic function. Original equation uses alpha.
        alpha_loss = -torch.mean(
            self.curr_log_alpha[module_id]
            * (fwd_out["logp_resampled"].detach() + self.target_entropy[module_id])
        )

        total_loss = actor_loss + critic_loss + alpha_loss
        # If twin Q networks should be used, add the critic loss of the twin Q network.
        if config.twin_q:
            # TODO (simon): Check, if we need to multiply the critic_loss then with 0.5.
            total_loss += critic_twin_loss

        # Log the TD-error with reduce=None, such that - in case we have n parallel
        # Learners - we will re-concatenate the produced TD-error tensors to yield
        # a 1:1 representation of the original batch.
        self.metrics.log_value(
            key=(module_id, TD_ERROR_KEY),
            value=td_error,
            reduce=None,
            clear_on_reduce=True,
        )
        # Log other important loss stats (reduce=mean (default), but with window=1
        # in order to keep them history free).
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: actor_loss,
                QF_LOSS_KEY: critic_loss,
                "alpha_loss": alpha_loss,
                "alpha_value": alpha,
                "log_alpha_value": torch.log(alpha),
                "target_entropy": self.target_entropy[module_id],
                LOGPS_KEY: torch.mean(fwd_out["logp_resampled"]),
                QF_MEAN_KEY: torch.mean(fwd_out["q_curr"]),
                QF_MAX_KEY: torch.max(fwd_out["q_curr"]),
                QF_MIN_KEY: torch.min(fwd_out["q_curr"]),
                TD_ERROR_MEAN_KEY: torch.mean(td_error),
            },
            key=module_id,
            window=1,  # <- single items (should not be mean/ema-reduced over time).
        )

        self._temp_losses[(module_id, POLICY_LOSS_KEY)] = actor_loss
        self._temp_losses[(module_id, QF_LOSS_KEY)] = critic_loss
        self._temp_losses[(module_id, "alpha_loss")] = alpha_loss

        # If twin Q networks should be used add a critic loss for the twin Q network.
        # Note, we need this in the `self.compute_gradients()` to optimize.
        if config.twin_q:
            self.metrics.log_value(
                key=(module_id, QF_TWIN_LOSS_KEY),
                value=critic_twin_loss,
                window=1,  # <- single items (should not be mean/ema-reduced over time).
            )
            self._temp_losses[(module_id, QF_TWIN_LOSS_KEY)] = critic_twin_loss

        return total_loss
