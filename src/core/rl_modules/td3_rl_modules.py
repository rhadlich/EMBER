"""TD3 (Twin Delayed DDPG) RLModule and Catalog.

Builds on the SAC infrastructure but replaces the stochastic policy with a
deterministic one and removes entropy/alpha-related logic.
"""
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.sac_learner import (
    QF_PREDS,
    QF_TWIN_PREDS,
)
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder, Model
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.rl_module.apis import TargetNetworkAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.torch.torch_distributions import TorchDeterministic
from ray.rllib.utils.annotations import override, OverrideToImplementCustomLogic
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TD3Catalog(SACCatalog):
    """Catalog for TD3 models.

    Identical to SACCatalog except the policy head outputs ``action_dim``
    (deterministic action) instead of ``2 * action_dim`` (mean + log-std),
    and the action distribution is ``TorchDeterministic``.
    """

    @OverrideToImplementCustomLogic
    def build_pi_head(self, framework: str) -> Model:
        action_dim = int(np.prod(self.action_space.shape, dtype=np.int32))
        self.pi_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.pi_and_qf_head_hiddens,
            hidden_layer_activation=self.pi_and_qf_head_activation,
            output_layer_dim=action_dim,
            output_layer_activation="tanh",
        )
        return self.pi_head_config.build(framework=framework)

    @override(SACCatalog)
    def get_action_dist_cls(self, framework: str) -> type:
        assert framework == "torch"
        return TorchDeterministic


class TD3TorchRLModule(DefaultSACTorchRLModule):
    """RLModule for TD3.

    Key differences from ``DefaultSACTorchRLModule``:
    - Deterministic policy (tanh-bounded, no log-std).
    - No log-probability computation.
    - Target-policy smoothing: clipped Gaussian noise is added to target
      actions when computing Q-targets during training.
    """

    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = TD3Catalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

        try:
            print("TD3TorchRLModule built type:", type(self))
            print("TD3TorchRLModule dist cls:", self.get_inference_action_dist_cls())
            print("TD3TorchRLModule catalog attr:", type(getattr(self, "catalog", None)))
        except Exception as e:
            print("TD3TorchRLModule post-init probe failed:", e)


    # ------------------------------------------------------------------
    # Inference / exploration
    # ------------------------------------------------------------------

    @override(RLModule)
    def _forward_inference(self, batch: Dict) -> Dict[str, Any]:
        pi_encoder_outs = self.pi_encoder(batch)
        action = self.pi(pi_encoder_outs[ENCODER_OUT])
        # Provide direct actions for connector-v2 sampling paths.
        # Keeping ACTION_DIST_INPUTS preserves compatibility with existing
        # realtime/minion code that reads model outputs as raw policy inputs.
        return {
            Columns.ACTIONS: action,
            Columns.ACTION_DIST_INPUTS: action,
        }

    @override(RLModule)
    def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        return self._forward_inference(batch)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    @override(RLModule)
    def _forward_train(self, batch: Dict) -> Dict[str, Any]:
        if self.inference_only:
            raise RuntimeError(
                "Trying to train a module that is not a learner module. "
                "Set the flag `inference_only=False` when building the module."
            )
        output: Dict[str, Any] = {}

        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

        # --- Critic predictions for the actual replay-buffer actions ---
        batch_curr_with_actions = {
            Columns.OBS: batch[Columns.OBS],
            Columns.ACTIONS: batch[Columns.ACTIONS],
        }
        output[QF_PREDS] = self._qf_forward_train_helper(
            batch_curr_with_actions, self.qf_encoder, self.qf,
        )
        if self.twin_q:
            output[QF_TWIN_PREDS] = self._qf_forward_train_helper(
                batch_curr_with_actions, self.qf_twin_encoder, self.qf_twin,
            )

        # --- Deterministic policy output for current observations ---
        pi_encoder_outs = self.pi_encoder(batch_curr)
        action_curr = self.pi(pi_encoder_outs[ENCODER_OUT])
        output[Columns.ACTION_DIST_INPUTS] = action_curr

        # Q(s, pi(s)) with Q-net gradients detached (straight-through for
        # policy gradients only, same trick SAC uses).
        q_batch_curr = {
            Columns.OBS: batch[Columns.OBS],
            Columns.ACTIONS: action_curr,
        }
        all_params = list(self.qf.parameters()) + list(self.qf_encoder.parameters())
        if self.twin_q:
            all_params += list(self.qf_twin.parameters()) + list(
                self.qf_twin_encoder.parameters()
            )
        for p in all_params:
            p.requires_grad = False
        output["q_curr"] = self._qf_forward_train_helper(
            q_batch_curr, self.qf_encoder, self.qf,
        )
        for p in all_params:
            p.requires_grad = True

        # --- Target policy smoothing ---
        # Compute target actions with noise.  The actual noise parameters
        # (target_policy_noise, target_noise_clip) live on the config and are
        # read in the Learner; here we just produce the *deterministic* target
        # action so the Learner can add noise later.
        pi_encoder_next_outs = self.pi_encoder(batch_next)
        # Use the *current* policy encoder + head (the target-network copy is
        # handled via TargetNetworkAPI in forward_target).
        action_next = self.pi(pi_encoder_next_outs[ENCODER_OUT]).detach()
        output["action_next_deterministic"] = action_next

        return output

    # ------------------------------------------------------------------
    # Target network forward pass (used by learner for Q-targets)
    # ------------------------------------------------------------------

    @override(TargetNetworkAPI)
    def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Return min(Q1_target, Q2_target) for the given (obs, action) batch."""
        target_qvs = self._qf_forward_train_helper(
            batch, self.target_qf_encoder, self.target_qf,
        )
        if self.twin_q:
            target_qvs = torch.min(
                target_qvs,
                self._qf_forward_train_helper(
                    batch, self.target_qf_twin_encoder, self.target_qf_twin,
                ),
            )
        return target_qvs

    # We also need a method to compute the *target policy* action (for target
    # policy smoothing).  We add a dedicated helper so the Learner can call it.
    def forward_target_policy(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Deterministic target-policy action for next-state observations.

        Uses the frozen target copies of pi_encoder and pi.
        """
        if not hasattr(self, "target_pi_encoder"):
            raise RuntimeError(
                "Target policy networks have not been created. "
                "Ensure make_target_networks() has been called."
            )
        pi_enc_out = self.target_pi_encoder(batch)
        return self.target_pi(pi_enc_out[ENCODER_OUT])

    # ------------------------------------------------------------------
    # Target-network bookkeeping – TD3 also needs target copies of the
    # policy (actor) networks for target-policy smoothing.
    # ------------------------------------------------------------------

    @override(DefaultSACTorchRLModule)
    def make_target_networks(self):
        super().make_target_networks()
        from ray.rllib.core.learner.utils import make_target_network
        self.target_pi_encoder = make_target_network(self.pi_encoder)
        self.target_pi = make_target_network(self.pi)

    @override(DefaultSACTorchRLModule)
    def get_non_inference_attributes(self):
        attrs = super().get_non_inference_attributes()
        attrs += ["target_pi_encoder", "target_pi"]
        return attrs

    @override(DefaultSACTorchRLModule)
    def get_target_network_pairs(self):
        pairs = super().get_target_network_pairs()
        pairs += [
            (self.pi_encoder, self.target_pi_encoder),
            (self.pi, self.target_pi),
        ]
        return pairs
