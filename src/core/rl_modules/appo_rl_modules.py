from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import copy

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical
from ray.rllib.core.rl_module.apis import (
    TARGET_NETWORK_ACTION_DIST_INPUTS,
    TargetNetworkAPI,
)
from ray.rllib.utils.typing import NetworkType

from ...utils.utils import ActionAdapter

import logging
from ...utils import logging_setup


class AppoMlpModule(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """
    RLModule to use with APPO and have full control of the network parameters.
    """

    # 1) Build the sub-nets here
    def setup(self):

        self.logger = logging.getLogger("MyRLApp.RLModule")

        self.logger.info("MLPActorCriticModule.setup()")

        self.action_adapter = ActionAdapter(self.action_space)

        self._build_networks()

        if not self.cont:
            # if the action space is not continuous, need to get the lens to feed into the action distribution classes.
            # the lens is just a way to tell the action distribution classes which part of the logits belongs to which
            # action.
            self.lens = tuple([int(k) for k in self.action_adapter.nvec])
            self.logger.debug(f"lens: {self.lens}")

    def make_target_networks(self) -> None:
        self._old_actor_encoder = copy.deepcopy(self.actor_encoder)
        self._old_actor_head = copy.deepcopy(self.actor_head)

        if self.action_adapter.mode in ("discrete1", "multidiscrete"):
            return
        elif self.action_adapter.mode == "continuous":
            self._old_log_std_module = copy.deepcopy(self.log_std_module)
        return

    def get_target_network_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        pairs = [
            (self.actor_encoder, self._old_actor_encoder),
            (self.actor_head, self._old_actor_head),
        ]
        if self.action_adapter.mode in ("discrete1", "multidiscrete"):
            return pairs
        elif self.action_adapter.mode == "continuous":
            pairs.append((self.log_std_module, self._old_log_std_module))
        return pairs

    def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obs = batch[Columns.OBS].float()

        old_actor_inputs_encoded = self._old_actor_encoder(obs)

        if self.action_adapter.mode in ("discrete1", "multidiscrete"):
            old_action_dist = self._old_actor_head(old_actor_inputs_encoded)
        elif self.action_adapter.mode == "continuous":
            old_mu = self._old_actor_head(old_actor_inputs_encoded)
            old_log_std = self._old_log_std_module.log_std.expand_as(old_mu)
            old_action_dist = torch.cat([old_mu, old_log_std], dim=-1)
        else:
            raise NotImplementedError(f"Unsupported obs space {self.observation_space}")

        return {TARGET_NETWORK_ACTION_DIST_INPUTS: old_action_dist}

    # 2) One shared forward – used by exploration, train, inference
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].float()

        # actor
        encoding = self.actor_encoder(obs)
        if self.cont:
            mu = self.actor_head(encoding)
            log_std = self.log_std_module.log_std.expand_as(mu)
            action_inputs = torch.cat([mu, log_std], dim=-1)
        else:
            action_inputs = self.actor_head(encoding)

        return {
            Columns.ACTION_DIST_INPUTS: action_inputs,
        }

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS].float()

        # critic
        values = self.value_head(self.actor_encoder(obs)).squeeze(-1)

        # actor
        encoding = self.actor_encoder(obs)
        if self.cont:
            mu = self.actor_head(encoding)
            log_std = self.log_std_module.log_std.expand_as(mu)
            action_inputs = torch.cat([mu, log_std], dim=-1)
        else:
            action_inputs = self.actor_head(encoding)

        return {
            Columns.ACTION_DIST_INPUTS: action_inputs,
            Columns.EMBEDDINGS: values,     # this will get passed down to compute_values()
        }

    def compute_values(self, batch, embeddings=None):
        # Re-use cached critic output if caller provides it
        if embeddings is None:
            obs = batch[Columns.OBS].float()
            return self.value_head(self.actor_encoder(obs)).squeeze(-1)

        assert isinstance(embeddings, torch.Tensor)
        return embeddings

    def _build_networks(self):
        self.logger.debug("In MLPActorCriticModule._build_networks()")

        if hasattr(self, "actor_encoder"):
            return

        self.logger.debug("Does not have actor_encoder, building it.")

        if isinstance(self.observation_space, gym.spaces.Box):
            obs_dim = int(np.prod(self.observation_space.shape))
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            obs_dim = 1
        else:
            raise NotImplementedError(f"Unsupported obs space {self.observation_space}")

        # extract action space information and make actor head

        if self.action_adapter.mode in ("discrete1", "multidiscrete"):
            # discrete action space case
            self.cont = False
            act_dim = self.action_adapter.nint
            self.actor_head = nn.Linear(64, act_dim)
            self.logger.debug(f"actor head: {self.actor_head}")
        elif self.action_adapter.mode == "continuous":
            # continuous action space case
            self.cont = True
            act_dim = self.action_space.shape[0]
            self.actor_head = nn.Linear(64, act_dim)
            # for the log head, this can be state-dependent or not. SAC uses state-dependent, default IMPALA uses
            # state-independent. to make it state-dependent, the log head would have to be a nn.Linear(XX, act_dim)
            # and its output needs to be clipped, something like torch.clamp(self.log_std_head(feat), min=-20., max=2.),
            # to keep the std from being too small or too large.
            self.log_std_module = LogStdModule(torch.zeros(act_dim, dtype=torch.float32))
            self.logger.debug(f"mu head: {self.actor_head}")
        else:
            raise NotImplementedError(f"Unsupported space {self.action_space}")

        activation = nn.ReLU()

        # -------- actor ----------
        self.actor_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), activation,
            nn.Linear(64, 64), activation,
            nn.Linear(64, 64), activation,
        )

        # -------- critic ----------
        self.critic_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), activation,
            nn.Linear(64, 64), activation,
            nn.Linear(64, 32), activation,
        )
        self.value_head = nn.Linear(64, 1)

        self.logger.debug(f"actor encoder: {self.actor_encoder}")

    def get_inference_action_dist_cls(self):
        if isinstance(self.action_space, gym.spaces.Tuple):
            return self._tuple_multi_cat_cls(self.lens)
        return super().get_inference_action_dist_cls()

    def get_exploration_action_dist_cls(self):
        if isinstance(self.action_space, gym.spaces.Tuple):
            return self._tuple_multi_cat_cls(self.lens)
        return super().get_exploration_action_dist_cls()

    def get_train_action_dist_cls(self):
        if isinstance(self.action_space, gym.spaces.Tuple):
            return self._tuple_multi_cat_cls(self.lens)
        return super().get_train_action_dist_cls()

    def _tuple_multi_cat_cls(self, lens):

        class FixedMultiCategorical(TorchMultiCategorical):
            @classmethod
            def from_logits(cls, logits):
                return super().from_logits(logits, lens)
        return FixedMultiCategorical


class LogStdModule(nn.Module):
    """Container that turns a nn.Parameter into a nn.Module."""
    def __init__(self, init_tensor: torch.Tensor):
        super().__init__()
        self.log_std = nn.Parameter(init_tensor.clone())

    def forward(self, *_):
        #  called only if you explicitly invoke it; otherwise treat
        #  self.log_std like any normal parameter.
        return self.log_std

