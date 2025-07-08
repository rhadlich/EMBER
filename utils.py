import numpy as np
import torch
from ray.rllib.utils.numpy import softmax
from gymnasium import spaces

from typing import Union, Type

from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchMultiCategorical,
    TorchDiagGaussian,
    TorchSquashedGaussian
)
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper

import logging
import logging_setup


def get_action_dist(
        action_dist_cls: Type[TorchDistributionWrapper],
        mu: Union[float, "torch.Tensor"],
        std: Union[float, "torch.Tensor"],
        low: Union[float, "torch.Tensor", np.ndarray] = None,
        high: Union[float, "torch.Tensor", np.ndarray] = None,
):
    if action_dist_cls == TorchDiagGaussian:
        return TorchDiagGaussian(
            loc=mu,
            scale=std,
        )
    elif action_dist_cls == TorchSquashedGaussian:
        return TorchSquashedGaussian(
            loc=mu,
            scale=std,
            low=low,
            high=high,
        )
    else:
        raise NotImplementedError(f"Unsupported action_dist_cls {action_dist_cls}")


class ActionAdapter:
    """
    Codec -- translate actions between
      • env   ⟷   replay buffer   ⟷   network
    Works for:
      1. spaces.Discrete(n)                 (1-D categorical)
      2. spaces.MultiDiscrete(nvec)         (k-D categorical)
      3. Tuple / list of Discrete           (treated like MultiDiscrete)
      4. spaces.Box(low, high, (k,))        (k-D continuous)
    """

    # ---------- initialisation ------------------------------------------------
    def __init__(self, action_space: spaces.Space, *, action_dist_cls: Type[TorchDistributionWrapper] = None):
        self.log = logging.getLogger('MyRLApp.ActionAdapter')

        self.action_dist_cls = action_dist_cls
        self.space = action_space

        # --- single-dim categorical ------------------------------------------
        if isinstance(action_space, spaces.Discrete):
            self.mode = "discrete1"
            self.nvec = np.array([action_space.n], dtype=np.int32)
            self.nint = action_space.n

        # --- k-dim categorical -----------------------------------------------
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.mode = "multidiscrete"
            self.nvec = np.asarray(action_space.nvec, dtype=np.int32)
            self.nint = int(action_space.nvec.sum())

        elif isinstance(action_space, spaces.Tuple) and all(
                isinstance(sp, spaces.Discrete) for sp in action_space
        ):
            self.mode = "multidiscrete"
            self.nvec = np.array([sp.n for sp in action_space], dtype=np.int32)
            self.nint = int(self.nvec.sum())

        # --- continuous -------------------------------------------------------
        elif isinstance(action_space, spaces.Box):
            self.mode = "continuous"
            self.low = action_space.low.astype(np.float32)
            self.high = action_space.high.astype(np.float32)
            self.scale = (self.high - self.low) / 2.0
            self.center = (self.high + self.low) / 2.0
            self.act_dim = action_space.shape[0]
        else:
            raise NotImplementedError(f"Unsupported space {action_space}")

        if self.mode.startswith("multi"):
            # pre-compute split points for slicing the concatenated logits
            self.cuts = np.cumsum(self.nvec)[:-1]

    def unsquash(self, action):
        """
        Take action in the normalized space (-1, 1) and convert to the range expected by the environment.
        """
        if self.mode == "continuous":
            act_unsquashed = self.low + (action + 1) * self.scale
            return np.clip(act_unsquashed, self.low, self.high)
        else:
            raise NotImplementedError(f"Cannot unsquash, unsupported mode {self.mode}")

    # ---------- representation fed back into the network ----------------------
    def encode(self, action):
        """
        Return the representation you append to the observation
        as “previous action”.

        * Discrete(1)      → one-hot (length n)
        * Multi-Discrete   → concat(one-hot(i, n_i))  (length ∑ n_i)
        * Continuous       → scaled to (-1, 1)
        """
        if self.mode == "discrete1":
            one_hot = np.zeros(self.nvec[0], dtype=np.float32)
            one_hot[int(action)] = 1.0
            return one_hot

        if self.mode == "multidiscrete":
            outs = []
            for comp, n in zip(np.asarray(action).astype(int), self.nvec):
                oh = np.zeros(n, dtype=np.float32)
                oh[comp] = 1.0
                outs.append(oh)
            return np.concatenate(outs, dtype=np.float32)

        # continuous
        return ((action - self.center) / self.scale).astype(np.float32)

    # ---------- forward pass → env action + log-prob --------------------------
    def sample_from_policy(
            self, net_out, deterministic=False, rng=np.random.default_rng()
    ):
        """
        * For categorical: `net_out` is a **flat** logits vector
            – length n  (single-dim)  or  ∑ n_i  (multi-dim).
        * For continuous: `net_out` is
              (μ, log σ)  tuple  OR  just μ  for deterministic nets.
        Returns
            action_for_env  (scalar, ndarray, or list)
            logp            (float)   or None if deterministic
        """

        # ---------- CATEGORICAL ------------------------------------------------
        if self.mode in ("discrete1", "multidiscrete"):
            logits = net_out.astype(np.float32)

            # split into per-dimension blocks (single-dim gives 1 block)
            blocks = (
                [logits]
                if self.mode == "discrete1"
                else np.split(logits, self.cuts)
            )

            actions = []
            logps = []

            for logit_vec, n in zip(blocks, self.nvec):
                if deterministic:
                    a = int(np.argmax(logit_vec))
                    prob = softmax(logit_vec)[a]
                else:
                    probs = softmax(logit_vec)
                    a = int(rng.choice(n, p=probs))
                    prob = probs[a]
                actions.append(a)
                logps.append(np.log(prob + 1e-8))

            if self.mode == "discrete1":
                return actions[0], logps[0], None
            # k-dim ➜ numpy int array for env; sum log-probs (independent dims)
            return np.array(actions, dtype=np.int32), float(np.sum(logps)), None

        # ---------- CONTINUOUS -------------------------------------------------
        # self.log.debug("ActionAdapter (sample_from_policy): in the continuous action section.")

        if isinstance(net_out, (tuple, np.ndarray)):
            # self.log.debug(f"ActionAdapter (sample_from_policy): net_out= {net_out}.")
            if isinstance(net_out, tuple):
                mu, log_sigma = [x.astype(np.float32) for x in net_out]
            else:
                if net_out.size == 2 * self.act_dim:
                    mu = net_out[:self.act_dim]
                    log_sigma = net_out[self.act_dim:]
                else:
                    raise NotImplementedError(f"Unexpected net_out size {net_out.size}; "
                                              f"expected {2 * self.act_dim} for action_dim={self.act_dim}")
            dist_inputs = np.concatenate([mu, log_sigma], axis=-1).astype(np.float32)
            if deterministic:
                act = ((mu + 1.0) / 2.0) * (self.high - self.low) + self.low
                logp = None
            else:
                # use RLlib's built-in class to perform sampling
                try:
                    with torch.no_grad():
                        mu_t = torch.from_numpy(mu)
                        log_st = torch.from_numpy(log_sigma)
                        low_t = torch.from_numpy(self.low)
                        high_t = torch.from_numpy(self.high)
                        dist = get_action_dist(self.action_dist_cls,
                                               mu=mu_t,
                                               std=log_st.exp(),
                                               low=low_t,
                                               high=high_t)
                        act_t = dist.sample()  # sample action normalized (using tanh) by the env bounds
                        logp_t = dist.logp(act_t)  # calculate log probability
                    act = act_t.numpy()
                    logp = logp_t.numpy()
                except Exception as e:
                    self.log.debug(f"ActionAdapter (sample_from_policy): got exception {e}")

        else:  # deterministic net
            mu = net_out.astype(np.float32)
            act = mu
            logp = None
            dist_inputs = None

        # # clip to env range
        # act = np.clip(act, -1, 1)

        # self.log.debug(f"ActionAdapter (sample_from_policy): outputs are: act={act}, logp={logp}, dist_inputs={dist_inputs}.")
        return act.astype(np.float32), logp, dist_inputs
