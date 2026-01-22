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
from ray.rllib.models.distributions import Distribution

import logging
import utils.logging_setup as logging_setup
import csv
import time
from datetime import datetime
from typing import Optional


# class DistributionHandler:
#     def __init__(
#             self,
#             action_dist_cls: Type[TorchDistributionWrapper],
#             mu: Union[float, "torch.Tensor"],
#             std: Union[float, "torch.Tensor"],
#             low: Union[float, "torch.Tensor", np.ndarray] = None,
#             high: Union[float, "torch.Tensor", np.ndarray] = None,
#     ):
#         self.action_dist_cls = action_dist_cls
#         self.mu = mu
#         self.std = std
#         self.low = low
#         self.high = high
#
#         # get distribution object
#         if action_dist_cls == TorchDiagGaussian:
#             self.dist = TorchDiagGaussian(
#                 loc=mu,
#                 scale=std,
#             )
#         elif action_dist_cls == TorchSquashedGaussian:
#             self.dist = TorchSquashedGaussian(
#                 loc=mu,
#                 scale=std,
#                 low=low,
#                 high=high,
#             )
#         else:
#             raise NotImplementedError(f"Unsupported action_dist_cls {action_dist_cls}")
#
#     def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Method for sampling the action and logp using the action distribution.
#
#         :return:
#             act_for_env: action scaled to the range expected by the environment.
#             act_norm: action in the range expected by the learner (to be passed to EnvRunner).
#             logp: log probability of the sampled action (to be passed to EnvRunner).
#         """
#         if self.action_dist_cls == TorchDiagGaussian:
#             act_raw = self.dist.sample()
#             act_norm = torch.clamp(act_raw, min=-1.0, max=1.0)
#             logp = self.dist.logp(act_norm)
#             act_for_env = ((act_norm + 1) / 2) * (self.high - self.low) + self.low
#             return act_for_env, act_norm, logp
#         elif self.action_dist_cls == TorchDiagGaussian:
#             act_norm = self.dist.sample()
#             act_for_env = act_norm
#             logp = self.dist.logp(act_norm)
#             return act_for_env, act_norm, logp


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
    def __init__(
            self,
            action_space: spaces.Space,
            *,
            action_dist_cls: Union[Type[TorchDistributionWrapper], Type[Distribution]]
            = None
    ):
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

        self.dist = None

    def get_action_dist(
            self,
            mu: Union[float, "torch.Tensor"],
            std: Union[float, "torch.Tensor"],
            *,
            action_dist_cls: Type[TorchDistributionWrapper] = None
    ):
        if not self.action_dist_cls and not action_dist_cls:
            raise NotImplementedError(f"Action distribution class not provided.")

        if action_dist_cls:
            self.action_dist_cls = action_dist_cls

        # get distribution object
        if self.action_dist_cls == TorchDiagGaussian:
            return TorchDiagGaussian(
                loc=mu,
                scale=std,
            )
        elif self.action_dist_cls == TorchSquashedGaussian:
            return TorchSquashedGaussian(
                loc=mu,
                scale=std,
                low=torch.from_numpy(self.low),
                high=torch.from_numpy(self.high),
            )
        else:
            raise NotImplementedError(f"Unsupported action_dist_cls {self.action_dist_cls}")

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize the action to match what the learner expects according to the action distribution class.
        """
        if self.action_dist_cls == TorchDiagGaussian:
            return torch.clamp(action, min=-1.0, max=1.0)
        elif self.action_dist_cls == TorchSquashedGaussian:
            # squashed gaussian already squashes to the env range
            return action
        else:
            raise NotImplementedError(f"Unsupported action_dist_cls {self.action_dist_cls}")

    def get_action_in_env_range(self, action: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get the action to be in the environment range if it is not already.
        """
        if isinstance(action, torch.Tensor):
            action = action.numpy().astype(np.float32)

        if self.action_dist_cls == TorchDiagGaussian:
            return ((action + 1.0) / 2) * (self.high - self.low) + self.low
        elif self.action_dist_cls == TorchSquashedGaussian:
            return action
        else:
            raise NotImplementedError(f"Unsupported action_dist_cls {self.action_dist_cls}")

    # ---------- forward pass → env action + log-prob --------------------------
    def sample_from_policy(
            self, net_out, deterministic=False, rng=np.random.default_rng()
    ):
        """
        * For categorical: `net_out` is a flat logits vector
            – length n  (single-dim)  or  ∑ n_i  (multi-dim).
        * For continuous: `net_out` is
              (μ, log σ)  tuple  OR  just μ  for deterministic nets.
        Returns
            action_norm     (scalar, ndarray, or list)
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
                        dist = self.get_action_dist(
                           mu=mu_t,
                           std=log_st.exp(),
                        )

                        # get action and logp normalized to the range expected by the learner
                        act_t = self.normalize_action(dist.sample())
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


        # self.log.debug(
        #     f"ActionAdapter (sample_from_policy): outputs are: act={act}, logp={logp}, dist_inputs={dist_inputs}.")
        return act.astype(np.float32), logp, dist_inputs


class TimingRecorder:
    """
    A reusable class for recording and saving timing measurements to CSV files.
    Can be used across multiple processes.
    """
    
    def __init__(self, csv_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the timing recorder.
        
        Args:
            csv_path: Path to the CSV file. If None, generates a timestamped filename.
            logger: Logger instance. If None, creates a default logger.
        """
        self.timing_data = []  # in-memory storage for timing records
        self.sequence_number = 0  # counter for sequence numbers
        
        if csv_path is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.timing_csv_path = f"timing_{timestamp_str}.csv"
        else:
            self.timing_csv_path = csv_path
        
        self.timing_csv_initialized = False  # track if CSV header has been written
        
        if logger is None:
            self.logger = logging.getLogger('MyRLApp.TimingRecorder')
        else:
            self.logger = logger
        
        self.logger.debug(f"TimingRecorder: Timing data will be saved to {self.timing_csv_path}")
    
    def record_timing(self, process_name: str, duration_ms: float, deterministic: Optional[bool] = None):
        """
        Record a timing measurement to in-memory storage.
        Does not save to CSV immediately to avoid interfering with nested operations.
        
        Args:
            process_name: Name of the process being timed
            duration_ms: Duration in milliseconds
            deterministic: Optional boolean flag indicating if the process was deterministic
        """
        self.sequence_number += 1
        record = {
            'timestamp': round(time.time(), 3),  # Round to 3 decimal places (millisecond precision)
            'sequence_number': self.sequence_number,
            'process_name': process_name,
            'duration_ms': round(duration_ms, 3),  # Round to 3 decimal places (0.001ms precision)
            'deterministic': deterministic if deterministic is not None else None
        }
        self.timing_data.append(record)
    
    def save_timing_data(self):
        """
        Save accumulated timing data to CSV file incrementally.
        This should be called at appropriate points (not during nested operations).
        """
        if not self.timing_data:
            return
        
        # Write header if this is the first time
        write_header = not self.timing_csv_initialized
        
        try:
            with open(self.timing_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'sequence_number', 'process_name', 'duration_ms', 'deterministic'])
                if write_header:
                    writer.writeheader()
                    self.timing_csv_initialized = True
                
                # Write all accumulated data
                for record in self.timing_data:
                    writer.writerow(record)
            
            # Clear the in-memory buffer after successful write
            self.timing_data = []
        except Exception as e:
            self.logger.warning(f"TimingRecorder: Failed to save timing data: {e}")
