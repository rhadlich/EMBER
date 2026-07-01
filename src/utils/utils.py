import numpy as np
import torch
from ray.rllib.utils.numpy import softmax
from gymnasium import spaces

from typing import Union, Type, Optional

POLICY_ACTION_NORM_LOW = -1.0
POLICY_ACTION_NORM_HIGH = 1.0

from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchMultiCategorical,
    TorchDiagGaussian,
    TorchSquashedGaussian,
    TorchDeterministic,
)
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.distributions import Distribution

import logging
import time
import utils.logging_setup as logging_setup
import csv
from datetime import datetime


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
            self.env_low = action_space.low.astype(np.float32)
            self.env_high = action_space.high.astype(np.float32)
            self.policy_low = np.full_like(self.env_low, POLICY_ACTION_NORM_LOW)
            self.policy_high = np.full_like(self.env_high, POLICY_ACTION_NORM_HIGH)
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
                low=torch.from_numpy(self.policy_low),
                high=torch.from_numpy(self.policy_high),
            )
        else:
            raise NotImplementedError(f"Unsupported action_dist_cls {self.action_dist_cls}")

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize the action to the policy range expected by the learner ([-1, 1]).
        """
        if self.mode == "continuous":
            return torch.clamp(action, min=-1.0, max=1.0)
        if self.action_dist_cls == TorchDiagGaussian:
            return torch.clamp(action, min=-1.0, max=1.0)
        raise NotImplementedError(f"Unsupported action_dist_cls {self.action_dist_cls}")

    def get_action_in_env_range(self, action: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Map policy-space actions in [-1, 1] to physical environment bounds.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().astype(np.float32)

        if self.mode == "continuous":
            # Lazy import avoids circular import via core.environments.__init__.
            from core.environments.env_adapter import EnvAdapter

            return EnvAdapter.denormalize_action(
                action,
                action_low=self.env_low,
                action_high=self.env_high,
            )

        raise NotImplementedError(
            f"get_action_in_env_range not implemented for mode {self.mode}"
        )

    # ---------- forward pass → env action + log-prob --------------------------
    def sample_from_policy(
            self, net_out, deterministic=False, rng=None
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

        # Lazily create an RNG if none was provided. Minion passes in a seeded
        # RNG for reproducible runs; this fallback is only used in contexts
        # where we don't control the caller.
        if rng is None:
            rng = np.random.default_rng()

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
                elif net_out.size == self.act_dim:
                    # Deterministic policy (e.g. TD3): output is the action
                    # directly, no log-std component.
                    act = np.clip(net_out, -1.0, 1.0).astype(np.float32)
                    logp = 0.0
                    dist_inputs = act.copy()
                    return act, logp, dist_inputs
                else:
                    raise NotImplementedError(f"Unexpected net_out size {net_out.size}; "
                                              f"expected {2 * self.act_dim} or {self.act_dim} for action_dim={self.act_dim}")
            dist_inputs = np.concatenate([mu, log_sigma], axis=-1).astype(np.float32)
            if deterministic:
                if self.action_dist_cls == TorchSquashedGaussian:
                    act = torch.tanh(torch.from_numpy(mu)).numpy().astype(np.float32)
                else:
                    act = self.normalize_action(torch.from_numpy(mu)).numpy().astype(np.float32)
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

                        # sample and logp in policy space [-1, 1]
                        act_t = dist.sample()
                        logp_t = dist.logp(act_t)
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


def get_rollout_field_slices(shm_properties: dict) -> dict[str, slice]:
    """Return per-field slices for one rollout row."""
    start = 0
    slices = {}
    for field in shm_properties["ROLLOUT_FIELD_ORDER"]:
        field_size = shm_properties["ROLLOUT_FIELD_DIMS"][field]
        slices[field] = slice(start, start + field_size)
        start += field_size
    return slices


def build_rollout_row(shm_properties: dict, field_values: dict[str, Union[float, np.ndarray]]) -> np.ndarray:
    """Serialize one rollout row according to the configured schema."""
    row = np.zeros(shm_properties["ELEMENTS_PER_ROLLOUT"], dtype=np.float32)
    field_slices = get_rollout_field_slices(shm_properties)
    for field in shm_properties["ROLLOUT_FIELD_ORDER"]:
        if field not in field_values:
            raise KeyError(f"Missing rollout field {field}")
        value = np.asarray(field_values[field], dtype=np.float32).reshape(-1)
        expected = shm_properties["ROLLOUT_FIELD_DIMS"][field]
        if value.size != expected:
            raise ValueError(
                f"Field {field} expected size {expected}, got {value.size}"
            )
        row[field_slices[field]] = value
    return row


class TimingRecorder:
    """
    A reusable class for recording and saving timing measurements to CSV files.
    Can be used across multiple processes.
    """
    
    def __init__(self, csv_path: Optional[str] = None, logger: Optional[logging.Logger] = None, enabled: bool = True):
        """
        Initialize the timing recorder.
        
        Args:
            csv_path: Path to the CSV file. If None, generates a timestamped filename.
            logger: Logger instance. If None, creates a default logger.
            enabled: When False, record_timing and save_timing_data are no-ops.
        """
        self.enabled = enabled
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
        
        if self.enabled:
            self.logger.debug(f"TimingRecorder: Timing data will be saved to {self.timing_csv_path}")
        else:
            self.logger.debug("TimingRecorder: Timing logging is disabled.")
    
    def record_timing(self, process_name: str, duration_ms: float, deterministic: Optional[bool] = None):
        """
        Record a timing measurement to in-memory storage.
        Does not save to CSV immediately to avoid interfering with nested operations.
        
        Args:
            process_name: Name of the process being timed
            duration_ms: Duration in milliseconds
            deterministic: Optional boolean flag indicating if the process was deterministic
        """
        if not self.enabled:
            return
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
        if not self.enabled or not self.timing_data:
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


class EpisodeLogger:
    """
    Records per-step episode data to a CSV file incrementally.
    The CSV header is derived from the first row written (lazy initialization).
    """

    def __init__(self, csv_path: str, logger: Optional[logging.Logger] = None):
        """
        Args:
            csv_path: Destination CSV file path.
            logger: Logger instance. If None, a default logger is created.
        """
        self.csv_path = csv_path
        self._rows: list[dict] = []
        self._initialized = False

        if logger is None:
            self.logger = logging.getLogger("MyRLApp.EpisodeLogger")
        else:
            self.logger = logger

        self.logger.debug(f"EpisodeLogger: Episode data will be saved to {self.csv_path}")

    def log_step(self, row: dict) -> None:
        """Append a single step's data to the in-memory buffer."""
        self._rows.append(row)

    def flush(self) -> None:
        """Write all buffered rows to the CSV file. Header is written on the first call."""
        if not self._rows:
            return

        fieldnames = list(self._rows[0].keys())
        write_header = not self._initialized

        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                    self._initialized = True
                for row in self._rows:
                    writer.writerow(row)
            self._rows = []
        except Exception as e:
            self.logger.warning(f"EpisodeLogger: Failed to flush episode data: {e}")

    def close(self) -> None:
        """Flush any remaining buffered rows."""
        self.flush()
