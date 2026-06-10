from multiprocessing import shared_memory
from typing import Optional, Union

import numpy as np
import pandas as pd
import struct
import onnxruntime as ort
import time
import gzip
import os
import torch
import copy
import random

from ray.rllib.utils.numpy import softmax
import gymnasium as gym
from core.environments import ENGINE_CONTINUOUS_ADAPTER_ID, get_env_adapter, reward_fn

from utils.utils import (
    ActionAdapter,
    TimingRecorder,
    build_rollout_row,
    get_rollout_field_slices,
)
from utils.shared_memory_utils import get_indices, set_indices

from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.core import DEFAULT_MODULE_ID

import logging
import utils.logging_setup as logging_setup
from pprint import pformat
from datetime import datetime
from core.safety.safety_filter import SafetyFilter

# Try to import zmq, but make it optional
try:
    import zmq
    zmq_available = True
except ImportError:
    zmq_available = False
    zmq = None


def _flatten_obs_array(obs) -> np.ndarray:
    """
    Function that flattens the observation so that it can be stored in the
    shared memory buffer.
    """
    # return np.append(obs["state"], obs["target"]).astype(np.float32)
    # return np.expand_dims(obs, 0).astype(np.float32)
    return obs.astype(np.float32)


def set_realtime_priority(priority: int = 80, logger=None):
    """
    Set real-time scheduling priority for the current process.
    
    Args:
        priority: SCHED_FIFO priority (1-99). Higher = more priority.
                  Use 80-90 range to avoid starving system processes.
        logger: Optional logger instance for logging messages.
    """
    try:
        import ctypes
        from ctypes import c_int
        
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        
        # Constants
        SCHED_FIFO = 1
        
        class sched_param(ctypes.Structure):
            _fields_ = [("sched_priority", c_int)]
        
        # Set priority
        param = sched_param(c_int(priority))
        result = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
        
        if result != 0:
            errno = ctypes.get_errno()
            if logger:
                if errno == 1:  # EPERM - Operation not permitted
                    logger.warning(
                        f"Failed to set real-time priority: Permission denied (errno {errno}). "
                        f"Please run the application with sudo to enable real-time scheduling."
                    )
                else:
                    logger.warning(f"Failed to set real-time priority: errno {errno}")
        else:
            if logger:
                logger.info(f"Set real-time priority to {priority} (SCHED_FIFO)")
    except Exception as e:
        # If setting real-time priority fails (e.g., not running as root, or not on Linux),
        # log a warning but continue execution
        if logger:
            logger.warning(f"Could not set real-time priority: {e}. Continuing with default scheduling.")


class Minion:
    def __init__(
            self,
            policy_shm_name: str,
            flag_shm_name: str,
            ep_shm_name: str,
            config,
    ):
        # create logger
        self.logger = logging.getLogger("MyRLApp.Minion")
        self.logger.info(f"Minion, PID={os.getpid()}")
        self.logger.debug("Minion: Started __init__()")

        self.config = config

        # Configure deterministic RNGs for this process, if seeds were provided
        # via env_config. When no seeds are set, behavior stays non-deterministic.
        env_cfg = getattr(self.config, "env_config", {}) or {}
        minion_seed = env_cfg.get("minion_seed")
        self._env_seed = env_cfg.get("env_seed")
        if minion_seed is not None:
            minion_seed = int(minion_seed)
            random.seed(minion_seed)
            np.random.seed(minion_seed)
            torch.manual_seed(minion_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(minion_seed)
            # Dedicated RNG for policy sampling so we don't rely on global state.
            self.rng = np.random.default_rng(minion_seed + 1)
        else:
            # Fallback RNG (non-deterministic across runs).
            self.rng = np.random.default_rng()

        # Set real-time priority (always defaults to 80 unless explicitly disabled)
        rt_priority = self.config.env_config.get("realtime_priority", 80)
        # Set priority if not explicitly disabled (None or False)
        if rt_priority is not None and rt_priority is not False:
            set_realtime_priority(priority=rt_priority, logger=self.logger)
        else:
            # Explicitly disabled, skip setting priority
            self.logger.debug("Real-time priority explicitly disabled in config")

        # initialize timing instrumentation (must be early, before any methods that use it)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"minion_timing_{timestamp_str}.csv"
        self.timing_recorder = TimingRecorder(csv_path=csv_path, logger=self.logger)

        # add attributes to object
        self.policy_shm_name = self.config.env_config['policy_shm_name']
        self.flag_shm_name = self.config.env_config['flag_shm_name']
        self.episode_shm_properties = self.config.env_config["ep_shm_properties"]
        self.actor_rollout_field_slices = get_rollout_field_slices(
            self.episode_shm_properties
        )
        self.policy_output_kind = self.episode_shm_properties.get(
            "policy_output_kind", "unknown"
        )
        self.exploration_noise = float(
            getattr(self.config, "exploration_noise", 0.0) or 0.0
        )
        self.initial_steps = int(
            getattr(self.config, "initial_steps", 0) or 0
        )
        self.initial_std = float(
            getattr(self.config, "initial_std", 0.5) or 0.5
        )
        self.noise_decay_k = float(
            getattr(self.config, "noise_decay_k", 0.01) or 0.01
        )
        self.noise_decay_schedule = str(
            getattr(self.config, "noise_decay_schedule", "linear") or "linear"
        ).lower()
        self.linear_decay_steps = int(
            getattr(self.config, "linear_decay_steps", 8000) or 8000
        )
        self.ep_shm_name = self.episode_shm_properties['name']
        self.enable_safety_filter = self.config.env_config.get("enable_safety_filter", True)
        self.filter_policy_shm_name = self.config.env_config.get('filter_policy_shm_name', 'filter_policy')
        self.filter_ep_shm_properties = self.config.env_config.get("filter_ep_shm_properties")
        if self.filter_ep_shm_properties is not None:
            self.filter_ep_shm_name = self.filter_ep_shm_properties['name']
        else:
            self.filter_ep_shm_name = None

        # connect to shared memory blocks
        self.f_shm = shared_memory.SharedMemory(name=self.flag_shm_name, create=False)  # this one has to be first
        self.f_buf = self.f_shm.buf
        self.ep_shm = shared_memory.SharedMemory(name=self.ep_shm_name, create=False)
        self.ep_buf = self.ep_shm.buf
        self.ep_arr = np.ndarray(shape=(self.episode_shm_properties["TOTAL_SIZE"],),
                                 dtype=np.float32,
                                 buffer=self.ep_buf,
                                 )
        while self.f_buf[0] == 1:  # wait until actor policy shared memory block has been created
            time.sleep(0.01)
        # connect to actor policy shared memory block and get buffer pointer
        self.p_shm = shared_memory.SharedMemory(name=policy_shm_name, create=False)
        self.p_buf = self.p_shm.buf

        self.logger.debug(f"Minion: ep_arr shape -> {self.ep_arr.shape}")
        self.logger.debug("Minion: connected to actor memory blocks")

        self.logger.debug("Minion: Getting initial actor network weights")
        self.ort_session = None
        # get initial actor network weights
        while self.f_buf[1] == 0:  # wait until actor weights-available flag is set to true
            time.sleep(0.01)
        self.ort_session, self.input_names, self.output_names = self._get_ort_session(model_type='actor')
        self.logger.debug(f"Minion: input_names: {self.input_names}, output_names: {self.output_names}")
        self.f_buf[1] = 0  # change actor new-weights-available flag to false

        self.logger.debug("Minion: Initialized actor ORT session")

        # initialize environment through adapter (continuous-only realtime path)
        env_adapter_id = self.config.env_config.get(
            "env_adapter_id", ENGINE_CONTINUOUS_ADAPTER_ID
        )
        env_adapter_kwargs = self.config.env_config.get("env_adapter_kwargs", {})
        self.env_adapter = get_env_adapter(env_adapter_id)
        self.env = self.env_adapter.build_env(
            reward_fn=reward_fn,
            env_kwargs=dict(env_adapter_kwargs),
        )
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise NotImplementedError(
                f"Unsupported observation space for realtime adapter path: "
                f"{self.env.observation_space}"
            )
        self.obs_is_discrete = False
        self.logger.debug("Minion: obs_is_discrete=False (continuous-only path)")

        # Seed the Gymnasium environment's internal RNG once so that all future
        # resets without an explicit seed follow a deterministic trajectory.
        if self._env_seed is not None:
            try:
                self.env.reset(seed=int(self._env_seed))
            except Exception as e:
                self.logger.debug(f"Minion: Failed to seed environment with env_seed={self._env_seed}: {e}")

        self.adapter_state = self.env_adapter.init_runtime_state(
            env=self.env,
            env_seed=(int(self._env_seed) if self._env_seed is not None else None),
        )
        self.history_features = self.adapter_state.history
        self.logger.debug("Minion: initialized adapter runtime state")

        # get random reference observation to check ort outputs and make sure weights change
        obs_shape = self.episode_shm_properties["STATE_ACTION_DIMS"]["state"]
        self.ref_obs = np.random.randn(32, obs_shape).astype(np.float32)
        self.old_policy_output = None

        # initialize action adapter, build module and extract action_dist_cls to sample actions properly
        spaces = {
            INPUT_ENV_SPACES: (self.config.observation_space, self.config.action_space),
            DEFAULT_MODULE_ID: (
                self.config.observation_space,
                self.config.action_space,
            ),
        }
        module_spec = self.config.get_rl_module_spec(
            spaces=spaces, inference_only=True
        )
        module = module_spec.build()
        self.action_dist_cls = module.get_inference_action_dist_cls()
        self.action_adapter = ActionAdapter(self.env.action_space, action_dist_cls=self.action_dist_cls)

        self.logger.debug("Minion: Initialized ENV.")

        self.safety_filter = None
        self.filter_ep_arr = None
        self.filter_ep_shm = None
        self.model_error = 0.0

        if self.enable_safety_filter:
            # Connect to filter policy shared memory block
            while self.f_buf[2] == 1:  # wait until filter policy shared memory block has been created (lock flag)
                time.sleep(0.01)
            self.filter_p_shm = shared_memory.SharedMemory(name=self.filter_policy_shm_name, create=False)
            self.filter_p_buf = self.filter_p_shm.buf

            self.logger.debug("Minion: Getting initial filter network weights")
            self.filter_ort_session = None
            # get initial filter network weights
            while self.f_buf[3] == 0:  # wait until filter weights-available flag is set to true
                time.sleep(0.01)
            self.filter_ort_session, self.filter_input_names, self.filter_output_names = self._get_ort_session(model_type='filter')
            self.logger.debug(f"Minion: filter_input_names: {self.filter_input_names}, filter_output_names: {self.filter_output_names}")
            self.f_buf[3] = 0  # change filter new-weights-available flag to false

            self.logger.debug("Minion: Initialized filter ORT session")

            try:
                # Initialize SafetyFilter with ORT session
                filter_dims = self.filter_ep_shm_properties.get("filter_dims", None)
                filter_state_dim = filter_dims.get("state", None)
                filter_action_dim = filter_dims.get("action", None)
                filter_sample_data_dir = self.config.env_config.get("filter_sample_data_dir")
                if filter_state_dim is None or filter_action_dim is None:
                    raise ValueError(f"Filter state or action dimension not set. Please check the observation and action spaces.")

                self.safety_filter = SafetyFilter(
                    state_dim=filter_state_dim,
                    action_dim=filter_action_dim,
                    ort_session=self.filter_ort_session,
                    input_names=self.filter_input_names,
                    output_names=self.filter_output_names,
                    sample_data_dir=filter_sample_data_dir,
                )
                self.model_error = 0.0  # Initial model error, will be updated from shared memory (float, not torch tensor)
            except Exception as e:
                self.logger.error(f"Minion: Could not initialize SafetyFilter: {e}")
                raise RuntimeError(f"Minion: Could not initialize SafetyFilter: {e}")

            self.logger.debug("Minion: Initialized SafetyFilter")

            # Connect to filter episode shared memory if available
            if self.filter_ep_shm_properties is not None:
                self.filter_ep_shm = shared_memory.SharedMemory(name=self.filter_ep_shm_name, create=False)
                self.filter_ep_buf = self.filter_ep_shm.buf
                self.filter_ep_arr = np.ndarray(shape=(self.filter_ep_shm_properties["TOTAL_SIZE"],),
                                                dtype=np.float32,
                                                buffer=self.filter_ep_buf,
                                                )
                self.logger.debug(f"Minion: filter_ep_arr shape -> {self.filter_ep_arr.shape}")
                self.logger.debug("Minion: connected to filter episode memory block")
        else:
            self.logger.debug("Minion: Safety filter disabled.")

        # set up data broadcasting to GUI (optional)
        self.pub = None
        self.zmq_ctx = None
        enable_zmq = self.config.env_config.get("enable_zmq", True)
        if enable_zmq and zmq_available and zmq is not None:
            try:
                self.zmq_ctx = zmq.Context()
                self.pub = self.zmq_ctx.socket(zmq.PUB)
                self.pub.bind("ipc:///tmp/engine.ipc")
                self.logger.info("Minion: ZMQ publisher initialized for GUI communication")
            except Exception as e:
                self.logger.warning(f"Minion: Failed to initialize ZMQ publisher: {e}. Continuing without ZMQ.")
                self.pub = None
                self.zmq_ctx = None
        elif enable_zmq and not zmq_available:
            self.logger.warning("Minion: ZMQ requested but not available (zmq not installed). Continuing without ZMQ.")
        else:
            self.logger.debug("Minion: ZMQ disabled via config")

        # start count
        self.batch_count = 0
        self.rollout_count = 0
        self.last_obs = None

        # Set EMA settings for reward scaling
        H = 50.0  # number of steps to reach 99% of the value, or half life of the exponential decay
        self.ema_beta = np.exp(-np.log(2) / H)
        self.current_var_ema = 1.0
        self.current_reward_scale = 1.0

        self.logger.debug("Minion: Done with __init__().")

    def _get_current_exploration_std(self) -> float:
        """Return the rollout-time actor-noise std after the random phase."""
        steps_since_random = max(self.rollout_count - self.initial_steps, 0)

        if self.noise_decay_schedule == "linear":
            progress = min(steps_since_random / max(self.linear_decay_steps, 1), 1.0)
            return self.initial_std + progress * (
                self.exploration_noise - self.initial_std
            )

        if self.noise_decay_schedule == "hyperbolic":
            return (
                self.exploration_noise
                + (self.initial_std - self.exploration_noise)
                / (1.0 + self.noise_decay_k * steps_since_random)
            )

        raise ValueError(
            f"Unsupported noise decay schedule: {self.noise_decay_schedule}"
        )

    def _ort_session_run(self, session, obs):
        tic = time.time()
        net_out = session.run(
            self.output_names,
            {self.input_names[0]: obs},
        )
        toc = time.time()
        duration_ms = (toc - tic) * 1000.0
        self.timing_recorder.record_timing('ort_session_run', duration_ms)
        return net_out

    def _weights_changed(self, new_sess, atol=1e-5):
        tic = time.time()

        out_new = []
        for obs in self.ref_obs:
            out_new.append(self._ort_session_run(new_sess, np.array([obs], np.float32))[0])
        out_new = np.array(out_new)

        if self.old_policy_output is None:
            self.old_policy_output = out_new
            toc = time.time()
            duration_ms = (toc - tic) * 1000.0
            self.timing_recorder.record_timing('weights_changed', duration_ms)
            return True

        diff = np.sum(np.abs(self.old_policy_output - out_new))

        msg = {
            "topic": "policy",
            "delta in minion": float(diff),
        }
        # self.logger.debug(f"Minion (train_and_eval_sequence): eval msg: {msg}.")
        if self.pub is not None:
            self.pub.send_json(msg)

        self.old_policy_output = out_new

        toc = time.time()
        duration_ms = (toc - tic) * 1000.0
        self.timing_recorder.record_timing('weights_changed', duration_ms)

        return diff > atol

    def _get_ort_session(self, model_type: str = 'actor'):
        """
        Get ORT session from shared memory for either actor or filter model.
        
        Args:
            model_type: 'actor' or 'filter' to specify which model to load
            
        Returns:
            Tuple of (ort_session, input_names, output_names)
        """
        if model_type == 'actor':
            p_buf = self.p_buf
            timing_name = 'get_ort_session'
        elif model_type == 'filter':
            p_buf = self.filter_p_buf
            timing_name = 'get_filter_ort_session'
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'actor' or 'filter'")
        
        tic = time.time()
        # get length of ort_compressed from header
        _len_ort = struct.unpack_from("<I", p_buf, 0)
        header_offset = 4  # number of bytes in the int

        ort_raw = p_buf[header_offset:header_offset + _len_ort[0]].tobytes()  # this is the actual ort model bytes

        ort_session = ort.InferenceSession(
            ort_raw,
            providers=[("CoreMLExecutionProvider",
                        {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}
                        ),
                       "CPUExecutionProvider", ]
        )
        output_names = [o.name for o in ort_session.get_outputs()]
        input_names = [i.name for i in ort_session.get_inputs()]

        # Read MSE loss for filter model (stored after the weights as float32)
        if model_type == 'filter':
            mse_offset = header_offset + _len_ort[0]
            mse_loss = struct.unpack_from("<f", p_buf, mse_offset)[0]
            self.model_error = float(mse_loss)  # Store as float, not torch tensor
            self.logger.debug(f"Minion: Loaded filter model with MSE={mse_loss:.6f}")

        toc = time.time()
        duration_ms = (toc - tic) * 1000.0
        self.timing_recorder.record_timing(timing_name, duration_ms)
        return ort_session, input_names, output_names

    def try_update_weights(self, model_type: str = 'actor') -> bool:
        """
        Update model weights (actor or filter) if available and not being written to.
        
        Args:
            model_type: 'actor' or 'filter' to specify which model to update
            
        Returns:
            True if weights were updated, False otherwise
        """
        if model_type == 'filter' and not getattr(self, 'enable_safety_filter', True):
            return False
        if model_type == 'actor':
            weights_flag_idx = 1
            lock_flag_idx = 0
            log_prefix = "ort session"
            timing_name = 'ort_weight_update'
            check_weights_changed = True
        elif model_type == 'filter':
            weights_flag_idx = 3
            lock_flag_idx = 2
            log_prefix = "filter ort session"
            timing_name = 'filter_ort_weight_update'
            check_weights_changed = False
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'actor' or 'filter'")
        
        # Check if weights are available and buffer is not locked
        if self.f_buf[weights_flag_idx] == 1 and self.f_buf[lock_flag_idx] == 0:
            self.f_buf[lock_flag_idx] = 1  # set lock flag to locked
            tic = time.time()
            
            # Get ORT session with new weights
            ort_session, input_names, output_names = self._get_ort_session(model_type=model_type)
            
            # Update instance attributes based on model type
            if model_type == 'actor':
                self.ort_session = ort_session
                self.input_names = input_names
                self.output_names = output_names
            else:  # filter
                self.filter_ort_session = ort_session
                self.filter_input_names = input_names
                self.filter_output_names = output_names
                # Update SafetyFilter with new session
                self.safety_filter.ort_session = ort_session
                self.safety_filter.input_names = input_names
                self.safety_filter.output_names = output_names
                # model_error is already updated in _get_ort_session()
            
            toc = time.time()

            # Check if policy changed (only for actor)
            if check_weights_changed:
                try:
                    policy_changed = self._weights_changed(ort_session)
                    self.logger.debug(f"Minion: Policy weights changed? {policy_changed}")
                except Exception as e:
                    self.logger.debug(f"Minion: Could not check policy weights due to error {e}")
                    raise RuntimeError from e

            self.f_buf[lock_flag_idx] = 0  # set lock flag to unlocked
            self.f_buf[weights_flag_idx] = 0  # reset weights-available flag to 0 (false, i.e. no new weights)
            
            # Time the full update process
            toc_full = time.time()
            duration_ms = (toc_full - tic) * 1000.0
            self.timing_recorder.record_timing(timing_name, duration_ms)
            # self.logger.debug(f"Minion: {log_prefix.capitalize()} weights updated.")
            
            # Save timing data after update completes
            self.timing_recorder.save_timing_data()
            return True
        else:
            return False

    def _try_update_ort_weights(self) -> bool:
        """Convenience wrapper for backward compatibility."""
        return self.try_update_weights(model_type='actor')

    def _try_update_filter_weights(self) -> bool:
        """Convenience wrapper for backward compatibility."""
        return self.try_update_weights(model_type='filter')

    def _write_fragment(
            self,
            data: np.ndarray,
            is_initial_state: Optional[bool] = False,
            buffer_type: str = 'actor',
    ) -> Optional[np.ndarray]:
        """
        Append ONE rollout into the ring buffer (actor or filter).
        Drops the oldest slot if ring is full.

        Structure of the ring is as follows:
        ┌────────────────────────────────────────────────────────────────┐
        │ Offset 0                                                       │
        │ ┌──────────────────────┐                                       │
        │ │ write_idx (uint32)   │   ← head pointer (next slot to write) │
        │ └──────────────────────┘                                       │
        │ ┌──────────────────────┐                                       │
        │ │ read_idx  (uint32)   │   ← tail pointer (next slot to read)  │
        │ └──────────────────────┘                                       │
        │ Offset 8                                                       │
        │ ┌─────────────────────────────────────────────────────────────┐│
        │ │ Slot 0 (SLOT_SIZE bytes):                                   ││
        │ │   ┌────────────────┐  ┌──────────────────────────────────┐  ││
        │ │   │ length (uint16)│  │ payload bytes (≤ PAYLOAD_SIZE)   │  ││
        │ │   └────────────────┘  └──────────────────────────────────┘  ││
        │ └─────────────────────────────────────────────────────────────┘│
        │ ┌─────────────────────────────────────────────────────────────┐│
        │ │ Slot 1 (SLOT_SIZE bytes): …                                 ││
        │ └─────────────────────────────────────────────────────────────┘│
        │                              …                                 │
        │ ┌─────────────────────────────────────────────────────────────┐│
        │ │ Slot N-1                                                    ││
        │ └─────────────────────────────────────────────────────────────┘│
        └────────────────────────────────────────────────────────────────┘

        The structure for each slot is:
        Slot k (size = HEADER_SLOT_SIZE + PAYLOAD_SIZE)
        ┌──────────────────────────────────────────────────────┐
        │ HEADER_SLOT_SIZE bytes:                              │
        │   • filled_count  (float32)      ← how many rollouts │
        │                        (# of rollouts,not byte count)│
        ├──────────────────────────────────────────────────────┤
        │ PAYLOAD_SIZE bytes:                                  │
        │   ┌ starting_state (state_dim floats)                │
        │   ├ rollout[0]:                                      │
        │   │   ┌ action    (state_dim floats)                 │
        │   │   ├ reward    (action_dim floats)                │
        │   │   └ state     (state_dim floats)                 │
        │   ├ rollout[1]:  (same layout)                       │
        ├ ...                                                  │
        │   └ rollout[batch_size − 1]:                         │
        │       └ same as rollout[0]                           │
        └──────────────────────────────────────────────────────┘
        
        Args:
            data: Array containing rollout data
            is_initial_state: Whether this is an initial state
            buffer_type: 'actor' or 'filter' to specify which buffer to write to
        """
        # Set up buffer-specific attributes
        if buffer_type == 'actor':
            ep_arr = self.ep_arr
            shm_properties = self.episode_shm_properties
            lock_index = 5
            timing_name = 'write_fragment'
            log_prefix = "write_fragment"
        elif buffer_type == 'filter':
            if not hasattr(self, 'filter_ep_arr') or self.filter_ep_arr is None:
                return None
            ep_arr = self.filter_ep_arr
            shm_properties = self.filter_ep_shm_properties
            lock_index = 6
            timing_name = 'write_filter_fragment'
            log_prefix = "write_filter_fragment"
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}. Must be 'actor' or 'filter'")
        
        tic = time.time()
        if is_initial_state:
            if buffer_type == 'filter':
                # Filter extracts state_dim from data
                state_dim = shm_properties["STATE_ACTION_DIMS"]["state"]
                assert data.shape == (state_dim,) and data.dtype == np.float32
            else:
                assert data.shape == (shm_properties["STATE_ACTION_DIMS"]['state'],) and data.dtype == np.float32
        else:
            assert data.shape == (shm_properties["ELEMENTS_PER_ROLLOUT"],) and data.dtype == np.float32

        # wait until the buffer is unlocked to read indices, then read and lock (locking happens in get_indices)
        while True:
            if self.f_buf[lock_index] == 0:
                write_idx, read_idx = get_indices(ep_arr, self.f_buf, logger=self.logger, lock_index=lock_index)
                break
            else:
                time.sleep(0.0001)

        slot_off = shm_properties["HEADER_SIZE"] + write_idx * shm_properties["SLOT_SIZE"]

        # Handle initial state
        if is_initial_state:
            if buffer_type == 'actor':
                initial_state_off = slot_off + shm_properties["HEADER_SLOT_SIZE"]
                ep_arr[initial_state_off: initial_state_off + len(data)] = data
                set_indices(ep_arr, write_idx, 'w', self.f_buf, lock_index=lock_index)
                self.logger.debug(f"Minion ({log_prefix}): Done writing initial state. Final writing index: {write_idx}.")
                return ep_arr
            else:
                return None

        # get how many rollouts have been filled in the current episode
        filled = int(ep_arr[slot_off])

        # Copy rollout into slot payload
        episode_off = slot_off + shm_properties["HEADER_SLOT_SIZE"] + filled * shm_properties["ELEMENTS_PER_ROLLOUT"]
        # Actor buffer stores an initial state at the beginning of each slot payload.
        # Filter buffer does NOT (its payload is exactly BATCH_SIZE * ELEMENTS_PER_ROLLOUT).
        if buffer_type == 'actor':
            episode_off += shm_properties["STATE_ACTION_DIMS"]["state"]
        ep_arr[episode_off: episode_off + shm_properties["ELEMENTS_PER_ROLLOUT"]] = data

        # Increment fill counter
        filled += 1
        ep_arr[slot_off] = filled

        # If this is the last rollout that can be added to this slot -> move write_idx to next slot
        if filled == shm_properties["BATCH_SIZE"]:
            next_w = (write_idx + 1) % shm_properties["NUM_SLOTS"]
            if next_w == read_idx:  # ring full → drop oldest
                buffer_name = "filter ring buffer" if buffer_type == 'filter' else "ring buffer"
                self.logger.debug(f"Minion: {buffer_name} got filled, writing data faster than algorithm can read.")
                read_idx = (read_idx + 1) % shm_properties["NUM_SLOTS"]

            # get offset for next slot
            write_idx = next_w
            slot_off = shm_properties["HEADER_SIZE"] + write_idx * shm_properties["SLOT_SIZE"]

            # Extract state for next slot's initial state
            if buffer_type == 'actor':
                next_obs_slice = self.actor_rollout_field_slices["next_obs"]
                state = data[next_obs_slice]


            if buffer_type == 'actor':
                # add initial state to next slot
                initial_state_off = slot_off + shm_properties["HEADER_SLOT_SIZE"]
                ep_arr[initial_state_off: initial_state_off + len(state)] = state

            # reset the fill counter
            ep_arr[slot_off] = 0

            # increment batch count (only for actor)
            if buffer_type == 'actor':
                self.batch_count += 1

        # Commit updated indices and unlock episode buffer (unlocking happens inside set_indices)
        set_indices(ep_arr, write_idx, 'w', self.f_buf, lock_index=lock_index)

        toc = time.time()
        duration_ms = (toc - tic) * 1000.0
        self.timing_recorder.record_timing(timing_name, duration_ms)
        return ep_arr

    def write_filter_fragment(
            self,
            data: np.ndarray,
            is_initial_state: Optional[bool] = False,
    ) -> Optional[np.ndarray]:
        """Convenience wrapper for backward compatibility."""
        return self._write_fragment(data, is_initial_state=is_initial_state, buffer_type='filter')
    
    def _get_obs_from_env_to_actor(self, obs: dict[str, float]) -> np.ndarray:
        """Format raw env observation into actor-model input."""
        obs_for_actor = self.env_adapter.obs_to_actor(
            obs=obs,
            runtime_state=self.adapter_state,
        )
        return np.expand_dims(obs_for_actor, axis=0)

    def _get_obs_from_env_to_filter_input(self, obs: dict[str, float]) -> np.ndarray:
        """Format raw env observation into filter-model input."""
        return self.env_adapter.obs_to_filter_input(
            obs=obs,
            runtime_state=self.adapter_state,
        )
    
    def _get_obs_from_env_to_filter_output(self, obs: dict[str, float]) -> np.ndarray:
        """Format env observation into filter-training output."""
        return self.env_adapter.obs_to_filter_output(obs=obs)
    
    def _get_action_from_actor_to_filter(self, action: np.ndarray) -> np.ndarray:
        """Format actor sampled action into filter action domain."""
        return self.env_adapter.action_actor_to_filter(
            action=action,
            action_adapter=self.action_adapter,
        )
    
    def _get_action_from_filter_to_env(self, obs: dict[str, float], action: np.ndarray) -> np.ndarray:
        """Format filter action output into environment-step action."""
        return self.env_adapter.action_filter_to_env(
            obs=obs,
            action=action,
            runtime_state=self.adapter_state,
        )

    def _update_history_features(self, action: np.ndarray, obs: dict[str, float]) -> None:
        """Update adapter-controlled history features."""
        self.env_adapter.update_history(
            action_in_env_range=action,
            obs=obs,
            runtime_state=self.adapter_state,
        )
    
    def _update_target(self, obs: dict[str, float], target: float) -> None:
        """Update target in env internals and current observation."""
        self.env_adapter.set_target(
            env=self.env,
            obs=obs,
            target=target,
            runtime_state=self.adapter_state,
        )

    def collect_rollout(
            self,
            initial_obs: Optional[dict[str, float]] = None,
            deterministic: Optional[bool] = False,
    ) -> Union[dict, list]:
        """
        function to collect a single rollout
        """
        tic_collect = time.time()

        if initial_obs is None:
            obs, info = self.env.reset()
            target = self.env_adapter.target_current(self.adapter_state)
            self._update_target(obs, target)
        else:
            obs = initial_obs

        # self.logger.debug("Minion: in collect_rollout")

        # format observation for actor and filter models
        try:
            obs_for_actor_model = self._get_obs_from_env_to_actor(obs)
        except Exception as e:
            self.logger.debug(f"Minion: Failed to get observation from environment to actor model: {e}")
            raise RuntimeError(f"Failed to get observation from environment to actor model: {e}")

        try:
            obs_for_filter_model = self._get_obs_from_env_to_filter_input(obs)
        except Exception as e:
            self.logger.debug(f"Minion: Failed to get observation from environment to filter model: {e}")
            raise RuntimeError(f"Failed to get observation from environment to filter model: {e}")

        # self.logger.debug(f"Minion: obs_for_actor_model: {obs_for_actor_model}")
        # self.logger.debug(f"Minion: obs_for_filter_model: {obs_for_filter_model}")

        tic2 = time.time()
        # perform inference to get action distribution
        try:
            # first [0] -> selects "output". second [0] -> selects 0th batch
            net_out = self._ort_session_run(self.ort_session, obs_for_actor_model)[0][0]
        except Exception as e:
            raise RuntimeError(f"Could not perform action inference due to error {e}")

        toc2 = time.time()
        # self.logger.debug(f"Minion time for inference only is {(toc2 - tic2)*1000:0.4f}ms")

        # self.logger.debug(f"Minion: performed action inference, net_out={net_out}, type={type(net_out)}")

        tic_policy = time.time()
        # sample action from policy distribution (use seeded RNG when provided)
        action_from_actor, logp, dist_inputs = self.action_adapter.sample_from_policy(
            net_out,
            deterministic=deterministic,
            rng=self.rng,
        )
        toc_policy = time.time()
        duration_policy_ms = (toc_policy - tic_policy) * 1000.0
        self.timing_recorder.record_timing('policy_sampling', duration_policy_ms, deterministic=deterministic)

        # self.logger.debug(
        #     f"Minion: sampled action: action_raw={action_from_actor}, logp={logp}, dist_inputs={dist_inputs}")

        # Apply exploration noise (for deterministic policies like TD3).
        # Phase 1: pure uniform-random actions for the first initial_steps rollouts.
        # Phase 2: policy + Gaussian noise with std controlled by the configured
        # decay schedule (linear by default, hyperbolic available for compatibility).
        #
        # Important distinction here between the requested sample being deterministic vs
        # the policy being deterministic. By nature of the algorithm, some policies
        # like TD3 have a deterministic output from the policy. If the requested sample
        # is deterministic, we do not add noise. But if the requested sample is not deterministic,
        # we add noise to the policy output.
        if (
            not deterministic
            and self.policy_output_kind == "deterministic"
        ):
            if self.rollout_count < self.initial_steps:
                action_from_actor = self.rng.uniform(
                    low=-1.0, high=1.0,
                    size=np.shape(action_from_actor),
                ).astype(np.float32)
            elif self.initial_std > 0.0:
                current_std = self._get_current_exploration_std()
                noise = self.rng.normal(
                    loc=0.0,
                    scale=current_std,
                    size=np.shape(action_from_actor),
                ).astype(np.float32)
                action_from_actor = np.clip(
                    action_from_actor + noise, -1.0, 1.0
                ).astype(np.float32)
                self.logger.debug(
                    f"Minion: Phase 2 exploration ({self.noise_decay_schedule}), "
                    f"std={current_std:.4f}, "
                    f"action_noisy={action_from_actor}"
                )
        
        try:
            # Keep a copy of the nominal action (pre-filter) for filter-training data.
            action_from_actor_for_filter_nominal = self._get_action_from_actor_to_filter(action_from_actor)
        except Exception as e:
            self.logger.debug(f"Minion: Failed to get action from actor to filter: {e}")
            raise RuntimeError(f"Failed to get action from actor to filter: {e}")
        
        self.logger.debug(f"Minion: action_from_actor_for_filter_nominal: {action_from_actor_for_filter_nominal}")

        # Apply safety filter to action (or use nominal when filter disabled)
        if hasattr(self, 'safety_filter') and self.safety_filter is not None:
            try:
                # SafetyFilter now uses numpy arrays directly (no torch conversion needed)
                action_filtered = self.safety_filter.compute_filtered_action(
                    obs_for_filter_model,       # numpy array: (state_dim,)
                    action_from_actor_for_filter_nominal,  # numpy array: (action_dim,)
                    self.model_error   # float scalar
                )
                self.logger.debug(f"Minion: Got filtered action: action_filtered={action_filtered}")
            except Exception as e:
                self.logger.debug(f"Minion: Safety filter failed: {e}, using original action")
                action_filtered = action_from_actor_for_filter_nominal
        else:
            action_filtered = action_from_actor_for_filter_nominal

        try:
            # Format action for environment
            action_for_env_filtered = self._get_action_from_filter_to_env(obs, action_filtered)
            nominal_action_for_env = self._get_action_from_filter_to_env(obs, action_from_actor_for_filter_nominal)
            # self.logger.debug(f"Minion: action_for_env_filtered: {action_for_env_filtered}")
            # self.logger.debug(f"Minion: nominal_action_for_env: {nominal_action_for_env}")
        except Exception as e:
            self.logger.debug(f"Minion: Failed to get action from filter to environment: {e}")
            raise RuntimeError(f"Failed to get action from filter to environment: {e}")

        try:
        # Time environment step
            tic_env = time.time()
            new_obs, reward, reward_vec, terminated, truncated, info = self.env.step(action_for_env_filtered, nominal_action_for_env)
            toc_env = time.time()
            duration_env_ms = (toc_env - tic_env) * 1000.0
            self.timing_recorder.record_timing('env_step', duration_env_ms)
        except Exception as e:
            self.logger.debug(f"Minion: Failed to step environment: {e}")
            raise RuntimeError(f"Failed to step environment: {e}")
        
        self.logger.debug(f"Minion: Observation: {new_obs}")
        
        # Collect filter training data (current_state, action, next_state, nominal_action) and write 
        # to filter buffer.
        # This is done here to make sure all sampled data is written to filter buffer, independent of whether
        # sampling is deterministic or not.
        if hasattr(self, 'filter_ep_arr') and self.filter_ep_arr is not None:
            try:
                current_state_filter = obs_for_filter_model.reshape(-1)
                next_state_filter = self._get_obs_from_env_to_filter_output(new_obs).reshape(-1)
                filter_dims = self.filter_ep_shm_properties["STATE_ACTION_DIMS"]
                expected_state_dim = int(filter_dims["state"])
                expected_action_dim = int(filter_dims["action"])
                expected_next_state_dim = int(filter_dims["next_state"])
                expected_nominal_action_dim = int(filter_dims["nominal_action"])
                if current_state_filter.shape[0] != expected_state_dim:
                    raise ValueError(
                        f"Filter current_state dim mismatch: got {current_state_filter.shape[0]}, "
                        f"expected {expected_state_dim}."
                    )
                if action_filtered.shape[0] != expected_action_dim:
                    raise ValueError(
                        f"Filter action_filtered dim mismatch: got {action_filtered.shape[0]}, "
                        f"expected {expected_action_dim}."
                    )
                if next_state_filter.shape[0] != expected_next_state_dim:
                    raise ValueError(
                        f"Filter next_state dim mismatch: got {next_state_filter.shape[0]}, "
                        f"expected {expected_next_state_dim}."
                    )
                if action_from_actor_for_filter_nominal.shape[0] != expected_nominal_action_dim:
                    raise ValueError(
                        f"Filter nominal_action dim mismatch: got {action_from_actor_for_filter_nominal.shape[0]}, "
                        f"expected {expected_nominal_action_dim}."
                    )
                # Store: (current_state, action_filtered, next_state, action_nominal)
                filter_data = np.concatenate([current_state_filter, action_filtered, next_state_filter, action_from_actor_for_filter_nominal]).astype(np.float32)
                self._write_fragment(filter_data, buffer_type='filter')
                self.logger.debug(f"Minion: Wrote the following data to filter buffer: {filter_data}")
            except Exception as e:
                self.logger.debug(f"Minion: Failed to write filter training data: {e}")

        # Increment rollout count
        self.rollout_count += 1

        # Finish timing the rollout and return the results
        toc_collect = time.time()
        duration_collect_ms = (toc_collect - tic_collect) * 1000.0
        self.timing_recorder.record_timing('collect_rollout', duration_collect_ms, deterministic=deterministic)
        # Save timing data after collect_rollouts completes
        self.timing_recorder.save_timing_data()
        return [new_obs, action_from_actor, reward, reward_vec, terminated, truncated, logp, net_out, dist_inputs, info]

    def _collect_and_process_rollout(
        self,
        obs,
        *,
        deterministic: bool = False,
        gui_topic: str = "engine",
        rewards_raw: list,
        rewards_adjusted: list,
        sigmas: list,
        reward_vecs: list,
        dist_inputs_list: list,
    ):
        obs, action, reward, reward_vec, _, _, logp, net_out, dist_inputs, info = (
            self.collect_rollout(initial_obs=obs, deterministic=deterministic)
        )

        adjusted_reward = self._scale_reward(reward)

        rewards_raw.append(reward)
        rewards_adjusted.append(adjusted_reward)
        sigmas.append(np.sqrt(self.current_reward_scale))
        reward_vecs.append(reward_vec)
        dist_inputs_list.append(np.asarray(net_out, dtype=np.float32).reshape(-1))

        msg = self.env_adapter.build_gui_message(
            topic=gui_topic,
            obs=obs,
            info=info,
            runtime_state=self.adapter_state,
        )
        if self.pub is not None:
            try:
                self.pub.send_json(msg)
            except Exception as e:
                self.logger.debug(f"Minion (_collect_and_process_rollout): {e}")

        self._update_history_features(self.action_adapter.get_action_in_env_range(action), obs)

        target = self.env_adapter.target_next(self.adapter_state)
        self._update_target(obs, target)

        return obs, action, reward, reward_vec, logp, net_out, dist_inputs, info

    def train_and_eval_sequence(
            self,
            train_batches: int = 1,
            eval_rollouts: int = 1,
    ):

        # Write initial state for actor buffer (filter does not need initial state)
        if self.last_obs is None:
            obs, info = self.env.reset()
            target = self.env_adapter.target_current(self.adapter_state)
            self._update_target(obs, target)
            self._write_fragment(self._get_obs_from_env_to_actor(obs).reshape(-1), is_initial_state=True, buffer_type='actor')
        else:
            obs = self.last_obs

        rewards_raw = []
        rewards_adjusted = []
        sigmas = []
        reward_vecs = []
        dist_inputs_list = []

        for i in range(int(train_batches * self.episode_shm_properties["BATCH_SIZE"])):
            obs, action, reward, reward_vec, logp, net_out, dist_inputs, info = (
                self._collect_and_process_rollout(
                    obs,
                    gui_topic="engine",
                    rewards_raw=rewards_raw,
                    rewards_adjusted=rewards_adjusted,
                    sigmas=sigmas,
                    reward_vecs=reward_vecs,
                    dist_inputs_list=dist_inputs_list,
                )
            )
            # Collect data to send to the learner according to the selected
            # algorithm's shared-memory rollout schema.
            obs_array = self._get_obs_from_env_to_actor(obs)
            rollout_fields = {
                "action": action,
                "reward": np.array([reward], dtype=np.float32),
                "next_obs": obs_array,
            }

            if self.episode_shm_properties["HAS_ACTION_LOGP"]:
                rollout_fields["action_logp"] = np.array(
                    [0.0 if logp is None else logp], dtype=np.float32
                )

            if self.episode_shm_properties["HAS_ACTION_DIST_INPUTS"]:
                rollout_fields["action_dist_inputs"] = net_out.astype(np.float32)

            current_packet = build_rollout_row(
                self.episode_shm_properties, rollout_fields
            )

            # Write data into the buffer
            try:
                self._write_fragment(current_packet, is_initial_state=False, buffer_type='actor')
            except Exception as e:
                self.logger.debug(f"Minion: Failed to write fragment to actor buffer: {e}")
                raise RuntimeError(f"Failed to write fragment to actor buffer: {e}")
            

        # Compute performance metrics
        try:
            train_rollouts_df = self._compute_performance_metrics(rewards_raw, rewards_adjusted, sigmas, np.vstack(reward_vecs), np.vstack(dist_inputs_list))
        except Exception as e:
            self.logger.debug(f"Minion: Failed to compute performance metrics for train rollouts: {e}")
            raise RuntimeError(f"Failed to compute performance metrics for train rollouts: {e}")

        rewards_raw_eval = []
        rewards_adjusted_eval = []
        sigmas_eval = []
        reward_vecs_eval = []
        dist_inputs_list_eval = []

        for i in range(eval_rollouts):
            obs, *_ = self._collect_and_process_rollout(
                    obs,
                    deterministic=True,
                    gui_topic="evaluation",
                    rewards_raw=rewards_raw_eval,
                    rewards_adjusted=rewards_adjusted_eval,
                    sigmas=sigmas_eval,
                    reward_vecs=reward_vecs_eval,
                    dist_inputs_list=dist_inputs_list_eval,
                )
        # Compute performance metrics for evaluation
        try:
            eval_rollouts_df = self._compute_performance_metrics(rewards_raw_eval, rewards_adjusted_eval, sigmas_eval, np.vstack(reward_vecs_eval), np.vstack(dist_inputs_list_eval))
        except Exception as e:
            self.logger.debug(f"Minion: Failed to compute performance metrics for eval rollouts: {e}")
            raise RuntimeError(f"Failed to compute performance metrics for eval rollouts: {e}")

        # Update reward scale
        self._update_reward_scale(rewards_raw)
        
        # Save timing data after train_and_eval_sequence completes
        self.timing_recorder.save_timing_data()

        # set last observation
        self.last_obs = copy.deepcopy(obs)

        return train_rollouts_df, eval_rollouts_df
    
    def _scale_reward(self, reward: float) -> float:
        """
        Helper function to scale reward. The current_reward_scale is the EMA of the reward variance.
        """
        return reward/self.current_reward_scale

    def _update_reward_scale(self, rewards: list) -> None:
        """
        Helper function to update reward scale. Small epsilon to avoid division by zero.
        """
        reward_variance = np.var(rewards)
        self.current_var_ema = np.max([self.ema_beta * self.current_var_ema + (1 - self.ema_beta) * reward_variance, 1e-6])
        self.current_reward_scale = np.sqrt(self.current_var_ema)
    
    def _compute_performance_metrics(
        self,
        rewards_raw: list,
        rewards_adjusted: list,
        sigmas: list,
        reward_vecs: np.ndarray,
        dist_inputs_vecs: np.ndarray,
    ) -> dict:
        """
        Helper function to compute performance metrics from rollouts.
        """
        metrics = {
            "return_raw": np.sum(rewards_raw),
            "return_adjusted": np.sum(rewards_adjusted),
            "reward_raw_mean": np.mean(rewards_raw),
            "reward_raw_std": np.std(rewards_raw),
            "reward_raw_min": np.min(rewards_raw),
            "reward_raw_max": np.max(rewards_raw),
            "reward_adjusted_mean": np.mean(rewards_adjusted),
            "reward_adjusted_std": np.std(rewards_adjusted),
            "reward_adjusted_min": np.min(rewards_adjusted),
            "reward_adjusted_max": np.max(rewards_adjusted),
            "sigma_mean": np.mean(sigmas),
            "sigma_std": np.std(sigmas),
            "sigma_min": np.min(sigmas),
            "sigma_max": np.max(sigmas),
            "load_tracking_mean": np.mean(reward_vecs[:, 0]),
            "load_tracking_std": np.std(reward_vecs[:, 0]),
            "load_tracking_min": np.min(reward_vecs[:, 0]),
            "load_tracking_max": np.max(reward_vecs[:, 0]),
            "safety_mean": np.mean(reward_vecs[:, 1]),
            "safety_std": np.std(reward_vecs[:, 1]),
            "safety_min": np.min(reward_vecs[:, 1]),
            "safety_max": np.max(reward_vecs[:, 1]),
            "filter_interference_mean": np.mean(reward_vecs[:, 2]),
            "filter_interference_std": np.std(reward_vecs[:, 2]),
            "filter_interference_min": np.min(reward_vecs[:, 2]),
            "filter_interference_max": np.max(reward_vecs[:, 2]),
        }
        for idx in range(dist_inputs_vecs.shape[1]):
            metrics[f"dist_inputs[{idx}]_mean"] = np.mean(dist_inputs_vecs[:, idx])
        return pd.DataFrame([metrics])


def main(policy_shm_name: str,
         flag_shm_name: str,
         ep_shm_name: str,
         config,
         ):
    """
    Function that runs minion to interact with the environment. Structure is:

    │ connect to shared memory blocks
    │ load initial policy network weights
    │ initialize environment (gym.Env or LabVIEW socket)
    │ Get initial network weights
    │ initialize episode collection buckets

    │ while True
    │ │ receive state (and maybe reward) from environment
    │ │ perform policy inference to sample actions
    │ │ send action to environment
    │ │ log state, action, reward into buckets
    │ │ if batch size or episode length reached
    │ │ │ write episode data to shared memory block
    │ │ │ clear buckets


    """

    actor = Minion(
        policy_shm_name,
        flag_shm_name,
        ep_shm_name,
        config
    )

    timesteps = 0
    weight_updates = 0
    # store_rollout = True

    # cumulative performance metrics over the full run
    total_train_rollouts_df = None
    total_eval_rollouts_df = None

    try:
        while True:
            # check for termination request from driver (f_buf[7] == 1)
            try:
                if actor.f_buf[7] == 1:
                    actor.logger.debug("Minion: Termination flag detected (f_buf[7]=1), exiting main loop.")
                    break
            except Exception as e:
                actor.logger.debug(f"Minion: Failed to read termination flag, exiting loop: {e}")
                break

            actor.logger.debug(f"Minion: Rollout count -> {actor.rollout_count}.")

            weights_updated = actor.try_update_weights(model_type='actor')
            if weights_updated:
                actor.logger.debug(f"Minion: Actor update number -> {weight_updates}.")
                weight_updates += 1
            else:
                actor.logger.debug("Minion: Actor weights not updated.")

            # Try to update filter weights
            filter_weights_updated = actor.try_update_weights(model_type='filter')
            if filter_weights_updated:
                actor.logger.debug("Minion: Filter weights updated.")
            else:
                actor.logger.debug("Minion: Filter weights not updated.")
            
            actor.logger.debug(f"Minion: Starting train and eval sequence.")

            # model_error is now read from filter_policy_shm along with weights in try_update_filter_weights()
            # No need to read from flag buffer anymore

            # perform train and eval routine
            try:
                train_rollouts_df, eval_rollouts_df = actor.train_and_eval_sequence(
                    train_batches=1,
                    eval_rollouts=2,
                )
            except Exception as e:
                actor.logger.debug(f"Minion: Failed to perform train and eval sequence: {e}")
                raise RuntimeError(f"Failed to perform train and eval sequence: {e}")

            actor.logger.debug(f"Minion: Train and eval sequence completed.")

            # set minion rollout flag to true to enable the algo.train() calls
            actor.f_buf[4] = 1  # minion data collection flag is now at index 4

            # Append performance metrics
            try:
                total_train_rollouts_df = pd.concat([total_train_rollouts_df, train_rollouts_df])
                total_eval_rollouts_df = pd.concat([total_eval_rollouts_df, eval_rollouts_df])
            except Exception as e:
                actor.logger.debug(f"Minion: Failed to append performance metrics: {e}")
                total_train_rollouts_df = train_rollouts_df
                total_eval_rollouts_df = eval_rollouts_df
            # logger.debug(f"Minion: Done with iteration {timesteps}")

            # if environment is the physical engine, wait for new state update and reward (simulated with a sleep)
            time.sleep(0.01)

            timesteps += 1

    except KeyboardInterrupt:
        actor.logger.debug("Minion: Program interrupted via KeyboardInterrupt.")
    finally:
        # Check to see if this process was terminated by the driver or if it crashed
        if actor.f_buf[7] != 1:
            # This means this process crashed, so terminate the driver process.
            actor.f_buf[7] = 1  # set termination flag to true

        actor.logger.debug("Minion: Cleaning up on exit.")
        # Persist aggregated rollout statistics, if any were collected
        try:
            if total_train_rollouts_df is not None:
                total_train_rollouts_df.to_csv("minion_train_rollouts.csv", index=False)
                actor.logger.debug("Minion: Saved total_train_rollouts_df to minion_train_rollouts.csv")
            if total_eval_rollouts_df is not None:
                total_eval_rollouts_df.to_csv("minion_eval_rollouts.csv", index=False)
                actor.logger.debug("Minion: Saved total_eval_rollouts_df to minion_eval_rollouts.csv")
        except Exception as e:
            actor.logger.debug(f"Minion: Failed to save rollout CSVs on exit: {e}")

        # Save any remaining timing data before exit
        try:
            actor.timing_recorder.save_timing_data()
        except Exception as e:
            actor.logger.debug(f"Minion: Failed to save timing data on exit: {e}")

        # Close shared memory buffers and release numpy views
        try:
            if hasattr(actor, "ep_arr"):
                del actor.ep_arr
            if hasattr(actor, "filter_ep_arr"):
                del actor.filter_ep_arr
        except Exception as e:
            actor.logger.debug(f"Minion: Failed to delete numpy views on exit: {e}")

        try:
            actor.ep_shm.close()
            actor.p_shm.close()
            actor.f_shm.close()
            actor.filter_p_shm.close()
            if hasattr(actor, "filter_ep_shm") and actor.filter_ep_shm is not None:
                actor.filter_ep_shm.close()
        except Exception as e:
            actor.logger.debug(f"Minion: Failed to close shared memory blocks on exit: {e}")

        # Close ZMQ resources if enabled
        try:
            if getattr(actor, "pub", None) is not None:
                actor.pub.close()
            if getattr(actor, "zmq_ctx", None) is not None:
                actor.zmq_ctx.term()
        except Exception as e:
            actor.logger.debug(f"Minion: Failed to close ZMQ resources on exit: {e}")

        actor.logger.debug("Minion: Clean exit completed.")
