from multiprocessing import shared_memory
from typing import Optional, Union

import numpy as np
import struct
import onnxruntime as ort
import time
import gzip
import os
import torch

from ray.rllib.utils.numpy import softmax
import gymnasium as gym
from core.environments.engine_env import reward_fn, EngineEnvDiscrete, EngineEnvContinuous

from utils.utils import ActionAdapter, TimingRecorder
from utils.shared_memory_utils import get_indices, set_indices, flatten_obs_onehot

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
    return np.expand_dims(obs, 0).astype(np.float32)


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
        # self.policy_shm_name = policy_shm_name
        # self.flag_shm_name = flag_shm_name
        # self.ep_shm_name = ep_shm_name
        self.policy_shm_name = self.config.env_config['policy_shm_name']
        self.flag_shm_name = self.config.env_config['flag_shm_name']
        self.episode_shm_properties = self.config.env_config["ep_shm_properties"]
        self.ep_shm_name = self.episode_shm_properties['name']
        self.filter_policy_shm_name = self.config.env_config.get('filter_policy_shm_name', 'filter_policy')
        self.filter_ep_shm_properties = self.config.env_config.get("filter_ep_shm_properties")
        if self.filter_ep_shm_properties is not None:
            self.filter_ep_shm_name = self.filter_ep_shm_properties['name']
        else:
            self.filter_ep_shm_name = None

        # set parameters for training and evaluation
        self.reset_env_each_n_batches = False
        self.n_batches_for_env_reset = 50

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

        # initialize environment
        env_type = self.config.env_config['env_type']
        if env_type == 'continuous':
            self.env = EngineEnvContinuous(reward=reward_fn)
        elif env_type == 'discrete':
            self.env = EngineEnvDiscrete(reward=reward_fn)
        else:
            raise NotImplementedError(f"Environment type not supported or not provided.")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.obs_is_discrete = True

            # extract action and observation spaces dimensions
            # self.sizes = [sp.n for sp in self.env.action_space]
            # self.cuts = np.cumsum(self.sizes)[:-1]
            self.len_imep = len(self.env.imep_space)
            self.len_mprr = len(self.env.mprr_space)
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_is_discrete = False
        else:
            raise NotImplementedError(f'Unknown observation_space {self.env.observation_space}')

        self.logger.debug(f"Minion: obs_is_discrete={self.obs_is_discrete}")

        # get random reference observation to check ort outputs and make sure weights change
        if self.obs_is_discrete:
            obs_shape = self.len_imep
        else:
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
            if filter_state_dim is None or filter_action_dim is None:
                raise ValueError(f"Filter state or action dimension not set. Please check the observation and action spaces.")
            
            self.safety_filter = SafetyFilter(
                state_dim=filter_state_dim,
                action_dim=filter_action_dim,
                ort_session=self.filter_ort_session,
                input_names=self.filter_input_names,
                output_names=self.filter_output_names
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

        # set up data broadcasting to GUI (optional)
        self.pub = None
        self.zmq_ctx = None
        enable_zmq = self.config.env_config.get("enable_zmq", True)
        if enable_zmq and zmq_available and zmq is not None:
            try:
                self.zmq_ctx = zmq.Context()
                self.pub = self.zmq_ctx.socket(zmq.PUB)
                self.pub.bind("ipc:///tmp/engine.ipc")
                self.logger.info("ZMQ publisher initialized for GUI communication")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ZMQ publisher: {e}. Continuing without ZMQ.")
                self.pub = None
                self.zmq_ctx = None
        elif enable_zmq and not zmq_available:
            self.logger.warning("ZMQ requested but not available (zmq not installed). Continuing without ZMQ.")
        else:
            self.logger.debug("ZMQ disabled via config")

        # start count
        self.batch_count = 0
        self.rollout_count = 0
        self.last_obs = None

        self.logger.debug("Minion: Done with __init__().")

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
        self.logger.debug("Minion: Called _weights_changed")
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
        self.logger.debug(f"Δpolicy={diff:.4e}")

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
        self.logger.debug("Minion: Finished _weights_changed")

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
            self.logger.debug("Minion: Called _get_ort_session (actor)")
            p_buf = self.p_buf
            timing_name = 'get_ort_session'
        elif model_type == 'filter':
            self.logger.debug("Minion: Called _get_ort_session (filter)")
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
            self.logger.debug(f"Minion: Updating {log_prefix} weights...")
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
            self.logger.debug(f"Minion: Time to update {log_prefix} weights is {(toc - tic)*1000:0.4f}ms")

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
            self.logger.debug(f"Minion: {log_prefix.capitalize()} weights updated.")
            
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
            self.logger.debug("Minion: Writing filter fragment")
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

        self.logger.debug(f"Minion: Writing fragment to slot {write_idx} of buffer {buffer_type}")

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

        self.logger.debug(f"Minion: Wrote data to slot {write_idx} of buffer {buffer_type}")

        # Increment fill counter
        filled += 1
        ep_arr[slot_off] = filled

        self.logger.debug(f"Minion: Incremented fill counter to {filled}")

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
                # Actor: either reset env or extract last state from current rollout
                if self.reset_env_each_n_batches and not self.batch_count % self.n_batches_for_env_reset:
                    # get next state from env.reset()
                    self.logger.debug(f"Minion: resetting env in batch {self.batch_count}")
                    obs, info = self.env.reset()
                    state = _flatten_obs_array(obs)
                else:
                    # extract last state from current rollout to use as initial observation for buffer slot
                    state_off = (shm_properties["STATE_ACTION_DIMS"]['action'] +
                                 shm_properties["STATE_ACTION_DIMS"]['reward'])
                    state = data[state_off:state_off + shm_properties["STATE_ACTION_DIMS"]['state']]

            self.logger.debug(f"Minion: Past reset and state extraction")

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
    
    def _get_obs_from_env_to_actor(self, obs: np.ndarray) -> np.ndarray:
        """
        Helper function to format observation for model inference.
        """
        if self.obs_is_discrete:
            # one-hot encode observation
            obs_for_actor = np.array([flatten_obs_onehot(obs, self.env.imep_space, self.env.mprr_space)],
                                        np.float32)
        else:
            obs_for_actor = np.array([obs], np.float32)

        if obs_for_actor.ndim == 1:
            obs_for_actor = np.expand_dims(obs_for_actor, axis=0)

        return obs_for_actor

    def _get_obs_from_env_to_filter(self, obs: np.ndarray) -> np.ndarray:
        """
        Helper function to format observation for filter model inference.
        """
        if self.obs_is_discrete:
            # already in values, so just return
            obs_for_filter = np.array([obs], np.float32)
        else:
            # already in values, so just return
            obs_for_filter = np.array([obs], np.float32)

        return obs_for_filter
    
    def _get_action_from_actor_to_filter(self, action: np.ndarray) -> np.ndarray:
        """
        Helper function to format action for filter.
        """
        
        if self.obs_is_discrete:
            # outputs in indices, so need to convert to values
            action_for_filter = self.env.action_ind_to_vals(action)
        else:
            # already in values, so just return
            action_for_filter = np.array([action], np.float32)

        return action_for_filter
    
    def _get_action_from_filter_to_env(self, action: np.ndarray) -> np.ndarray:
        """
        Helper function to format action for environment. Directly takes the output of the
        safety filter (numpy array of values) and converts it to the format expected by the 
        environment.
        """
        if self.action_adapter.mode == "continuous":
            action_for_env = np.concatenate(([550, ], self.action_adapter.get_action_in_env_range(action)),
                                            dtype=np.float32)
        else:
            action_for_env = self.env.action_vals_to_ind(np.concatenate(([550, ], action), dtype=np.float32))
        return action_for_env
    

    def collect_rollouts(
            self,
            n_rollouts: int = 1,
            initial_obs: Optional[np.ndarray] = None,
            deterministic: Optional[bool] = False,
    ) -> Union[dict, list]:
        """
        function to collect a set number of rollouts
        """
        tic_collect = time.time()

        if initial_obs is None:
            obs, info = self.env.reset()
        else:
            obs = initial_obs

        if n_rollouts > 1:
            rollouts = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "terminateds": [],
                "truncateds": [],
                "action_dist_inputs": [],
                "action_logps": [],
                "info": []
            }  # dict to store the rollouts

        for i in range(n_rollouts):

            self.logger.debug("Minion: in loop to collect rollouts")

            # format observation for actor and filter models
            try:
                obs_for_actor_model = self._get_obs_from_env_to_actor(obs)
            except Exception as e:
                self.logger.debug(f"Minion: Failed to get observation from environment to actor model: {e}")
                raise RuntimeError(f"Failed to get observation from environment to actor model: {e}")

            try:
                obs_for_filter_model = self._get_obs_from_env_to_filter(obs)
            except Exception as e:
                self.logger.debug(f"Minion: Failed to get observation from environment to filter model: {e}")
                raise RuntimeError(f"Failed to get observation from environment to filter model: {e}")

            self.logger.debug(f"Minion: obs_for_actor_model: {obs_for_actor_model}")
            self.logger.debug(f"Minion: obs_for_filter_model: {obs_for_filter_model}")

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
            # sample action from policy distribution
            action_from_actor, logp, dist_inputs = self.action_adapter.sample_from_policy(net_out,
                                                                                       deterministic=deterministic)
            toc_policy = time.time()
            duration_policy_ms = (toc_policy - tic_policy) * 1000.0
            self.timing_recorder.record_timing('policy_sampling', duration_policy_ms, deterministic=deterministic)

            self.logger.debug(
                f"Minion: sampled action: action_raw={action_from_actor}, logp={logp}, dist_inputs={dist_inputs}")
            
            try:
                # Keep a copy of the nominal action (pre-filter) for filter-training data.
                action_from_actor_nominal = self._get_action_from_actor_to_filter(action_from_actor)
            except Exception as e:
                self.logger.debug(f"Minion: Failed to get action from actor to filter: {e}")
                raise RuntimeError(f"Failed to get action from actor to filter: {e}")

            # Apply safety filter to action
            if hasattr(self, 'safety_filter') and self.safety_filter is not None:
                try:
                    # SafetyFilter now uses numpy arrays directly (no torch conversion needed)
                    action_filtered = self.safety_filter.compute_filtered_action(
                        obs_for_filter_model,       # numpy array: (state_dim,)
                        action_from_actor_nominal,  # numpy array: (action_dim,)
                        self.model_error   # float scalar
                    )
                    self.logger.debug(f"Minion: Got filtered action: action_filtered={action_filtered}")
                except Exception as e:
                    self.logger.debug(f"Minion: Safety filter failed: {e}, using original action")
            
            try:
            # Format action for environment
                action_for_env_filtered = self._get_action_from_filter_to_env(action_filtered)
                self.logger.debug(f"Minion: action_for_env_filtered: {action_for_env_filtered}")
            except Exception as e:
                self.logger.debug(f"Minion: Failed to get action from filter to environment: {e}")
                raise RuntimeError(f"Failed to get action from filter to environment: {e}")

            try:
            # Time environment step
                tic_env = time.time()
                new_obs, reward, terminated, truncated, info = self.env.step(action_for_env_filtered)
                toc_env = time.time()
                duration_env_ms = (toc_env - tic_env) * 1000.0
                self.timing_recorder.record_timing('env_step', duration_env_ms)
                self.logger.debug(f"Minion: Stepped environment.")
            except Exception as e:
                self.logger.debug(f"Minion: Failed to step environment: {e}")
                raise RuntimeError(f"Failed to step environment: {e}")
            
            # Collect filter training data (current_state, action, next_state, nominal_action) and write 
            # to filter buffer.
            # This is done here to make sure all sampled data is written to buffer, independent of whether
            # sampling is deterministic or not.
            if hasattr(self, 'filter_ep_arr') and self.filter_ep_arr is not None:
                try:
                    current_state = obs_for_filter_model        # Current state before action
                    next_state = self._get_obs_from_env_to_filter(new_obs)   # Next state after action
                    # Store: (current_state, action_filtered, next_state, action_nominal)
                    filter_data = np.concatenate([current_state, action_filtered, next_state, action_from_actor_nominal]).astype(np.float32)
                    self._write_fragment(filter_data, buffer_type='filter')
                    self.logger.debug(f"Minion: Wrote the following data to filter buffer: {filter_data}")
                except Exception as e:
                    self.logger.debug(f"Minion: Failed to write filter training data: {e}")

            # Increment rollout count
            self.rollout_count += 1

            if n_rollouts == 1:
                toc_collect = time.time()
                duration_collect_ms = (toc_collect - tic_collect) * 1000.0
                self.timing_recorder.record_timing('collect_rollouts', duration_collect_ms, deterministic=deterministic)
                # Save timing data after collect_rollouts completes
                self.timing_recorder.save_timing_data()
                return [new_obs, action_from_actor, reward, terminated, truncated, logp, net_out, info]
            else:
                rollouts["obs"].append(obs)
                rollouts["actions"].append(action_from_actor)
                rollouts["rewards"].append(reward)
                rollouts["terminateds"].append(terminated)
                rollouts["truncateds"].append(truncated)
                rollouts["action_logps"].append(logp)
                rollouts["action_dist_inputs"].append(net_out)
                rollouts["info"].append(info)


        toc_collect = time.time()
        duration_collect_ms = (toc_collect - tic_collect) * 1000.0
        self.timing_recorder.record_timing('collect_rollouts', duration_collect_ms, deterministic=deterministic)
        # Save timing data after collect_rollouts completes
        self.timing_recorder.save_timing_data()
        return rollouts

    def train_and_eval_sequence(
            self,
            train_batches: int = 1,
            eval_rollouts: int = 1,
    ):
        self.logger.debug("Minion: in train_and_eval_sequence")

        # Write initial state for actor buffer (filter does not need initial state)
        if not self.last_obs:
            obs, info = self.env.reset()
            self._write_fragment(_flatten_obs_array(obs), is_initial_state=True, buffer_type='actor')
        else:
            obs = self.last_obs
        
        self.logger.debug("Minion: going to collect train batches")

        for i in range(int(train_batches * self.episode_shm_properties["BATCH_SIZE"])):
            # self.logger.debug(f"Minion: in loop to collect batch, iter={i}")
            tic = time.time()
            obs, action, reward, terminated, truncated, logp, net_out, info = (
                self.collect_rollouts(initial_obs=obs))
            toc = time.time()
            # self.logger.debug(f"Minion: time to collect full rollout is {(toc - tic)*1000:0.4f}ms")


            # self.logger.debug(f"Minion (train_and_eval_sequence): received new rollout.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): obs: {obs}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): action: {action}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): reward: {reward}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): logp: {logp}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): info: {info}.")

            obs_flat = _flatten_obs_array(obs)  # flatten to shape needed by memory buffer
            current_packet = np.concatenate((
                action,
                np.array([reward], dtype=np.float32),
                obs_flat,
                np.array([logp], dtype=np.float32),
                net_out.astype(np.float32),
            )).astype(np.float32)

            # self.logger.debug(f"Minion (train_and_eval_sequence): built packet.")

            # write it into the buffer
            self._write_fragment(current_packet, is_initial_state=False, buffer_type='actor')

            # self.logger.debug(f"Minion (train_and_eval_sequence): wrote to buffer.")

            # send training results to be logged in the GUI
            msg = {
                "topic": "engine",
                "current imep": float(info["current imep"]),
                "mprr": float(info["mprr"]),
                "target": float(obs)
            }
            # self.logger.debug(f"Minion (train_and_eval_sequence): msg: {msg}.")
            if self.pub is not None:
                try:
                    self.pub.send_json(msg)
                except Exception as e:
                    self.logger.debug(f"Minion (train_and_eval_sequence): {e}")

            # self.logger.debug(f"Minion (train_and_eval_sequence): sent to GUI.")

        # set last observation
        self.last_obs = obs

        for i in range(eval_rollouts):
            obs, action, _, _, _, _, _, info = (
                self.collect_rollouts(initial_obs=obs, deterministic=True))

            # send evaluation results to be logged in the GUI
            msg = {
                "topic": "evaluation",
                "current imep": float(info["current imep"]),
                "mprr": float(info["mprr"]),
                "target": float(obs)
            }
            # self.logger.debug(f"Minion (train_and_eval_sequence): eval msg: {msg}.")
            if self.pub is not None:
                self.pub.send_json(msg)
        
        # Save timing data after train_and_eval_sequence completes
        self.timing_recorder.save_timing_data()


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

    try:
        while True:

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

            # model_error is now read from filter_policy_shm along with weights in try_update_filter_weights()
            # No need to read from flag buffer anymore

            # perform train and eval routine
            actor.train_and_eval_sequence(
                train_batches=1,
                eval_rollouts=1,
            )

            # set minion rollout flag to true to enable the algo.train() calls
            actor.f_buf[4] = 1  # minion data collection flag is now at index 4

            # logger.debug(f"Minion: Done with iteration {timesteps}")

            # if environment is the physical engine, wait for new state update and reward (simulated with a sleep)
            time.sleep(0.01)

            timesteps += 1

    except KeyboardInterrupt:
        # Save any remaining timing data before exit
        actor.timing_recorder.save_timing_data()
        # close socket connection
        del actor.ep_arr
        actor.logger.debug("Program interrupted")
