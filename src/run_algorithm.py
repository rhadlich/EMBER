import argparse
import json
import logging
import os
import shutil
import pprint
import random
import re
import time
import gzip
import base64
import struct
import signal
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from multiprocessing import shared_memory

import ray
import torch
import torch.nn as nn
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, WANDB_ENV_VAR
from ray.rllib.core import DEFAULT_MODULE_ID, Columns
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.annotations import OldAPIStack
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter
from ray.tune.result import TRAINING_ITERATION
from ray.rllib.core.rl_module.rl_module import RLModule

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.rllib.offline.dataset_reader import DatasetReader

from configs.args import custom_args
from onnxruntime.tools import convert_onnx_models_to_ort as c2o
from pathlib import Path
from core.safety.safety_filter import StatePredictor, FilterStorageBuffer, SafetyFilter
from utils.shared_memory_utils import get_indices, set_indices

import logging
import utils.logging_setup as logging_setup

# Try to import zmq, but make it optional
try:
    import zmq
    zmq_available = True
except ImportError:
    zmq_available = False
    zmq = None


def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def _get_current_onnx_model(module: RLModule,

                            *,
                            outdir: str = "model.onnx",
                            logger,
                            ):
    """
    Function to extract the policy model converted to ort.
    """
    # if os.path.exists(outdir):
    #     shutil.rmtree(outdir)
    assert module.get_ctor_args_and_kwargs()[1]["inference_only"]

    # act_dim = module.action_space.shape[0]
    # obs_dim = module.observation_space.shape[0]
    # dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(module,
                      {"batch": {"obs": torch.randn(1, *module.observation_space.shape)}},
                      outdir,
                      export_params=True)
    # convert .onnx to .ort (optimized for faster loading and inference in the minion)
    styles = [c2o.OptimizationStyle.Fixed]
    c2o.convert_onnx_models_to_ort(
        model_path_or_dir=Path(outdir),  # may also be a directory
        output_dir=None,  # None = same folder as the .onnx
        optimization_styles=styles,
        # target_platform="arm",  # Only in the Raspberry Pi
    )
    with open("model.ort", "rb") as f:
        ort_raw = f.read()

    return ort_raw


def _get_filter_onnx_model(state_predictor_model,
                            *,
                            outdir: str = "filter_model.onnx",
                            logger,
                            ):
    """
    Function to extract the StatePredictor model converted to ort.
    
    Args:
        state_predictor_model: PyTorch StatePredictor model
        outdir: Output directory for ONNX file
        logger: Logger instance
        
    Returns:
        ort_raw: Raw bytes of the ORT model
    """
    state_predictor_model.eval()
    
    # Get model dimensions
    state_dim = state_predictor_model.state_dim
    action_dim = state_predictor_model.action_dim
    
    # Create dummy inputs for export
    dummy_x = torch.randn(1, state_dim, dtype=torch.float32)
    dummy_u = torch.randn(1, action_dim, dtype=torch.float32)
    
    # Wrap forward to match ONNX export requirements
    # We need to export with two inputs (x, u) and return (x_next, f_x, G_x_flat)
    class WrappedPredictor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x, u):
            x_next, f_x, G_x = self.model(x, u)
            # Flatten G_x for ONNX export (reshape happens in inference)
            G_x_flat = G_x.reshape(x.shape[0], -1)
            return x_next, f_x, G_x_flat
    
    wrapped_model = WrappedPredictor(state_predictor_model)
    
    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        (dummy_x, dummy_u),
        outdir,
        export_params=True,
        input_names=['state', 'action'],
        output_names=['x_next', 'f_x', 'G_x_flat'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action': {0: 'batch_size'},
            'x_next': {0: 'batch_size'},
            'f_x': {0: 'batch_size'},
            'G_x_flat': {0: 'batch_size'},
        }
    )
    
    # Convert .onnx to .ort (optimized for faster loading and inference in the minion)
    styles = [c2o.OptimizationStyle.Fixed]
    c2o.convert_onnx_models_to_ort(
        model_path_or_dir=Path(outdir),  # may also be a directory
        output_dir=None,  # None = same folder as the .onnx
        optimization_styles=styles,
        # target_platform="arm",  # Only in the Raspberry Pi
    )
    
    ort_path = outdir.replace('.onnx', '.ort')
    with open(ort_path, "rb") as f:
        ort_raw = f.read()
    
    logger.debug(f"Filter model converted to ORT: {len(ort_raw)} bytes")
    return ort_raw


def run_rllib_shared_memory(
        base_config: "AlgorithmConfig",
        args: Optional[argparse.Namespace] = None,
        *,
        stop: Optional[Dict] = None,
        success_metric: Optional[Dict] = None,
        trainable: Optional[Type] = None,
        tune_callbacks: Optional[List] = None,
        keep_config: bool = False,
        keep_ray_up: bool = False,
        scheduler=None,
        progress_reporter=None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    """Given an algorithm config and some command line args, runs an experiment.

    Use the custom_args function from the define_args.py script to generate "args".

    The function sets up an Algorithm object from the given config (altered by the
    contents of `args`), then runs the Algorithm via Tune (or manually, if
    `args.no_tune` is set to True) using the stopping criteria in `stop`.

    At the end of the experiment, if `args.as_test` is True, checks, whether the
    Algorithm reached the `success_metric` (if None, use `env_runners/
    episode_return_mean` with a minimum value of `args.stop_reward`).

    See https://github.com/ray-project/ray/tree/master/rllib/examples for an overview
    of all supported command line options.

    Args:
        base_config: The AlgorithmConfig object to use for this experiment. This base
            config will be automatically "extended" based on some of the provided
            `args`. For example, `args.num_env_runners` is used to set
            `config.num_env_runners`, etc...
        args: A argparse.Namespace object, ideally returned by calling
            `args = add_rllib_example_script_args()`. It must have the following
            properties defined: `stop_iters`, `stop_reward`, `stop_timesteps`,
            `no_tune`, `verbose`, `checkpoint_freq`, `as_test`. Optionally, for WandB
            logging: `wandb_key`, `wandb_project`, `wandb_run_name`.
        stop: An optional dict mapping ResultDict key strings (using "/" in case of
            nesting, e.g. "env_runners/episode_return_mean" for referring to
            `result_dict['env_runners']['episode_return_mean']` to minimum
            values, reaching of which will stop the experiment). Default is:
            {
            "env_runners/episode_return_mean": args.stop_reward,
            "training_iteration": args.stop_iters,
            "num_env_steps_sampled_lifetime": args.stop_timesteps,
            }
        success_metric: Only relevant if `args.as_test` is True.
            A dict mapping a single(!) ResultDict key string (using "/" in
            case of nesting, e.g. "env_runners/episode_return_mean" for referring
            to `result_dict['env_runners']['episode_return_mean']` to a single(!)
            minimum value to be reached in order for the experiment to count as
            successful. If `args.as_test` is True AND this `success_metric` is not
            reached with the bounds defined by `stop`, will raise an Exception.
        trainable: The Trainable sub-class to run in the tune.Tuner. If None (default),
            use the registered RLlib Algorithm class specified by args.algo.
        tune_callbacks: A list of Tune callbacks to configure with the tune.Tuner.
            In case `args.wandb_key` is provided, appends a WandB logger to this
            list.
        keep_config: Set this to True, if you don't want this utility to change the
            given `base_config` in any way and leave it as-is. This is helpful
            for those example scripts which demonstrate how to set config settings
            that are otherwise taken care of automatically in this function (e.g.
            `num_env_runners`).

    Returns:
        The last ResultDict from a --no-tune run OR the tune.Tuner.fit()
        results.
    """
    if args is None:
        parser = custom_args()
        args = parser.parse_args()

    # If run --as-release-test, --as-test must also be set.
    if args.as_release_test:
        args.as_test = True

    logger = logging.getLogger("MyRLApp.custom_runner")
    logger.info(f"custom_runner, PID={os.getpid()}")

    # Pin to CPU core if specified
    if args.cpu_core_learner is not None:
        from ray_primitives import pin_to_core
        pin_to_core(args.cpu_core_learner)
        logger.info(f"Pinned learner process to CPU core {args.cpu_core_learner}")

    # pass main driver PID down to EnvRunner

    logger.debug("custom_run: Started custom run function")

    # Initialize Ray.
    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}},
    )

    logger.debug("custom_run: Concluded ray.init()")

    # Define one or more stopping criteria.
    if stop is None:
        stop = {
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
                args.stop_timesteps
            ),
            TRAINING_ITERATION: args.stop_iters,
        }

    config = base_config

    # Enhance the `base_config`, based on provided `args`.
    if not keep_config:
        # Set the framework.
        config.framework(args.framework)

        # Add an env specifier (only if not already set in config)?
        if args.env is not None and config.env is None:
            config.environment(args.env)

        # Disable the new API stack?
        if not args.enable_new_api_stack:
            config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )

        # Define EnvRunner scaling and behavior.
        if args.num_env_runners is not None:
            config.env_runners(num_env_runners=args.num_env_runners)
        if args.num_envs_per_env_runner is not None:
            config.env_runners(num_envs_per_env_runner=args.num_envs_per_env_runner)
        if args.create_local_env_runner is not None:
            config.env_runners(create_local_env_runner=args.create_local_env_runner)

        # Define compute resources used automatically (only using the --num-learners
        # and --num-gpus-per-learner args).
        # New stack.
        if config.enable_rl_module_and_learner:
            if args.num_gpus is not None and args.num_gpus > 0:
                raise ValueError(
                    "--num-gpus is not supported on the new API stack! To train on "
                    "GPUs, use the command line options `--num-gpus-per-learner=1` and "
                    "`--num-learners=[your number of available GPUs]`, instead."
                )

            # Do we have GPUs available in the cluster?
            num_gpus_available = ray.cluster_resources().get("GPU", 0)
            # Number of actual Learner instances (including the local Learner if
            # `num_learners=0`).
            num_actual_learners = (
                                      args.num_learners
                                      if args.num_learners is not None
                                      else config.num_learners
                                  ) or 1  # 1: There is always a local Learner, if num_learners=0.
            # How many were hard-requested by the user
            # (through explicit `--num-gpus-per-learner >= 1`).
            num_gpus_requested = (args.num_gpus_per_learner or 0) * num_actual_learners
            # Number of GPUs needed, if `num_gpus_per_learner=None` (auto).
            num_gpus_needed_if_available = (
                                               args.num_gpus_per_learner
                                               if args.num_gpus_per_learner is not None
                                               else 1
                                           ) * num_actual_learners
            # Define compute resources used.
            config.resources(num_gpus=0)  # old API stack setting
            if args.num_learners is not None:
                config.learners(num_learners=args.num_learners)

            # User wants to use aggregator actors per Learner.
            if args.num_aggregator_actors_per_learner is not None:
                config.learners(
                    num_aggregator_actors_per_learner=(
                        args.num_aggregator_actors_per_learner
                    )
                )

            # User wants to use GPUs if available, but doesn't hard-require them.
            if args.num_gpus_per_learner is None:
                if num_gpus_available >= num_gpus_needed_if_available:
                    config.learners(num_gpus_per_learner=1)
                else:
                    config.learners(num_gpus_per_learner=0)
            # User hard-requires n GPUs, but they are not available -> Error.
            elif num_gpus_available < num_gpus_requested:
                raise ValueError(
                    "You are running your script with --num-learners="
                    f"{args.num_learners} and --num-gpus-per-learner="
                    f"{args.num_gpus_per_learner}, but your cluster only has "
                    f"{num_gpus_available} GPUs!"
                )

            # All required GPUs are available -> Use them.
            else:
                config.learners(num_gpus_per_learner=args.num_gpus_per_learner)

            # Set CPUs per Learner.
            if args.num_cpus_per_learner is not None:
                config.learners(num_cpus_per_learner=args.num_cpus_per_learner)

        # Old stack (override only if arg was provided by user).
        elif args.num_gpus is not None:
            config.resources(num_gpus=args.num_gpus)

        # Evaluation setup.
        if args.evaluation_interval > 0:
            config.evaluation(
                evaluation_num_env_runners=args.evaluation_num_env_runners,
                evaluation_interval=args.evaluation_interval,
                evaluation_duration=args.evaluation_duration,
                evaluation_duration_unit=args.evaluation_duration_unit,
                evaluation_parallel_to_training=args.evaluation_parallel_to_training,
            )

        # Set the log-level (if applicable).
        if args.log_level is not None:
            config.debugging(log_level=args.log_level)

        # Set the output dir (if applicable).
        if args.output is not None:
            config.offline_data(output=args.output)

    logger.debug("custom_run: Done with setting config. Going into args.no_tune")

    signal.signal(signal.SIGTERM, on_sigterm)

    # Run the experiment w/o Tune (directly operate on the RLlib Algorithm object).
    # THIS IS WHAT WILL BE RUN ON THE RASPBERRY PI
    if args.no_tune:
        assert not args.as_test and not args.as_release_test

        # create flag shared memory block here
        # flag buffer has 8 bytes:
        #   0 -> actor weights lock flag (locked_state=1)
        #   1 -> actor weights available flag (true_state=1)
        #   2 -> filter weights lock flag (locked_state=1)
        #   3 -> filter weights available flag (true_state=1)
        #   4 -> minion data collection flag (has it started collecting data?, true_state=1)
        #   5 -> actor episode buffer lock flag (needed because of race conditions with reading and writing, locked_state=1)
        #   6 -> filter episode buffer lock flag (locked_state=1)
        #   7 -> reserved (can be used for model_error or other purposes)
        f_shm = shared_memory.SharedMemory(
            create=True,
            name=args.flag_shm_name,
            size=8,
        )
        f_buf = f_shm.buf

        f_buf[0] = 1  # set actor weights lock flag to locked
        f_buf[1] = 0  # set actor weights-available flag to false
        f_buf[2] = 1  # set filter weights lock flag to locked
        f_buf[3] = 0  # set filter weights-available flag to false
        f_buf[4] = 0  # set minion flag to false (minion has not started collecting rollouts)
        f_buf[5] = 0  # set actor episode lock flag to unlocked
        f_buf[6] = 0  # set filter episode lock flag to unlocked
        f_buf[7] = 0  # reserved

        logger.debug("custom_run: created flag memory buffer")

        # build algorithm, EnvRunner is created in this call
        algo = config.build()

        logger.debug("custom_run: done with config.build()")

        # extract dimensions of weights in the networks
        ort_raw = _get_current_onnx_model(algo.get_module(), logger=logger)
        policy_nbytes = len(ort_raw)

        logger.debug(f"custom_run: ort_raw length is {policy_nbytes}")

        # create policy shared memory blocks
        # need to include one more float32 as the buffer header to contain length of ort_compressed.
        # python creates the length of the buffer to be the smallest number of pages that can hold the requested number
        # of bytes, but not the size requested (on Mac at least)
        header_offset = 4
        p_shm = shared_memory.SharedMemory(
            create=True,
            name=args.policy_shm_name,
            size=policy_nbytes + header_offset
        )

        logger.debug("custom_run: created policy memory buffer")

        # get reference to policy buffer
        p_buf = p_shm.buf

        logger.debug(f"custom_run: buffer length is {len(p_buf)}")

        # store initial weights and remove lock flags
        struct.pack_into("<I", p_buf, 0, policy_nbytes)
        p_buf[header_offset:header_offset + len(ort_raw)] = ort_raw  # insert raw weights
        f_buf[0] = 0  # set lock flag to unlocked
        f_buf[1] = 1  # set weights-available flag to 1 (true)

        logger.debug("custom_run: stored initial model weights")

        # Initialize filter model (StatePredictor)
        filter_state_dim = config.observation_space.shape[0]
        filter_action_dim = config.action_space.shape[0]
        filter_num_hidden = config.env_config.get("filter_num_hidden", 2)
        filter_hidden_exp = config.env_config.get("filter_hidden_exp", 7)
        filter_dropout = config.env_config.get("filter_dropout", 0.0)
        
        logger.debug(f"custom_run: Initializing filter model with state_dim={filter_state_dim}, "
                    f"action_dim={filter_action_dim}, num_hidden={filter_num_hidden}, "
                    f"hidden_exp={filter_hidden_exp}, dropout={filter_dropout}")
        
        # Initialize filter storage buffer configuration
        filter_storage_max_samples = config.env_config.get("filter_storage_max_samples", 50000)
        filter_storage_min_samples = config.env_config.get("filter_storage_min_samples", 1000)
        filter_storage_training_batch_size = config.env_config.get("filter_storage_training_batch_size", None)
        filter_storage_critical_fraction = config.env_config.get("filter_storage_critical_fraction", 0.20)
        filter_storage_critical_capacity_fraction = config.env_config.get("filter_storage_critical_capacity_fraction", 0.20)
        filter_storage_h_critical_threshold = config.env_config.get("filter_storage_h_critical_threshold", 2.0)
        filter_storage_intervention_l2_threshold = config.env_config.get("filter_storage_intervention_l2_threshold", 0.10)
        
        logger.debug(f"custom_run: Filter storage buffer config: max_samples={filter_storage_max_samples}, "
                    f"min_samples={filter_storage_min_samples}")
        logger.debug(
            "custom_run: Filter storage critical config: "
            f"critical_fraction={filter_storage_critical_fraction}, "
            f"critical_capacity_fraction={filter_storage_critical_capacity_fraction}, "
            f"h_critical_threshold={filter_storage_h_critical_threshold}, "
            f"intervention_l2_threshold={filter_storage_intervention_l2_threshold}"
        )
        
        # Create PyTorch StatePredictor model
        filter_model = StatePredictor(
            state_dim=filter_state_dim,
            action_dim=filter_action_dim,
            num_hidden=filter_num_hidden,
            hidden_exp=filter_hidden_exp,
            dropout=filter_dropout
        )
        filter_model.eval()
        
        # Convert filter model to ONNX/ORT
        filter_ort_raw = _get_filter_onnx_model(filter_model, logger=logger, outdir="filter_model.onnx")
        filter_policy_nbytes = len(filter_ort_raw)
        
        logger.debug(f"custom_run: filter_ort_raw length is {filter_policy_nbytes}")
        
        # Create filter policy shared memory block
        # Size: header (4 bytes) + ORT model + MSE loss (4 bytes float32)
        filter_policy_shm_name = config.env_config.get("filter_policy_shm_name", "filter_policy")
        filter_p_shm = shared_memory.SharedMemory(
            create=True,
            name=filter_policy_shm_name,
            size=filter_policy_nbytes + header_offset + 4  # +4 for MSE float32
        )
        
        logger.debug("custom_run: created filter policy memory buffer")
        
        # Get reference to filter policy buffer
        filter_p_buf = filter_p_shm.buf
        
        logger.debug(f"custom_run: filter buffer length is {len(filter_p_buf)}")
        
        # Store initial filter weights and remove lock flags
        struct.pack_into("<I", filter_p_buf, 0, filter_policy_nbytes)
        filter_p_buf[header_offset:header_offset + len(filter_ort_raw)] = filter_ort_raw  # insert raw weights
        # Store initial MSE loss (0.0 for initial untrained model)
        mse_offset = header_offset + filter_policy_nbytes
        struct.pack_into("<f", filter_p_buf, mse_offset, 0.0)
        f_buf[2] = 0  # set filter lock flag to unlocked
        f_buf[3] = 1  # set filter weights-available flag to 1 (true)
        
        logger.debug("custom_run: stored initial filter model weights with MSE=0.0")

        # Create filter episode shared memory block for training data
        filter_ep_shm_properties = config.env_config.get("filter_ep_shm_properties")
        filter_ep_shm = None
        filter_ep_arr = None
        if filter_ep_shm_properties is not None:
            filter_ep_shm_name = filter_ep_shm_properties.get("name", "filter_episodes")
            try:
                # Try to connect first (might already exist from env runner)
                filter_ep_shm = shared_memory.SharedMemory(name=filter_ep_shm_name, create=False)
                filter_ep_buf = filter_ep_shm.buf
                filter_ep_arr = np.ndarray(shape=(filter_ep_shm_properties["TOTAL_SIZE"],),
                                           dtype=np.float32,
                                           buffer=filter_ep_buf)
                logger.debug(f"custom_run: Connected to existing filter episode shared memory: {filter_ep_arr.shape}")
            except FileNotFoundError:
                # Create if it doesn't exist
                filter_ep_shm = shared_memory.SharedMemory(
                    create=True,
                    name=filter_ep_shm_name,
                    size=filter_ep_shm_properties["TOTAL_SIZE_BYTES"]
                )
                filter_ep_buf = filter_ep_shm.buf
                filter_ep_arr = np.ndarray(shape=(filter_ep_shm_properties["TOTAL_SIZE"],),
                                           dtype=np.float32,
                                           buffer=filter_ep_buf)
                # Initialize write and read indices to 0
                filter_ep_arr[:2] = 0
                logger.debug(f"custom_run: Created filter episode shared memory: {filter_ep_arr.shape}")
        else:
            logger.warning("custom_run: filter_ep_shm_properties not found in config, filter training will be disabled")

        # Initialize filter storage buffer
        # Calculate sample dimension: state + action_filtered + next_state + action_nominal
        filter_storage_sample_dim = None
        barrier_only_safety_filter = None
        if filter_ep_shm_properties is not None:
            state_dim = filter_ep_shm_properties["STATE_ACTION_DIMS"]["state"]
            action_dim = filter_ep_shm_properties["STATE_ACTION_DIMS"]["action"]
            filter_storage_sample_dim = sum(filter_ep_shm_properties["filter_dims"].values())
            logger.debug(f"custom_run: Filter storage sample_dim={filter_storage_sample_dim} (state={state_dim}, action={action_dim})")

            # ORT-free SafetyFilter instance used only for compute_h() during Critical classification.
            barrier_only_safety_filter = SafetyFilter(
                state_dim=state_dim,
                action_dim=action_dim,
                ort_session=None,
                input_names=None,
                output_names=None,
            )
        
        filter_storage_buffer = FilterStorageBuffer(
            max_samples=filter_storage_max_samples,
            min_samples_for_training=filter_storage_min_samples,
            sample_dim=filter_storage_sample_dim,
            state_dim=state_dim if filter_ep_shm_properties is not None else None,
            action_dim=action_dim if filter_ep_shm_properties is not None else None,
            critical_fraction=filter_storage_critical_fraction,
            critical_capacity_fraction=filter_storage_critical_capacity_fraction,
            h_critical_threshold=filter_storage_h_critical_threshold,
            intervention_l2_threshold=filter_storage_intervention_l2_threshold,
            compute_h_fn=barrier_only_safety_filter.compute_h if barrier_only_safety_filter is not None else None,
        )
        logger.debug("custom_run: Initialized filter storage buffer")
        
        # Store filter model and shared memory references for training loop
        filter_model_refs = {
            'model': filter_model,
            'filter_p_shm': filter_p_shm,
            'filter_p_buf': filter_p_buf,
            'filter_policy_shm_name': filter_policy_shm_name,
            'state_dim': filter_state_dim,
            'action_dim': filter_action_dim,
            'storage_buffer': filter_storage_buffer,
            'storage_training_batch_size': filter_storage_training_batch_size,
        }

        results = None

        logger.debug("custom_run: waiting until minion starts collecting rollouts.")
        # wait until the minion has started collecting rollouts
        while f_buf[4] == 0:  # minion data collection flag is now at index 4
            time.sleep(0.1)
        logger.debug("custom_run: minion is now collecting rollouts")

        # debugging
        # logger.debug(f"custom_run: circular_buffer_num_batches -> {algo.config.circular_buffer_num_batches}")
        # logger.debug(
        #     f"custom_run: circular_buffer_iterations_per_batch -> {algo.config.circular_buffer_iterations_per_batch}")
        # logger.debug(f"custom_run: target_network_update_freq -> {algo.config.target_network_update_freq}")
        # logger.debug(
        #     f"custom_run: num_aggregator_actors_per_learner -> {algo.config.num_aggregator_actors_per_learner}")
        # logger.debug(
        #     f"custom_run: num_envs_per_env_runner -> {algo.config.num_envs_per_env_runner}")
        # logger.debug(f"custom_run: _skip_learners -> {algo.config._skip_learners}")
        # logger.debug(f"custom_run: enable_rl_module_and_learner? {algo.config.enable_rl_module_and_learner}")
        # logger.debug(f"custom_run: broadcast_env_runner_states? {algo.config.broadcast_env_runner_states}")
        # logger.debug(f"custom_run: num_learners -> {algo.config.num_learners}")
        # logger.debug(f"custom_run: min_sample_timesteps_per_iteration -> {algo.config.min_sample_timesteps_per_iteration}")
        # logger.debug(f"custom_run: min_env_steps_per_iteration -> {algo.config.min_env_steps_per_iteration}")
        # logger.debug(f"custom_run: min_time_s_per_iteration -> {algo.config.min_time_s_per_iteration}")

        merge = (
                        not algo.config.enable_env_runner_and_connector_v2
                        and algo.config.use_worker_filter_stats
                ) or (
                        algo.config.enable_env_runner_and_connector_v2
                        and (
                                algo.config.merge_env_runner_states is True
                                or (
                                        algo.config.merge_env_runner_states == "training_only"
                                        and not algo.config.in_evaluation
                                )
                        )
                )
        broadcast = (
                            not algo.config.enable_env_runner_and_connector_v2
                            and algo.config.update_worker_filter_stats
                    ) or (
                            algo.config.enable_env_runner_and_connector_v2
                            and algo.config.broadcast_env_runner_states
                    )
        logger.debug(f"custom_run: merge -> {merge}")
        logger.debug(f"custom_run: broadcast -> {broadcast}")

        module = algo.get_module()
        dist_cls = module.get_inference_action_dist_cls()
        logger.debug(f"policy dist_class: {dist_cls}, {dist_cls.__name__}")

        # set up data broadcasting to GUI (optional)
        pub = None
        ctx = None
        if args.enable_zmq and zmq_available and zmq is not None:
            try:
                ctx = zmq.Context()
                pub = ctx.socket(zmq.PUB)
                pub.bind("ipc:///tmp/training.ipc")
                logger.info("ZMQ publisher initialized for GUI communication")
            except Exception as e:
                logger.warning(f"Failed to initialize ZMQ publisher: {e}. Continuing without ZMQ.")
                pub = None
                ctx = None
        elif args.enable_zmq and not zmq_available:
            logger.warning("ZMQ requested but not available (zmq not installed). Continuing without ZMQ.")
        else:
            logger.debug("ZMQ disabled via --enable-zmq flag")

        # Helper functions for filter training
        def _read_filter_batch():
            """Read completed batches from filter episode shared memory buffer.
            Returns batches as numpy arrays of shape (batch_size, state_dim + action_dim + state_dim)
            or None if no complete batch available.
            Also adds batches to storage buffer for accumulation.
            """
            if filter_ep_arr is None or filter_ep_shm_properties is None:
                return None

            # wait until the buffer is unlocked to read indices
            while True:
                if f_buf[6] == 0:  # filter episode buffer lock flag
                    write_idx, read_idx = get_indices(filter_ep_arr, f_buf, lock_index=6)
                    break
                else:
                    time.sleep(0.0001)

            # Check if any batches available
            if write_idx == read_idx:
                set_indices(filter_ep_arr, read_idx, 'r', f_buf, lock_index=6)
                return None  # ring empty

            # Calculate number of batches
            if write_idx < read_idx:
                num_batches = ((filter_ep_shm_properties["NUM_SLOTS"] - 1) - read_idx) + write_idx + 1
            else:
                num_batches = write_idx - read_idx

            batches = []
            for i in range(num_batches):
                slot_off = filter_ep_shm_properties["HEADER_SIZE"] + read_idx * filter_ep_shm_properties["SLOT_SIZE"]
                filled = int(filter_ep_arr[slot_off])
                
                # Ensure batch is complete
                if filled < filter_ep_shm_properties["BATCH_SIZE"]:
                    set_indices(filter_ep_arr, read_idx, 'r', f_buf, lock_index=6)
                    return batches if batches else None

                # Extract data from ring
                data_start = slot_off + filter_ep_shm_properties["HEADER_SLOT_SIZE"]
                payload = np.copy(filter_ep_arr[data_start: data_start + filter_ep_shm_properties["PAYLOAD_SIZE"]])
                
                # Extract initial state and remove it
                state_dim = filter_ep_shm_properties["STATE_ACTION_DIMS"]["state"]
                initial_state = payload[:state_dim]
                payload = payload[state_dim:]
                
                # Reshape to (batch_size, elements_per_rollout)
                batch = payload.reshape(filter_ep_shm_properties["BATCH_SIZE"], 
                                       filter_ep_shm_properties["ELEMENTS_PER_ROLLOUT"])
                
                batches.append(batch)
                
                # Add batch to storage buffer
                if filter_model_refs['storage_buffer'] is not None:
                    filter_model_refs['storage_buffer'].add_batch(batch)
                
                # Advance read_idx
                read_idx = (read_idx + 1) % filter_ep_shm_properties["NUM_SLOTS"]

            # Commit new indices and unlock
            set_indices(filter_ep_arr, read_idx, 'r', f_buf, lock_index=6)
            return batches

        def _sample_from_storage_buffer():
            """Sample batches from storage buffer for training.
            Returns batches as numpy arrays or None if not enough samples available.
            """
            storage_buffer = filter_model_refs.get('storage_buffer')
            if storage_buffer is None:
                return None
            
            # Determine batch size for sampling
            training_batch_size = filter_model_refs.get('storage_training_batch_size')
            if training_batch_size is None:
                # Use ring buffer batch size as default
                if filter_ep_shm_properties is not None:
                    training_batch_size = filter_ep_shm_properties.get("BATCH_SIZE", 128)
                else:
                    training_batch_size = 128
            
            # Sample batches from storage buffer
            sampled_batches = storage_buffer.sample_batches(n_batches=1, batch_size=training_batch_size)
            return sampled_batches

        def _train_filter_model():
            """Train filter model on available batch data.
            First accumulates data from ring buffer to storage buffer, then samples from storage buffer
            for training if enough samples are available. Falls back to ring buffer batches if storage buffer is too small.
            Returns loss value or None if no data available.
            """
            if filter_model_refs['model'] is None:
                return None

            # First, read from ring buffer and accumulate in storage buffer
            ring_batches = _read_filter_batch()  # This also adds batches to storage buffer
            
            # Try to sample from storage buffer first
            batches = None
            storage_buffer = filter_model_refs.get('storage_buffer')
            if storage_buffer is not None and storage_buffer.size() >= storage_buffer.min_samples_for_training:
                sampled = _sample_from_storage_buffer()
                if sampled is not None and len(sampled) > 0:
                    batches = sampled
                    logger.debug(f"custom_run: Using storage buffer for training ({storage_buffer.size()} samples)")
            
            # Fall back to ring buffer batches if storage buffer doesn't have enough samples
            if batches is None or len(batches) == 0:
                batches = ring_batches
                if batches is None or len(batches) == 0:
                    return None
                logger.debug(f"custom_run: Using ring buffer for training (storage buffer has {storage_buffer.size() if storage_buffer else 0} samples)")

            model = filter_model_refs['model']
            model.train()  # Set to training mode
            
            # Initialize optimizer if not already done
            if not hasattr(_train_filter_model, 'optimizer'):
                _train_filter_model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            optimizer = _train_filter_model.optimizer
            criterion = torch.nn.MSELoss()
            
            total_loss = 0.0
            num_samples = 0
            
            for batch in batches:
                # Split batch into state, action, next_state
                state_dim = filter_ep_shm_properties["STATE_ACTION_DIMS"]["state"]
                action_dim = filter_ep_shm_properties["STATE_ACTION_DIMS"]["action"]
                
                states = torch.from_numpy(batch[:, :state_dim]).float()
                actions = torch.from_numpy(batch[:, state_dim:state_dim + action_dim]).float()
                # next_state is the segment immediately following action_filtered (ignore appended action_nominal)
                next_state_start = state_dim + action_dim
                next_state_end = next_state_start + state_dim
                next_states = torch.from_numpy(batch[:, next_state_start:next_state_end]).float()
                
                # Forward pass
                predicted_next_states, _, _ = model(states, actions)
                
                # Compute loss
                loss = criterion(predicted_next_states, next_states)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += states.shape[0]
            
            model.eval()  # Set back to eval mode
            
            avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
            logger.debug(f"custom_run: Filter model training loss: {avg_loss:.6f}")
            
            return avg_loss

        def _update_filter_policy_shm(mse_loss):
            """Update filter policy weights in shared memory along with MSE loss.
            
            Args:
                mse_loss: The MSE loss value from training to store with the weights.
            """
            if filter_model_refs['model'] is None:
                return
            
            model = filter_model_refs['model']
            model.eval()
            
            # Convert to ONNX/ORT
            filter_ort_raw = _get_filter_onnx_model(model, logger=logger, outdir="filter_model.onnx")
            filter_policy_nbytes = len(filter_ort_raw)
            filter_p_buf = filter_model_refs['filter_p_buf']
            header_offset = 4
            mse_offset = header_offset + filter_policy_nbytes  # Store MSE after the ORT weights
            
            # Check if buffer is large enough (weights + 4 bytes for MSE float32)
            required_size = mse_offset + 4
            if len(filter_p_buf) < required_size:
                logger.warning(f"Filter policy buffer too small: need {required_size}, have {len(filter_p_buf)}")
                return
            
            # Check size matches (excluding MSE storage)
            _len_ort_expected = struct.unpack_from("<I", filter_p_buf, 0)
            if _len_ort_expected[0] != filter_policy_nbytes:
                logger.warning(f"Filter model size mismatch: expected {_len_ort_expected[0]}, got {filter_policy_nbytes}")
                return
            
            # Wait for lock and update
            while f_buf[2] == 1:  # filter weights lock flag
                time.sleep(0.0001)
            
            f_buf[2] = 1  # set lock flag to locked
            filter_p_buf[header_offset:header_offset + len(filter_ort_raw)] = filter_ort_raw
            # Store MSE loss as float32 after the weights
            struct.pack_into("<f", filter_p_buf, mse_offset, float(mse_loss))
            f_buf[2] = 0  # set lock flag to unlocked
            f_buf[3] = 1  # set filter weights-available flag to 1 (true)
            
            logger.debug(f"custom_run: Updated filter policy weights in shared memory with MSE={mse_loss:.6f}")

        try:
            # start counter
            train_iter = 0
            while True:
                logger.debug("custom_run: in the train loop now.")

                # perform one logical iteration of actor training
                results = algo.train()

                # Train filter model (alternating training)
                filter_loss = _train_filter_model()
                if filter_loss is not None:
                    _update_filter_policy_shm(filter_loss)
                    logger.debug(f"custom_run: Filter training completed, loss={filter_loss:.6f}")

                state = algo.learner_group.get_state(components="learner")
                if 'metrics_logger' in state['learner']:
                    stats = state['learner']['metrics_logger']['stats']
                    try:
                        logger.debug(f"step {train_iter:>4}: "
                                     f"Qloss={list(stats['default_policy--qf_loss']['values'])} | "
                                     f"Ploss={list(stats['default_policy--policy_loss']['values'])} | "
                                     f"α={list(stats['default_policy--alpha_value']['values'])} "
                                     f"(αloss={list(stats['default_policy--alpha_loss']['values'])}) | "
                                     f"Qµ={list(stats['default_policy--qf_mean']['values'])}"
                                     )
                    except Exception as e:
                        logger.debug(f"could not print stats due to error {e}")

                seq_learn = state["learner"]["weights_seq_no"]

                logger.debug(f"custom_run: learner weights_seq_no="
                             f"{seq_learn}")

                # walk_keys(results)
                # logger.debug(f"custom_run: results.keys()={results.keys()}.")
                logger.debug("custom_run: printing BehaviourAudit.")
                try:
                    msg = {
                        "topic": "policy",
                        "ratio_max": float(results["env_runners"]["ratio_max"]),
                    }
                    # send results to be logged in the GUI
                    if pub is not None:
                        pub.send_json(msg)

                    logger.debug(f'ratio_max={results["env_runners"]["ratio_max"]}, '
                                 f'ratio_p99={results["env_runners"]["ratio_p99"]}, '
                                 f'delta_logp={results["env_runners"]["delta_logp"]}')
                except KeyError:
                    logger.debug("Could not find the keys in results dictionary")


                # attempt at debugging
                # logger.debug("custom_run: ran algo.train()")
                # target_updates = algo._counters["num_target_updates"]
                # last_update = algo._counters["last_target_update_ts"]
                # cur_ts = algo._counters[
                #     (
                #         "num_agent_steps_sampled"
                #         if algo.config.count_steps_by == "agent_steps"
                #         else "num_env_steps_sampled"
                #     )
                # ]
                # logger.debug(f"custom_run: enable_rl_module_and_learner? {algo.config.enable_rl_module_and_learner}")
                # logger.debug(f"custom_run: tentative update frequency: {algo.config.num_epochs * algo.config.minibatch_buffer_size}")
                # logger.debug(f"custom_run: update math: {cur_ts - last_update}")
                # logger.debug(f"custom_run: number of target updates: {target_updates}")
                # last_synch = algo.metrics.peek(
                #     "num_training_step_calls_since_last_synch_worker_weights",
                # )
                # logger.debug(f"custom_run: num training steps since last synch: {last_synch}")
                # logger.debug(f"custom_run: num_weights_broadcast -> {algo.metrics.peek('num_weight_broadcasts')}")

                msg = {"topic": "training", "iteration": train_iter}  # to send to GUI

                # print results
                if ENV_RUNNER_RESULTS in results:
                    mean_return = results[ENV_RUNNER_RESULTS].get(
                        EPISODE_RETURN_MEAN, np.nan
                    )
                    logger.debug(f"iter={train_iter} R={mean_return}")
                    msg.update({"mean_return": float(mean_return)})
                    # print(f"iter={train_iter} R={mean_return}", end="")
                if EVALUATION_RESULTS in results:
                    Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
                        EPISODE_RETURN_MEAN
                    ]
                    print(f" R(eval)={Reval}", end="")
                    msg.update({"eval_return": float(Reval)})
                print()

                # send results to be logged in the GUI
                if pub is not None:
                    pub.send_json(msg)

                # increment counter
                train_iter += 1

        except KeyboardInterrupt:
            if not keep_ray_up:
                # del f_arr
                ray.shutdown()

        return results
