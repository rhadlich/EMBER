"""Example of running against a shared-memory-connected external env performing its own inference.

How to run this script
----------------------
`python setup_run.py --algo "algo_name (like PPO, SAC, etc.)" --model-mode "create or load" --gui True --rllib-module-name "rllib_module_name" --filter-model-name "filter_model_name" --cpu-core-minion "core#"`

Determinism
-----------
Pass a global seed via `--seed <int>` to get fully reproducible training runs
for a fixed algorithm/env configuration. Running this script twice with the
same CLI arguments (including `--seed`) should produce identical training
metrics (e.g., `env_runners/episode_return_mean`) and filter MSE sequences.
"""
import numpy as np
import os
import subprocess
import sys

from gymnasium import spaces
from core.environments.engine_env import EngineEnvDiscrete, EngineEnvContinuous, reward_fn

from env_runner import SharedMemoryEnvRunner
# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )
from ray.tune.registry import get_trainable_cls
from ray.rllib.core.rl_module import RLModuleSpec

from configs.args import get_full_parser
from run_algorithm import run_rllib_shared_memory

from utils.utils import ActionAdapter
from core.rl_modules.impala_rl_modules import ImpalaMlpModule

import logging

parser = get_full_parser()
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=1,  # number of remote EnvRunners
    num_cpus_per_env_runner=1,  # how many cpus per remote EnvRunner
    create_local_env_runner=True,  # only have remote EnvRunners
    create_env_on_local_worker=False,  # don't sample from env if local worker is created
    num_learners=0,  # only have the learner in the main driver
    num_cpus_per_learner=0,  # this will be ignored if num_learners is 0
    num_cpus=3,  # for ray_init call inside test_utils
    num_gpus_per_learner=0,
    # algo='APPO',
)
parser.add_argument(
    "--policy_shm_name",
    type=str,
    default="policy",
    help="Name of shm for policy weights to use",
)
parser.add_argument(
    "--flag_shm_name",
    type=str,
    default="flag",
    help="Name of flag shm to use",
)

if __name__ == "__main__":
    args = parser.parse_args()

    logger = logging.getLogger("MyRLApp.setup_run")
    logger.info(f"setup_run, PID={os.getpid()}")

    # If a global seed is provided, make this driver process deterministic.
    # RLlib will additionally receive this seed via AlgorithmConfig.debugging(seed=...).
    if getattr(args, "seed", None) is not None:
        import random
        import torch

        seed = int(args.seed)
        # Python stdlib, NumPy, and Torch RNGs.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Spawn GUI if requested (implies enable_zmq for telemetry)
    if getattr(args, "gui", False):
        args.enable_zmq = True
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rlapp_path = os.path.join(script_dir, "RLapp.py")
        cmd = [sys.executable, rlapp_path, "--subscriber-only"]
        subprocess.Popen(cmd)
        logger.info("Spawned GUI in subscriber-only mode")

    # make environment to have access to observation and action spaces
    if args.env_type.lower() == 'continuous':
        env = EngineEnvContinuous(reward=reward_fn)
    elif args.env_type.lower() == 'discrete':
        env = EngineEnvDiscrete(reward=reward_fn)
    else:
        raise NotImplementedError(f"Environment type not supported or not provided.")
    obs_space = env.observation_space
    action_space = env.action_space

    if isinstance(obs_space, spaces.Discrete):
        imep_space = env.imep_space
        mprr_space = env.mprr_space

        # patch up the dimensions issue when running a discrete observation space
        flat_dim = len(imep_space)
        obs_space_onehot = spaces.Box(
                low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32
            )
        obs_is_discrete = True
    elif isinstance(obs_space, spaces.Box):
        obs_space_onehot = None
        imep_space = env.imep_lims
        mprr_space = env.mprr_lims
        obs_is_discrete = False
    else:
        raise NotImplementedError(f"Unsupported observation space {obs_space}")

    adapter = ActionAdapter(action_space)

    # get action space size in one-hot representation (if discrete). will need this to create
    # the episode buffer such that it can hold the probability distribution
    if adapter.mode in ("discrete1", "multidiscrete"):
        # logits will be the same length as action_onehot_size
        action_dist_size = int(sum(adapter.nvec))
    else:
        # non-discrete action space, need mean and standard deviation of the distribution instead of logits
        action_dist_size = 2*action_space.shape[0]

    # Define name and properties of episode ring buffer to pass down to EnvRunner.
    # Define the size of each rollout tuple.
    bytes_per_float = np.dtype("float32").itemsize  # number of bytes in rollout data type
    dims = {
        "action": 2,    # Currently injection timing and injection duration
        "reward": 1,
        # Current state is [prev desired IMEP, prev achieved IMEP, next desired IMEP]
        "state": 3,                             # this will be the next state AFTER taking "action"
        "action_dist_size": action_dist_size,
        "logp": 1                               # scalar by definition
    }  # the length of the vector of each component of the rollout
    BATCH_SIZE = 32  # number of rollouts per batch (episode)
    NUM_SLOTS = 8  # ring depth (i.e. number of episodes)

    # the length of the vector of each component of the rollout
    ELEMENTS_PER_ROLLOUT = sum(dims.values())
    BYTES_PER_ROLLOUT = ELEMENTS_PER_ROLLOUT * bytes_per_float
    
    # Added state to PAYLOAD_SIZE because need to include the starting state for each episode. This will be simply
    # the state observation at the end of the last episode/batch.
    PAYLOAD_SIZE = ELEMENTS_PER_ROLLOUT * BATCH_SIZE + dims["state"]
    HEADER_SIZE = 2  # write_idx, read_idx
    HEADER_SLOT_SIZE = 1  # one float32 to store how many rollouts already in slot
    SLOT_SIZE = HEADER_SLOT_SIZE + PAYLOAD_SIZE
    TOTAL_SIZE = HEADER_SIZE + NUM_SLOTS*SLOT_SIZE
    TOTAL_SIZE_BYTES = int(TOTAL_SIZE * bytes_per_float)
    logger.debug(f"action_dist_size: {action_dist_size}")
    logger.debug(f"ELEMENTS_PER_ROLLOUT: {ELEMENTS_PER_ROLLOUT}")
    logger.debug(f"PAYLOAD_SIZE: {PAYLOAD_SIZE}")
    logger.debug(f"SLOT_SIZE: {SLOT_SIZE}")
    logger.debug(f"TOTAL_SIZE: {TOTAL_SIZE}")

    ep_shm_properties = {
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_SLOTS": NUM_SLOTS,
        "ELEMENTS_PER_ROLLOUT": ELEMENTS_PER_ROLLOUT,
        "BYTES_PER_ROLLOUT": BYTES_PER_ROLLOUT,
        "PAYLOAD_SIZE": PAYLOAD_SIZE,
        "HEADER_SIZE": HEADER_SIZE,
        "HEADER_SLOT_SIZE": HEADER_SLOT_SIZE,
        "SLOT_SIZE": SLOT_SIZE,
        "TOTAL_SIZE": TOTAL_SIZE,
        "TOTAL_SIZE_BYTES": TOTAL_SIZE_BYTES,
        "STATE_ACTION_DIMS": dims,
        "BYTES_PER_FLOAT": bytes_per_float,
        "name": "episodes",
        "action_dist_size": action_dist_size,
    }

    enable_safety_filter = not getattr(args, "disable_safety_filter", False)

    # Define filter training data ring buffer properties (4x larger than actor buffer)
    # Filter data format: (current_state, action_filtered, next_state, action_nominal)
    filter_ep_shm_properties = None
    if enable_safety_filter:
        filter_dims = {
            "state": dims["state"],
            "action": dims["action"],
            "next_state": dims["state"],
            "nominal_action": dims["action"],
        }
        FILTER_BATCH_SIZE = BATCH_SIZE * 4  # 4x larger (128 vs 32)
        FILTER_NUM_SLOTS = NUM_SLOTS * 4  # 4x larger (32 vs 8)
        FILTER_ELEMENTS_PER_ROLLOUT = sum(filter_dims.values())  # state + action_filtered + next_state + action_nominal
        FILTER_BYTES_PER_ROLLOUT = FILTER_ELEMENTS_PER_ROLLOUT * bytes_per_float
        FILTER_PAYLOAD_SIZE = FILTER_ELEMENTS_PER_ROLLOUT * FILTER_BATCH_SIZE  # filter buffer does not need initial state
        FILTER_HEADER_SIZE = HEADER_SIZE  # same structure
        FILTER_HEADER_SLOT_SIZE = HEADER_SLOT_SIZE
        FILTER_SLOT_SIZE = FILTER_HEADER_SLOT_SIZE + FILTER_PAYLOAD_SIZE
        FILTER_TOTAL_SIZE = FILTER_HEADER_SIZE + FILTER_NUM_SLOTS * FILTER_SLOT_SIZE
        FILTER_TOTAL_SIZE_BYTES = int(FILTER_TOTAL_SIZE * bytes_per_float)
        FILTER_N_BATCHES_FOR_TRAINING_ITERATION = 16

        logger.debug(f"Filter buffer: BATCH_SIZE={FILTER_BATCH_SIZE}, NUM_SLOTS={FILTER_NUM_SLOTS}")
        logger.debug(f"Filter ELEMENTS_PER_ROLLOUT: {FILTER_ELEMENTS_PER_ROLLOUT}")
        logger.debug(f"Filter PAYLOAD_SIZE: {FILTER_PAYLOAD_SIZE}")
        logger.debug(f"Filter TOTAL_SIZE_BYTES: {FILTER_TOTAL_SIZE_BYTES}")

        filter_ep_shm_properties = {
            "BATCH_SIZE": FILTER_BATCH_SIZE,
            "NUM_SLOTS": FILTER_NUM_SLOTS,
            "ELEMENTS_PER_ROLLOUT": FILTER_ELEMENTS_PER_ROLLOUT,
            "BYTES_PER_ROLLOUT": FILTER_BYTES_PER_ROLLOUT,
            "PAYLOAD_SIZE": FILTER_PAYLOAD_SIZE,
            "HEADER_SIZE": FILTER_HEADER_SIZE,
            "HEADER_SLOT_SIZE": FILTER_HEADER_SLOT_SIZE,
            "SLOT_SIZE": FILTER_SLOT_SIZE,
            "TOTAL_SIZE": FILTER_TOTAL_SIZE,
            "TOTAL_SIZE_BYTES": FILTER_TOTAL_SIZE_BYTES,
            "STATE_ACTION_DIMS": filter_dims,
            "BYTES_PER_FLOAT": bytes_per_float,
            "N_BATCHES_FOR_TRAINING_ITERATION": FILTER_N_BATCHES_FOR_TRAINING_ITERATION,
            "name": "filter_episodes",
            "filter_dims": filter_dims,
        }

    # Build env_config with enable_safety_filter always; filter-related keys only when filter enabled.
    env_config = {
        "policy_shm_name": args.policy_shm_name,
        "flag_shm_name": args.flag_shm_name,
        "ep_shm_properties": ep_shm_properties,
        "imep_space": imep_space,
        "mprr_space": mprr_space,
        "obs_is_discrete": obs_is_discrete,
        "env_type": args.env_type.lower(),
        "cpu_core_env_runner": args.cpu_core_env_runner,
        "cpu_core_minion": args.cpu_core_minion,
        "enable_zmq": args.enable_zmq,
        "realtime_priority": 80,  # Default real-time priority for minion
        "enable_safety_filter": enable_safety_filter,
    }

    # Propagate a single global seed from CLI into per-component seeds used by
    # the EnvRunner/minion/env processes. If no seed is provided, behavior
    # remains non-deterministic (current default).
    if getattr(args, "seed", None) is not None:
        base_seed = int(args.seed)
        env_config["global_seed"] = base_seed
        env_config["env_seed"] = base_seed + 1
        env_config["minion_seed"] = base_seed + 2
        env_config["filter_seed"] = base_seed + 3

    if enable_safety_filter:
        env_config["filter_ep_shm_properties"] = filter_ep_shm_properties
        env_config["filter_policy_shm_name"] = getattr(args, "filter_policy_shm_name", "filter_policy")
        env_config["filter_num_hidden"] = getattr(args, "filter_num_hidden", 2)
        env_config["filter_hidden_exp"] = getattr(args, "filter_hidden_exp", 7)
        env_config["filter_dropout"] = getattr(args, "filter_dropout", 0.0)

    # Define the RLlib config.
    base_config = (
        get_trainable_cls(args.algo)
        # IMPALAConfig(algo_class=IMPALADebug)
        .get_default_config()
        .api_stack(
            enable_rl_module_and_learner=True,  # turn RLModule on
            enable_env_runner_and_connector_v2=True,  # turn connector-v2 on
        )
        .environment(
            observation_space=obs_space_onehot or obs_space,
            action_space=action_space,
            normalize_actions=(True if adapter.mode == "continuous" else False),
            clip_actions=(True if adapter.mode == "continuous" else False),
            clip_rewards=False,
            env_config=env_config,
        )
        .env_runners(
            # Point RLlib to the custom EnvRunner to be used here.
            env_runner_cls=SharedMemoryEnvRunner,
            num_env_runners=args.num_env_runners,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            create_local_env_runner=args.create_local_env_runner,
            create_env_on_local_worker=args.create_env_on_local_worker,
        )
        # Give RLlib a deterministic seed so that trials with the same CLI
        # arguments (including --seed) produce identical results.
        .debugging(seed=getattr(args, "seed", None))
    )

    import importlib
    try:
        mod = importlib.import_module(f"configs.algorithms.{args.algo.lower()}_cfg")
        if hasattr(mod, "update_config"):
            base_config = mod.update_config(base_config, args)
    except ModuleNotFoundError:
        pass

    # Model save/load: resolve names and build spec for compatibility checks.
    env_type = getattr(args, "env_type", "continuous")
    args.rllib_module_name = getattr(args, "rllib_module_name", None) or f"{args.algo}_{env_type}_rllib_module"
    args.filter_model_name = getattr(args, "filter_model_name", None) or f"{args.algo}_{env_type}_filter"
    args.model_mode = getattr(args, "model_mode", "create")

    obs_space_final = obs_space_onehot or obs_space
    obs_shape = list(obs_space_final.shape) if hasattr(obs_space_final, "shape") else [getattr(obs_space_final, "n", None)]
    action_shape = list(action_space.shape) if hasattr(action_space, "shape") else [int(sum(adapter.nvec))] if adapter.mode in ("discrete1", "multidiscrete") else None

    args.rllib_module_spec = {
        "algo": args.algo,
        "framework": args.framework,
        "env_type": env_type,
        "obs_is_discrete": obs_is_discrete,
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "adapter_mode": adapter.mode,
    }
    try:
        mod = importlib.import_module(f"configs.algorithms.{args.algo.lower()}_cfg")
        if hasattr(mod, "get_rllib_module_spec"):
            arch = mod.get_rllib_module_spec(base_config)
            if arch is not None:
                args.rllib_module_spec["rllib_module_arch"] = arch
    except Exception:
        pass  # Algo may not define it
    logger.debug(f"rllib_module_spec: {args.rllib_module_spec}")

    if enable_safety_filter:
        filter_num_hidden = getattr(args, "filter_num_hidden", 2)
        filter_hidden_exp = getattr(args, "filter_hidden_exp", 7)
        filter_dropout = getattr(args, "filter_dropout", 0.0)
        args.filter_spec = {
            "filter_state_dim": filter_dims["state"],
            "filter_action_dim": filter_dims["action"],
            "filter_num_hidden": filter_num_hidden,
            "filter_hidden_exp": filter_hidden_exp,
            "filter_dropout": filter_dropout,
        }
        logger.debug(f"filter_spec: {args.filter_spec}")
    else:
        args.filter_spec = None
    run_rllib_shared_memory(base_config, args)
