"""Example of running against a shared-memory-connected external env performing its own inference.

How to run this script
----------------------
`python setup_run.py --algo "algo_name (like PPO, SAC, etc.)" --model-mode "create or load" --gui True --rllib-module-name "rllib_module_name" --filter-model-name "filter_model_name" --cpu-core-minion "core#"`
python src/setup_run.py --algo 'SAC' --rllib-module-name 'rllib_module' --filter-model-name 'filter_model' --disable-safety-filter --model-mode 'create' --gui True --seed 123 --stop-iters 2000
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
from core.environments.throughput_env import ThroughputEngineEnvContinuous

from env_runner import SharedMemoryEnvRunner
# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.core.rl_module import RLModuleSpec

from configs.args import get_full_parser
from run_algorithm import run_rllib_shared_memory
from run_algorithm_throughput import run_rllib_throughput

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
parser.add_argument(
    "--runtime-profile",
    type=str,
    default="realtime",
    choices=["realtime", "throughput"],
    help=(
        "Execution profile. 'realtime' keeps the shared-memory/minion pipeline. "
        "'throughput' uses standard RLlib sampling for intra-node cluster speed."
    ),
)


def _get_buffer_action_dim(action_space, adapter) -> int:
    if adapter.mode == "continuous":
        return int(action_space.shape[0])
    if adapter.mode == "discrete1":
        return 1
    if adapter.mode == "multidiscrete":
        return len(adapter.nvec)
    raise NotImplementedError(f"Unsupported adapter mode {adapter.mode}")


def _run_throughput_profile(args, logger) -> None:
    """Run a non-realtime profile using RLlib-native sampling on Ray workers."""
    if args.env_type.lower() != "continuous":
        raise NotImplementedError(
            "Throughput profile currently supports only env_type=continuous."
        )

    # The parser defaults are tuned for realtime and intentionally conservative.
    # In throughput mode, treat these as "unset" unless the user explicitly passed
    # the CLI flag, allowing autoscaling to use node capacity.
    cli_args = set(sys.argv[1:])

    def _flag_present(flag: str) -> bool:
        return any(a == flag or a.startswith(f"{flag}=") for a in cli_args)

    if not _flag_present("--num-cpus"):
        args.num_cpus = 0
    if not _flag_present("--num-env-runners"):
        args.num_env_runners = None
    if not _flag_present("--num-cpus-per-env-runner"):
        args.num_cpus_per_env_runner = None
    if not _flag_present("--num-learners"):
        args.num_learners = None
    if not _flag_present("--num-cpus-per-learner"):
        args.num_cpus_per_learner = None
    if not _flag_present("--num-gpus-per-learner"):
        args.num_gpus_per_learner = None

    env = EngineEnvContinuous(reward=reward_fn)
    obs_space = env.observation_space
    action_space = env.action_space
    adapter = ActionAdapter(action_space)

    import importlib

    try:
        algo_cfg_mod = importlib.import_module(
            f"configs.algorithms.{args.algo.lower()}_cfg"
        )
    except ModuleNotFoundError:
        algo_cfg_mod = None

    env_name = "engine_env_throughput_continuous"
    register_env(
        env_name,
        lambda env_cfg: ThroughputEngineEnvContinuous(env_cfg),
    )

    env_config = {
        "env_type": args.env_type.lower(),
        "max_episode_steps": int(getattr(args, "throughput_max_episode_steps", 32)),
    }
    if getattr(args, "seed", None) is not None:
        base_seed = int(args.seed)
        env_config["global_seed"] = base_seed
        env_config["env_seed"] = base_seed + 1

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env=env_name,
            env_config=env_config,
            observation_space=obs_space,
            action_space=action_space,
            normalize_actions=(True if adapter.mode == "continuous" else False),
            clip_actions=(True if adapter.mode == "continuous" else False),
            clip_rewards=False,
        )
        .debugging(seed=getattr(args, "seed", None))
    )
    base_config = base_config.rl_module(
        model_config={
            "throughput_apply_exploration_noise": True,
            "throughput_exploration_noise": float(getattr(args, "exploration_noise", 0.1)),
            "throughput_initial_steps": int(getattr(args, "initial_steps", 0)),
            "throughput_initial_std": float(getattr(args, "initial_std", 0.5)),
            "throughput_noise_decay_k": float(getattr(args, "noise_decay_k", 1e-5)),
            "throughput_noise_decay_schedule": str(
                getattr(args, "noise_decay_schedule", "linear")
            ),
            "throughput_linear_decay_steps": int(
                getattr(args, "linear_decay_steps", 100000)
            ),
        }
    )

    throughput_env_runner_kwargs = {}
    if getattr(args, "num_envs_per_env_runner", None) is not None:
        throughput_env_runner_kwargs["num_envs_per_env_runner"] = int(
            args.num_envs_per_env_runner
        )
    if getattr(args, "throughput_rollout_fragment_length", None) is not None:
        throughput_env_runner_kwargs["rollout_fragment_length"] = int(
            args.throughput_rollout_fragment_length
        )
    if throughput_env_runner_kwargs:
        base_config = base_config.env_runners(**throughput_env_runner_kwargs)

    # Respect explicit user overrides, but let throughput runner autoscale if unset.
    base_config = base_config.env_runners(
        num_env_runners=args.num_env_runners,
        num_cpus_per_env_runner=args.num_cpus_per_env_runner,
        create_local_env_runner=False,
        create_env_on_local_worker=False,
    )
    if args.num_learners is not None:
        base_config = base_config.learners(num_learners=args.num_learners)
    if args.num_cpus_per_learner is not None:
        base_config = base_config.learners(
            num_cpus_per_learner=args.num_cpus_per_learner
        )
    if args.num_gpus_per_learner is not None:
        base_config = base_config.learners(
            num_gpus_per_learner=args.num_gpus_per_learner
        )

    if algo_cfg_mod is not None and hasattr(algo_cfg_mod, "update_config"):
        base_config = algo_cfg_mod.update_config(base_config, args)

    # Apply throughput-specific manual overrides last so they always win over
    # algorithm defaults from update_config().
    throughput_training_kwargs = {}
    if getattr(args, "throughput_train_batch_size_per_learner", None) is not None:
        throughput_training_kwargs["train_batch_size_per_learner"] = int(
            args.throughput_train_batch_size_per_learner
        )
    if getattr(args, "throughput_num_steps_sampled_before_learning_starts", None) is not None:
        throughput_training_kwargs["num_steps_sampled_before_learning_starts"] = int(
            args.throughput_num_steps_sampled_before_learning_starts
        )
    if getattr(args, "throughput_training_intensity", None) is not None:
        throughput_training_kwargs["training_intensity"] = float(
            args.throughput_training_intensity
        )
    if throughput_training_kwargs:
        base_config = base_config.training(**throughput_training_kwargs)

    throughput_reporting_kwargs = {}
    if getattr(args, "throughput_min_sample_timesteps_per_iteration", None) is not None:
        throughput_reporting_kwargs["min_sample_timesteps_per_iteration"] = int(
            args.throughput_min_sample_timesteps_per_iteration
        )
    if getattr(args, "throughput_min_train_timesteps_per_iteration", None) is not None:
        throughput_reporting_kwargs["min_train_timesteps_per_iteration"] = int(
            args.throughput_min_train_timesteps_per_iteration
        )
    if throughput_reporting_kwargs:
        base_config = base_config.reporting(**throughput_reporting_kwargs)

    logger.info("Running throughput profile (RLlib-native sampling).")
    run_rllib_throughput(base_config, args)

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

    if args.runtime_profile == "throughput":
        if getattr(args, "gui", False):
            logger.warning(
                "GUI/ZMQ telemetry is not used in throughput profile; training will continue."
            )
        _run_throughput_profile(args, logger)
        sys.exit(0)

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

    # Import the algo config module early so that custom algorithms (e.g. TD3)
    # can register themselves with ray.tune before get_trainable_cls() is called.
    import importlib
    try:
        _algo_cfg_mod = importlib.import_module(
            f"configs.algorithms.{args.algo.lower()}_cfg"
        )
    except ModuleNotFoundError:
        _algo_cfg_mod = None

    action_dim = _get_buffer_action_dim(action_space, adapter)
    # Runtime observations stored in shared memory are the previous action signal
    # plus three IMEP-related values.
    state_dim = action_dim + 3

    if _algo_cfg_mod is not None and hasattr(_algo_cfg_mod, "get_actor_episode_buffer_spec"):
        ep_shm_properties = _algo_cfg_mod.get_actor_episode_buffer_spec(
            state_dim=state_dim,
            action_space=action_space,
            action_adapter=adapter,
            batch_size=8,
            num_slots=32,
            name="episodes",
        )
    else:
        raise NotImplementedError(
            f"Algorithm {args.algo} does not define get_actor_episode_buffer_spec()."
        )

    # Define name and properties of episode ring buffer to pass down to EnvRunner.
    bytes_per_float = np.dtype("float32").itemsize
    dims = ep_shm_properties["STATE_ACTION_DIMS"]
    BATCH_SIZE = ep_shm_properties["BATCH_SIZE"]
    NUM_SLOTS = ep_shm_properties["NUM_SLOTS"]
    logger.debug(f"actor rollout schema: {ep_shm_properties['ROLLOUT_FIELD_ORDER']}")
    logger.debug(f"actor rollout field dims: {ep_shm_properties['ROLLOUT_FIELD_DIMS']}")
    logger.debug(f"ELEMENTS_PER_ROLLOUT: {ep_shm_properties['ELEMENTS_PER_ROLLOUT']}")
    logger.debug(f"PAYLOAD_SIZE: {ep_shm_properties['PAYLOAD_SIZE']}")
    logger.debug(f"SLOT_SIZE: {ep_shm_properties['SLOT_SIZE']}")
    logger.debug(f"TOTAL_SIZE: {ep_shm_properties['TOTAL_SIZE']}")

    enable_safety_filter = not getattr(args, "disable_safety_filter", False)

    # Define filter training data ring buffer properties (4x larger than actor buffer)
    # Filter data format: (current_state, action_filtered, next_state, action_nominal)
    filter_ep_shm_properties = None
    if enable_safety_filter:
        filter_dims = {
            "state": state_dim,
            "action": action_dim,
            "next_state": state_dim,
            "nominal_action": action_dim,
        }
        FILTER_BATCH_SIZE = BATCH_SIZE * 4  # 4x larger (128 vs 32)
        FILTER_NUM_SLOTS = NUM_SLOTS * 4  # 4x larger (32 vs 8)
        FILTER_ELEMENTS_PER_ROLLOUT = sum(filter_dims.values())  # state + action_filtered + next_state + action_nominal
        FILTER_BYTES_PER_ROLLOUT = FILTER_ELEMENTS_PER_ROLLOUT * bytes_per_float
        FILTER_PAYLOAD_SIZE = FILTER_ELEMENTS_PER_ROLLOUT * FILTER_BATCH_SIZE  # filter buffer does not need initial state
        FILTER_HEADER_SIZE = ep_shm_properties["HEADER_SIZE"]
        FILTER_HEADER_SLOT_SIZE = ep_shm_properties["HEADER_SLOT_SIZE"]
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

    if _algo_cfg_mod is not None and hasattr(_algo_cfg_mod, "update_config"):
        base_config = _algo_cfg_mod.update_config(base_config, args)


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
    if _algo_cfg_mod is not None and hasattr(_algo_cfg_mod, "get_rllib_module_spec"):
        try:
            arch = _algo_cfg_mod.get_rllib_module_spec(base_config)
            if arch is not None:
                args.rllib_module_spec["rllib_module_arch"] = arch
        except Exception:
            pass
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
