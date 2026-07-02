import argparse
from typing import Optional
import importlib

# Directory containing .h5 files used for predictor input normalization (mean/std).
# Replace this placeholder with your HDF5 data directory.
DEFAULT_SAMPLE_DATA_DIR = '/Users/rodrigohadlich/Documents/Lab Documents/Methanol/Training Dataset/Training v1/digital_twin_processed_data/hdf5_data/test'

# Directory containing .h5 files used for safety-filter action normalization (min/max).
# Replace this placeholder with your safety-filter HDF5 data directory.
DEFAULT_FILTER_SAMPLE_DATA_DIR = '/Users/rodrigohadlich/Documents/Lab Documents/Methanol/Training Dataset/New Training/safety_filter_processed_data/hdf5_data/test'


def get_full_parser():
    parser = custom_args()
    tmp, _ = parser.parse_known_args()
    try:
        mod = importlib.import_module(f"configs.algorithms.{tmp.algo.lower()}_cfg")
        if hasattr(mod, "add_cli_args"):
            mod.add_cli_args(parser)  # inject algo-specific flags
    except ModuleNotFoundError:
        pass  # no extra flags for this algo

    return parser


def custom_args(
    parser: Optional[argparse.ArgumentParser] = None,
    default_reward: float = 100.0,
    default_iters: int = 200,
    default_timesteps: int = 100000,
) -> argparse.ArgumentParser:
    """Adds RLlib-typical (and common) examples scripts command line args to a parser.

    Args:
        parser: The parser to add the arguments to. If None, create a new one.
        default_reward: The default value for the --stop-reward option.
        default_iters: The default value for the --stop-iters option.
        default_timesteps: The default value for the --stop-timesteps option.

    Returns:
        The altered (or newly created) parser object.
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    # Algo and Algo config options.
    parser.add_argument(
        "--algo", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--enable-new-api-stack",
        action="store_true",
        help="Whether to use the `enable_rl_module_and_learner` config setting.",
    )
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="The gym.Env identifier to run the experiment with.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=None,
        help="The number of (remote) EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--create-local-env-runner",
        type=bool,
        default=False,
        help="If True, create a local EnvRunner instance, besides the num_env_runners remote EnvRunner actors.",
    )
    parser.add_argument(
        "--create-env-on-local-worker",
        type=bool,
        default=False,
        help="If True and create_local_env_runner is also True, will have the local EnvRunner also sample rollouts"
             "from environment.",
    )
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=None,
        help="The number of (vectorized) environments per EnvRunner. Note that "
        "this is identical to the batch size for (inference) action computations.",
    )
    parser.add_argument(
        "--throughput-max-episode-steps",
        type=int,
        default=32,
        help="Episode horizon used by throughput profile environment.",
    )
    parser.add_argument(
        "--throughput-rollout-fragment-length",
        type=int,
        default=None,
        help="Sampling fragment length per EnvRunner (throughput profile).",
    )
    parser.add_argument(
        "--throughput-train-batch-size-per-learner",
        type=int,
        default=None,
        help="Learner train batch size per update (throughput profile).",
    )
    parser.add_argument(
        "--throughput-num-steps-sampled-before-learning-starts",
        type=int,
        default=None,
        help="Warmup sampled steps before learning starts (throughput profile).",
    )
    parser.add_argument(
        "--throughput-min-sample-timesteps-per-iteration",
        type=int,
        default=None,
        help="Minimum env steps sampled before train() returns (throughput profile).",
    )
    parser.add_argument(
        "--throughput-min-train-timesteps-per-iteration",
        type=int,
        default=None,
        help="Minimum train timesteps before train() returns (throughput profile).",
    )
    parser.add_argument(
        "--throughput-training-intensity",
        type=float,
        default=None,
        help="Replay ratio target (train/sampled timesteps) in throughput profile.",
    )
    parser.add_argument(
        "--throughput-target-min-hold-len",
        type=int,
        default=None,
        help=(
            "Minimum flat-hold steps for the IMEP target curve in throughput profile. "
            "Default: 15 (fast-cycling for throughput training)."
        ),
    )
    parser.add_argument(
        "--throughput-target-max-hold-len",
        type=int,
        default=None,
        help=(
            "Maximum flat-hold steps for the IMEP target curve in throughput profile. "
            "Default: 60 (fast-cycling for throughput training)."
        ),
    )
    parser.add_argument(
        "--throughput-target-min-transition-len",
        type=int,
        default=None,
        help=(
            "Minimum transition steps for the IMEP target curve in throughput profile. "
            "Default: 20."
        ),
    )
    parser.add_argument(
        "--throughput-target-max-transition-len",
        type=int,
        default=None,
        help=(
            "Maximum transition steps for the IMEP target curve in throughput profile. "
            "Default: 90."
        ),
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=0,
        help="If 0 (default), will run as single-agent. If > 0, will run as "
        "multi-agent with the environment simply cloned n times and each agent acting "
        "independently at every single timestep. The overall reward for this "
        "experiment is then the sum over all individual agents' rewards.",
    )

    # Evaluation options.
    parser.add_argument(
        "--evaluation-num-env-runners",
        type=int,
        default=0,
        help="The number of evaluation (remote) EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=0,
        help="Every how many iterations to run one round of evaluation. "
        "Use 0 (default) to disable evaluation.",
    )
    parser.add_argument(
        "--evaluation-duration",
        type=lambda v: v if v == "auto" else int(v),
        default=10,
        help="The number of evaluation units to run each evaluation round. "
        "Use `--evaluation-duration-unit` to count either in 'episodes' "
        "or 'timesteps'. If 'auto', will run as many as possible during train pass ("
        "`--evaluation-parallel-to-training` must be set then).",
    )
    parser.add_argument(
        "--evaluation-duration-unit",
        type=str,
        default="episodes",
        choices=["episodes", "timesteps"],
        help="The evaluation duration unit to count by. One of 'episodes' or "
        "'timesteps'. This unit will be run `--evaluation-duration` times in each "
        "evaluation round. If `--evaluation-duration=auto`, this setting does not "
        "matter.",
    )
    parser.add_argument(
        "--evaluation-parallel-to-training",
        action="store_true",
        help="Whether to run evaluation parallel to training. This might help speed up "
        "your overall iteration time. Be aware that when using this option, your "
        "reported evaluation results are referring to one iteration before the current "
        "one.",
    )

    # RLlib logging options.
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The output directory to write trajectories to, which are collected by "
        "the algo's EnvRunners.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,  # None -> use default
        choices=["INFO", "DEBUG", "WARN", "ERROR"],
        help="The log-level to be used by the RLlib logger.",
    )

    # tune.Tuner options.
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many (tune.Tuner.fit()) experiments to execute - if possible in "
        "parallel.",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=None,
        help="How many (tune.Tuner) trials to run concurrently.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="The verbosity level for the `tune.Tuner()` running the experiment.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help=(
            "The frequency (in training iterations) with which to create checkpoints. "
            "Note that if --wandb-key is provided, all checkpoints will "
            "automatically be uploaded to WandB."
        ),
    )
    parser.add_argument(
        "--checkpoint-at-end",
        action="store_true",
        help=(
            "Whether to create a checkpoint at the very end of the experiment. "
            "Note that if --wandb-key is provided, all checkpoints will "
            "automatically be uploaded to WandB."
        ),
    )

    # WandB logging options.
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="The WandB API key to use for uploading results.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="The WandB project name to use.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="The WandB run name to use.",
    )

    # Experiment stopping and testing criteria.
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=default_reward,
        help="Reward at which the script should stop training.",
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=default_iters,
        help="The number of iterations to train.",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=default_timesteps,
        help="The number of (environment sampling) timesteps to train.",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test. If set, --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters, otherwise this "
        "script will throw an exception at the end.",
    )
    parser.add_argument(
        "--as-release-test",
        action="store_true",
        help="Whether this script should be run as a release test. If set, "
        "all that applies to the --as-test option is true, plus, a short JSON summary "
        "will be written into a results file whose location is given by the ENV "
        "variable `TEST_OUTPUT_JSON`.",
    )

    # Learner scaling options.
    parser.add_argument(
        "--num-learners",
        type=int,
        default=None,
        help="The number of Learners to use. If `None`, use the algorithm's default "
        "value.",
    )
    parser.add_argument(
        "--num-cpus-per-learner",
        type=float,
        default=None,
        help="The number of CPUs per Learner to use. If `None`, use the algorithm's "
        "default value.",
    )
    parser.add_argument(
        "--num-gpus-per-learner",
        type=float,
        default=None,
        help="The number of GPUs per Learner to use. If `None` and there are enough "
        "GPUs for all required Learners (--num-learners), use a value of 1, "
        "otherwise 0.",
    )
    parser.add_argument(
        "--num-aggregator-actors-per-learner",
        type=int,
        default=None,
        help="The number of Aggregator actors to use per Learner. If `None`, use the "
        "algorithm's default value.",
    )

    # Ray init options.
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Global random seed for fully deterministic training. "
            "If omitted, runs are non-deterministic."
        ),
    )
    parser.add_argument(
        "--cpu-core-learner",
        type=int,
        default=None,
        help="CPU core ID to pin the learner process to (0-indexed). Only works on Linux.",
    )
    parser.add_argument(
        "--cpu-core-env-runner",
        type=int,
        default=None,
        help="CPU core ID to pin the EnvRunner process to (0-indexed). Only works on Linux.",
    )
    parser.add_argument(
        "--cpu-core-minion",
        type=int,
        default=None,
        help="CPU core ID to pin the minion process to (0-indexed). Only works on Linux.",
    )
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument(
        "--enable-zmq",
        type=str_to_bool,
        default=False,
        help="Enable ZMQ publishing for GUI communication (default: False when running setup_run.py directly).",
    )
    parser.add_argument(
        "--gui",
        type=str_to_bool,
        default=False,
        help="Spawn the PyQt6 monitoring GUI (implies --enable-zmq True).",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    # Safety filter.
    parser.add_argument(
        "--disable-safety-filter",
        action="store_true",
        default=False,
        help="Disable the safety filter: no filter model load/train/save, and actor action is used as-is (no filter correction). Default: filter enabled.",
    )

    parser.add_argument(
        "--enable-benchmark-eval",
        type=str_to_bool,
        default=True,
        help=(
            "Run fixed-profile benchmark evaluation in the minion after each "
            "train/eval sequence (realtime engine path only). Default: True."
        ),
    )

    # Model save/load options.
    parser.add_argument(
        "--rllib-module-name",
        type=str,
        default=None,
        help="Basename for the full RLlib module (actor, critic, target networks). Saved under models/ directory. "
        "If not set, defaults are used based on algo and env_type.",
    )
    parser.add_argument(
        "--filter-model-name",
        type=str,
        default=None,
        help="Basename for the safety filter StatePredictor model. Saved under models/ directory. "
        "If not set, defaults are used based on algo and env_type.",
    )
    parser.add_argument(
        "--rllib-load-dir",
        type=str,
        default=None,
        help=(
            "Directory to load RLlib checkpoints/spec from. If provided, the script "
            "must successfully load the RLlib model from this directory or it will fail. "
            "Accepts absolute or project-relative paths."
        ),
    )
    parser.add_argument(
        "--filter-load-dir",
        type=str,
        default=None,
        help=(
            "Directory to load safety filter checkpoints/spec from. If provided, the "
            "script must successfully load the filter model from this directory or it "
            "will fail. Accepts absolute or project-relative paths."
        ),
    )
    parser.add_argument(
        "--predictor-checkpoint",
        type=str,
        default=None,
        help=(
            "Predictor .pth file (absolute path, project-relative path, or filename under "
            "src/core/digital_twin/models/) used by engine environments. "
            "If omitted, the default predictor checkpoint is used."
        ),
    )
    parser.add_argument(
        "--sample-data-dir",
        type=str,
        default=DEFAULT_SAMPLE_DATA_DIR,
        help=(
            "Directory containing HDF5 (.h5) files whose normalization/feature_mean and "
            "normalization/feature_std are used by the engine predictor. "
            f"Default: {DEFAULT_SAMPLE_DATA_DIR!r}"
        ),
    )
    parser.add_argument(
        "--filter-sample-data-dir",
        type=str,
        default=DEFAULT_FILTER_SAMPLE_DATA_DIR,
        help=(
            "Directory containing safety-filter HDF5 (.h5) files whose "
            "normalization/action_min and normalization/action_max are used "
            "to normalize actions before SafetyFilter inference. "
            f"Default: {DEFAULT_FILTER_SAMPLE_DATA_DIR!r}"
        ),
    )
    return parser