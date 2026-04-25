import argparse
import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import ray
import torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.tune.result import TRAINING_ITERATION

from configs.args import custom_args


def _auto_configure_throughput_resources(config, args, logger):
    """Autoscale learner/env-runner resources for single-node throughput."""
    cluster = ray.cluster_resources()
    node_cpus = max(1, int(cluster.get("CPU", os.cpu_count() or 1)))
    node_gpus = int(cluster.get("GPU", 0))

    # Learner placement: prefer 1 learner process on CPU-only, or one per GPU.
    if args.num_learners is None:
        num_learners = node_gpus if node_gpus > 0 else 1
    else:
        num_learners = max(0, int(args.num_learners))

    if args.num_gpus_per_learner is None:
        num_gpus_per_learner = 1 if node_gpus >= max(1, num_learners) and node_gpus > 0 else 0
    else:
        num_gpus_per_learner = float(args.num_gpus_per_learner)

    if args.num_cpus_per_learner is None:
        # Keep learner CPU reservation conservative so samplers still scale.
        num_cpus_per_learner = 1 if node_cpus < 8 else 2
    else:
        num_cpus_per_learner = float(args.num_cpus_per_learner)

    cpus_per_env_runner = (
        float(args.num_cpus_per_env_runner)
        if args.num_cpus_per_env_runner is not None
        else 1.0
    )
    cpus_reserved = 1.0 + (num_learners * num_cpus_per_learner)
    auto_env_runners = max(
        1,
        int((node_cpus - cpus_reserved) // max(cpus_per_env_runner, 1.0)),
    )
    num_env_runners = (
        int(args.num_env_runners)
        if args.num_env_runners is not None
        else auto_env_runners
    )

    logger.info(
        "Throughput autoscaling resolved: CPUs=%s GPUs=%s, num_learners=%s, "
        "cpus_per_learner=%s, gpus_per_learner=%s, num_env_runners=%s, "
        "cpus_per_env_runner=%s",
        node_cpus,
        node_gpus,
        num_learners,
        num_cpus_per_learner,
        num_gpus_per_learner,
        num_env_runners,
        cpus_per_env_runner,
    )

    config = config.learners(
        num_learners=num_learners,
        num_cpus_per_learner=num_cpus_per_learner,
        num_gpus_per_learner=num_gpus_per_learner,
    )
    config = config.env_runners(
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=cpus_per_env_runner,
        create_local_env_runner=False,
        create_env_on_local_worker=False,
    )
    return config


def run_rllib_throughput(
    base_config,
    args: Optional[argparse.Namespace] = None,
    *,
    stop: Optional[Dict] = None,
    keep_config: bool = False,
    keep_ray_up: bool = False,
):
    """Run RLlib training directly on Ray workers (no shared-memory realtime path)."""
    if args is None:
        parser = custom_args()
        args = parser.parse_args()

    seed = getattr(args, "seed", None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    logger = logging.getLogger("MyRLApp.run_algorithm_throughput")
    logger.info("run_rllib_throughput, PID=%s", os.getpid())

    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}},
    )

    try:
        config = base_config
        if not keep_config:
            config = config.framework(args.framework)
            if args.log_level is not None:
                config = config.debugging(log_level=args.log_level)
            if args.output is not None:
                config = config.offline_data(output=args.output)
            config = _auto_configure_throughput_resources(config, args, logger)

        if stop is None:
            stop = {
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
                f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": args.stop_timesteps,
                TRAINING_ITERATION: args.stop_iters,
            }

        algo = config.build()
        try:
            max_iters = int(stop.get(TRAINING_ITERATION, args.stop_iters or 200))
            for train_iter in range(max_iters):
                results = algo.train()
                mean_return = np.nan
                if ENV_RUNNER_RESULTS in results:
                    mean_return = results[ENV_RUNNER_RESULTS].get(EPISODE_RETURN_MEAN, np.nan)
                logger.info("throughput iter=%s return_mean=%s", train_iter, mean_return)
        finally:
            algo.stop()
    finally:
        if not keep_ray_up:
            ray.shutdown()
