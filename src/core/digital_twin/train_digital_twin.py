import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from core.digital_twin.architectures import MLP, MSEWithDp, ResidualMLP
from core.digital_twin.datasets import InMemoryRowDataset
from core.training import distributed as dist_utils
from core.training.hpo import (
    HPOGeneral,
    RayTunePruningConfig,
    build_asha_scheduler,
    build_combined_stopper,
    build_tune_search_space,
    flatten_tuner_result_grid,
    ray_trials_to_hpo_logger,
)
from core.training.loaders import create_dataloaders
from core.training.trainer import Trainer, resolve_device


def _history_curve(history_rows, key):
    return [float(row[key]) for row in history_rows if key in row]


def _build_seed_progression_rows(seed_idx, seed_value, trial_idx, train_history, val_history):
    rows = []
    max_epochs = max(len(train_history), len(val_history))
    for epoch_idx in range(max_epochs):
        train_row = train_history[epoch_idx] if epoch_idx < len(train_history) else {}
        val_row = val_history[epoch_idx] if epoch_idx < len(val_history) else {}
        rows.append(
            {
                "seed_idx": int(seed_idx),
                "seed": int(seed_value),
                "trial_idx": int(trial_idx),
                "epoch": int(epoch_idx),
                "train_mse": float(train_row["mse"]) if "mse" in train_row else np.nan,
                "train_mae": float(train_row["mae"]) if "mae" in train_row else np.nan,
                "val_mse": float(val_row["mse"]) if "mse" in val_row else np.nan,
                "val_mae": float(val_row["mae"]) if "mae" in val_row else np.nan,
            }
        )
    return rows


def _set_global_seed(seed: int, rank: int = 0, strict_reproducibility: bool = False) -> int:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
        if strict_reproducibility:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    return effective_seed


def _build_model(
    architecture: Literal["mlp", "residual_mlp"],
    data_shape: int,
    label_shape: int,
    num_layers: int,
    layer_exp: int,
    dropout: float,
):
    if architecture == "mlp":
        return MLP(
            input_dim=data_shape,
            output_dim=label_shape,
            num_hidden=num_layers,
            hidden_exp=layer_exp,
            dropout=dropout,
        )
    if architecture == "residual_mlp":
        return ResidualMLP(
            input_dim=data_shape,
            output_dim=label_shape,
            num_blocks=num_layers,
            hidden_exp=layer_exp,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported architecture '{architecture}'.")


def _default_hpo_param_configs():
    return {
        "num_layers": {"type": "int", "low": 1, "high": 10},
        "layer_exp": {"type": "int", "low": 7, "high": 12},
        "dropout": {"type": "int", "low": 0, "high": 3},
        "learning_rate": {
            "type": "float",
            "low": 1e-6,
            "high": 1e-3,
            "scale": "log",
        },
    }


def main(
    total_epochs,
    root_dir,
    node_type,
    architecture,
    method,
    num_layers,
    layer_exp,
    learning_rate,
    batch_size,
    dropout,
    n_trials,
    hpo_iters,
    distributed=False,
    output_dir=None,
    device_name="auto",
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    alpha=0.5,
    per_epoch_validation=True,
    seed=None,
    n_seeds=1,
    use_amp=None,
    strict_reproducibility=False,
    validation_batch_size=256,
    hpo_backend="legacy",
    ray_asha_grace_period=3,
    ray_asha_reduction_factor=2.0,
    ray_plateau_patience=6,
    ray_plateau_min_delta=1e-4,
    ray_overfit_ratio_threshold=0.25,
    ray_overfit_patience=3,
    ray_cpus_per_trial=1.0,
    ray_gpus_per_trial=None,
):
    if n_seeds < 1:
        raise ValueError("--n_seeds must be >= 1.")
    if hpo_iters != 0 and n_seeds != 1:
        raise ValueError("--n_seeds is only supported when --hpo_iters == 0.")
    if hpo_iters == 0 and n_seeds > 1 and seed is None:
        raise ValueError("--n_seeds > 1 requires an explicit --seed value.")

    device = resolve_device(device_name)
    if distributed:
        dist_utils.init_process_group(method)

    rank = dist_utils.get_rank() if distributed else 0
    world_size = dist_utils.get_size() if distributed else 1

    if device.type == "cuda" and not strict_reproducibility:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if use_amp is None:
        use_amp = device.type == "cuda"
    use_amp = bool(use_amp) and device.type == "cuda"

    effective_seed = None
    if seed is not None:
        effective_seed = _set_global_seed(
            seed, rank=rank, strict_reproducibility=strict_reproducibility
        )
        if rank == 0:
            print(
                f"Random seed set to {seed}"
                + (f" (effective rank-0 seed {effective_seed})" if distributed else "")
            )

    train_dataset = InMemoryRowDataset(
        os.path.join(root_dir, "train"),
        allow_uneven_distribution=False,
        shuffle=True,
        size=1,
        rank=0,
    )
    validation_dataset = InMemoryRowDataset(
        os.path.join(root_dir, "validation"),
        allow_uneven_distribution=True,
        shuffle=False,
        size=world_size,
        rank=rank,
    )

    train_loader, train_size, validation_loader, validation_size = create_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=batch_size,
        size=world_size,
        rank=rank,
        distributed=distributed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        train_drop_last=True,
        validation_batch_size=validation_batch_size,
        seed=effective_seed,
    )
    data_sample, label_sample = next(iter(train_loader))
    data_shape = int(data_sample.shape[-1])
    label_shape = int(label_sample.shape[-1])

    if rank == 0:
        print(f"Using device: {device}")
        print(f"AMP fp16: {'enabled' if use_amp else 'disabled'}")
        print(f"Strict reproducibility: {'on' if strict_reproducibility else 'off'}")
        print(f"Validation batch size: {validation_batch_size}")
        print(f"Train rows: {train_size}")
        print(f"Validation rows: {validation_size}")
        print(f"Data shape: {data_shape}, label shape: {label_shape}")

    mse = nn.MSELoss(reduction="mean")
    mae = nn.L1Loss()
    criterion = MSEWithDp(alpha=alpha)

    training_iters = max(1, hpo_iters)
    hpo_logger = None
    param_configs = _default_hpo_param_configs()
    hpo_metrics = [
        "mse_dp",
        "mse",
        "mae",
        "mse_dp_epoch_train",
        "mse_epoch_train",
        "mae_epoch_train",
        "mse_dp_epoch_val",
        "mse_epoch_val",
        "mae_epoch_val",
    ]
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "models"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_step = 10
    scheduler_gamma = 0.5

    if hpo_backend not in {"legacy", "ray"}:
        raise ValueError(f"Unsupported hpo_backend '{hpo_backend}'. Use 'legacy' or 'ray'.")

    if hpo_backend == "ray":
        if distributed:
            raise ValueError("Ray backend currently supports only non-distributed training.")
        if hpo_iters <= 0:
            raise ValueError("Ray backend requires --hpo_iters > 0 (number of sampled configs).")
        if not per_epoch_validation:
            raise ValueError("Ray backend requires --per-epoch-validation enabled for pruning.")
        if n_trials != 1:
            raise ValueError("Ray backend currently requires --n_trials=1.")

        from ray import tune

        if ray_gpus_per_trial is None:
            ray_gpus_per_trial = 1.0 if device.type == "cuda" else 0.0

        pruning_cfg = RayTunePruningConfig(
            grace_period=ray_asha_grace_period,
            reduction_factor=ray_asha_reduction_factor,
            plateau_patience=ray_plateau_patience,
            plateau_min_delta=ray_plateau_min_delta,
            overfit_ratio_threshold=ray_overfit_ratio_threshold,
            overfit_patience=ray_overfit_patience,
        )
        scheduler = build_asha_scheduler(
            metric="val_loss",
            mode="min",
            max_t=total_epochs,
            grace_period=pruning_cfg.grace_period,
            reduction_factor=pruning_cfg.reduction_factor,
        )
        stopper = build_combined_stopper(pruning_cfg, val_metric="val_loss", train_metric="train_loss")
        search_space = build_tune_search_space(param_configs)

        def _trainable(config, train_dataset, validation_dataset):
            trial_seed = int(effective_seed if effective_seed is not None else 0)
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)
            local_train_loader, _, local_validation_loader, _ = create_dataloaders(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                batch_size=batch_size,
                size=1,
                rank=0,
                distributed=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                train_drop_last=True,
                validation_batch_size=validation_batch_size,
                seed=trial_seed,
            )
            local_data_sample, local_label_sample = next(iter(local_train_loader))
            local_data_shape = int(local_data_sample.shape[-1])
            local_label_shape = int(local_label_sample.shape[-1])

            trial_num_layers = int(config["num_layers"])
            trial_layer_exp = int(config["layer_exp"])
            trial_dropout = float(config["dropout"]) * 0.1
            trial_lr = float(config["learning_rate"])

            local_model = _build_model(
                architecture=architecture,
                data_shape=local_data_shape,
                label_shape=local_label_shape,
                num_layers=trial_num_layers,
                layer_exp=trial_layer_exp,
                dropout=trial_dropout,
            ).to(device)
            local_criterion = MSEWithDp(alpha=alpha)
            local_mse = nn.MSELoss(reduction="mean")
            local_mae = nn.L1Loss()
            optimizer_lr = trial_lr * batch_size / 64
            local_optimizer = optim.AdamW(local_model.parameters(), lr=optimizer_lr)
            local_scheduler = optim.lr_scheduler.StepLR(
                local_optimizer, step_size=scheduler_step, gamma=scheduler_gamma
            )
            initial_model_state = copy.deepcopy(local_model.state_dict())
            initial_optimizer_state = copy.deepcopy(local_optimizer.state_dict())
            local_model.load_state_dict(initial_model_state)
            local_optimizer.load_state_dict(initial_optimizer_state)

            trainer = Trainer(
                model=local_model,
                train_data=local_train_loader,
                val_data=local_validation_loader,
                optimizer=local_optimizer,
                scheduler=local_scheduler,
                criterion=local_criterion,
                metric_fns={"mse": local_mse, "mae": local_mae},
                val_fn=None,
                epoch_end_callback=lambda metrics: tune.report(metrics),
                train_method="default",
                validate_each_epoch=True,
                distributed=False,
                rank=0,
                world_size=1,
                device=device,
                use_amp=use_amp,
            )
            trainer.train(total_epochs)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    _trainable,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                ),
                resources={"cpu": float(ray_cpus_per_trial), "gpu": float(ray_gpus_per_trial)},
            ),
            tune_config=tune.TuneConfig(
                num_samples=training_iters,
                scheduler=scheduler,
            ),
            run_config=tune.RunConfig(
                name=f"digital_twin_hpo_{node_type}",
                storage_path=str(output_dir / "ray_results"),
                stop=stopper,
                verbose=1,
            ),
            param_space=search_space,
        )
        result_grid = tuner.fit()
        trial_results = flatten_tuner_result_grid(result_grid)
        ray_hpo_logger = ray_trials_to_hpo_logger(
            trial_results=trial_results, param_configs=param_configs, seed=effective_seed
        )
        ray_hpo_logger.save_log(ray_hpo_logger.build_log_path(output_dir))

        if rank == 0:
            best_result = result_grid.get_best_result(metric="val_loss", mode="min")
            print(f"Best Ray Tune config: {best_result.config}")
            print(f"Best Ray Tune val_loss: {best_result.metrics.get('val_loss')}")

        if distributed and dist.is_initialized():
            destroy_process_group()
        return

    record_seed_progression = hpo_iters == 0 and n_seeds > 1
    seed_schedule = [seed]
    if record_seed_progression:
        seed_schedule = [int(seed) * (seed_idx + 1) for seed_idx in range(n_seeds)]
    seed_progression_rows = []

    for seed_idx, run_seed in enumerate(seed_schedule):
        run_effective_seed = None
        if run_seed is not None:
            run_effective_seed = _set_global_seed(
                int(run_seed), rank=rank, strict_reproducibility=strict_reproducibility
            )
            if rank == 0 and record_seed_progression:
                print(
                    f"Seed sweep {seed_idx + 1}/{len(seed_schedule)}: "
                    f"seed={run_seed}, effective_rank0_seed={run_effective_seed}"
                )

        train_loader, _, validation_loader, _ = create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            size=world_size,
            rank=rank,
            distributed=distributed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            train_drop_last=True,
            validation_batch_size=validation_batch_size,
            seed=run_effective_seed,
        )

        for hpo_iter in range(training_iters):
            if hpo_iters:
                if hpo_logger is None:
                    hpo_logger = HPOGeneral(
                        param_configs=param_configs,
                        metrics=hpo_metrics,
                        seed=effective_seed,
                    )
                sample = hpo_logger.sample()
                num_layers = int(sample["num_layers"])
                layer_exp = int(sample["layer_exp"])
                dropout = float(sample["dropout"]) * 0.1
                learning_rate = float(sample["learning_rate"])
                if rank == 0:
                    print(f"HPO sample {hpo_iter}: {sample}")

            model = _build_model(
                architecture=architecture,
                data_shape=data_shape,
                label_shape=label_shape,
                num_layers=num_layers,
                layer_exp=layer_exp,
                dropout=dropout,
            ).to(device)
            val_fn = None
            train_method = "default"

            if distributed:
                model = DDP(model, device_ids=None, output_device=None)

            filename_model = output_dir / f"model_weights_{node_type}_new.pth"
            if rank == 0:
                base_model = model.module if distributed else model
                torch.save(
                    {
                        "model_state_dict": base_model.state_dict(),
                        "model_config": {
                            "architecture": architecture,
                            "input_dim": data_shape,
                            "output_dim": label_shape,
                            "num_hidden": num_layers,
                            "hidden_exp": layer_exp,
                            "dropout": dropout,
                        },
                        "random_seed": run_effective_seed,
                        "normalization": {
                            "expected_feature_order": ["inj_pressure", "inj_timing", "inj_duration"],
                        },
                    },
                    filename_model,
                )

            optimizer_lr = learning_rate * world_size if distributed else learning_rate
            optimizer_lr *= batch_size / 64
            optimizer = optim.AdamW(model.parameters(), lr=optimizer_lr)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_step, gamma=scheduler_gamma
            )
            filename_optimizer = output_dir / f"optimizer_weights_{node_type}.pth"
            if rank == 0:
                torch.save(optimizer.state_dict(), filename_optimizer)
            if distributed:
                dist.barrier()

            loss_store = np.zeros([n_trials])
            mse_store = np.zeros([n_trials])
            mae_store = np.zeros([n_trials])
            mse_dp_epoch_train_store = []
            mse_epoch_train_store = []
            mae_epoch_train_store = []
            mse_dp_epoch_val_store = []
            mse_epoch_val_store = []
            mae_epoch_val_store = []

            for trial_idx in range(n_trials):
                if distributed:
                    dist.barrier()

                checkpoint = torch.load(filename_model, map_location=device)
                state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
                load_target = model.module if distributed else model
                load_target.load_state_dict(state_dict)
                optimizer.load_state_dict(torch.load(filename_optimizer, map_location=device))
                model.train()

                trainer = Trainer(
                    model=model,
                    train_data=train_loader,
                    val_data=validation_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    metric_fns={"mse": mse, "mae": mae},
                    val_fn=val_fn,
                    train_method=train_method,
                    validate_each_epoch=per_epoch_validation,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    use_amp=use_amp,
                )

                tic = time.time()
                trainer.train(total_epochs)
                toc = time.time()

                loss_store[trial_idx] = float(trainer.val_loss.detach().cpu().numpy())
                mse_store[trial_idx] = float(trainer.val_metrics["mse"].detach().cpu().numpy())
                mae_store[trial_idx] = float(trainer.val_metrics["mae"].detach().cpu().numpy())
                train_history = trainer.history.get("train", [])
                val_history = trainer.history.get("val", [])
                mse_dp_epoch_train_store.append(_history_curve(train_history, "loss"))
                mse_epoch_train_store.append(_history_curve(train_history, "mse"))
                mae_epoch_train_store.append(_history_curve(train_history, "mae"))
                mse_dp_epoch_val_store.append(_history_curve(val_history, "loss"))
                mse_epoch_val_store.append(_history_curve(val_history, "mse"))
                mae_epoch_val_store.append(_history_curve(val_history, "mae"))

                if record_seed_progression:
                    seed_progression_rows.extend(
                        _build_seed_progression_rows(
                            seed_idx=seed_idx,
                            seed_value=run_seed,
                            trial_idx=trial_idx,
                            train_history=train_history,
                            val_history=val_history,
                        )
                    )

                if rank == 0:
                    print(
                        f"HPO iter={hpo_iter} trial={trial_idx} val_loss={trainer.val_loss} "
                        f"time={toc - tic:.2f}s world_size={world_size}"
                    )
                    if not hpo_iters:
                        base_model = model.module if distributed else model
                        torch.save(
                            {
                                "model_state_dict": base_model.state_dict(),
                                "model_config": {
                                    "architecture": architecture,
                                    "input_dim": data_shape,
                                    "output_dim": label_shape,
                                    "num_hidden": num_layers,
                                    "hidden_exp": layer_exp,
                                    "dropout": dropout,
                                },
                                "random_seed": run_effective_seed,
                            },
                            filename_model,
                        )
                if distributed:
                    dist.barrier()

            if hpo_logger is not None:
                hpo_logger.log_performance(loss_store, metric="mse_dp")
                hpo_logger.log_performance(mse_store, metric="mse")
                hpo_logger.log_performance(mae_store, metric="mae")
                hpo_logger.log_performance(
                    np.asarray(mse_dp_epoch_train_store, dtype=float), metric="mse_dp_epoch_train"
                )
                hpo_logger.log_performance(
                    np.asarray(mse_epoch_train_store, dtype=float), metric="mse_epoch_train"
                )
                hpo_logger.log_performance(
                    np.asarray(mae_epoch_train_store, dtype=float), metric="mae_epoch_train"
                )
                hpo_logger.log_performance(
                    np.asarray(mse_dp_epoch_val_store, dtype=float), metric="mse_dp_epoch_val"
                )
                hpo_logger.log_performance(
                    np.asarray(mse_epoch_val_store, dtype=float), metric="mse_epoch_val"
                )
                hpo_logger.log_performance(
                    np.asarray(mae_epoch_val_store, dtype=float), metric="mae_epoch_val"
                )
                hpo_logger.save_log(hpo_logger.build_log_path(output_dir))

    if record_seed_progression and rank == 0 and seed_progression_rows:
        seed_progression_path = output_dir / f"seed_progression_{node_type}.parquet"
        pd.DataFrame(seed_progression_rows).to_parquet(seed_progression_path, index=False)
        print(f"Wrote seed progression log to {seed_progression_path}")
    if record_seed_progression and distributed:
        dist.barrier()

    if distributed and dist.is_initialized():
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train digital twin pressure predictor model")
    parser.add_argument("total_epochs", type=int, help="Total epochs to train the model")
    parser.add_argument("root_dir", type=str, help="Root directory with train/validation datasets")
    parser.add_argument("node_type", type=str, help="Model tag used in checkpoint filename")
    parser.add_argument(
        "--architecture",
        default="mlp",
        choices=["mlp", "residual_mlp"],
        type=str,
        help=(
            "Model architecture. 'mlp' uses --num_layers as hidden-layer count. "
            "'residual_mlp' uses --num_layers as residual-block count (each block has 2 linear layers)."
        ),
    )
    parser.add_argument("--method", default="dummy", type=str, help="Distributed init method")
    parser.add_argument(
        "--num_layers",
        default=4,
        type=int,
        help=(
            "Depth control. For 'mlp': number of hidden layers after the input layer. "
            "For 'residual_mlp': number of residual blocks."
        ),
    )
    parser.add_argument("--num_nodes_exp", default=10, type=int, help="Exponential width factor")
    parser.add_argument("--lr", default=0.0003, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--p", default=0.1, type=float, help="Dropout probability")
    parser.add_argument("--n_trials", default=1, type=int, help="Number of consecutive trials")
    parser.add_argument("--hpo_iters", default=0, type=int, help="Number of HPO samples")
    parser.add_argument(
        "--hpo-backend",
        default="legacy",
        choices=["legacy", "ray"],
        type=str,
        help="HPO backend. 'legacy' keeps random search, 'ray' enables Tune + pruning.",
    )
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "models"),
        type=str,
        help="Directory for checkpoints",
    )
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep workers alive")
    parser.add_argument("--prefetch_factor", default=2, type=int, help="Prefetch factor")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device",
    )
    parser.add_argument("--alpha", default=0.5, type=float, help="MSEWithDp alpha")
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable fp16 automatic mixed precision (CUDA only). "
            "Default: enabled when device is CUDA, otherwise disabled."
        ),
    )
    parser.add_argument(
        "--strict-reproducibility",
        dest="strict_reproducibility",
        action="store_true",
        help=(
            "Force cuDNN deterministic mode and disable the autotuner. "
            "Slower but bitwise reproducible for a given seed."
        ),
    )
    parser.add_argument(
        "--validation_batch_size",
        default=256,
        type=int,
        help="Validation DataLoader batch size (default: 256).",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Global random seed for reproducible and recoverable training runs.",
    )
    parser.add_argument(
        "--n_seeds",
        default=1,
        type=int,
        help=(
            "Number of seed-multiple runs (1 keeps current behavior). "
            "Only supported when --hpo_iters=0."
        ),
    )
    parser.add_argument(
        "--per-epoch-validation",
        dest="per_epoch_validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run and print validation metrics each epoch (default: enabled).",
    )
    parser.add_argument(
        "--ray-asha-grace-period",
        default=3,
        type=int,
        help="Minimum epochs before ASHA can prune a trial.",
    )
    parser.add_argument(
        "--ray-asha-reduction-factor",
        default=2.0,
        type=float,
        help="ASHA reduction factor (eta).",
    )
    parser.add_argument(
        "--ray-plateau-patience",
        default=6,
        type=int,
        help="Epochs without meaningful val-loss improvement before stopping.",
    )
    parser.add_argument(
        "--ray-plateau-min-delta",
        default=1e-4,
        type=float,
        help="Minimum val-loss improvement to reset plateau patience.",
    )
    parser.add_argument(
        "--ray-overfit-ratio-threshold",
        default=0.25,
        type=float,
        help="Overfitting threshold on (val_loss-train_loss)/train_loss.",
    )
    parser.add_argument(
        "--ray-overfit-patience",
        default=3,
        type=int,
        help="Consecutive epochs over overfit threshold before stopping.",
    )
    parser.add_argument(
        "--ray-cpus-per-trial",
        default=1.0,
        type=float,
        help="CPU resources allocated to each Ray Tune trial.",
    )
    parser.add_argument(
        "--ray-gpus-per-trial",
        default=None,
        type=float,
        help="GPU resources per Ray trial (default auto: 1 on CUDA, else 0).",
    )
    args = parser.parse_args()

    main(
        args.total_epochs,
        args.root_dir,
        args.node_type,
        args.architecture,
        args.method,
        args.num_layers,
        args.num_nodes_exp,
        args.lr,
        args.batch_size,
        args.p,
        args.n_trials,
        args.hpo_iters,
        distributed=args.distributed,
        output_dir=args.output_dir,
        device_name=args.device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        alpha=args.alpha,
        per_epoch_validation=args.per_epoch_validation,
        seed=args.seed,
        n_seeds=args.n_seeds,
        use_amp=args.use_amp,
        strict_reproducibility=args.strict_reproducibility,
        validation_batch_size=args.validation_batch_size,
        hpo_backend=args.hpo_backend,
        ray_asha_grace_period=args.ray_asha_grace_period,
        ray_asha_reduction_factor=args.ray_asha_reduction_factor,
        ray_plateau_patience=args.ray_plateau_patience,
        ray_plateau_min_delta=args.ray_plateau_min_delta,
        ray_overfit_ratio_threshold=args.ray_overfit_ratio_threshold,
        ray_overfit_patience=args.ray_overfit_patience,
        ray_cpus_per_trial=args.ray_cpus_per_trial,
        ray_gpus_per_trial=args.ray_gpus_per_trial,
    )
