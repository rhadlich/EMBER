from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from core.safety.checkpoint import (
    model_config_to_filter_spec,
    resolve_training_output_dir,
    save_filter_checkpoint,
)
from core.safety.datasets import SafetyInMemoryRowDataset, peek_hdf5
from core.safety.safety_filter import StatePredictor
from core.training import distributed as dist_utils
from core.training.hpo import (
    HPOGeneral,
    RayTunePruningConfig,
    build_asha_scheduler,
    build_combined_stopper,
    build_tune_search_space,
    flatten_tuner_result_grid,
)
from core.training.loaders import create_dataloaders
from core.training.trainer import Trainer, resolve_device


def _save_training_checkpoint(
    output_dir: Path,
    *,
    state_dict: dict,
    model_config: dict,
    random_seed: int | None,
) -> None:
    filter_spec = model_config_to_filter_spec(model_config)
    weights_path, spec_path = save_filter_checkpoint(
        output_dir,
        state_dict=state_dict,
        filter_spec=filter_spec,
        random_seed=random_seed,
    )
    legacy_path = output_dir / "model_weights_filter_new.pth"
    torch.save(
        {
            "model_state_dict": state_dict,
            "model_config": model_config,
            "random_seed": random_seed,
        },
        legacy_path,
    )
    print(f"Saved runtime filter checkpoint to {weights_path} and {spec_path}")
    print(f"Saved legacy evaluation checkpoint to {legacy_path}")


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


def _default_hpo_param_configs():
    return {
        "num_hidden": {"type": "int", "low": 1, "high": 6},
        "hidden_exp": {"type": "int", "low": 5, "high": 10},
        "dropout": {"type": "int", "low": 0, "high": 3},
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-2,
            "scale": "log",
        },
    }


def _build_predictor(
    state_dim: int,
    action_dim: int,
    output_dim: int,
    num_hidden: int,
    hidden_exp: int,
    dropout: float,
) -> StatePredictor:
    return StatePredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        num_hidden=num_hidden,
        hidden_exp=hidden_exp,
        dropout=dropout,
        output_dim=output_dim,
    )


class StatePredictorTrainAdapter(nn.Module):
    def __init__(self, predictor: StatePredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, source):
        state, action = source
        x_next, _, _ = self.predictor(state, action)
        return x_next


def _ray_trials_to_safety_hpo_logger(
    *,
    trial_results: list[dict[str, Any]],
    param_configs: dict[str, dict[str, Any]],
    seed: Optional[int],
) -> HPOGeneral:
    metrics = [
        "loss",
        "mse",
        "mae",
        "loss_epoch_train",
        "mse_epoch_train",
        "mae_epoch_train",
        "loss_epoch_val",
        "mse_epoch_val",
        "mae_epoch_val",
    ]
    logger = HPOGeneral(param_configs=param_configs, metrics=metrics, seed=seed)
    max_epochs = 0
    for trial in trial_results:
        max_epochs = max(max_epochs, len(trial.get("train_history", [])))

    for trial in trial_results:
        params = trial["params"]
        for name in param_configs:
            logger.logger["hyperparameter"][name].append(params[name])

        train_history = trial.get("train_history", [])
        val_history = trial.get("val_history", [])
        train_loss = _history_curve(train_history, "loss")
        train_mse = _history_curve(train_history, "mse")
        train_mae = _history_curve(train_history, "mae")
        val_loss = _history_curve(val_history, "loss")
        val_mse = _history_curve(val_history, "mse")
        val_mae = _history_curve(val_history, "mae")

        logger.logger["performance"]["loss"].append(float(val_loss[-1]) if val_loss else float("nan"))
        logger.logger["performance"]["mse"].append(float(val_mse[-1]) if val_mse else float("nan"))
        logger.logger["performance"]["mae"].append(float(val_mae[-1]) if val_mae else float("nan"))

        def _pad(values):
            arr = np.full((max_epochs,), np.nan, dtype=float)
            n = min(len(values), max_epochs)
            if n > 0:
                arr[:n] = np.asarray(values[:n], dtype=float)
            return arr

        logger.logger["performance"]["loss_epoch_train"].append(_pad(train_loss))
        logger.logger["performance"]["mse_epoch_train"].append(_pad(train_mse))
        logger.logger["performance"]["mae_epoch_train"].append(_pad(train_mae))
        logger.logger["performance"]["loss_epoch_val"].append(_pad(val_loss))
        logger.logger["performance"]["mse_epoch_val"].append(_pad(val_mse))
        logger.logger["performance"]["mae_epoch_val"].append(_pad(val_mae))

    return logger


def main(
    root_dir: str,
    output_path: str,
    total_epochs: int,
    batch_size: int,
    num_hidden: int,
    hidden_exp: int,
    dropout: float,
    learning_rate: float,
    device_name: str,
    method: str,
    distributed: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    per_epoch_validation: bool,
    seed: Optional[int],
    n_seeds: int = 1,
    use_amp: Optional[bool] = None,
    strict_reproducibility: bool = False,
    validation_batch_size: int = 256,
    n_trials: int = 1,
    hpo_iters: int = 0,
    hpo_backend: str = "legacy",
    ray_asha_grace_period: int = 3,
    ray_asha_reduction_factor: float = 2.0,
    ray_plateau_patience: int = 6,
    ray_plateau_min_delta: float = 1e-4,
    ray_overfit_ratio_threshold: float = 0.25,
    ray_overfit_patience: int = 3,
    ray_cpus_per_trial: float = 1.0,
    ray_gpus_per_trial: Optional[float] = None,
):
    if n_seeds < 1:
        raise ValueError("--n_seeds must be >= 1.")
    if hpo_iters != 0 and n_seeds != 1:
        raise ValueError("--n_seeds is only supported when --hpo_iters == 0.")
    if hpo_iters == 0 and n_seeds > 1 and seed is None:
        raise ValueError("--n_seeds > 1 requires an explicit --seed value.")

    device = resolve_device(device_name)
    output_dir = resolve_training_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    train_data_dir = str(Path(root_dir) / "train")
    validation_data_dir = str(Path(root_dir) / "validation")

    train_state_dim, train_action_dim, train_output_dim = peek_hdf5(train_data_dir)
    val_state_dim, val_action_dim, val_output_dim = peek_hdf5(validation_data_dir)
    if (
        train_state_dim != val_state_dim
        or train_action_dim != val_action_dim
        or train_output_dim != val_output_dim
    ):
        raise ValueError(
            "Train/validation dimension mismatch: "
            f"train=(state={train_state_dim}, action={train_action_dim}, output={train_output_dim}), "
            f"validation=(state={val_state_dim}, action={val_action_dim}, output={val_output_dim})"
        )
    state_dim, action_dim, output_dim = train_state_dim, train_action_dim, train_output_dim

    if rank == 0:
        print(
            "Inferred dimensions from dataset: "
            f"state_dim={state_dim}, action_dim={action_dim}, output_dim={output_dim}"
        )

    train_data = SafetyInMemoryRowDataset(
        train_data_dir,
        allow_uneven_distribution=False,
        shuffle=True,
        size=1,
        rank=0,
    )
    val_data = SafetyInMemoryRowDataset(
        validation_data_dir,
        allow_uneven_distribution=True,
        shuffle=False,
        size=world_size,
        rank=rank,
    )

    train_loader, train_size, val_loader, val_size = create_dataloaders(
        train_dataset=train_data,
        validation_dataset=val_data,
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
    if rank == 0:
        print(f"Using device: {device}")
        print(f"AMP fp16: {'enabled' if use_amp else 'disabled'}")
        print(f"Strict reproducibility: {'on' if strict_reproducibility else 'off'}")
        print(f"Train rows: {train_size}")
        print(f"Validation rows: {val_size}")

    criterion = nn.MSELoss()
    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()
    param_configs = _default_hpo_param_configs()
    training_iters = max(1, hpo_iters)
    scheduler_step = max(1, total_epochs // 3)
    scheduler_gamma = 0.5
    hpo_logger = None

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

            local_train_loader, _, local_val_loader, _ = create_dataloaders(
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

            trial_num_hidden = int(config["num_hidden"])
            trial_hidden_exp = int(config["hidden_exp"])
            trial_dropout = float(config["dropout"]) * 0.1
            trial_lr = float(config["learning_rate"])

            local_predictor = _build_predictor(
                state_dim=state_dim,
                action_dim=action_dim,
                output_dim=output_dim,
                num_hidden=trial_num_hidden,
                hidden_exp=trial_hidden_exp,
                dropout=trial_dropout,
            ).to(device)
            local_model = StatePredictorTrainAdapter(local_predictor).to(device)
            local_optimizer = optim.AdamW(local_model.parameters(), lr=trial_lr * batch_size / 64)
            local_scheduler = optim.lr_scheduler.StepLR(
                local_optimizer, step_size=scheduler_step, gamma=scheduler_gamma
            )
            trainer = Trainer(
                model=local_model,
                train_data=local_train_loader,
                val_data=local_val_loader,
                optimizer=local_optimizer,
                scheduler=local_scheduler,
                criterion=criterion,
                metric_fns={"mse": mse_metric, "mae": mae_metric},
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
                tune.with_parameters(_trainable, train_dataset=train_data, validation_dataset=val_data),
                resources={"cpu": float(ray_cpus_per_trial), "gpu": float(ray_gpus_per_trial)},
            ),
            tune_config=tune.TuneConfig(num_samples=training_iters, scheduler=scheduler),
            run_config=tune.RunConfig(
                name="safety_filter_hpo",
                storage_path=str(output_dir / "ray_results"),
                stop=stopper,
                verbose=1,
            ),
            param_space=search_space,
        )
        result_grid = tuner.fit()
        trial_results = flatten_tuner_result_grid(result_grid)
        ray_hpo_logger = _ray_trials_to_safety_hpo_logger(
            trial_results=trial_results,
            param_configs=param_configs,
            seed=effective_seed,
        )
        if rank == 0:
            ray_hpo_logger.save_log(ray_hpo_logger.build_log_path(output_dir))
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

        train_loader, _, val_loader, _ = create_dataloaders(
            train_dataset=train_data,
            validation_dataset=val_data,
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
            trial_num_hidden = int(num_hidden)
            trial_hidden_exp = int(hidden_exp)
            trial_dropout = float(dropout)
            trial_learning_rate = float(learning_rate)

            if hpo_iters:
                if hpo_logger is None:
                    hpo_logger = HPOGeneral(
                        param_configs=param_configs,
                        metrics=[
                            "loss",
                            "mse",
                            "mae",
                            "loss_epoch_train",
                            "mse_epoch_train",
                            "mae_epoch_train",
                            "loss_epoch_val",
                            "mse_epoch_val",
                            "mae_epoch_val",
                        ],
                        seed=effective_seed,
                    )
                sample = hpo_logger.sample()
                trial_num_hidden = int(sample["num_hidden"])
                trial_hidden_exp = int(sample["hidden_exp"])
                trial_dropout = float(sample["dropout"]) * 0.1
                trial_learning_rate = float(sample["learning_rate"])
                if rank == 0:
                    print(f"HPO sample {hpo_iter}: {sample}")

            predictor = _build_predictor(
                state_dim=state_dim,
                action_dim=action_dim,
                output_dim=output_dim,
                num_hidden=trial_num_hidden,
                hidden_exp=trial_hidden_exp,
                dropout=trial_dropout,
            ).to(device)
            model = StatePredictorTrainAdapter(predictor).to(device)
            if distributed:
                model = DDP(model, device_ids=None, output_device=None)

            filename_model = output_dir / "model_weights_filter_new.pth"
            if rank == 0:
                base_model = model.module if distributed else model
                model_config = {
                    "state_dim": state_dim,
                    "output_dim": output_dim,
                    "action_dim": action_dim,
                    "num_hidden": trial_num_hidden,
                    "hidden_exp": trial_hidden_exp,
                    "dropout": trial_dropout,
                }
                _save_training_checkpoint(
                    output_dir,
                    state_dict=base_model.state_dict(),
                    model_config=model_config,
                    random_seed=run_effective_seed,
                )

            optimizer_lr = trial_learning_rate * world_size if distributed else trial_learning_rate
            optimizer_lr *= batch_size / 64
            optimizer = optim.AdamW(model.parameters(), lr=optimizer_lr)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_step, gamma=scheduler_gamma
            )
            filename_optimizer = output_dir / f"optimizer_weights_filter.pth"
            if rank == 0:
                torch.save(optimizer.state_dict(), filename_optimizer)
            if distributed:
                dist.barrier()

            loss_store = np.zeros([n_trials], dtype=float)
            mse_store = np.zeros([n_trials], dtype=float)
            mae_store = np.zeros([n_trials], dtype=float)
            loss_epoch_train_store = []
            mse_epoch_train_store = []
            mae_epoch_train_store = []
            loss_epoch_val_store = []
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
                    val_data=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    metric_fns={"mse": mse_metric, "mae": mae_metric},
                    train_method="default",
                    validate_each_epoch=per_epoch_validation,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    use_amp=use_amp,
                )
                trainer.train(total_epochs)

                val_loss_scalar = float(trainer.val_loss.detach().cpu().item())
                val_mse_scalar = float(trainer.val_metrics["mse"].detach().cpu().item())
                val_mae_scalar = float(trainer.val_metrics["mae"].detach().cpu().item())
                loss_store[trial_idx] = val_loss_scalar
                mse_store[trial_idx] = val_mse_scalar
                mae_store[trial_idx] = val_mae_scalar

                train_history = trainer.history.get("train", [])
                val_history = trainer.history.get("val", [])
                loss_epoch_train = _history_curve(train_history, "loss")
                mse_epoch_train = _history_curve(train_history, "mse")
                mae_epoch_train = _history_curve(train_history, "mae")
                loss_epoch_val = _history_curve(val_history, "loss")
                mse_epoch_val = _history_curve(val_history, "mse")
                mae_epoch_val = _history_curve(val_history, "mae")
                loss_epoch_train_store.append(loss_epoch_train)
                mse_epoch_train_store.append(mse_epoch_train)
                mae_epoch_train_store.append(mae_epoch_train)
                loss_epoch_val_store.append(loss_epoch_val)
                mse_epoch_val_store.append(mse_epoch_val)
                mae_epoch_val_store.append(mae_epoch_val)

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
                        f"HPO iter={hpo_iter} trial={trial_idx} "
                        f"val_loss={val_loss_scalar:.6e} world_size={world_size}"
                    )
                    if not hpo_iters:
                        base_model = model.module if distributed else model
                        model_config = {
                            "state_dim": state_dim,
                            "output_dim": output_dim,
                            "action_dim": action_dim,
                            "num_hidden": trial_num_hidden,
                            "hidden_exp": trial_hidden_exp,
                            "dropout": trial_dropout,
                        }
                        _save_training_checkpoint(
                            output_dir,
                            state_dict=base_model.state_dict(),
                            model_config=model_config,
                            random_seed=run_effective_seed,
                        )

                if distributed:
                    dist.barrier()

            if hpo_logger is not None:
                hpo_logger.log_performance(loss_store, metric="loss")
                hpo_logger.log_performance(mse_store, metric="mse")
                hpo_logger.log_performance(mae_store, metric="mae")
                hpo_logger.log_performance(
                    np.asarray(loss_epoch_train_store, dtype=float), metric="loss_epoch_train"
                )
                hpo_logger.log_performance(
                    np.asarray(mse_epoch_train_store, dtype=float), metric="mse_epoch_train"
                )
                hpo_logger.log_performance(
                    np.asarray(mae_epoch_train_store, dtype=float), metric="mae_epoch_train"
                )
                hpo_logger.log_performance(
                    np.asarray(loss_epoch_val_store, dtype=float), metric="loss_epoch_val"
                )
                hpo_logger.log_performance(
                    np.asarray(mse_epoch_val_store, dtype=float), metric="mse_epoch_val"
                )
                hpo_logger.log_performance(
                    np.asarray(mae_epoch_val_store, dtype=float), metric="mae_epoch_val"
                )
                if rank == 0:
                    hpo_logger.save_log(hpo_logger.build_log_path(output_dir))

    if record_seed_progression and rank == 0 and seed_progression_rows:
        seed_progression_path = output_dir / "seed_progression_safety_filter.parquet"
        pd.DataFrame(seed_progression_rows).to_parquet(seed_progression_path, index=False)
        print(f"Wrote seed progression log to {seed_progression_path}")
    if record_seed_progression and distributed:
        dist.barrier()

    if distributed and dist.is_initialized():
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StatePredictor with shared Trainer")
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing train/validation safety HDF5 folders.",
    )
    parser.add_argument(
        "--output_path",
        default=str(Path(__file__).resolve().parent / "models"),
        type=str,
        help="Output directory for filter.pt, filter_spec.json, and legacy .pth checkpoints",
    )
    parser.add_argument("--total_epochs", default=20, type=int, help="Training epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--num_hidden", default=2, type=int, help="Hidden block count")
    parser.add_argument("--hidden_exp", default=7, type=int, help="Width exponent")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--n_trials", default=1, type=int, help="Number of consecutive trials")
    parser.add_argument("--hpo_iters", default=0, type=int, help="Number of HPO samples")
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
        "--hpo-backend",
        default="legacy",
        choices=["legacy", "ray"],
        type=str,
        help="HPO backend. 'legacy' keeps random search, 'ray' enables Tune + pruning.",
    )
    parser.add_argument("--method", default="dummy", type=str, help="Distributed init method")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep workers alive")
    parser.add_argument("--prefetch_factor", default=2, type=int, help="Prefetch factor")
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
        "--per-epoch-validation",
        dest="per_epoch_validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run and print validation metrics each epoch (default: enabled).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device",
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
        root_dir=args.root_dir,
        output_path=args.output_path,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        num_hidden=args.num_hidden,
        hidden_exp=args.hidden_exp,
        dropout=args.dropout,
        learning_rate=args.lr,
        device_name=args.device,
        method=args.method,
        distributed=args.distributed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        per_epoch_validation=args.per_epoch_validation,
        seed=args.seed,
        n_seeds=args.n_seeds,
        use_amp=args.use_amp,
        strict_reproducibility=args.strict_reproducibility,
        validation_batch_size=args.validation_batch_size,
        n_trials=args.n_trials,
        hpo_iters=args.hpo_iters,
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
