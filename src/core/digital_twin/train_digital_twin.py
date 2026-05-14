import argparse
import os
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from core.digital_twin.architectures import MLP, MSEWithDp, ResidualMLP
from core.digital_twin.datasets import InMemoryRowDataset
from core.training import distributed as dist_utils
from core.training.hpo import HPOGeneral
from core.training.loaders import create_dataloaders
from core.training.trainer import Trainer, resolve_device


def _history_curve(history_rows, key):
    return [float(row[key]) for row in history_rows if key in row]


def _set_global_seed(seed: int, rank: int = 0) -> int:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
        )
    raise ValueError(f"Unsupported architecture '{architecture}'.")


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
    beta=0.0,
    per_epoch_validation=True,
    seed=None,
):
    device = resolve_device(device_name)
    if distributed:
        dist_utils.init_process_group(method)

    rank = dist_utils.get_rank() if distributed else 0
    world_size = dist_utils.get_size() if distributed else 1
    effective_seed = None
    if seed is not None:
        effective_seed = _set_global_seed(seed, rank=rank)
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
        validation_batch_size=1,
        seed=effective_seed,
    )
    data_sample, label_sample = next(iter(train_loader))
    data_shape = int(data_sample.shape[-1])
    label_shape = int(label_sample.shape[-1])

    if rank == 0:
        print(f"Using device: {device}")
        print(f"Train rows: {train_size}")
        print(f"Validation rows: {validation_size}")
        print(f"Data shape: {data_shape}, label shape: {label_shape}")

    mse = nn.MSELoss(reduction="mean")
    mae = nn.L1Loss()
    criterion = MSEWithDp(alpha=alpha, beta=beta)

    training_iters = max(1, hpo_iters)
    hpo_logger = None
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "models"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_step = 10
    scheduler_gamma = 0.5

    for hpo_iter in range(training_iters):
        if hpo_iters:
            if hpo_logger is None:
                hpo_logger = HPOGeneral(
                    param_configs={
                        "num_layers": {"type": "int", "low": 2, "high": 10},
                        "layer_exp": {"type": "int", "low": 7, "high": 10},
                        "dropout": {"type": "int", "low": 0, "high": 3},
                        "learning_rate": {
                            "type": "float",
                            "low": 1e-6,
                            "high": 1e-3,
                            "scale": "log",
                        },
                    },
                    metrics=[
                        "mse_dp",
                        "mse",
                        "mae",
                        "mse_dp_epoch_train",
                        "mse_epoch_train",
                        "mae_epoch_train",
                        "mse_dp_epoch_val",
                        "mse_epoch_val",
                        "mae_epoch_val",
                    ],
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
                    "random_seed": effective_seed,
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
                            "random_seed": effective_seed,
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
            hpo_logger.save_log(str(output_dir / "hpo_log.parquet"))

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
    parser.add_argument("--beta", default=0.0, type=float, help="MSEWithDp beta")
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
        beta=args.beta,
        per_epoch_validation=args.per_epoch_validation,
        seed=args.seed,
    )
