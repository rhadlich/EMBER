import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

from core.safety.safety_filter import StatePredictor
from core.safety.datasets import SafetyInMemoryRowDataset
from core.training import distributed as dist_utils
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


class StatePredictorTrainAdapter(nn.Module):
    def __init__(self, predictor: StatePredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, source):
        state, action = source
        x_next, _, _ = self.predictor(state, action)
        return x_next


def main(
    dataset_path: str,
    output_path: str,
    total_epochs: int,
    batch_size: int,
    state_dim: int,
    action_dim: int,
    num_hidden: int,
    hidden_exp: int,
    dropout: float,
    learning_rate: float,
    train_ratio: float,
    device_name: str,
    method: str,
    distributed: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    per_epoch_validation: bool,
    seed: int | None,
):
    device = resolve_device(device_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    dataset = SafetyInMemoryRowDataset(
        dataset_path,
        allow_uneven_distribution=True,
        size=1,
        rank=0,
    )
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    if train_len == 0 or val_len == 0:
        raise ValueError("Dataset split produced empty train or validation split.")

    split_generator = None
    if effective_seed is not None:
        split_generator = torch.Generator().manual_seed(effective_seed)
    train_data, val_data = random_split(
        dataset, [train_len, val_len], generator=split_generator
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
        validation_batch_size=batch_size,
        seed=effective_seed,
    )

    predictor = StatePredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        num_hidden=num_hidden,
        hidden_exp=hidden_exp,
        dropout=dropout,
    ).to(device)
    model = StatePredictorTrainAdapter(predictor).to(device)
    if distributed:
        model = DDP(model, device_ids=None, output_device=None)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 3), gamma=0.5)
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metric_fns={"mse": nn.MSELoss(), "mae": nn.L1Loss()},
        validate_each_epoch=per_epoch_validation,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    trainer.train(total_epochs)
    train_history = trainer.history.get("train", [])
    val_history = trainer.history.get("val", [])
    mse_epoch_train = _history_curve(train_history, "mse")
    mae_epoch_train = _history_curve(train_history, "mae")
    mse_epoch_val = _history_curve(val_history, "mse")
    mae_epoch_val = _history_curve(val_history, "mae")

    if rank == 0:
        base_predictor = model.module.predictor if distributed else predictor
        torch.save(
            {
                "model_state_dict": base_predictor.state_dict(),
                "model_config": {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "num_hidden": num_hidden,
                    "hidden_exp": hidden_exp,
                    "dropout": dropout,
                },
                "dataset_path": dataset_path,
                "random_seed": effective_seed,
                "training_history": {
                    "mse_epoch_train": mse_epoch_train,
                    "mae_epoch_train": mae_epoch_train,
                    "mse_epoch_val": mse_epoch_val,
                    "mae_epoch_val": mae_epoch_val,
                },
            },
            str(output_path),
        )
        print(f"Saved StatePredictor checkpoint to {output_path}")

    if distributed and torch.distributed.is_initialized():
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StatePredictor with shared Trainer")
    parser.add_argument("dataset_path", type=str, help="Path to .npz with states/actions/next_states arrays")
    parser.add_argument(
        "--output_path",
        default=str(Path(__file__).resolve().parent / "models" / "filter.pt"),
        type=str,
        help="Checkpoint path",
    )
    parser.add_argument("--total_epochs", default=20, type=int, help="Training epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--state_dim", required=True, type=int, help="State dimension")
    parser.add_argument("--action_dim", required=True, type=int, help="Action dimension")
    parser.add_argument("--num_hidden", default=2, type=int, help="Hidden block count")
    parser.add_argument("--hidden_exp", default=7, type=int, help="Width exponent")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--train_ratio", default=0.9, type=float, help="Train split ratio")
    parser.add_argument("--method", default="dummy", type=str, help="Distributed init method")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep workers alive")
    parser.add_argument("--prefetch_factor", default=2, type=int, help="Prefetch factor")
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
    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_hidden=args.num_hidden,
        hidden_exp=args.hidden_exp,
        dropout=args.dropout,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
        device_name=args.device,
        method=args.method,
        distributed=args.distributed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        per_epoch_validation=args.per_epoch_validation,
        seed=args.seed,
    )
