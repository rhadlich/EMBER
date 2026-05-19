#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.safety.datasets import SafetyInMemoryRowDataset
from core.safety.safety_filter import StatePredictor


# %%
# =========================
# Interactive configuration
# =========================
DEFAULT_TEST_DIR = Path(
    "/Users/rodrigohadlich/Documents/Lab Documents/Methanol/Training Dataset/New Training/filter_processed_data/hdf5_data/test"
)
DEFAULT_MODEL_DIR = Path("/Users/rodrigohadlich/EMBER/src/core/safety/models")

# Set to a filename inside MODEL_DIR, e.g. "model_weights_filter_new.pth".
# Leave as None to auto-pick the latest model_weights_filter*.pth.
CHECKPOINT_NAME = None

# Inference settings
DEVICE = "cpu"  # one of: auto, cpu, mps, cuda
BATCH_SIZE = 256
NUM_WORKERS = 0

# Optional fallback config for old checkpoints that do not contain "model_config".
FALLBACK_STATE_DIM = None
FALLBACK_OUTPUT_DIM = None
FALLBACK_ACTION_DIM = None
FALLBACK_NUM_HIDDEN = None
FALLBACK_HIDDEN_EXP = None
FALLBACK_DROPOUT = None

# Main paths used by the run cell below
MODEL_DIR = DEFAULT_MODEL_DIR
TEST_DIR = DEFAULT_TEST_DIR


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device '{device_name}'. Use one of: auto, cpu, mps, cuda.")


def select_checkpoint(model_dir: Path, checkpoint_name: str | None) -> Path:
    if checkpoint_name:
        checkpoint_path = model_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    matches = sorted(
        model_dir.glob("model_weights_filter*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        matches = sorted(
            model_dir.glob("model_weights*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not matches:
        raise FileNotFoundError(
            f"No model checkpoint matching 'model_weights_filter*.pth' or 'model_weights*.pth' found in {model_dir}"
        )
    return matches[0]


def _adapt_state_dict_for_predictor(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("predictor.") for k in state_dict):
        return {k.removeprefix("predictor."): v for k, v in state_dict.items()}
    return state_dict


def build_model_from_checkpoint(
    checkpoint: dict,
    fallback_state_dim: int | None,
    fallback_output_dim: int | None,
    fallback_action_dim: int | None,
    fallback_num_hidden: int | None,
    fallback_hidden_exp: int | None,
    fallback_dropout: float | None,
) -> StatePredictor:
    cfg = checkpoint.get("model_config")
    if cfg is None:
        if None in (
            fallback_state_dim,
            fallback_output_dim,
            fallback_action_dim,
            fallback_num_hidden,
            fallback_hidden_exp,
            fallback_dropout,
        ):
            raise ValueError(
                "Checkpoint has no model_config. Provide --state-dim, --output-dim, "
                "--action-dim, --num-hidden, --hidden-exp, and --dropout."
            )
        cfg = {
            "state_dim": fallback_state_dim,
            "output_dim": fallback_output_dim,
            "action_dim": fallback_action_dim,
            "num_hidden": fallback_num_hidden,
            "hidden_exp": fallback_hidden_exp,
            "dropout": fallback_dropout,
        }

    state_dim = int(cfg["state_dim"])
    output_dim = int(cfg["output_dim"])
    action_dim = int(cfg["action_dim"])
    num_hidden = int(cfg["num_hidden"])
    hidden_exp = int(cfg["hidden_exp"])
    dropout = float(cfg["dropout"])

    model = StatePredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        output_dim=output_dim,
        num_hidden=num_hidden,
        hidden_exp=hidden_exp,
        dropout=dropout,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _adapt_state_dict_for_predictor(state_dict)
    model.load_state_dict(state_dict)
    return model


def evaluate_test_metrics(
    model: StatePredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    total_elements = 0

    with torch.no_grad():
        for (states, actions), targets in loader:
            states = states.to(device=device, dtype=torch.float32)
            actions = actions.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            preds, _, _ = model(states, actions)
            diff = preds - targets
            mse_sum += torch.sum(diff**2).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            total_elements += diff.numel()

    mse = mse_sum / total_elements
    rmse = float(np.sqrt(mse))
    mae = mae_sum / total_elements
    return mse, rmse, mae


def run_evaluation(
    *,
    model_dir: Path,
    checkpoint_name: str | None,
    test_dir: Path,
    batch_size: int,
    num_workers: int,
    device_name: str,
    fallback_state_dim: int | None,
    fallback_output_dim: int | None,
    fallback_action_dim: int | None,
    fallback_num_hidden: int | None,
    fallback_hidden_exp: int | None,
    fallback_dropout: float | None,
):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    device = resolve_device(device_name)
    checkpoint_path = select_checkpoint(model_dir, checkpoint_name)
    print(f"Using device: {device}")
    print(f"Using checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        fallback_state_dim=fallback_state_dim,
        fallback_output_dim=fallback_output_dim,
        fallback_action_dim=fallback_action_dim,
        fallback_num_hidden=fallback_num_hidden,
        fallback_hidden_exp=fallback_hidden_exp,
        fallback_dropout=fallback_dropout,
    ).to(device)

    test_dataset = SafetyInMemoryRowDataset(
        source=str(test_dir),
        allow_uneven_distribution=True,
        shuffle=False,
        size=1,
        rank=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    mse, rmse, mae = evaluate_test_metrics(model=model, loader=test_loader, device=device)
    print(f"Test rows: {len(test_dataset)}")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    return {
        "checkpoint_path": checkpoint_path,
        "num_test_rows": len(test_dataset),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


# %%
# Run this cell (or click Run on the file) to evaluate on test data.
results = run_evaluation(
    model_dir=MODEL_DIR,
    checkpoint_name=CHECKPOINT_NAME,
    test_dir=TEST_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device_name=DEVICE,
    fallback_state_dim=FALLBACK_STATE_DIM,
    fallback_output_dim=FALLBACK_OUTPUT_DIM,
    fallback_action_dim=FALLBACK_ACTION_DIM,
    fallback_num_hidden=FALLBACK_NUM_HIDDEN,
    fallback_hidden_exp=FALLBACK_HIDDEN_EXP,
    fallback_dropout=FALLBACK_DROPOUT,
)
results
