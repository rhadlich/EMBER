#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import h5py as h5
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.safety.datasets import SafetyInMemoryRowDataset, list_h5_files
from core.safety.safety_filter import StatePredictor


# %%
# =========================
# Interactive configuration
# =========================
DEFAULT_TEST_DIR = Path(
    '/Users/rodrigohadlich/Documents/Lab Documents/Methanol/Training Dataset/New Training/filter_processed_data/hdf5_data_filter/test'
)
DEFAULT_MODEL_DIR = Path("/Users/rodrigohadlich/EMBER/src/core/safety/run_logs")
SEED_PROGRESSION_DIR = Path("/Users/rodrigohadlich/EMBER/src/core/safety/run_logs/seed_progression_safety_filter.parquet")

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

# Plot settings (joint scatter + MAE vs IMEP, matching visualize_predictions.py style)
JOINT_SCATTER_HEIGHT = 7
JOINT_SCATTER_SIZE = 15
JOINT_SCATTER_ALPHA = 0.7
JOINT_SCATTER_BINS = 30
JOINT_SCATTER_CMAP = "viridis"
JOINT_SCATTER_MARGINAL_COLOR = "#2A788E"
JOINT_SCATTER_MAE_CBAR_LIMITS: tuple[float, float] | None = None
# Per-example MAE histogram x-axis and bin edges (values above max go in the last bin).
MAE_HISTOGRAM_MIN = 0.0
MAE_HISTOGRAM_MAX = 2.0
WORST_FRACTION = 0.1
NUM_WORST_CASES_TO_PRINT = 50
# Seed progression MSE axis scale ("linear" or "log").
SEED_PROGRESSION_MSE_SCALE = "linear"
# Seed progression plot axis bounds (set to None for auto).
SEED_PROGRESSION_XLIM: tuple[float, float] | None = None
SEED_PROGRESSION_MSE_YLIM = (0.25, 0.45)
SEED_PROGRESSION_MAE_YLIM = (0.28, 0.45)

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
) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    total_elements = 0
    sample_mae: list[float] = []

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
            sample_mae.extend(torch.mean(torch.abs(diff), dim=1).detach().cpu().tolist())

    mse = mse_sum / total_elements
    rmse = float(np.sqrt(mse))
    mae = mae_sum / total_elements
    return mse, rmse, mae, np.asarray(sample_mae, dtype=np.float64)


def load_raw_inspection_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load raw actions (ID1, SOI2, ID2) and IMEP from HDF5 test files."""
    action_blocks: list[np.ndarray] = []
    imep_blocks: list[np.ndarray] = []

    for filename in list_h5_files(str(data_dir)):
        with h5.File(filename, "r") as fin:
            states = fin["features/data"][...]
            actions = fin["features/action"][...]
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"Expected 2D arrays in {filename}, got states={states.shape}, actions={actions.shape}")
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"Row mismatch in {filename}: states={states.shape}, actions={actions.shape}")
        if actions.shape[1] < 3:
            raise ValueError(f"Expected at least 3 action columns in {filename}, got {actions.shape[1]}")

        action_blocks.append(actions[:, :3].astype(np.float64, copy=False))
        imep_blocks.append(states[:, 0].astype(np.float64, copy=False))

    return np.concatenate(action_blocks, axis=0), np.concatenate(imep_blocks, axis=0)


def align_imep_with_predictions(
    imep: np.ndarray,
    *aligned_arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    IMEP achieved at index k is stored at row k+1 in the HDF5 file.
    Shift back so row k carries IMEP for index k (drops the last row).
    """
    if len(imep) < 2:
        raise ValueError("Need at least 2 rows to align IMEP with per-example metrics.")
    for arr in aligned_arrays:
        if arr.shape[0] != imep.shape[0]:
            raise ValueError("All arrays must have the same number of rows as IMEP before alignment.")
    aligned_imep = imep[1:]
    trimmed = tuple(arr[:-1] for arr in aligned_arrays)
    return (aligned_imep, *trimmed)


def build_inspection_dataframe(
    *,
    actions: np.ndarray,
    imep: np.ndarray,
    sample_mae: np.ndarray,
) -> pd.DataFrame:
    if actions.shape[0] != imep.shape[0] or actions.shape[0] != sample_mae.shape[0]:
        raise ValueError("actions, imep, and sample_mae must have the same number of rows after alignment.")
    if actions.shape[1] != 3:
        raise ValueError(f"Expected 3 action columns (ID1, SOI2, ID2), got {actions.shape[1]}.")

    return pd.DataFrame(
        {
            "sample_idx": np.arange(actions.shape[0]),
            "prediction_mae": sample_mae,
            "imep": imep,
            "ID1": actions[:, 0],
            "SOI2": actions[:, 1],
            "ID2": actions[:, 2],
        }
    )


def build_worst_predictions_dataframe(
    full_df: pd.DataFrame,
    worst_fraction: float,
) -> pd.DataFrame:
    if "prediction_mae" not in full_df.columns:
        raise KeyError("Missing required column 'prediction_mae' in inspection dataframe.")
    if not 0 < worst_fraction <= 1:
        raise ValueError("worst_fraction must be in (0, 1].")

    num_rows = len(full_df)
    num_worst = max(1, int(np.ceil(num_rows * worst_fraction)))
    worst_indices = np.argsort(full_df["prediction_mae"].to_numpy())[-num_worst:][::-1]
    return full_df.iloc[worst_indices].reset_index(drop=True)


def print_worst_mae_cases(worst_df: pd.DataFrame, num_cases_to_print: int) -> None:
    required_cols = ["sample_idx", "prediction_mae", "imep", "ID1", "SOI2", "ID2"]
    missing_cols = [c for c in required_cols if c not in worst_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in worst dataframe: {missing_cols}")

    top_n = max(1, min(num_cases_to_print, len(worst_df)))
    print(f"Worst {top_n} cases (by per-example MAE) - actions and IMEP:")
    print(
        worst_df.loc[:, required_cols]
        .head(top_n)
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )


def plot_joint_scatter(
    x,
    y,
    c,
    x_label: str = "x",
    y_label: str = "y",
    cbar_label: str = "value",
    height: float = 7,
    size: float = 10,
    alpha: float = 0.7,
    bins: int = 30,
    cmap: str = "viridis",
    marginal_color: str = "#2A788E",
    c_limits: tuple[float, float] | None = None,
):
    try:
        import importlib

        import matplotlib.pyplot as plt

        sns = importlib.import_module("seaborn")
    except ImportError as exc:
        raise ImportError(
            "seaborn and matplotlib are required for joint scatter plots. Install with: pip install seaborn matplotlib"
        ) from exc

    g = sns.JointGrid(x=x, y=y, height=height)
    vmin, vmax = (c_limits if c_limits is not None else (None, None))

    sc = g.ax_joint.scatter(
        x,
        y,
        c=c,
        s=size,
        alpha=alpha,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    sns.histplot(x=x, ax=g.ax_marg_x, bins=bins, color=marginal_color)
    sns.histplot(y=y, ax=g.ax_marg_y, bins=bins, color=marginal_color)
    g.ax_joint.set_xlabel(x_label)
    g.ax_joint.set_ylabel(y_label)
    g.figure.subplots_adjust(left=0.14, bottom=0.14, right=0.86, top=0.96)
    joint_pos = g.ax_joint.get_position()
    marg_y_pos = g.ax_marg_y.get_position()
    cbar_pad = 0.01
    cbar_width = 0.02
    cbar_left = min(0.98 - cbar_width, marg_y_pos.x1 + cbar_pad)
    cax = g.figure.add_axes([cbar_left, joint_pos.y0, cbar_width, joint_pos.height])
    cbar = plt.colorbar(sc, cax=cax)
    if c_limits is not None:
        cbar.extend = "both"
    cbar.set_label(cbar_label)
    return g, sc


def plot_joint_scatter_error_map(full_df: pd.DataFrame) -> list:
    required_cols = ["ID1", "ID2", "SOI2", "prediction_mae"]
    missing_cols = [c for c in required_cols if c not in full_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for joint scatter plots: {missing_cols}")

    plot_specs = [
        ("ID1", "ID2"),
        ("ID1", "SOI2"),
        ("ID2", "SOI2"),
    ]
    grids = []
    for x_col, y_col in plot_specs:
        g, _ = plot_joint_scatter(
            x=full_df[x_col].to_numpy(),
            y=full_df[y_col].to_numpy(),
            c=full_df["prediction_mae"].to_numpy(),
            x_label=x_col,
            y_label=y_col,
            cbar_label="Per-example MAE",
            height=JOINT_SCATTER_HEIGHT,
            size=JOINT_SCATTER_SIZE,
            alpha=JOINT_SCATTER_ALPHA,
            bins=JOINT_SCATTER_BINS,
            cmap=JOINT_SCATTER_CMAP,
            marginal_color=JOINT_SCATTER_MARGINAL_COLOR,
            c_limits=JOINT_SCATTER_MAE_CBAR_LIMITS,
        )
        grids.append(g)
    return grids


def print_mae_quartiles(full_inspection_df: pd.DataFrame) -> None:
    if "prediction_mae" not in full_inspection_df.columns:
        raise KeyError("Missing required column 'prediction_mae' in full inspection dataframe.")

    q1, q2, q3, q95 = full_inspection_df["prediction_mae"].quantile([0.25, 0.5, 0.75, 0.95])
    print("Per-example MAE quartiles (full inspection dataframe):")
    print(f"  Q1 : {q1:.3f}")
    print(f"  Q2 : {q2:.3f}")
    print(f"  Q3 : {q3:.3f}")
    print(f"  P95: {q95:.3f}")


def plot_mae_histogram(full_inspection_df: pd.DataFrame):
    if "prediction_mae" not in full_inspection_df.columns:
        raise KeyError("Missing required column 'prediction_mae' in full inspection dataframe.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    if MAE_HISTOGRAM_MIN >= MAE_HISTOGRAM_MAX:
        raise ValueError("MAE_HISTOGRAM_MIN must be less than MAE_HISTOGRAM_MAX.")

    mae_values = full_inspection_df["prediction_mae"].to_numpy()
    q1, q2, q3, q95 = full_inspection_df["prediction_mae"].quantile([0.25, 0.5, 0.75, 0.95])
    num_overflow = int(np.sum(mae_values > MAE_HISTOGRAM_MAX))
    binned_values = np.clip(mae_values, MAE_HISTOGRAM_MIN, MAE_HISTOGRAM_MAX)
    bin_edges = np.linspace(MAE_HISTOGRAM_MIN, MAE_HISTOGRAM_MAX, JOINT_SCATTER_BINS + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        binned_values,
        bins=bin_edges,
        color=JOINT_SCATTER_MARGINAL_COLOR,
        edgecolor="white",
        alpha=0.9,
    )
    ax.axvline(q1, color="#E45756", linestyle="--", linewidth=1.5, label=f"Q1 = {q1:.6f}")
    ax.axvline(q2, color="#F58518", linestyle="--", linewidth=1.5, label=f"Q2 = {q2:.6f}")
    ax.axvline(q3, color="#54A24B", linestyle="--", linewidth=1.5, label=f"Q3 = {q3:.6f}")
    ax.axvline(q95, color="#4C78A8", linestyle="-.", linewidth=1.5, label=f"P95 = {q95:.6f}")
    ax.set_xlim(MAE_HISTOGRAM_MIN, MAE_HISTOGRAM_MAX)
    ax.set_xlabel("Per-example MAE")
    ax.set_ylabel("Count")
    overflow_note = f", {num_overflow} in last bin (MAE > {MAE_HISTOGRAM_MAX})" if num_overflow else ""
    ax.set_title(f"Per-example MAE distribution (n={len(mae_values)}{overflow_note})")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_mae_vs_imep(full_df: pd.DataFrame):
    required_cols = ["prediction_mae", "imep"]
    missing_cols = [c for c in required_cols if c not in full_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for MAE-vs-IMEP scatter: {missing_cols}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    x_all = full_df["imep"].to_numpy(dtype=np.float64)
    y_all = full_df["prediction_mae"].to_numpy(dtype=np.float64)
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    if not np.any(valid):
        raise ValueError("No finite rows found for MAE-vs-IMEP plot.")

    x = x_all[valid]
    y = y_all[valid]
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(
        x,
        y,
        gridsize=JOINT_SCATTER_BINS * 8,
        cmap=JOINT_SCATTER_CMAP,
        mincnt=1,
    )
    ax.set_xlabel("IMEP (features/data col 0, index-aligned)")
    ax.set_ylabel("Per-example MAE")
    ax.set_title(f"Per-example MAE vs IMEP hexbin (n={len(x)})")
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_inspection_figures(full_df: pd.DataFrame) -> dict:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    print("\nGenerating per-example MAE histogram...")
    mae_histogram_fig = plot_mae_histogram(full_df)
    print("Generating joint scatter plots colored by per-example MAE...")
    joint_scatter_grids = plot_joint_scatter_error_map(full_df)
    print("Generating MAE vs IMEP scatter plot...")
    mae_vs_imep_fig = plot_mae_vs_imep(full_df)
    plt.show()
    return {
        "mae_histogram_fig": mae_histogram_fig,
        "joint_scatter_grids": joint_scatter_grids,
        "mae_vs_imep_fig": mae_vs_imep_fig,
    }


def load_seed_progression_dataframe(seed_progression_path: Path) -> pd.DataFrame:
    if not seed_progression_path.exists():
        raise FileNotFoundError(f"Seed progression parquet not found: {seed_progression_path}")

    df = pd.read_parquet(seed_progression_path)
    required_cols = [
        "seed_idx",
        "seed",
        "trial_idx",
        "epoch",
        "train_mse",
        "val_mse",
        "train_mae",
        "val_mae",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Seed progression parquet is missing required columns: {missing}. "
            f"Expected at least {required_cols}."
        )
    if df.empty:
        raise ValueError(f"Seed progression parquet is empty: {seed_progression_path}")

    out = df.loc[:, required_cols].copy()
    for col in ["seed_idx", "seed", "trial_idx", "epoch"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["train_mse"] = pd.to_numeric(out["train_mse"], errors="coerce")
    out["val_mse"] = pd.to_numeric(out["val_mse"], errors="coerce")
    out["train_mae"] = pd.to_numeric(out["train_mae"], errors="coerce")
    out["val_mae"] = pd.to_numeric(out["val_mae"], errors="coerce")
    out = out.dropna(subset=["seed_idx", "seed", "trial_idx", "epoch"])
    if out.empty:
        raise ValueError("No valid seed/trial/epoch rows found in seed progression parquet.")
    out["seed_idx"] = out["seed_idx"].astype(int)
    out["seed"] = out["seed"].astype(int)
    out["trial_idx"] = out["trial_idx"].astype(int)
    out["epoch"] = out["epoch"].astype(int)
    return out.sort_values(["seed_idx", "trial_idx", "epoch"]).reset_index(drop=True)


def plot_seed_progression_metrics(seed_progression_df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting seed progression. Install with: pip install matplotlib"
        ) from exc

    run_cols = ["seed_idx", "seed", "trial_idx"]
    val_grid = seed_progression_df.pivot_table(
        index="epoch",
        columns=run_cols,
        values="val_mse",
        aggfunc="last",
    ).sort_index()
    train_grid = seed_progression_df.pivot_table(
        index="epoch",
        columns=run_cols,
        values="train_mse",
        aggfunc="last",
    ).reindex(index=val_grid.index, columns=val_grid.columns)
    val_mae_grid = seed_progression_df.pivot_table(
        index="epoch",
        columns=run_cols,
        values="val_mae",
        aggfunc="last",
    ).reindex(index=val_grid.index, columns=val_grid.columns)
    train_mae_grid = seed_progression_df.pivot_table(
        index="epoch",
        columns=run_cols,
        values="train_mae",
        aggfunc="last",
    ).reindex(index=val_grid.index, columns=val_grid.columns)

    if val_grid.empty:
        raise ValueError("Seed progression dataframe has no validation MSE rows to plot.")

    val_matrix = val_grid.to_numpy(dtype=np.float64)
    train_matrix = train_grid.to_numpy(dtype=np.float64)
    val_mae_matrix = val_mae_grid.to_numpy(dtype=np.float64)
    train_mae_matrix = train_mae_grid.to_numpy(dtype=np.float64)
    epochs = val_grid.index.to_numpy(dtype=np.int64)

    final_val_by_run = np.full((val_matrix.shape[1],), np.nan, dtype=np.float64)
    for run_idx in range(val_matrix.shape[1]):
        run_curve = val_matrix[:, run_idx]
        finite_idx = np.flatnonzero(np.isfinite(run_curve))
        if finite_idx.size:
            final_val_by_run[run_idx] = run_curve[finite_idx[-1]]

    valid_run_idx = np.flatnonzero(np.isfinite(final_val_by_run))
    if valid_run_idx.size == 0:
        raise ValueError("No run has a finite final validation MSE.")

    median_rank = np.argsort(final_val_by_run[valid_run_idx])[valid_run_idx.size // 2]
    median_run_idx = int(valid_run_idx[median_rank])
    median_run_key = val_grid.columns[median_run_idx]

    val_min = np.nanmin(val_matrix[:, valid_run_idx], axis=1)
    val_max = np.nanmax(val_matrix[:, valid_run_idx], axis=1)
    train_min = np.nanmin(train_matrix[:, valid_run_idx], axis=1)
    train_max = np.nanmax(train_matrix[:, valid_run_idx], axis=1)
    val_mae_min = np.nanmin(val_mae_matrix[:, valid_run_idx], axis=1)
    val_mae_max = np.nanmax(val_mae_matrix[:, valid_run_idx], axis=1)
    train_mae_min = np.nanmin(train_mae_matrix[:, valid_run_idx], axis=1)
    train_mae_max = np.nanmax(train_mae_matrix[:, valid_run_idx], axis=1)

    median_val_curve = val_matrix[:, median_run_idx]
    median_train_curve = train_matrix[:, median_run_idx]
    median_val_mae_curve = val_mae_matrix[:, median_run_idx]
    median_train_mae_curve = train_mae_matrix[:, median_run_idx]

    mse_scale = str(SEED_PROGRESSION_MSE_SCALE).lower()
    if mse_scale not in {"linear", "log"}:
        raise ValueError(
            f"Invalid SEED_PROGRESSION_MSE_SCALE='{SEED_PROGRESSION_MSE_SCALE}'. "
            "Use 'linear' or 'log'."
        )
    if mse_scale == "log":
        # Log scale requires strictly positive values.
        train_min = np.where(train_min > 0, train_min, np.nan)
        train_max = np.where(train_max > 0, train_max, np.nan)
        val_min = np.where(val_min > 0, val_min, np.nan)
        val_max = np.where(val_max > 0, val_max, np.nan)
        median_train_curve = np.where(median_train_curve > 0, median_train_curve, np.nan)
        median_val_curve = np.where(median_val_curve > 0, median_val_curve, np.nan)
        if not np.any(np.isfinite(median_train_curve)) and not np.any(np.isfinite(median_val_curve)):
            raise ValueError(
                "Cannot use log-scale MSE axis because all median MSE values are non-positive/invalid."
            )

    fig, ax_mse = plt.subplots(figsize=(11, 5.2))
    ax_mae = ax_mse.twinx()

    ax_mse.fill_between(epochs, train_min, train_max, color="#1f77b4", alpha=0.16)
    ax_mse.plot(epochs, median_train_curve, color="#1f77b4", linewidth=2.2, label="Train MSE median")
    ax_mse.fill_between(epochs, val_min, val_max, color="#ff7f0e", alpha=0.16)
    ax_mse.plot(epochs, median_val_curve, color="#ff7f0e", linewidth=2.2, label="Val MSE median")

    ax_mae.fill_between(epochs, train_mae_min, train_mae_max, color="#2ca02c", alpha=0.12)
    ax_mae.plot(
        epochs,
        median_train_mae_curve,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="--",
        label="Train MAE median",
    )
    ax_mae.fill_between(epochs, val_mae_min, val_mae_max, color="#d62728", alpha=0.12)
    ax_mae.plot(
        epochs,
        median_val_mae_curve,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="Val MAE median",
    )

    if mse_scale == "log":
        ax_mse.set_yscale("log")
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel(f"MSE ({mse_scale} scale)")
    ax_mae.set_ylabel("MAE")
    if SEED_PROGRESSION_XLIM is not None:
        ax_mse.set_xlim(SEED_PROGRESSION_XLIM)
    if SEED_PROGRESSION_MSE_YLIM is not None:
        ax_mse.set_ylim(SEED_PROGRESSION_MSE_YLIM)
    if SEED_PROGRESSION_MAE_YLIM is not None:
        ax_mae.set_ylim(SEED_PROGRESSION_MAE_YLIM)
    ax_mse.grid(alpha=0.25)

    handles_mse, labels_mse = ax_mse.get_legend_handles_labels()
    handles_mae, labels_mae = ax_mae.get_legend_handles_labels()
    ax_mse.legend(handles_mse + handles_mae, labels_mse + labels_mae, loc="best")

    fig.suptitle(
        "Seed sweep train/validation MSE+MAE curves "
        f"(median run by final val MSE: seed_idx={median_run_key[0]}, seed={median_run_key[1]}, trial={median_run_key[2]})"
    )
    fig.tight_layout()
    return fig, {
        "median_seed_idx": int(median_run_key[0]),
        "median_seed": int(median_run_key[1]),
        "median_trial_idx": int(median_run_key[2]),
        "median_final_val_mse": float(final_val_by_run[median_run_idx]),
        "num_runs": int(valid_run_idx.size),
    }


def find_best_seed_mae(seed_progression_df: pd.DataFrame, metric_col: str = "val_mae") -> dict:
    if metric_col not in seed_progression_df.columns:
        raise KeyError(f"Missing MAE column '{metric_col}' in seed progression dataframe.")
    valid_rows = seed_progression_df.loc[np.isfinite(seed_progression_df[metric_col].to_numpy())]
    if valid_rows.empty:
        raise ValueError(f"No finite values found for MAE column '{metric_col}'.")
    best_row = valid_rows.loc[valid_rows[metric_col].idxmin()]
    return {
        "metric": metric_col,
        "mae": float(best_row[metric_col]),
        "seed_idx": int(best_row["seed_idx"]),
        "seed": int(best_row["seed"]),
        "trial_idx": int(best_row["trial_idx"]),
        "epoch": int(best_row["epoch"]),
    }


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
    seed_progression_path: Path | None = SEED_PROGRESSION_DIR,
    worst_fraction: float = WORST_FRACTION,
    num_worst_cases_to_print: int = NUM_WORST_CASES_TO_PRINT,
):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not 0 < worst_fraction <= 1:
        raise ValueError("worst_fraction must be in (0, 1].")
    if num_worst_cases_to_print <= 0:
        raise ValueError("num_worst_cases_to_print must be > 0")

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

    mse, rmse, mae, sample_mae = evaluate_test_metrics(model=model, loader=test_loader, device=device)
    print(f"Test rows: {len(test_dataset)}")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    raw_actions, raw_imep = load_raw_inspection_arrays(test_dir)
    if raw_actions.shape[0] != sample_mae.shape[0] or raw_imep.shape[0] != sample_mae.shape[0]:
        raise ValueError(
            "Row count mismatch between HDF5 inspection arrays and model evaluation: "
            f"actions={raw_actions.shape[0]}, imep={raw_imep.shape[0]}, sample_mae={sample_mae.shape[0]}"
        )

    aligned_imep, aligned_actions, aligned_sample_mae = align_imep_with_predictions(
        raw_imep,
        raw_actions,
        sample_mae,
    )
    dropped_rows = len(sample_mae) - len(aligned_sample_mae)
    print(
        f"IMEP alignment: shifted back by one row (dropped last {dropped_rows} example(s) for plotting)."
    )

    full_inspection_df = build_inspection_dataframe(
        actions=aligned_actions,
        imep=aligned_imep,
        sample_mae=aligned_sample_mae,
    )
    print_mae_quartiles(full_inspection_df)
    worst_predictions_df = build_worst_predictions_dataframe(
        full_inspection_df,
        worst_fraction=worst_fraction,
    )
    print(f"\nWorst {len(worst_predictions_df)} rows dataframe created.")
    print_worst_mae_cases(worst_predictions_df, num_cases_to_print=num_worst_cases_to_print)
    plot_results = plot_inspection_figures(full_inspection_df)

    seed_progression_fig = None
    seed_progression_summary = None
    best_val_mae_seed_info = None
    if seed_progression_path is not None:
        print(f"\nLoading seed progression parquet: {seed_progression_path}")
        seed_progression_df = load_seed_progression_dataframe(seed_progression_path)
        best_val_mae_seed_info = find_best_seed_mae(seed_progression_df, metric_col="val_mae")
        print(
            "Lowest validation MAE from seed progression: "
            f"{best_val_mae_seed_info['mae']:.6e} at epoch={best_val_mae_seed_info['epoch']} "
            f"(seed_idx={best_val_mae_seed_info['seed_idx']}, seed={best_val_mae_seed_info['seed']}, "
            f"trial={best_val_mae_seed_info['trial_idx']})."
        )
        print("Plotting seed progression train/validation MSE+MAE (median run with min-max envelopes)...")
        seed_progression_fig, seed_progression_summary = plot_seed_progression_metrics(seed_progression_df)
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except ImportError:
            pass

    return {
        "checkpoint_path": checkpoint_path,
        "num_test_rows": len(test_dataset),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "full_inspection_df": full_inspection_df,
        "worst_predictions_df": worst_predictions_df,
        "seed_progression_summary": seed_progression_summary,
        "best_val_mae_seed_info": best_val_mae_seed_info,
        "seed_progression_fig": seed_progression_fig,
        **plot_results,
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
    seed_progression_path=SEED_PROGRESSION_DIR,
)
results
