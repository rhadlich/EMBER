#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pickle
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import h5py as h5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from core.digital_twin.architectures import MLP, ResidualMLP
from core.digital_twin.datasets import InMemoryRowDataset
from core.digital_twin.engine_metrics import calculate_pressure_metrics, engine_geometry
from core.training.trainer import resolve_device


# =========================
# Interactive configuration
# =========================
# Model/checkpoint settings (same style as visualize_predictions.py).
MODEL_DIR = Path('/Users/rodrigohadlich/Documents/Lab Documents/Thesis Material/Methanol DL Modeling/Dataset v3/Feature Rich/Model')
# Set to a filename inside MODEL_DIR, e.g. "model_weights_engine_new.pth".
# Leave as None to auto-pick latest model_weights*.pth.
CHECKPOINT_NAME = None

# Data settings:
# DATA_ROOT_DIR is expected to contain split subfolders such as train/validation/test.
DATA_ROOT_DIR = Path(
    '/Users/rodrigohadlich/Documents/Lab Documents/Thesis Material/Methanol DL Modeling/Dataset v3/Feature Rich/Dataset/hdf5_data'
)
# Split used as explained/evaluated dataset.
ANALYSIS_SUBSET = "validation"  # e.g. "train", "validation", "test"
# Split used for background/reference points (ALE + SHAP).
BACKGROUND_SUBSET = "no_injection"

OUTPUT_DIR = Path("/Users/rodrigohadlich/EMBER/src/core/training/xai_outputs")

# If your model has multiple outputs and EXPLANATION_TARGETS is empty,
# choose which output index to explain. Set to None to explain output mean.
TARGET_OUTPUT_INDEX: Optional[int] = None
# Digested engine metrics to explain from predicted pressure traces.
# Supported entries: "imep", "mprr", "qnet", "ca50".
EXPLANATION_TARGETS: tuple[str, ...] = ("imep", "mprr", "qnet", "ca50")
METRIC_TARGET_ORDER: tuple[str, ...] = ("imep", "mprr", "qnet", "ca50")
# Optional map from raw explanation target key -> display label used in plots.
# Example: {"imep": "IMEP [bar]", "ca50": "CA50 [deg aTDC]"}
EXPLANATION_TARGET_LABEL_MAP: dict[str, str] = {
    "imep": r"IMEP",
    "mprr": r"MPRR",
    "qnet": r"Q_net",
    "ca50": r"CA50",
}

# Which analyses to run.
RUN_INTEGRATED_GRADIENTS = False
RUN_IG_TRACE = False
RUN_ALE = False
RUN_SHAP = True
RUN_PERMUTATION_IMPORTANCE = False
RUN_MULTI_OUTPUT_SHAP = True

# Runtime behavior
DEVICE = "cpu"  # auto, cpu, mps, cuda
BATCH_SIZE = 256
SHAP_BACKGROUND_SIZE = 200
SHAP_EVAL_SIZE = 400
ALE_GRID_SIZE = 20
PERMUTATION_REPEATS = 12
RANDOM_SEED = 42
USE_PREDICTOR_CACHE = True
PREDICTOR_CACHE_MAX_ENTRIES = 64
USE_IG_CACHE = False
IG_TRACE_STEPS = 50
IG_TRACE_EVAL_SIZE = 2000
IG_BATCH_CHUNK = 32
IG_OUTPUT_CHUNK = 256
IG_RETURN_FULL_ATTR = False
IG_USE_JACFWD = True

# Plot/output controls
SHOW_PLOTS = True
SAVE_PLOTS = True

# SHAP caching controls (to avoid recomputing expensive SHAP values).
LOAD_SHAP_FROM_CACHE = False
SAVE_SHAP_TO_CACHE = True
SHAP_CACHE_DIR = OUTPUT_DIR / "shap_cache"

# Optional map from raw HDF5 feature name -> display label used in plots.
# Example: {"imep_prev": "IMEP (prev cycle)"}
FEATURE_LABEL_MAP: dict[str, str] = {
    "ID1_prev": r"$ID1_{k-1}$",
    "ID2_prev": r"$ID2_{k-1}$",
    "SOI2_prev": r"$SOI2_{k-1}$",
    "Q_net_prev": r"$Q_{net, k-1}$",
    "IMEP_ma": r"IMEP$_{moving average}$",
    "P_max_prev": r"$P_{max, k-1}$",
    "CA50_prev": r"$CA50_{k-1}$",
    "CA10_90_prev": r"$CA10-90_{k-1}$",
    "P_int_IVC_prev": r"$P_{int, k-1}$",
    "mprr_prev": r"$MPRR_{k-1}$",
    "ID1": r"$ID1_{k}$",
    "ID2": r"$ID2_{k}$",
    "SOI2": r"$SOI2_{k}$",
    "skewness": "Skewness",
}

# Optional CAD window for IG trace heatmap plotting.
# Set to None to keep full range, or (cad_min_deg, cad_max_deg), e.g. (-20.0, 80.0).
IG_TRACE_CAD_RANGE: Optional[tuple[float, float]] = (-20.0, 40.0)

# Optional fallback config for old checkpoints that do not contain "model_config".
FALLBACK_INPUT_DIM = None
FALLBACK_OUTPUT_DIM = None
FALLBACK_NUM_HIDDEN = None
FALLBACK_HIDDEN_EXP = None
FALLBACK_DROPOUT = None

@dataclass
class XAIBundle:
    model: nn.Module
    x_train: np.ndarray
    y_train_traces: np.ndarray
    x_eval: np.ndarray
    y_eval_traces: np.ndarray
    feature_names: list[str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._")
    return token or "unnamed"


def _build_shap_cache_path(kind: str, tag: str) -> Path:
    return SHAP_CACHE_DIR / (
        f"{_safe_token(kind)}"
        f"__analysis_{_safe_token(ANALYSIS_SUBSET)}"
        f"__background_{_safe_token(BACKGROUND_SUBSET)}"
        f"__bg{int(SHAP_BACKGROUND_SIZE)}"
        f"__eval{int(SHAP_EVAL_SIZE)}"
        f"__seed{int(RANDOM_SEED)}"
        f"__{_safe_token(tag)}.pkl"
    )


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "wb") as fout:
        pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fin:
        obj = pickle.load(fin)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in cache file {path}, got {type(obj)}")
    return obj


def _extract_tensor_output(raw_output: Any) -> torch.Tensor:
    if torch.is_tensor(raw_output):
        return raw_output
    if isinstance(raw_output, (tuple, list)):
        for item in raw_output:
            if torch.is_tensor(item):
                return item
    raise TypeError(
        "Model forward output is not a tensor and no tensor was found in tuple/list output."
    )


def _reduce_to_single_output(output: torch.Tensor, target_output_index: Optional[int]) -> torch.Tensor:
    if output.ndim == 1:
        return output
    if output.ndim != 2:
        raise ValueError(f"Expected model output shape (N,) or (N, C), got {tuple(output.shape)}")
    if target_output_index is None:
        return output.mean(dim=1)
    if target_output_index < 0 or target_output_index >= output.shape[1]:
        raise IndexError(
            f"target_output_index={target_output_index} out of bounds for output shape {tuple(output.shape)}"
        )
    return output[:, target_output_index]


def _predict_raw_numpy(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    x = np.asarray(x, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            out = _extract_tensor_output(model(xb))
            preds.append(out.detach().cpu().numpy())
    concatenated = np.concatenate(preds, axis=0)
    if concatenated.ndim == 1:
        return concatenated.reshape(-1, 1)
    if concatenated.ndim != 2:
        raise ValueError(f"Expected model output shape (N, C), got {concatenated.shape}")
    return concatenated


def select_checkpoint(model_dir: Path, checkpoint_name: str | None) -> Path:
    if checkpoint_name:
        checkpoint_path = model_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    matches = sorted(
        model_dir.glob("model_weights*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No model checkpoint matching 'model_weights*.pth' found in {model_dir}"
        )
    return matches[0]


def _infer_architecture_from_state_dict(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any(k.startswith("input_layer.") for k in keys):
        return "residual_mlp"
    if any(k.startswith("network.") for k in keys):
        return "mlp"
    raise ValueError(
        "Could not infer model architecture from state_dict keys. "
        "Use a checkpoint with model_config or known MLP/ResidualMLP weights."
    )


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    fallback_input_dim: int | None,
    fallback_output_dim: int | None,
    fallback_num_hidden: int | None,
    fallback_hidden_exp: int | None,
    fallback_dropout: float | None,
) -> nn.Module:
    cfg = checkpoint.get("model_config")
    if cfg is None:
        if None in (
            fallback_input_dim,
            fallback_output_dim,
            fallback_num_hidden,
            fallback_hidden_exp,
            fallback_dropout,
        ):
            raise ValueError(
                "Checkpoint has no model_config. Provide fallback input/output dims and architecture settings."
            )
        cfg = {
            "input_dim": fallback_input_dim,
            "output_dim": fallback_output_dim,
            "num_hidden": fallback_num_hidden,
            "hidden_exp": fallback_hidden_exp,
            "dropout": fallback_dropout,
        }

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    architecture = cfg.get("architecture")
    if architecture is None:
        architecture = _infer_architecture_from_state_dict(state_dict)

    input_dim = int(cfg["input_dim"])
    output_dim = int(cfg["output_dim"])
    num_hidden = int(cfg["num_hidden"])
    hidden_exp = int(cfg["hidden_exp"])
    dropout = float(cfg["dropout"])

    if architecture == "mlp":
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden=num_hidden,
            hidden_exp=hidden_exp,
            dropout=dropout,
        )
    elif architecture == "residual_mlp":
        model = ResidualMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_blocks=num_hidden,
            hidden_exp=hidden_exp,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Expected one of: mlp, residual_mlp."
        )

    model.load_state_dict(state_dict)
    return model


def _resolve_subset_dir(root_dir: Path, subset: str) -> Path:
    subset_as_path = Path(subset).expanduser()
    if subset_as_path.exists():
        return subset_as_path

    candidate = root_dir / subset
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve subset '{subset}'. Checked '{subset_as_path}' and '{candidate}'."
    )


def _load_split_arrays(
    split_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    ds = InMemoryRowDataset(
        source=str(split_dir),
        allow_uneven_distribution=True,
        shuffle=False,
        size=1,
        rank=0,
    )
    x = ds.data.detach().cpu().numpy().astype(np.float32, copy=False)
    y = ds.labels.detach().cpu().numpy().astype(np.float32, copy=False)
    return x, y


def _normalize_explanation_targets(raw_targets: tuple[str, ...]) -> list[str]:
    if not raw_targets:
        return []
    normalized = [name.strip().lower() for name in raw_targets if name.strip()]
    if not normalized:
        return []
    valid = {"imep", "mprr", "qnet", "ca50"}
    unknown = sorted(set(normalized) - valid)
    if unknown:
        raise ValueError(
            f"Unsupported EXPLANATION_TARGETS entries: {unknown}. "
            f"Supported targets are: {sorted(valid)}"
        )
    return normalized


def _extract_metric_targets(
    traces: np.ndarray,
    target_names: list[str],
    *,
    volume: np.ndarray,
    vd: float,
) -> dict[str, np.ndarray]:
    metrics = calculate_pressure_metrics(traces, volume=volume, vd=vd)
    return {name: metrics[name].reshape(-1).astype(np.float32, copy=False) for name in target_names}


class MultiMetricPredictor:
    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        volume: np.ndarray,
        vd: float,
        batch_size: int,
        metric_names: list[str],
        use_cache: bool = True,
        max_cache_entries: int = 64,
        target_output_index: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.volume = volume
        self.vd = vd
        self.batch_size = batch_size
        self.metric_names = metric_names
        self.target_output_index = target_output_index
        self.use_cache = use_cache
        self.max_cache_entries = max(1, int(max_cache_entries))
        self._trace_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._metrics_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.stats: dict[str, int] = {
            "trace_forward_calls": 0,
            "trace_cache_hits": 0,
            "metrics_cache_hits": 0,
        }

    @staticmethod
    def _fingerprint(x: np.ndarray) -> str:
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        digest = hashlib.blake2b(
            x_arr.view(np.uint8),
            digest_size=16,
            person=b"xai_predictor_v1",
        ).hexdigest()
        return f"{x_arr.shape}|{digest}"

    @staticmethod
    def _maybe_add_cache(cache: OrderedDict[str, np.ndarray], key: str, value: np.ndarray, max_entries: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def predict_trace_np(self, x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float32)
        key = self._fingerprint(x_np)
        if self.use_cache and key in self._trace_cache:
            self.stats["trace_cache_hits"] += 1
            self._trace_cache.move_to_end(key)
            return self._trace_cache[key]
        trace = _predict_raw_numpy(
            self.model,
            x_np,
            self.device,
            batch_size=self.batch_size,
        )
        self.stats["trace_forward_calls"] += 1
        if self.use_cache:
            self._maybe_add_cache(self._trace_cache, key, trace, self.max_cache_entries)
        return trace

    def predict_metrics_np(self, x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float32)
        key = self._fingerprint(x_np)
        if self.use_cache and key in self._metrics_cache:
            self.stats["metrics_cache_hits"] += 1
            self._metrics_cache.move_to_end(key)
            return self._metrics_cache[key]
        traces = self.predict_trace_np(x_np)
        metrics_map = _extract_metric_targets(
            traces,
            self.metric_names,
            volume=self.volume,
            vd=self.vd,
        )
        matrix = np.stack([metrics_map[name] for name in self.metric_names], axis=1).astype(np.float32, copy=False)
        if self.use_cache:
            self._maybe_add_cache(self._metrics_cache, key, matrix, self.max_cache_entries)
        return matrix

    def predict_metric_np(self, x: np.ndarray, metric_name: str) -> np.ndarray:
        if metric_name not in self.metric_names:
            raise KeyError(f"Metric '{metric_name}' is not configured in predictor metric_names={self.metric_names}")
        matrix = self.predict_metrics_np(x)
        idx = self.metric_names.index(metric_name)
        return matrix[:, idx]

    def predict_reduced_output_np(self, x: np.ndarray) -> np.ndarray:
        traces = self.predict_trace_np(x)
        reduced = _reduce_to_single_output(
            torch.from_numpy(traces),
            self.target_output_index,
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        return reduced

    def predict_trace_torch(self, x_torch: torch.Tensor) -> torch.Tensor:
        raw = _extract_tensor_output(self.model(x_torch))
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.ndim != 2:
            raise ValueError(f"Expected model output shape (N, C), got {tuple(raw.shape)}")
        return raw


def _extract_feature_names(split_dir: Path, feature_dim: int) -> list[str]:
    h5_files = sorted(split_dir.glob("*.h5"))
    if not h5_files:
        return [f"feature_{i}" for i in range(feature_dim)]

    with h5.File(h5_files[0], "r") as fin:
        if "features/names" not in fin:
            return [f"feature_{i}" for i in range(feature_dim)]
        raw_names = fin["features/names"][...]

    names: list[str] = []
    for item in raw_names:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        elif isinstance(item, np.bytes_):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))

    if len(names) == feature_dim:
        return names
    return [f"feature_{i}" for i in range(feature_dim)]


def _apply_feature_label_map(feature_names: list[str]) -> list[str]:
    if not FEATURE_LABEL_MAP:
        return feature_names
    return [FEATURE_LABEL_MAP.get(name, name) for name in feature_names]


def _target_display_label(target_name: str) -> str:
    return EXPLANATION_TARGET_LABEL_MAP.get(target_name, target_name)


def load_xai_bundle(device: torch.device) -> XAIBundle:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")
    if not DATA_ROOT_DIR.exists():
        raise FileNotFoundError(f"Data root directory does not exist: {DATA_ROOT_DIR}")

    checkpoint_path = select_checkpoint(MODEL_DIR, CHECKPOINT_NAME)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(
        checkpoint,
        fallback_input_dim=FALLBACK_INPUT_DIM,
        fallback_output_dim=FALLBACK_OUTPUT_DIM,
        fallback_num_hidden=FALLBACK_NUM_HIDDEN,
        fallback_hidden_exp=FALLBACK_HIDDEN_EXP,
        fallback_dropout=FALLBACK_DROPOUT,
    )
    model = model.to(device)
    model.eval()

    analysis_dir = _resolve_subset_dir(DATA_ROOT_DIR, ANALYSIS_SUBSET)
    background_dir = _resolve_subset_dir(DATA_ROOT_DIR, BACKGROUND_SUBSET)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Analysis subset dir: {analysis_dir}")
    print(f"Background subset dir: {background_dir}")

    x_eval, y_eval_traces = _load_split_arrays(analysis_dir)
    x_train, y_train_traces = _load_split_arrays(background_dir)

    feature_names = _extract_feature_names(analysis_dir, x_eval.shape[1])
    feature_names = _apply_feature_label_map(feature_names)
    return XAIBundle(
        model=model,
        x_train=x_train,
        y_train_traces=y_train_traces,
        x_eval=x_eval,
        y_eval_traces=y_eval_traces,
        feature_names=feature_names,
    )


def _plot_importance(
    values: np.ndarray,
    feature_names: list[str],
    *,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    order = np.argsort(values)[::-1]
    vals = np.asarray(values)[order]
    names = np.asarray(feature_names)[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _plot_grouped_importance(
    values_by_output: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    if not values_by_output:
        raise ValueError("values_by_output must not be empty.")

    output_names = list(values_by_output.keys())
    matrix = np.stack([np.asarray(values_by_output[name], dtype=np.float64) for name in output_names], axis=0)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D importance matrix (O, F), got shape {matrix.shape}")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch. Expected {len(feature_names)} columns, got {matrix.shape[1]}"
        )

    # Normalize each parameter/output independently so feature importances sum to 1.
    denom = matrix.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    normalized = matrix / denom

    n_features = len(feature_names)
    n_outputs = len(output_names)
    x = np.arange(n_features, dtype=np.float64)
    group_width = 0.84
    bar_width = group_width / max(1, n_outputs)
    first_center_offset = -0.5 * group_width + 0.5 * bar_width

    fig_w = max(10.0, 0.45 * n_features)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    for output_idx, output_name in enumerate(output_names):
        offsets = x + first_center_offset + output_idx * bar_width
        ax.bar(offsets, normalized[output_idx], width=bar_width, label=output_name)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Normalized SHAP importance")
    # ax.set_title(title)
    ax.legend(title="Parameter")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _crank_angles_for_pressure_trace(n_outputs: int, cad_step_deg: float = 0.1) -> np.ndarray:
    cad_full, _, _ = engine_geometry(cad_step_deg=cad_step_deg)
    if n_outputs == cad_full.shape[0]:
        return cad_full
    return np.arange(n_outputs, dtype=np.float64) * cad_step_deg - 360.0


def _plot_shap_beeswarm(
    shap_values: Any,
    *,
    title: str,
    save_path: Optional[Path] = None,
    max_display: int = 20,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "matplotlib and shap are required for SHAP beeswarm plots. "
            "Install with: pip install matplotlib shap"
        ) from exc

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    if fig.axes:
        fig.axes[0].set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _slice_shap_explanation_for_output(shap_values: Any, output_index: int) -> Any:
    import shap

    values = np.asarray(shap_values.values)
    if values.ndim != 3:
        return shap_values

    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim == 2:
        sliced_base = base_values[:, output_index]
    elif base_values.ndim == 1:
        sliced_base = base_values
    else:
        sliced_base = base_values[..., output_index]

    return shap.Explanation(
        values=values[:, :, output_index],
        base_values=sliced_base,
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )


def _plot_ig_attribution_heatmap(
    attribution_matrix: np.ndarray,
    feature_names: list[str],
    crank_angles: np.ndarray,
    *,
    title: str,
    save_path: Optional[Path] = None,
    interpolation: str = "nearest",
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    matrix = np.asarray(attribution_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected attribution matrix with shape (F, T), got {matrix.shape}")
    if matrix.shape[0] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch. Expected {len(feature_names)} rows, got {matrix.shape[0]}"
        )
    if matrix.shape[1] != crank_angles.shape[0]:
        raise ValueError(
            f"Crank-angle dimension mismatch. Expected {crank_angles.shape[0]} columns, got {matrix.shape[1]}"
        )

    max_val = float(np.nanmax(matrix)) if matrix.size else 0.0
    positive_vals = matrix[matrix > 0.0]
    min_positive = float(np.nanmin(positive_vals)) if positive_vals.size else 0.0
    if max_val > 0.0 and min_positive > 0.0:
        # LogNorm requires strictly positive vmin/vmax.
        norm = LogNorm(vmin=min_positive, vmax=max_val)
    else:
        norm = None

    fig_h = max(4.0, 0.35 * len(feature_names))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation=interpolation,
        extent=[
            float(crank_angles[0]),
            float(crank_angles[-1]),
            -0.5,
            len(feature_names) - 0.5,
        ],
        cmap="viridis",
        norm=norm,
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Crank angle (deg)")
    ax.set_ylabel("Feature")
    # ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |attribution|")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def run_integrated_gradients(
    *,
    model: nn.Module,
    x_eval: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    target_output_index: Optional[int] = None,
    baseline: Optional[np.ndarray] = None,
    analysis_label: str = "output",
    output_tag: str = "output",
) -> Dict[str, Any]:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("captum is required for Integrated Gradients. Install with: pip install captum") from exc

    class _IGWrapper(nn.Module):
        def __init__(self, wrapped_model: nn.Module, output_index: Optional[int]):
            super().__init__()
            self.wrapped_model = wrapped_model
            self.output_index = output_index

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raw = _extract_tensor_output(self.wrapped_model(x))
            return _reduce_to_single_output(raw, self.output_index)

    wrapper = _IGWrapper(model, target_output_index).to(device)
    wrapper.eval()

    x_tensor = torch.from_numpy(np.asarray(x_eval, dtype=np.float32)).to(device)
    if baseline is None:
        baseline = np.zeros((1, x_eval.shape[1]), dtype=np.float32)
    baseline_tensor = torch.from_numpy(np.asarray(baseline, dtype=np.float32)).to(device)

    ig = IntegratedGradients(wrapper)
    attributions, convergence_delta = ig.attribute(
        inputs=x_tensor,
        baselines=baseline_tensor,
        return_convergence_delta=True,
    )
    attr_np = attributions.detach().cpu().numpy()
    mean_abs_attr = np.mean(np.abs(attr_np), axis=0)

    if SAVE_PLOTS or SHOW_PLOTS:
        save_path = (
            OUTPUT_DIR / f"integrated_gradients_feature_importance_{output_tag}.png"
            if SAVE_PLOTS
            else None
        )
        _plot_importance(
            mean_abs_attr,
            feature_names,
            title=f"Integrated Gradients ({analysis_label}, mean |attribution|)",
            save_path=save_path,
        )

    return {
        "attributions": attr_np,
        "mean_abs_attribution": mean_abs_attr,
        "convergence_delta_mean_abs": float(np.mean(np.abs(convergence_delta.detach().cpu().numpy()))),
    }


def run_integrated_gradients_trace(
    *,
    predictor: MultiMetricPredictor,
    x_eval: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    baseline: Optional[np.ndarray] = None,
    steps: int = 24,
    eval_size: int = 64,
    batch_chunk: int = 8,
    output_chunk: int = 256,
    return_full_attr: bool = False,
    analysis_label: str = "pressure_trace",
    use_jacfwd: bool = False,
    cad_range: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    steps = max(2, int(steps))
    eval_size = max(1, int(eval_size))
    batch_chunk = max(1, int(batch_chunk))
    output_chunk = max(1, int(output_chunk))

    x_subset = np.asarray(x_eval, dtype=np.float32)[: min(eval_size, x_eval.shape[0])]
    x_tensor = torch.from_numpy(x_subset).to(device)
    if baseline is None:
        baseline_np = np.zeros((x_tensor.shape[1],), dtype=np.float32)
    else:
        baseline_np = np.asarray(baseline, dtype=np.float32).reshape(-1)
    if baseline_np.shape[0] != x_tensor.shape[1]:
        raise ValueError(
            f"Baseline dimension mismatch. Expected {x_tensor.shape[1]}, got {baseline_np.shape[0]}"
        )
    baseline_tensor = torch.from_numpy(baseline_np).to(device)

    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device)
    n_samples = x_tensor.shape[0]
    output_dim = int(predictor.predict_trace_np(x_subset[:1]).shape[1])
    input_dim = x_tensor.shape[1]

    mean_abs_input_sum = np.zeros((input_dim,), dtype=np.float64)
    mean_abs_output_sum = np.zeros((output_dim,), dtype=np.float64)
    mean_abs_feature_crank_sum = np.zeros((input_dim, output_dim), dtype=np.float64)
    full_attr_chunks: list[np.ndarray] = []
    n_processed = 0
    using_forward_mode = False

    if use_jacfwd:
        try:
            from torch.func import jacfwd, vmap  # type: ignore

            def single_forward(x_single: torch.Tensor) -> torch.Tensor:
                out = predictor.predict_trace_torch(x_single.unsqueeze(0))
                return out.squeeze(0)

            jac_fn = jacfwd(single_forward)
            batched_jac_fn = vmap(jac_fn)
            using_forward_mode = True
        except Exception:
            batched_jac_fn = None
    else:
        batched_jac_fn = None

    for start in range(0, n_samples, batch_chunk):
        x_batch = x_tensor[start : start + batch_chunk]
        delta = x_batch - baseline_tensor.unsqueeze(0)
        ig_batch = torch.zeros((x_batch.shape[0], output_dim, input_dim), device=device, dtype=torch.float32)

        for alpha in alphas:
            x_step = baseline_tensor.unsqueeze(0) + alpha * delta
            x_step = x_step.requires_grad_(True)

            if using_forward_mode and batched_jac_fn is not None:
                jac = batched_jac_fn(x_step)
            else:
                jac_per_sample = []
                for sample_idx in range(x_step.shape[0]):
                    jac_s = torch.autograd.functional.jacobian(
                        lambda z: predictor.predict_trace_torch(z.unsqueeze(0)).squeeze(0),
                        x_step[sample_idx],
                    )
                    jac_per_sample.append(jac_s)
                jac = torch.stack(jac_per_sample, dim=0)
            ig_batch = ig_batch + jac * delta[:, None, :] / float(steps)

        ig_np = ig_batch.detach().cpu().numpy()
        mean_abs_input_sum += np.abs(ig_np).sum(axis=(0, 1))
        mean_abs_feature_crank_sum += np.abs(ig_np).sum(axis=0).T
        for out_start in range(0, output_dim, output_chunk):
            out_end = min(output_dim, out_start + output_chunk)
            chunk = np.abs(ig_np[:, out_start:out_end, :])
            mean_abs_output_sum[out_start:out_end] += chunk.sum(axis=(0, 2))

        if return_full_attr:
            full_attr_chunks.append(ig_np)
        n_processed += ig_np.shape[0]

    denom_input = max(1, n_processed * output_dim)
    denom_output = max(1, n_processed * input_dim)
    mean_abs_input = (mean_abs_input_sum / float(denom_input)).astype(np.float32)
    mean_abs_output = (mean_abs_output_sum / float(denom_output)).astype(np.float32)
    mean_abs_feature_crank = (mean_abs_feature_crank_sum / float(max(1, n_processed))).astype(np.float32)
    crank_angles = _crank_angles_for_pressure_trace(output_dim)

    if cad_range is not None:
        cad_min, cad_max = sorted((float(cad_range[0]), float(cad_range[1])))
        cad_mask = (crank_angles >= cad_min) & (crank_angles <= cad_max)
        if not np.any(cad_mask):
            raise ValueError(
                "Requested IG CAD range has no overlap with computed crank angles: "
                f"[{cad_min:.1f}, {cad_max:.1f}]"
            )
        crank_angles_plot = crank_angles[cad_mask]
        mean_abs_feature_crank_plot = mean_abs_feature_crank[:, cad_mask]
        heatmap_title_suffix = f", CAD {cad_min:.1f} to {cad_max:.1f} deg"
    else:
        crank_angles_plot = crank_angles
        mean_abs_feature_crank_plot = mean_abs_feature_crank
        heatmap_title_suffix = ""

    if SAVE_PLOTS or SHOW_PLOTS:
        heatmap_save = (
            OUTPUT_DIR / "integrated_gradients_trace_feature_crank_heatmap.png" if SAVE_PLOTS else None
        )
        _plot_ig_attribution_heatmap(
            mean_abs_feature_crank_plot,
            feature_names,
            crank_angles_plot,
            title=f"Integrated Gradients ({analysis_label}, feature × crank angle{heatmap_title_suffix})",
            save_path=heatmap_save,
        )
        bar_save = (
            OUTPUT_DIR / "integrated_gradients_trace_feature_importance.png" if SAVE_PLOTS else None
        )
        _plot_importance(
            mean_abs_input,
            feature_names,
            title=f"Integrated Gradients ({analysis_label}, mean |attribution| across trace)",
            save_path=bar_save,
        )

    result: Dict[str, Any] = {
        "mean_abs_attribution_by_input": mean_abs_input,
        "mean_abs_attribution_by_output": mean_abs_output,
        "mean_abs_attribution_feature_crank": mean_abs_feature_crank,
        "crank_angles_deg": crank_angles.astype(np.float32, copy=False),
        "plotted_crank_angles_deg": crank_angles_plot.astype(np.float32, copy=False),
        "num_eval": int(n_processed),
        "steps": steps,
        "mode": "jacfwd_vmap" if using_forward_mode else "autograd_jacobian_fallback",
    }
    if return_full_attr:
        result["attributions"] = np.concatenate(full_attr_chunks, axis=0) if full_attr_chunks else np.empty((0,))
    return result


def run_ale(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_train: np.ndarray,
    feature_names: list[str],
    grid_size: int = 20,
) -> Dict[str, Any]:
    try:
        from PyALE import ale
    except ImportError:
        try:
            from pyale import ale  # type: ignore
        except ImportError as exc:
            raise ImportError("pyale is required for ALE. Install with: pip install pyALE") from exc

    x_train_df = pd.DataFrame(x_train, columns=feature_names)
    def model_predict_for_ale(x_df: pd.DataFrame) -> np.ndarray:
        return predict_fn(x_df.to_numpy(dtype=np.float32, copy=False))

    ale_results: Dict[str, Any] = {}
    for feature in feature_names:
        res = ale(
            X=x_train_df,
            model=model_predict_for_ale,
            feature=[feature],
            grid_size=grid_size,
            include_CI=False,
            plot=SHOW_PLOTS,
        )
        ale_results[feature] = res
    return {"ale_per_feature": ale_results}


def run_shap(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_train: np.ndarray,
    x_eval: np.ndarray,
    feature_names: list[str],
    background_size: int = 200,
    eval_size: int = 400,
    analysis_label: str = "output",
    output_tag: str = "output",
    cache_path: Optional[Path] = None,
    load_from_cache: bool = True,
    save_to_cache: bool = True,
) -> Dict[str, Any]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP analysis. Install with: pip install shap") from exc

    n_bg = min(background_size, x_train.shape[0])
    n_eval = min(eval_size, x_eval.shape[0])
    loaded_from_cache = False

    if cache_path is not None and load_from_cache and cache_path.exists():
        try:
            payload = _load_pickle(cache_path)
            shap_values = payload["shap_values"]
            mean_abs_shap = np.asarray(payload["mean_abs_shap"], dtype=np.float32)
            n_bg = int(payload.get("num_background", n_bg))
            n_eval = int(payload.get("num_eval", n_eval))
            cached_features = list(payload.get("feature_names", feature_names))
            if cached_features != list(feature_names):
                raise ValueError("Feature names in SHAP cache do not match current feature names.")
            loaded_from_cache = True
            print(f"Loaded SHAP cache: {cache_path}")
        except Exception as exc:
            print(f"Failed to load SHAP cache ({cache_path}); recomputing SHAP. Error: {exc}")

    if not loaded_from_cache:
        bg_idx = np.random.default_rng(RANDOM_SEED).choice(x_train.shape[0], n_bg, replace=False)
        eval_idx = np.random.default_rng(RANDOM_SEED + 1).choice(x_eval.shape[0], n_eval, replace=False)

        x_background = x_train[bg_idx]
        x_target = x_eval[eval_idx]

        def predict_for_shap(arr: np.ndarray) -> np.ndarray:
            return predict_fn(np.asarray(arr, dtype=np.float32))

        explainer = shap.explainers.Permutation(
            predict_for_shap,
            x_background,
            feature_names=feature_names,
        )
        shap_values = explainer(x_target)

        values = shap_values.values
        if values.ndim == 3:
            values = values[..., 0]
        mean_abs_shap = np.mean(np.abs(values), axis=0)

        if cache_path is not None and save_to_cache:
            try:
                _save_pickle(
                    cache_path,
                    {
                        "shap_values": shap_values,
                        "mean_abs_shap": np.asarray(mean_abs_shap, dtype=np.float32),
                        "num_background": int(n_bg),
                        "num_eval": int(n_eval),
                        "feature_names": list(feature_names),
                        "analysis_label": analysis_label,
                        "output_tag": output_tag,
                    },
                )
                print(f"Saved SHAP cache: {cache_path}")
            except Exception as exc:
                print(f"Failed to save SHAP cache ({cache_path}). Error: {exc}")

    if SAVE_PLOTS or SHOW_PLOTS:
        beeswarm_save = OUTPUT_DIR / f"shap_beeswarm_{output_tag}.png" if SAVE_PLOTS else None
        _plot_shap_beeswarm(
            shap_values,
            title=f"SHAP beeswarm ({analysis_label})",
            save_path=beeswarm_save,
        )
        bar_save = OUTPUT_DIR / f"shap_feature_importance_{output_tag}.png" if SAVE_PLOTS else None
        _plot_importance(
            mean_abs_shap,
            feature_names,
            title=f"SHAP ({analysis_label}, mean |value|)",
            save_path=bar_save,
        )

    return {
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "num_background": n_bg,
        "num_eval": n_eval,
    }


def run_shap_multi_output(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_train: np.ndarray,
    x_eval: np.ndarray,
    feature_names: list[str],
    output_names: list[str],
    output_display_names: Optional[list[str]] = None,
    background_size: int = 200,
    eval_size: int = 400,
    cache_path: Optional[Path] = None,
    load_from_cache: bool = True,
    save_to_cache: bool = True,
) -> Dict[str, Any]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP analysis. Install with: pip install shap") from exc

    if output_display_names is None:
        output_display_names = output_names
    if len(output_display_names) != len(output_names):
        raise ValueError(
            "output_display_names length must match output_names length. "
            f"Got {len(output_display_names)} vs {len(output_names)}."
        )

    n_bg = min(background_size, x_train.shape[0])
    n_eval = min(eval_size, x_eval.shape[0])
    loaded_from_cache = False

    if cache_path is not None and load_from_cache and cache_path.exists():
        try:
            payload = _load_pickle(cache_path)
            shap_values = payload["shap_values"]
            n_bg = int(payload.get("num_background", n_bg))
            n_eval = int(payload.get("num_eval", n_eval))
            cached_features = list(payload.get("feature_names", feature_names))
            if cached_features != list(feature_names):
                raise ValueError("Feature names in SHAP cache do not match current feature names.")
            cached_output_names = list(payload.get("output_names", output_names))
            if cached_output_names != list(output_names):
                raise ValueError("Output names in SHAP cache do not match current output names.")
            print(f"Loaded shared multi-output SHAP cache: {cache_path}")
            loaded_from_cache = True
        except Exception as exc:
            print(
                f"Failed to load shared multi-output SHAP cache ({cache_path}); "
                f"recomputing SHAP. Error: {exc}"
            )

    if not loaded_from_cache:
        bg_idx = np.random.default_rng(RANDOM_SEED).choice(x_train.shape[0], n_bg, replace=False)
        eval_idx = np.random.default_rng(RANDOM_SEED + 1).choice(x_eval.shape[0], n_eval, replace=False)

        x_background = x_train[bg_idx]
        x_target = x_eval[eval_idx]

        def predict_for_shap(arr: np.ndarray) -> np.ndarray:
            return predict_fn(np.asarray(arr, dtype=np.float32))

        explainer = shap.explainers.Permutation(
            predict_for_shap,
            x_background,
            feature_names=feature_names,
            output_names=output_names,
        )
        shap_values = explainer(x_target)

        if cache_path is not None and save_to_cache:
            try:
                _save_pickle(
                    cache_path,
                    {
                        "shap_values": shap_values,
                        "num_background": int(n_bg),
                        "num_eval": int(n_eval),
                        "feature_names": list(feature_names),
                        "output_names": list(output_names),
                    },
                )
                print(f"Saved shared multi-output SHAP cache: {cache_path}")
            except Exception as exc:
                print(f"Failed to save shared multi-output SHAP cache ({cache_path}). Error: {exc}")

    values = np.asarray(shap_values.values)
    if values.ndim != 3:
        raise ValueError(f"Expected multi-output SHAP values with shape (N, F, O), got {values.shape}")
    if values.shape[2] != len(output_names):
        raise ValueError(
            f"Unexpected SHAP output dimension. Expected {len(output_names)} outputs, got {values.shape[2]}"
        )

    per_output: dict[str, Any] = {}
    grouped_mean_abs_display: dict[str, np.ndarray] = {}
    for output_idx, output_name in enumerate(output_names):
        output_display = output_display_names[output_idx]
        mean_abs = np.mean(np.abs(values[:, :, output_idx]), axis=0)
        grouped_mean_abs_display[output_display] = mean_abs
        if SAVE_PLOTS or SHOW_PLOTS:
            output_tag = output_name.lower().replace("/", "_")
            beeswarm_save = OUTPUT_DIR / f"shap_beeswarm_{output_tag}.png" if SAVE_PLOTS else None
            output_explanation = _slice_shap_explanation_for_output(shap_values, output_idx)
            _plot_shap_beeswarm(
                output_explanation,
                title=f"SHAP beeswarm ({output_display})",
                save_path=beeswarm_save,
            )
        per_output[output_name] = {
            "mean_abs_shap": mean_abs,
            "output_index": output_idx,
        }

    if SAVE_PLOTS or SHOW_PLOTS:
        grouped_save = OUTPUT_DIR / "shap_feature_importance_grouped_normalized.png" if SAVE_PLOTS else None
        _plot_grouped_importance(
            grouped_mean_abs_display,
            feature_names,
            title="SHAP normalized feature importance by parameter",
            save_path=grouped_save,
        )

    return {
        "shap_values": shap_values,
        "per_output": per_output,
        "grouped_mean_abs_shap": grouped_mean_abs_display,
        "num_background": n_bg,
        "num_eval": n_eval,
    }


class _TorchPredictorEstimator:
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
    ):
        self.predict_fn = predict_fn

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_TorchPredictorEstimator":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_fn(np.asarray(x, dtype=np.float32))

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return r2_score(y, y_pred)


def run_permutation_importance(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 12,
    analysis_label: str = "output",
    output_tag: str = "output",
) -> Dict[str, Any]:
    estimator = _TorchPredictorEstimator(
        predict_fn,
    )
    result = permutation_importance(
        estimator,
        x_eval,
        y_eval,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="neg_mean_squared_error",
    )
    mean_importance = result.importances_mean

    if SAVE_PLOTS or SHOW_PLOTS:
        save_path = OUTPUT_DIR / f"permutation_importance_{output_tag}.png" if SAVE_PLOTS else None
        _plot_importance(
            mean_importance,
            feature_names,
            title=f"Permutation Importance ({analysis_label}, neg MSE drop)",
            save_path=save_path,
        )

    return {
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "raw_result": result,
    }


def main() -> Dict[str, Any]:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    _ensure_dir(OUTPUT_DIR)
    if LOAD_SHAP_FROM_CACHE or SAVE_SHAP_TO_CACHE:
        _ensure_dir(SHAP_CACHE_DIR)

    device = resolve_device(DEVICE)
    bundle = load_xai_bundle(device=device)
    model = bundle.model
    _cad, volume, vd = engine_geometry(cad_step_deg=0.1)
    method_times_sec: dict[str, float] = {}

    target_names = _normalize_explanation_targets(EXPLANATION_TARGETS)
    use_metric_targets = len(target_names) > 0
    metric_names = [name for name in METRIC_TARGET_ORDER if name in target_names]
    target_display_names = {name: _target_display_label(name) for name in target_names}
    predictor = MultiMetricPredictor(
        model=model,
        device=device,
        volume=volume,
        vd=vd,
        batch_size=BATCH_SIZE,
        metric_names=metric_names if metric_names else list(METRIC_TARGET_ORDER),
        use_cache=USE_PREDICTOR_CACHE,
        max_cache_entries=PREDICTOR_CACHE_MAX_ENTRIES,
        target_output_index=TARGET_OUTPUT_INDEX,
    )

    if use_metric_targets:
        print(f"Using digested metric targets: {target_names}")
        y_eval_targets = _extract_metric_targets(
            bundle.y_eval_traces,
            target_names,
            volume=volume,
            vd=vd,
        )
    else:
        fallback_name = (
            f"output_{TARGET_OUTPUT_INDEX}" if TARGET_OUTPUT_INDEX is not None else "output_mean"
        )
        target_names = [fallback_name]
        target_display_names = {fallback_name: _target_display_label(fallback_name)}
        y_eval_reduced = _reduce_to_single_output(
            torch.from_numpy(bundle.y_eval_traces),
            TARGET_OUTPUT_INDEX,
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        y_eval_targets = {fallback_name: y_eval_reduced}

    print(f"Device: {device}")
    print(f"Eval samples: {bundle.x_eval.shape[0]}")
    print(
        "Predictor cache settings | "
        f"enabled={USE_PREDICTOR_CACHE} | max_entries={PREDICTOR_CACHE_MAX_ENTRIES}"
    )

    def predict_target(x: np.ndarray, target_name: str) -> np.ndarray:
        if use_metric_targets:
            return predictor.predict_metric_np(x, target_name)
        return predictor.predict_reduced_output_np(x)

    results: Dict[str, Any] = {}
    target_results: Dict[str, Any] = {}
    shared_shap_per_target: dict[str, Dict[str, Any]] = {}
    if RUN_SHAP and use_metric_targets and RUN_MULTI_OUTPUT_SHAP and metric_names:
        try:
            y_pred_eval_matrix = predictor.predict_metrics_np(bundle.x_eval)
            y_true_eval_matrix = np.stack([y_eval_targets[name] for name in metric_names], axis=1)
            finite_shared = np.isfinite(y_true_eval_matrix).all(axis=1) & np.isfinite(y_pred_eval_matrix).all(axis=1)
            if np.any(finite_shared):
                start_time = time.perf_counter()
                print("\nRunning shared multi-output SHAP...")
                shared_cache_tag = "__".join(metric_names)
                shared_cache_path = _build_shap_cache_path("shap_multi_output", shared_cache_tag)
                shared_shap = run_shap_multi_output(
                    predict_fn=predictor.predict_metrics_np,
                    x_train=bundle.x_train,
                    x_eval=bundle.x_eval[finite_shared],
                    feature_names=bundle.feature_names,
                    output_names=metric_names,
                    output_display_names=[target_display_names.get(name, name) for name in metric_names],
                    background_size=SHAP_BACKGROUND_SIZE,
                    eval_size=SHAP_EVAL_SIZE,
                    cache_path=shared_cache_path,
                    load_from_cache=LOAD_SHAP_FROM_CACHE,
                    save_to_cache=SAVE_SHAP_TO_CACHE,
                )
                method_times_sec["shap_multi_output"] = time.perf_counter() - start_time
                shared_shap_per_target = {
                    name: {
                        "mean_abs_shap": shared_shap["per_output"][name]["mean_abs_shap"],
                        "shared_explainer": True,
                        "output_index": shared_shap["per_output"][name]["output_index"],
                        "num_background": shared_shap["num_background"],
                        "num_eval": shared_shap["num_eval"],
                    }
                    for name in metric_names
                }
                results["shap_multi_output"] = shared_shap
                print("Shared multi-output SHAP done.")
            else:
                print("Shared multi-output SHAP skipped: no rows with all-finite metric targets.")
        except Exception as exc:
            print(f"Shared multi-output SHAP failed; falling back to per-target SHAP. Error: {exc}")
            results["shap_multi_output_error"] = str(exc)

    for target_name in target_names:
        target_display = target_display_names.get(target_name, target_name)
        target_key = target_name.lower()
        output_tag = target_key.replace("/", "_")
        y_eval = y_eval_targets[target_name]
        predict_fn = lambda x, tn=target_name: predict_target(x, tn)

        y_pred = predict_fn(bundle.x_eval)
        finite_mask = np.isfinite(y_eval) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            print(f"[{target_name}] No finite rows to evaluate; skipping this target.")
            target_results[target_name] = {
                "error": "No finite rows available after metric extraction.",
            }
            continue
        if np.any(~finite_mask):
            dropped = int((~finite_mask).sum())
            print(f"[{target_name}] Dropping {dropped} non-finite rows before analysis.")

        x_eval_valid = bundle.x_eval[finite_mask]
        y_eval_valid = y_eval[finite_mask]
        eval_mse = mean_squared_error(y_eval_valid, y_pred[finite_mask])
        eval_r2 = r2_score(y_eval_valid, y_pred[finite_mask])
        print(f"[{target_name}] Baseline regression metrics | MSE={eval_mse:.6f} | R2={eval_r2:.6f}")

        target_out: Dict[str, Any] = {
            "baseline_mse": float(eval_mse),
            "baseline_r2": float(eval_r2),
            "num_eval_rows": int(finite_mask.sum()),
        }

        if RUN_INTEGRATED_GRADIENTS:
            if use_metric_targets:
                msg = (
                    "Integrated Gradients is skipped for digested metrics because IG requires "
                    "a differentiable scalar output directly from the model."
                )
                target_out["integrated_gradients_error"] = msg
                print(f"[{target_name}] {msg}")
            else:
                print(f"[{target_name}] Running Integrated Gradients...")
                try:
                    target_out["integrated_gradients"] = run_integrated_gradients(
                        model=model,
                        x_eval=x_eval_valid,
                        feature_names=bundle.feature_names,
                        device=device,
                        target_output_index=TARGET_OUTPUT_INDEX,
                        baseline=np.mean(bundle.x_train, axis=0, keepdims=True).astype(np.float32),
                        analysis_label=target_display,
                        output_tag=output_tag,
                    )
                    print(f"[{target_name}] Integrated Gradients done.")
                except Exception as exc:
                    target_out["integrated_gradients_error"] = str(exc)
                    print(f"[{target_name}] Integrated Gradients failed: {exc}")

        if RUN_ALE:
            print(f"[{target_name}] Running ALE...")
            try:
                start_time = time.perf_counter()
                target_out["ale"] = run_ale(
                    predict_fn=predict_fn,
                    x_train=bundle.x_train,
                    feature_names=bundle.feature_names,
                    grid_size=ALE_GRID_SIZE,
                )
                method_times_sec[f"ale_{target_key}"] = time.perf_counter() - start_time
                print(f"[{target_name}] ALE done.")
            except Exception as exc:
                target_out["ale_error"] = str(exc)
                print(f"[{target_name}] ALE failed: {exc}")

        if RUN_SHAP:
            if target_name in shared_shap_per_target:
                target_out["shap"] = shared_shap_per_target[target_name]
                print(f"[{target_name}] SHAP done via shared multi-output explainer.")
            else:
                print(f"[{target_name}] Running SHAP...")
                try:
                    start_time = time.perf_counter()
                    target_cache_path = _build_shap_cache_path("shap_single_output", output_tag)
                    target_out["shap"] = run_shap(
                        predict_fn=predict_fn,
                        x_train=bundle.x_train,
                        x_eval=x_eval_valid,
                        feature_names=bundle.feature_names,
                        background_size=SHAP_BACKGROUND_SIZE,
                        eval_size=SHAP_EVAL_SIZE,
                        analysis_label=target_display,
                        output_tag=output_tag,
                        cache_path=target_cache_path,
                        load_from_cache=LOAD_SHAP_FROM_CACHE,
                        save_to_cache=SAVE_SHAP_TO_CACHE,
                    )
                    method_times_sec[f"shap_{target_key}"] = time.perf_counter() - start_time
                    print(f"[{target_name}] SHAP done.")
                except Exception as exc:
                    target_out["shap_error"] = str(exc)
                    print(f"[{target_name}] SHAP failed: {exc}")

        if RUN_PERMUTATION_IMPORTANCE:
            print(f"[{target_name}] Running permutation importance...")
            try:
                start_time = time.perf_counter()
                target_out["permutation_importance"] = run_permutation_importance(
                    predict_fn=predict_fn,
                    x_eval=x_eval_valid,
                    y_eval=y_eval_valid,
                    feature_names=bundle.feature_names,
                    n_repeats=PERMUTATION_REPEATS,
                    analysis_label=target_display,
                    output_tag=output_tag,
                )
                method_times_sec[f"permutation_{target_key}"] = time.perf_counter() - start_time
                print(f"[{target_name}] Permutation importance done.")
            except Exception as exc:
                target_out["permutation_importance_error"] = str(exc)
                print(f"[{target_name}] Permutation importance failed: {exc}")

        target_results[target_name] = target_out

    if RUN_IG_TRACE:
        print("\nRunning Integrated Gradients trace mode...")
        ig_predictor = MultiMetricPredictor(
            model=model,
            device=device,
            volume=volume,
            vd=vd,
            batch_size=BATCH_SIZE,
            metric_names=list(METRIC_TARGET_ORDER),
            use_cache=USE_IG_CACHE,
            max_cache_entries=PREDICTOR_CACHE_MAX_ENTRIES,
            target_output_index=TARGET_OUTPUT_INDEX,
        )
        try:
            start_time = time.perf_counter()
            results["integrated_gradients_trace"] = run_integrated_gradients_trace(
                predictor=ig_predictor,
                x_eval=bundle.x_eval,
                feature_names=bundle.feature_names,
                device=device,
                baseline=np.mean(bundle.x_train, axis=0).astype(np.float32),
                steps=IG_TRACE_STEPS,
                eval_size=IG_TRACE_EVAL_SIZE,
                batch_chunk=IG_BATCH_CHUNK,
                output_chunk=IG_OUTPUT_CHUNK,
                return_full_attr=IG_RETURN_FULL_ATTR,
                use_jacfwd=IG_USE_JACFWD,
                cad_range=IG_TRACE_CAD_RANGE,
            )
            method_times_sec["integrated_gradients_trace"] = time.perf_counter() - start_time
            print("Integrated Gradients trace mode done.")
        except Exception as exc:
            results["integrated_gradients_trace_error"] = str(exc)
            print(f"Integrated Gradients trace mode failed: {exc}")

    results["targets"] = target_results
    results["target_names"] = target_names
    results["timings_sec"] = method_times_sec
    results["predictor_stats"] = predictor.stats
    print(
        "Predictor stats | "
        f"trace_forward_calls={predictor.stats['trace_forward_calls']} | "
        f"trace_cache_hits={predictor.stats['trace_cache_hits']} | "
        f"metrics_cache_hits={predictor.stats['metrics_cache_hits']}"
    )
    if method_times_sec:
        print("Timing summary (seconds):")
        for key in sorted(method_times_sec):
            print(f"  {key}: {method_times_sec[key]:.3f}")
    print(f"\nFinished. Plot outputs (if enabled) are in: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    analysis_results = main()
