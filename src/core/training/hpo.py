import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch.distributed as dist

try:  # Optional dependency for legacy-only users.
    from ray.tune.stopper import Stopper as _RayStopper
except Exception:  # pragma: no cover - fallback when ray is unavailable
    class _RayStopper:  # type: ignore[too-many-ancestors]
        pass


def suggest_float(low, high, scale="linear", rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    if scale == "linear":
        num = rng.random() * (high - low) + low
    elif scale == "log":
        if low <= 0 or high <= 0:
            raise ValueError("Log scale sampling requires low > 0 and high > 0")
        if low > high:
            raise ValueError(f"Expected low <= high, got low={low}, high={high}")
        num = np.exp(rng.uniform(np.log(low), np.log(high)))
    else:
        raise ValueError("scale must be linear or log")
    return float(num)


def suggest_int(low, high, step=1, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    rescale = int((high - low) / step)
    sample = int(rng.integers(rescale + 1))
    return int(sample * step + low)


def suggest_categorical(cats, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    sample = int(rng.integers(len(cats)))
    return cats[sample]


class HPOGeneral:
    def __init__(
        self,
        param_configs: Dict[str, Dict[str, Any]],
        metrics: Optional[Sequence[str]] = ("metric",),
        seed: Optional[int] = None,
    ):
        self.param_configs = param_configs
        self.pg_avail = dist.is_available() and dist.is_initialized()
        self.run_id = self._generate_run_id()
        self._rng = np.random.default_rng(seed)
        self.logger: Dict[str, Any] = {
            "hyperparameter": {name: [] for name in param_configs},
            "performance": {m: [] for m in (metrics or [])},
        }

    @staticmethod
    def _random_run_id() -> str:
        return f"{secrets.randbelow(1000):03d}"

    def _generate_run_id(self) -> str:
        if self.pg_avail:
            run_id = self._random_run_id() if dist.get_rank() == 0 else None
            payload = [run_id]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
        return self._random_run_id()

    def unique_log_path(self, filename: str) -> str:
        path = Path(filename)
        run_suffix = f"_{self.run_id}"
        if path.stem.endswith(run_suffix):
            return str(path)
        ext = path.suffix
        return str(path.with_name(f"{path.stem}{run_suffix}{ext}"))

    def build_log_path(
        self,
        directory: Union[str, Path],
        stem: str = "hpo_log",
        ext: str = "parquet",
    ) -> str:
        ext = ext.lstrip(".")
        return self.unique_log_path(str(Path(directory) / f"{stem}.{ext}"))

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [HPOGeneral._serialize_value(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [HPOGeneral._serialize_value(item) for item in value]
        if isinstance(value, tuple):
            return [HPOGeneral._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: HPOGeneral._serialize_value(item) for key, item in value.items()}
        return value

    def _check_log_alignment(self, metric: str = None) -> None:
        hparam_counts = [len(v) for v in self.logger["hyperparameter"].values()]
        n_samples = hparam_counts[0] if hparam_counts else 0

        if metric is None:
            for key in self.logger["performance"].keys():
                n_perf = len(self.logger["performance"][key])
                if n_samples - n_perf >= 1:
                    raise RuntimeError(
                        f"Hyperparameters sampled {n_samples} times but '{key}' has {n_perf} "
                        "entries. Log performance before sampling again."
                    )
        else:
            n_perf = len(self.logger["performance"][metric])
            if n_perf >= n_samples:
                raise RuntimeError(
                    f"'{metric}' has {n_perf} entries but only {n_samples} hyperparameter samples. "
                    "Sample again before logging performance."
                )

    def sample(self) -> Dict[str, Any]:
        self._check_log_alignment()

        params = {}
        for name, cfg in self.param_configs.items():
            param_type = cfg.get("type")
            if param_type == "float":
                val = suggest_float(
                    cfg["low"], cfg["high"], cfg.get("scale", "linear"), rng=self._rng
                )
            elif param_type == "int":
                val = suggest_int(cfg["low"], cfg["high"], cfg.get("step", 1), rng=self._rng)
            elif param_type == "categorical":
                val = suggest_categorical(cfg["choices"], rng=self._rng)
            else:
                raise ValueError(f"Unknown type '{param_type}' for hyperparameter '{name}'")
            if isinstance(val, np.generic):
                val = val.item()
            params[name] = val
            self.logger["hyperparameter"][name].append(val)

        return self.distribute_parameters(params)

    def distribute_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pg_avail:
            return params
        obj = [params]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def log_performance(self, value: Union[float, list, dict, np.ndarray], *, metric="metric"):
        self._check_log_alignment(metric)
        if metric not in self.logger["performance"]:
            raise KeyError(
                f"Metric '{metric}' not initialized. Define metrics in HPOGeneral constructor."
            )
        self.logger["performance"][metric].append(value)

    def save_log(self, filename: str) -> str:
        filename = self.unique_log_path(filename)
        pg_avail = self.pg_avail and dist.is_available() and dist.is_initialized()
        is_writer = (dist.get_rank() == 0) if pg_avail else True
        if is_writer:
            raw_data = {**self.logger["hyperparameter"], **self.logger["performance"]}
            data = {
                key: [self._serialize_value(item) for item in values]
                for key, values in raw_data.items()
            }
            df = pd.DataFrame(data)
            ext = Path(filename).suffix.lstrip(".").lower()
            if ext == "csv":
                df.to_csv(filename, index=False)
            elif ext == "json":
                df.to_json(filename, orient="records")
            elif ext in ("pkl", "pickle"):
                df.to_pickle(filename)
            elif ext == "parquet":
                df.to_parquet(filename, index=False)
            else:
                df.to_parquet(filename)
            print(f"Wrote HPO log to {filename} (run_id={self.run_id})")
        if pg_avail:
            dist.barrier()
        return filename


@dataclass
class RayTunePruningConfig:
    grace_period: int = 3
    reduction_factor: float = 2.0
    plateau_patience: int = 6
    plateau_min_delta: float = 1e-4
    overfit_ratio_threshold: float = 0.25
    overfit_patience: int = 3
    overfit_eps: float = 1e-8


class PlateauOverfitStopper(_RayStopper):
    """Stops trials with stagnating validation loss or persistent overfitting."""

    def __init__(
        self,
        *,
        metric: str,
        train_metric: str,
        patience: int,
        min_delta: float,
        overfit_ratio_threshold: float,
        overfit_patience: int,
        eps: float = 1e-8,
    ) -> None:
        self.metric = metric
        self.train_metric = train_metric
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.overfit_ratio_threshold = float(overfit_ratio_threshold)
        self.overfit_patience = int(overfit_patience)
        self.eps = float(eps)
        self._state: Dict[str, Dict[str, Any]] = {}

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        val_value = result.get(self.metric)
        train_value = result.get(self.train_metric)
        if val_value is None or train_value is None:
            return False

        val_loss = float(val_value)
        train_loss = max(float(train_value), self.eps)
        trial_state = self._state.setdefault(
            trial_id, {"best": float("inf"), "bad_epochs": 0, "overfit_epochs": 0}
        )

        improved = val_loss < (trial_state["best"] - self.min_delta)
        if improved:
            trial_state["best"] = val_loss
            trial_state["bad_epochs"] = 0
        else:
            trial_state["bad_epochs"] += 1

        overfit_ratio = (val_loss - train_loss) / train_loss
        if overfit_ratio > self.overfit_ratio_threshold:
            trial_state["overfit_epochs"] += 1
        else:
            trial_state["overfit_epochs"] = 0

        hit_plateau = trial_state["bad_epochs"] >= self.patience
        hit_overfit = trial_state["overfit_epochs"] >= self.overfit_patience
        return hit_plateau or hit_overfit

    def stop_all(self) -> bool:
        return False


def _ensure_ray_imports():
    try:
        from ray.tune.schedulers import ASHAScheduler  # noqa: F401
        from ray.tune.stopper import CombinedStopper, Stopper  # noqa: F401
        from ray import tune  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise ImportError(
            "Ray Tune is required for --hpo-backend ray. Install ray[tune] or ray with Tune extras."
        ) from exc


def build_tune_search_space(param_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    _ensure_ray_imports()
    from ray import tune

    space: Dict[str, Any] = {}
    for name, cfg in param_configs.items():
        param_type = cfg.get("type")
        if param_type == "float":
            scale = cfg.get("scale", "linear")
            low = float(cfg["low"])
            high = float(cfg["high"])
            if scale == "log":
                space[name] = tune.loguniform(low, high)
            elif scale == "linear":
                space[name] = tune.uniform(low, high)
            else:
                raise ValueError(f"Unsupported float scale '{scale}' for parameter '{name}'")
        elif param_type == "int":
            low = int(cfg["low"])
            high = int(cfg["high"])
            step = int(cfg.get("step", 1))
            if step <= 0:
                raise ValueError(f"Invalid step={step} for parameter '{name}'")
            space[name] = tune.qrandint(low, high + 1, step)
        elif param_type == "categorical":
            choices = list(cfg["choices"])
            space[name] = tune.choice(choices)
        else:
            raise ValueError(f"Unknown type '{param_type}' for hyperparameter '{name}'")
    return space


def build_asha_scheduler(
    *,
    metric: str,
    mode: str,
    max_t: int,
    grace_period: int,
    reduction_factor: float,
):
    _ensure_ray_imports()
    from ray.tune.schedulers import ASHAScheduler

    return ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=max_t,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        time_attr="training_iteration",
    )


def build_combined_stopper(
    pruning: RayTunePruningConfig,
    *,
    val_metric: str = "val_loss",
    train_metric: str = "train_loss",
):
    _ensure_ray_imports()
    from ray.tune.stopper import CombinedStopper

    return CombinedStopper(
        PlateauOverfitStopper(
            metric=val_metric,
            train_metric=train_metric,
            patience=pruning.plateau_patience,
            min_delta=pruning.plateau_min_delta,
            overfit_ratio_threshold=pruning.overfit_ratio_threshold,
            overfit_patience=pruning.overfit_patience,
            eps=pruning.overfit_eps,
        )
    )


def _pad_curve(values: Sequence[float], max_len: int) -> np.ndarray:
    padded = np.full((max_len,), np.nan, dtype=float)
    n = min(len(values), max_len)
    if n > 0:
        padded[:n] = np.asarray(values[:n], dtype=float)
    return padded


def _extract_metric_curve(history: Sequence[Dict[str, Any]], metric: str) -> List[float]:
    return [float(row[metric]) for row in history if metric in row]


def _extract_scalar_metric(curve: Sequence[float]) -> float:
    if len(curve) == 0:
        return float("nan")
    return float(curve[-1])


def ray_trials_to_hpo_logger(
    *,
    trial_results: Sequence[Dict[str, Any]],
    param_configs: Dict[str, Dict[str, Any]],
    seed: Optional[int] = None,
) -> HPOGeneral:
    metrics = [
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
    logger = HPOGeneral(param_configs=param_configs, metrics=metrics, seed=seed)
    max_epochs = 0
    for trial in trial_results:
        train_history = trial.get("train_history", [])
        max_epochs = max(max_epochs, len(train_history))

    for trial in trial_results:
        params = trial["params"]
        for name in param_configs:
            logger.logger["hyperparameter"][name].append(params[name])

        train_history = trial.get("train_history", [])
        val_history = trial.get("val_history", [])
        train_loss_curve = _extract_metric_curve(train_history, "loss")
        train_mse_curve = _extract_metric_curve(train_history, "mse")
        train_mae_curve = _extract_metric_curve(train_history, "mae")
        val_loss_curve = _extract_metric_curve(val_history, "loss")
        val_mse_curve = _extract_metric_curve(val_history, "mse")
        val_mae_curve = _extract_metric_curve(val_history, "mae")

        logger.logger["performance"]["mse_dp"].append(_extract_scalar_metric(val_loss_curve))
        logger.logger["performance"]["mse"].append(_extract_scalar_metric(val_mse_curve))
        logger.logger["performance"]["mae"].append(_extract_scalar_metric(val_mae_curve))
        logger.logger["performance"]["mse_dp_epoch_train"].append(
            _pad_curve(train_loss_curve, max_epochs)
        )
        logger.logger["performance"]["mse_epoch_train"].append(
            _pad_curve(train_mse_curve, max_epochs)
        )
        logger.logger["performance"]["mae_epoch_train"].append(
            _pad_curve(train_mae_curve, max_epochs)
        )
        logger.logger["performance"]["mse_dp_epoch_val"].append(
            _pad_curve(val_loss_curve, max_epochs)
        )
        logger.logger["performance"]["mse_epoch_val"].append(_pad_curve(val_mse_curve, max_epochs))
        logger.logger["performance"]["mae_epoch_val"].append(_pad_curve(val_mae_curve, max_epochs))

    return logger


def flatten_tuner_result_grid(result_grid) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for result in result_grid:
        metrics_df = result.metrics_dataframe
        params = dict(result.config)
        train_history: List[Dict[str, Any]] = []
        val_history: List[Dict[str, Any]] = []
        if metrics_df is not None and not metrics_df.empty:
            for _, row in metrics_df.iterrows():
                epoch = None
                if "epoch" in row and pd.notna(row["epoch"]):
                    epoch = int(row["epoch"])
                elif "training_iteration" in row and pd.notna(row["training_iteration"]):
                    epoch = int(row["training_iteration"] - 1)
                if epoch is None:
                    continue
                train_entry = {"epoch": epoch}
                val_entry = {"epoch": epoch}

                if "train_loss" in row and pd.notna(row["train_loss"]):
                    train_entry["loss"] = float(row["train_loss"])
                if "train_mse" in row and pd.notna(row["train_mse"]):
                    train_entry["mse"] = float(row["train_mse"])
                if "train_mae" in row and pd.notna(row["train_mae"]):
                    train_entry["mae"] = float(row["train_mae"])
                if len(train_entry) > 1:
                    train_history.append(train_entry)

                if "val_loss" in row and pd.notna(row["val_loss"]):
                    val_entry["loss"] = float(row["val_loss"])
                if "val_mse" in row and pd.notna(row["val_mse"]):
                    val_entry["mse"] = float(row["val_mse"])
                if "val_mae" in row and pd.notna(row["val_mae"]):
                    val_entry["mae"] = float(row["val_mae"])
                if len(val_entry) > 1:
                    val_history.append(val_entry)

        trials.append(
            {
                "params": params,
                "train_history": train_history,
                "val_history": val_history,
                "best_val_loss": float(result.metrics.get("val_loss", np.nan)),
            }
        )
    return trials
