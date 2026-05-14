from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch.distributed as dist


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
        self._rng = np.random.default_rng(seed)
        self.param_configs = param_configs
        self.pg_avail = dist.is_available() and dist.is_initialized()
        self.logger: Dict[str, Any] = {
            "hyperparameter": {name: [] for name in param_configs},
            "performance": {m: [] for m in (metrics or [])},
        }

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

    def save_log(self, filename: str) -> None:
        pg_avail = self.pg_avail and dist.is_available() and dist.is_initialized()
        is_writer = (dist.get_rank() == 0) if pg_avail else True
        if is_writer:
            data = {**self.logger["hyperparameter"], **self.logger["performance"]}
            df = pd.DataFrame(data)
            ext = filename.lower().split(".")[-1]
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
        if pg_avail:
            dist.barrier()
