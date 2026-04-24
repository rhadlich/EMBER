import numpy as np
import pandas as pd
from typing import Any, Dict, Sequence, Optional
import torch.distributed as dist
from typing import Union


def suggest_float(low, high, scale='linear'):
    if scale == 'linear':
        num = np.random.rand() * (high - low) + low
    elif scale == 'log':
        num = 10 ** (np.random.uniform(low, high + 1))
    else:
        raise ValueError('scale must be linear or log')
    return num


def suggest_int(low, high, step=1):
    rescale = int((high - low) / step)
    sample = np.random.randint(rescale + 1)
    return int(sample * step + low)


def suggest_categorical(cats: list):
    sample = np.random.randint(len(cats))
    return cats[sample]


class HPOGeneral:
    def __init__(
        self,
        param_configs: Dict[str, Dict[str, Any]],
        metrics: Optional[Sequence[str]] = 'metric',
        seed: Optional[int] = None,
    ):
        """
        Args:
            param_configs: Mapping of hyperparameter names to their config dicts.
                Each config dict must have:
                  - 'type': one of 'float', 'int', 'categorical'
                  - if 'float': 'low' and 'high' keys, optional 'scale' key can be 'linear' or 'log'.
                  - if 'int': 'low' and 'high' keys, optional 'step' key (int) for the step size.
                  - if 'categorical': 'choices' key
            metrics: Optional list of performance metric names to track (e.g. ['mse','mae']). Defaults to 'metric'.
            seed: Optional random seed for reproducibility.
        """
        # Seed NumPy RNG
        if seed is not None:
            np.random.seed(seed)

        self.param_configs = param_configs
        self.pg_avail = dist.is_available() and dist.is_initialized()

        # Initialize logger:
        # 'hyperparameter': dict mapping param names to lists of sampled values
        # 'performance': dict mapping metric names to lists of values
        self.logger: Dict[str, Any] = {
            "hyperparameter": {name: [] for name in param_configs},
            "performance": {m: [] for m in (metrics or [])},
        }

    def _check_in_logger(self, name: str, cat: str) -> bool:
        """
        Check whether a given key exists in the specified logger category.

        Args:
            name: The key or value to check.
            cat: Either 'performance' or 'hyperparameter'.
        Returns:
            True if name exists in self.logger[cat], False otherwise.
        """
        if cat not in self.logger:
            raise ValueError(
                f"Category '{cat}' not found in logger. Available: {list(self.logger.keys())}"
            )
        entries = self.logger[cat]
        if isinstance(entries, dict):
            return name in entries
        elif isinstance(entries, list):
            return name in entries
        return False

    def _check_log_alignment(self, metric: str = None) -> None:
        """
        Ensure that number of hyperparameter samples is at most one greater than
        number of performance entries for each metric, and vice versa.
        Raises RuntimeError if alignment is violated.

        Args:
            metric: which metric to check if logging
        """
        # All hyperparameters sampled each round, so pick any
        hparam_counts = [len(v) for v in self.logger["hyperparameter"].values()]
        n_samples = hparam_counts[0] if hparam_counts else 0

        if metric is None:
            for key in self.logger["performance"].keys():
                n_perf = len(self.logger["performance"][key])
                if n_samples - n_perf >= 1 and metric is None:
                    raise RuntimeError(
                        f"Hyperparameters sampled {n_samples} times but '{key}' has {n_perf} entries."
                        f" Log performance before sampling again."
                    )
        else:
            n_perf = len(self.logger["performance"][metric])
            if n_perf >= n_samples:
                raise RuntimeError(
                    f"'{metric}' has {n_perf} entries but only {n_samples} hyperparameter samples."
                    f" Sample again before logging performance."
                )

    def sample(self) -> Dict[str, Any]:
        """
        Sample all hyperparameters according to their configs, log them, and broadcast.
        Returns:
            Dict of sampled hyperparameters.
        """
        # before anything else, check alignment
        self._check_log_alignment()

        params = {}
        for name, cfg in self.param_configs.items():
            t = cfg.get("type")
            if t == "float":
                val = suggest_float(cfg["low"], cfg["high"])
            elif t == "int":
                val = suggest_int(cfg["low"], cfg["high"])
            elif t == "categorical":
                val = suggest_categorical(cfg["choices"])
            else:
                raise ValueError(f"Unknown type '{t}' for hyperparameter '{name}'")
            params[name] = val
            self.logger["hyperparameter"][name].append(val)

        return self.distribute_parameters(params)

    def distribute_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcasts the sampled parameters from rank 0 to all ranks via broadcast_object_list.
        """
        if not self.pg_avail:
            return params
        obj = [params]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def log_performance(self, value: Union[float, list, dict, np.ndarray], *, metric: str = 'metric', ) -> None:
        """
        Log a performance value under an existing metric and validate alignment.

        Args:
            value: Numeric performance value. Can also be a list, dict, or np.ndarray of performance as training
                progresses (or across different trials of the same hyperparameter).
            metric (optional): A pre-defined metric name (e.g. 'mse'). Defaults to 'metric'.
        """
        # before anything else, check alignment
        self._check_log_alignment(metric)

        if metric not in self.logger["performance"]:
            raise KeyError(
                f"Metric '{metric}' not initialized. Define in __init__ via 'metrics' argument."
            )
        self.logger["performance"][metric].append(value)

    def save_log(self, filename: str) -> None:
        """
        Persist logs to disk. Format inferred from file extension:
          - '.csv': comma-separated values (primitive types only)
          - '.json': full JSON array (for simple lists; may not preserve nested arrays perfectly)
          - '.pkl': Python pickle (recommended for arbitrary nested data)
          - '.parquet': Apache Parquet via pyarrow (preserves lists with parquet list type)

        For full fidelity of numpy arrays or lists in single cells, use '.pkl' or '.parquet'.
        """
        if dist.get_rank() == 0:
            # Merge logs into tabular form
            data = {**self.logger["hyperparameter"], **self.logger["performance"]}
            df = pd.DataFrame(data)
            ext = filename.lower().split('.')[-1]
            if ext == "csv":
                df.to_csv(filename, index=False)
            elif ext == "json":
                # Writes a JSON array; nested lists will be encoded as JSON lists
                df.to_json(filename, orient="records")
            elif ext in ("pkl", "pickle"):
                # Recommended: preserves native Python and numpy objects in cells
                df.to_pickle(filename)
            elif ext == "parquet":
                # Requires pyarrow; preserves list types as parquet lists
                df.to_parquet(filename, index=False)
            else:
                # default to parquet for full fidelity and robustness for package versions.
                df.to_parquet(filename)
        dist.barrier()

    def get_logs(self) -> Dict[str, Any]:
        """
        Retrieve the current logger state.
        """
        return self.logger
