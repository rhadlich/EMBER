"""Fixed IMEP load profiles and reward-agnostic benchmark metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.environments.target_curve_generator import generate_profile

DEFAULT_PROFILE_SEEDS = (10001, 10002, 10003, 10004, 10005)
DEFAULT_PROFILE_LENGTH = 500
DEFAULT_BURN_IN = 5
DEFAULT_MPRR_LIMIT = 7.0
DEFAULT_CA50_STABILITY_WINDOW = 5

# Weights for composite score (lower is better).
DEFAULT_SCORE_WEIGHTS = {
    "load_tracking_mae": 1.0,
    "mprr_excess_mean": 0,
    "ca50_rolling_std_mean": 0.0,
    "injection_duration_mean": 0.0,
}


@dataclass(frozen=True)
class BenchmarkProfileSet:
    """Named collection of fixed target curves."""

    profiles: dict[str, np.ndarray]
    seeds: tuple[int, ...]
    profile_length: int
    generator_params: dict[str, Any]

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self.profiles.keys())


def _realtime_generator_params(env) -> dict[str, Any]:
    """Match ``EngineContinuousAdapter.init_runtime_state`` target timing."""
    imep_lo, imep_hi = env.imep_sample_lims
    return {
        "low": float(imep_lo),
        "high": float(imep_hi),
        "min_hold_len": 30,
        "max_hold_len": 50,
        "min_transition_len": 40,
        "max_transition_len": 100,
    }


def load_or_create_profiles(
    path: Path,
    *,
    env,
    profile_seeds: tuple[int, ...] = DEFAULT_PROFILE_SEEDS,
    profile_length: int = DEFAULT_PROFILE_LENGTH,
) -> BenchmarkProfileSet:
    """Load stored profiles or generate and persist them on first use."""
    path = Path(path)
    generator_params = _realtime_generator_params(env)

    if path.is_file():
        with np.load(path, allow_pickle=True) as data:
            seeds = tuple(int(s) for s in data["seeds"].tolist())
            length = int(data["profile_length"])
            profiles = {
                key: np.asarray(data[key], dtype=np.float64)
                for key in data.files
                if key.startswith("profile_")
            }
        return BenchmarkProfileSet(
            profiles=profiles,
            seeds=seeds,
            profile_length=length,
            generator_params=generator_params,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, np.ndarray] = {}
    for idx, seed in enumerate(profile_seeds):
        profiles[f"profile_{idx}"] = generate_profile(
            profile_length,
            seed=seed,
            **generator_params,
        )

    save_profiles(path, profiles, profile_seeds, profile_length, generator_params)
    return BenchmarkProfileSet(
        profiles=profiles,
        seeds=profile_seeds,
        profile_length=profile_length,
        generator_params=generator_params,
    )


def save_profiles(
    path: Path,
    profiles: dict[str, np.ndarray],
    seeds: tuple[int, ...],
    profile_length: int,
    generator_params: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in profiles.items()}
    arrays["seeds"] = np.asarray(seeds, dtype=np.int64)
    arrays["profile_length"] = np.asarray(profile_length, dtype=np.int64)
    np.savez(path, **arrays)

    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "profile_length": profile_length,
                "seeds": list(seeds),
                "generator_params": generator_params,
            },
            indent=2,
        )
    )


def record_step_metrics(
    *,
    target: float,
    obs: dict[str, float],
    physical_action: np.ndarray,
    policy_action: np.ndarray | None = None,
    cumulative_injection: float | None = None,
) -> dict[str, float]:
    """Compute reward-agnostic per-step benchmark metrics."""
    achieved_imep = float(obs["achieved_imep"])
    mprr = float(obs["achieved_mprr"])
    physical_action = np.asarray(physical_action, dtype=np.float64).reshape(-1)
    id1 = float(physical_action[0])
    soi2 = float(physical_action[1])
    id2 = float(physical_action[2])
    injection_duration = id1 + id2
    load_error = abs(achieved_imep - target)
    mprr_excess = max(0.0, mprr - DEFAULT_MPRR_LIMIT)

    row: dict[str, float] = {
        "target": target,
        "achieved_imep": achieved_imep,
        "load_error": load_error,
        "mprr": mprr,
        "mprr_excess": mprr_excess,
        "mprr_within_limit": float(mprr <= DEFAULT_MPRR_LIMIT),
        "ca50": float(obs["achieved_CA50"]),
        "env_id1": id1,
        "env_soi2": soi2,
        "env_id2": id2,
        "injection_duration": injection_duration,
    }
    if cumulative_injection is not None:
        row["cumulative_injection"] = float(cumulative_injection)
    if policy_action is not None:
        policy_action = np.asarray(policy_action, dtype=np.float64).reshape(-1)
        for idx, value in enumerate(policy_action):
            row[f"policy_action_{idx}"] = float(value)
    return row


def load_traces_from_npz(path: Path | str) -> dict[str, dict[str, np.ndarray]]:
    """Load benchmark trace arrays grouped by profile id."""
    path = Path(path)
    profiles: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if "/" not in key:
                continue
            profile_id, metric_name = key.split("/", 1)
            profiles.setdefault(profile_id, {})[metric_name] = np.asarray(
                data[key], dtype=np.float64
            )
    return profiles


def traces_to_step_records(
    profile_traces: dict[str, dict[str, np.ndarray]],
    profile_id: str,
) -> list[dict[str, float]]:
    """Reconstruct per-step dict records from npz trace arrays."""
    metrics = profile_traces[profile_id]
    if not metrics:
        return []
    n_steps = len(next(iter(metrics.values())))
    return [
        {name: float(values[step_idx]) for name, values in metrics.items()}
        for step_idx in range(n_steps)
    ]


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([], dtype=np.float64)
    out = np.empty(len(values) - window + 1, dtype=np.float64)
    for idx in range(len(out)):
        out[idx] = float(np.std(values[idx : idx + window]))
    return out


def aggregate_profile_metrics(
    step_metrics: list[dict[str, float]],
    *,
    burn_in: int = DEFAULT_BURN_IN,
    ca50_window: int = DEFAULT_CA50_STABILITY_WINDOW,
) -> dict[str, float]:
    """Aggregate per-step records for one profile."""
    if not step_metrics:
        raise ValueError("step_metrics must not be empty")

    effective = step_metrics[burn_in:] if burn_in > 0 else step_metrics
    if not effective:
        effective = step_metrics

    load_errors = np.asarray([row["load_error"] for row in effective], dtype=np.float64)
    mprr = np.asarray([row["mprr"] for row in effective], dtype=np.float64)
    mprr_excess = np.asarray([row["mprr_excess"] for row in effective], dtype=np.float64)
    ca50 = np.asarray([row["ca50"] for row in effective], dtype=np.float64)
    injection_duration = np.asarray(
        [row["injection_duration"] for row in effective], dtype=np.float64
    )
    rolling_std = _rolling_std(ca50, ca50_window)

    return {
        "load_tracking_mae": float(np.mean(load_errors)),
        "load_tracking_max": float(np.max(load_errors)),
        "mprr_max": float(np.max(mprr)),
        "mprr_excess_mean": float(np.mean(mprr_excess)),
        "mprr_compliance_fraction": float(np.mean(mprr <= DEFAULT_MPRR_LIMIT)),
        "ca50_std": float(np.std(ca50)),
        "ca50_rolling_std_mean": float(np.mean(rolling_std)) if len(rolling_std) else 0.0,
        "injection_duration_mean": float(np.mean(injection_duration)),
        "n_steps": float(len(step_metrics)),
        "n_steps_scored": float(len(effective)),
    }


def composite_score(
    aggregates: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Scalar score for ranking benchmark runs (lower is better)."""
    weights = weights or DEFAULT_SCORE_WEIGHTS
    return float(
        sum(weights[key] * aggregates[key] for key in weights)
    )


def aggregate_run_metrics(
    profile_aggregates: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Mean profile-level aggregates across all benchmark profiles."""
    if not profile_aggregates:
        raise ValueError("profile_aggregates must not be empty")

    keys = next(iter(profile_aggregates.values())).keys()
    return {
        key: float(np.mean([metrics[key] for metrics in profile_aggregates.values()]))
        for key in keys
    }


# =========================
# Interactive configuration
# =========================
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILES_DIR = _PROJECT_ROOT / "benchmark_profiles"
OUTPUT_DIR = PROFILES_DIR / "plots"
DEFAULT_PROFILES_FILENAME = "representative_profiles.npz"
SHOW = True
SAVE = True
DPI = 150


def _is_profile_array_key(key: str) -> bool:
    return (
        key.startswith("profile_")
        and key != "profile_length"
        and key.rsplit("_", 1)[-1].isdigit()
    )


def _resolve_profile_store_path(path: Path) -> Path:
    """Resolve a stored profile ``.npz`` from a file or directory path."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Profile path does not exist: {path}")

    preferred = path / DEFAULT_PROFILES_FILENAME
    if preferred.is_file():
        return preferred

    candidates = sorted(path.glob("*.npz"))
    for candidate in candidates:
        with np.load(candidate, allow_pickle=True) as data:
            if any(_is_profile_array_key(key) for key in data.files):
                return candidate

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"No stored benchmark profiles found in {path}. "
        f"Expected {DEFAULT_PROFILES_FILENAME} or another .npz with profile arrays."
    )


def load_profiles_from_directory(path: Path | str) -> BenchmarkProfileSet:
    """Load stored benchmark profiles from a directory or ``.npz`` file."""
    store_path = _resolve_profile_store_path(Path(path))
    with np.load(store_path, allow_pickle=True) as data:
        seeds = tuple(int(s) for s in data["seeds"].tolist())
        length = int(data["profile_length"])
        profiles = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in data.files
            if _is_profile_array_key(key)
        }

    if not profiles:
        raise ValueError(f"No profile arrays found in {store_path}")

    generator_params: dict[str, Any] = {}
    meta_path = store_path.with_suffix(".json")
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text())
        params = metadata.get("generator_params")
        if isinstance(params, dict):
            generator_params = params

    return BenchmarkProfileSet(
        profiles=profiles,
        seeds=seeds,
        profile_length=length,
        generator_params=generator_params,
    )


def _profile_sort_key(profile_id: str) -> tuple[int, str]:
    suffix = profile_id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix), profile_id
    return 10_000, profile_id


def plot_stored_profiles(
    profile_set: BenchmarkProfileSet,
    output_dir: Path,
    *,
    show: bool = False,
    save: bool = True,
    dpi: int = 150,
) -> Path:
    """Plot fixed IMEP target curves from a stored profile set."""
    import matplotlib.pyplot as plt

    profile_ids = sorted(profile_set.profile_ids, key=_profile_sort_key)
    n_cols = 1
    n_rows = int(np.ceil(len(profile_ids) / n_cols))

    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10 * n_cols, 2 * n_rows), squeeze=False
    )
    cmap = plt.get_cmap("tab10")
    for idx, profile_id in enumerate(profile_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        targets = profile_set.profiles[profile_id]
        steps = np.arange(len(targets))
        seed = profile_set.seeds[idx] if idx < len(profile_set.seeds) else "?"
        ax.plot(steps, targets, color=cmap(idx % 10), linewidth=1.4)
        ax.set_title(f"{profile_id} (seed={seed})")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Target IMEP")
        ax.grid(True, alpha=0.3)

    for idx in range(len(profile_ids), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.tight_layout()
    out_path = output_dir / "stored_profiles.png"
    if save:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    profile_set = load_profiles_from_directory(PROFILES_DIR)
    out_path = plot_stored_profiles(
        profile_set,
        OUTPUT_DIR,
        show=SHOW,
        save=SAVE,
        dpi=DPI,
    )
    if SAVE:
        print(f"Saved profile plot to {out_path}")
    elif SHOW:
        print("Displayed profile plot (SAVE=False).")


if __name__ == "__main__":
    main()
