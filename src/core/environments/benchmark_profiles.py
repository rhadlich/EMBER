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
    "mprr_excess_mean": 2.0,
    "ca50_rolling_std_mean": 0.01,
    "injection_duration_mean": 0.5,
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
    env_action: np.ndarray,
) -> dict[str, float]:
    """Compute reward-agnostic per-step benchmark metrics."""
    achieved_imep = float(obs["achieved_imep"])
    mprr = float(obs["achieved_mprr"])
    id1 = float(env_action[0])
    id2 = float(env_action[2])
    injection_duration = id1 + id2
    load_error = abs(achieved_imep - target)
    mprr_excess = max(0.0, mprr - DEFAULT_MPRR_LIMIT)

    return {
        "target": target,
        "achieved_imep": achieved_imep,
        "load_error": load_error,
        "mprr": mprr,
        "mprr_excess": mprr_excess,
        "mprr_within_limit": float(mprr <= DEFAULT_MPRR_LIMIT),
        "ca50": float(obs["achieved_CA50"]),
        "id1": id1,
        "id2": id2,
        "injection_duration": injection_duration,
    }


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
