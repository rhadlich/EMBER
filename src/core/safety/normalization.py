from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import h5py as h5
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def list_h5_files(source: str) -> list[str]:
    files = sorted([os.path.join(source, x) for x in os.listdir(source) if x.endswith(".h5")])
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {source}")
    return files


def _resolve_sample_hdf5_path(sample_data_dir: Union[str, Path]) -> str:
    path = Path(sample_data_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return str(path)
    if path.is_dir():
        return list_h5_files(str(path))[0]
    raise FileNotFoundError(
        f"Safety filter sample data path not found (expected directory or .h5 file): {path}"
    )


@dataclass(frozen=True)
class SafetyFilterNormalization:
    """HDF5-backed normalization for safety-filter state and action inputs."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    action_range: np.ndarray
    state_dim: int
    action_dim: int
    sample_data_path: str

    @classmethod
    def from_sample_data_dir(
        cls,
        sample_data_dir: Union[str, Path],
        *,
        state_dim: int,
        action_dim: int,
    ) -> SafetyFilterNormalization:
        sample_data_path = _resolve_sample_hdf5_path(sample_data_dir)
        with h5.File(sample_data_path, "r") as fin:
            feature_mean = fin["normalization/feature_mean"][...].reshape(-1)
            feature_std = fin["normalization/feature_std"][...].reshape(-1)
            action_min = fin["normalization/action_min"][...].reshape(-1)
            action_max = fin["normalization/action_max"][...].reshape(-1)

        if feature_mean.size != state_dim or feature_std.size != state_dim:
            raise ValueError(
                "Safety filter feature normalization shape mismatch: "
                f"expected state_dim={state_dim}, "
                f"got feature_mean={feature_mean.shape}, feature_std={feature_std.shape} "
                f"from {sample_data_path}"
            )
        if action_min.size != action_dim or action_max.size != action_dim:
            raise ValueError(
                "Safety filter action normalization shape mismatch: "
                f"expected action_dim={action_dim}, "
                f"got action_min={action_min.shape}, action_max={action_max.shape} "
                f"from {sample_data_path}. If you migrated to the 2D safety-filter "
                "schema (SOI2, ID2), regenerate filter sample data and normalization "
                "stats with action_dim=2."
            )
        if np.any(feature_std == 0):
            raise ValueError(
                f"Found zeros in feature_std in {sample_data_path}; cannot normalize safely."
            )

        action_range = action_max - action_min
        if np.any(action_range == 0):
            raise ValueError(
                f"Found zeros in (action_max - action_min) in {sample_data_path}; "
                "cannot normalize safely."
            )

        return cls(
            feature_mean=feature_mean.astype(np.float32, copy=True),
            feature_std=feature_std.astype(np.float32, copy=True),
            action_min=action_min.astype(np.float32, copy=True),
            action_max=action_max.astype(np.float32, copy=True),
            action_range=action_range.astype(np.float32, copy=True),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            sample_data_path=sample_data_path,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        action_min: np.ndarray,
        action_max: np.ndarray,
        state_dim: int,
        action_dim: int,
        sample_data_path: str = "",
    ) -> SafetyFilterNormalization:
        action_range = action_max - action_min
        if np.any(feature_std == 0):
            raise ValueError("Found zeros in feature_std; cannot normalize safely.")
        if np.any(action_range == 0):
            raise ValueError("Found zeros in (action_max - action_min); cannot normalize safely.")

        return cls(
            feature_mean=np.asarray(feature_mean, dtype=np.float32).reshape(-1),
            feature_std=np.asarray(feature_std, dtype=np.float32).reshape(-1),
            action_min=np.asarray(action_min, dtype=np.float32).reshape(-1),
            action_max=np.asarray(action_max, dtype=np.float32).reshape(-1),
            action_range=np.asarray(action_range, dtype=np.float32).reshape(-1),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            sample_data_path=sample_data_path,
        )

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        return (state - self.feature_mean) / self.feature_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        return 2.0 * (action - self.action_min) / self.action_range - 1.0

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        action_norm = np.asarray(action_norm, dtype=np.float32)
        return 0.5 * (action_norm + 1.0) * self.action_range + self.action_min
