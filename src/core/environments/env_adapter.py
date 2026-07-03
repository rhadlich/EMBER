from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class AdapterRuntimeState:
    """Mutable runtime state carried by a concrete adapter."""

    history: dict[str, Any]
    target_gen: Any


class EnvAdapter(ABC):
    """Contract for plugging environment-specific behavior into realtime pipeline."""
    ACTOR_NORM_LOW: float = -1.0
    ACTOR_NORM_HIGH: float = 1.0

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Stable adapter identifier used in env_config."""

    @abstractmethod
    def build_env(self, *, reward_fn, env_kwargs: dict[str, Any]) -> gym.Env:
        """Construct a Gymnasium environment instance."""

    @abstractmethod
    def get_actor_state_features(self) -> list[str]:
        """Ordered actor-side feature names used to build shared-memory schema."""

    @abstractmethod
    def get_actor_obs_bounds(self, *, env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
        """Return per-feature raw min/max bounds in `obs_to_actor` feature order."""

    @abstractmethod
    def get_filter_state_features(self) -> list[str]:
        """Ordered filter-input feature names used to build filter schema."""

    @abstractmethod
    def get_filter_output_features(self) -> list[str]:
        """Ordered filter-output feature names used to build filter schema."""

    def get_filter_action_features(self, *, env: gym.Env) -> list[str]:
        """Ordered filter-action feature names used for filter action schema."""
        if not isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError(
                f"Filter action features not implemented for action space {env.action_space}"
            )
        action_dim = int(np.prod(env.action_space.shape))
        return [f"action_{idx}" for idx in range(action_dim)]

    @abstractmethod
    def init_runtime_state(
        self,
        *,
        env: gym.Env,
        env_seed: int | None,
    ) -> AdapterRuntimeState:
        """Create runtime state (history and target controller)."""

    @abstractmethod
    def target_current(self, runtime_state: AdapterRuntimeState) -> float:
        """Return current target value after reset."""

    @abstractmethod
    def target_next(self, runtime_state: AdapterRuntimeState) -> float:
        """Advance and return next target value."""

    @abstractmethod
    def set_target(
        self,
        *,
        env: gym.Env,
        obs: dict[str, float],
        target: float,
        runtime_state: AdapterRuntimeState,
    ) -> None:
        """Apply setpoint target to env internals and observation dict."""

    @abstractmethod
    def obs_to_actor(
        self,
        *,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        """Map raw env observation into actor model input vector."""

    @abstractmethod
    def obs_to_filter_input(
        self,
        *,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        """Map env observation/history into filter model input vector."""

    @abstractmethod
    def obs_to_filter_output(self, *, obs: dict[str, float]) -> np.ndarray:
        """Map env observation into filter-training output vector."""

    @abstractmethod
    def action_actor_to_filter(
        self,
        *,
        action: np.ndarray,
        action_adapter: Any,
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        """Map sampled actor action into the effective filter/plant action domain."""

    @abstractmethod
    def action_filter_to_env(
        self,
        *,
        obs: dict[str, float],
        action: np.ndarray,
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        """Map filter action output into environment step input."""

    @abstractmethod
    def update_history(
        self,
        *,
        action_in_env_range: np.ndarray,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> None:
        """Update adapter history state after each collected rollout."""

    @abstractmethod
    def build_gui_message(
        self,
        *,
        topic: str,
        obs: dict[str, float],
        info: dict[str, Any],
        runtime_state: AdapterRuntimeState,
    ) -> dict[str, float | str]:
        """Create telemetry payload for ZMQ/GUI publishing."""

    def get_normalized_actor_observation_space(self, *, env: gym.Env) -> gym.spaces.Box:
        """Return normalized actor observation space aligned with `obs_to_actor` output."""
        low_raw, _ = self.get_actor_obs_bounds(env=env)
        shape = tuple(low_raw.shape)
        low = np.full(shape, self.ACTOR_NORM_LOW, dtype=np.float32)
        high = np.full(shape, self.ACTOR_NORM_HIGH, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def normalize_actor_obs(
        self,
        *,
        obs_vec: np.ndarray,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
    ) -> np.ndarray:
        """Scale actor observation vector from raw bounds to [ACTOR_NORM_LOW, ACTOR_NORM_HIGH]."""
        obs_vec = np.asarray(obs_vec, dtype=np.float32)
        obs_low = np.asarray(obs_low, dtype=np.float32)
        obs_high = np.asarray(obs_high, dtype=np.float32)

        if obs_vec.shape != obs_low.shape or obs_vec.shape != obs_high.shape:
            raise ValueError(
                "Actor observation normalization shape mismatch: "
                f"obs={obs_vec.shape}, low={obs_low.shape}, high={obs_high.shape}"
            )

        span = obs_high - obs_low
        if np.any(span < 0.0):
            raise ValueError(
                "Actor observation normalization bounds must satisfy high >= low "
                f"for all features. Got low={obs_low}, high={obs_high}."
            )

        normalized_01 = np.zeros_like(obs_vec, dtype=np.float32)
        non_constant = span > 0.0
        normalized_01[non_constant] = (
            (obs_vec[non_constant] - obs_low[non_constant]) / span[non_constant]
        )
        scaled = self.ACTOR_NORM_LOW + normalized_01 * (
            self.ACTOR_NORM_HIGH - self.ACTOR_NORM_LOW
        )
        constant_value = (self.ACTOR_NORM_LOW + self.ACTOR_NORM_HIGH) / 2.0
        scaled[~non_constant] = constant_value
        return np.clip(scaled, self.ACTOR_NORM_LOW, self.ACTOR_NORM_HIGH).astype(
            np.float32
        )

    def get_normalized_action_space(self, *, env: gym.Env) -> gym.spaces.Box:
        """Return normalized actor action space aligned with policy/replay-buffer actions."""
        if not isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError(
                f"Normalized action space not implemented for {env.action_space}"
            )
        shape = tuple(env.action_space.shape)
        low = np.full(shape, self.ACTOR_NORM_LOW, dtype=np.float32)
        high = np.full(shape, self.ACTOR_NORM_HIGH, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def normalize_action(
        action_phys: np.ndarray,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        norm_low: float = ACTOR_NORM_LOW,
        norm_high: float = ACTOR_NORM_HIGH,
    ) -> np.ndarray:
        """Scale physical action vector to [norm_low, norm_high]."""
        action_phys = np.asarray(action_phys, dtype=np.float32)
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        if action_phys.shape != action_low.shape or action_phys.shape != action_high.shape:
            raise ValueError(
                "Action normalization shape mismatch: "
                f"action={action_phys.shape}, low={action_low.shape}, "
                f"high={action_high.shape}"
            )

        span = action_high - action_low
        if np.any(span < 0.0):
            raise ValueError(
                "Action normalization bounds must satisfy high >= low "
                f"for all dimensions. Got low={action_low}, high={action_high}."
            )

        normalized_01 = np.zeros_like(action_phys, dtype=np.float32)
        non_constant = span > 0.0
        normalized_01[non_constant] = (
            (action_phys[non_constant] - action_low[non_constant])
            / span[non_constant]
        )
        scaled = norm_low + normalized_01 * (norm_high - norm_low)
        constant_value = (norm_low + norm_high) / 2.0
        scaled[~non_constant] = constant_value
        return np.clip(scaled, norm_low, norm_high).astype(np.float32)

    @staticmethod
    def denormalize_action(
        action_norm: np.ndarray,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        norm_low: float = ACTOR_NORM_LOW,
        norm_high: float = ACTOR_NORM_HIGH,
    ) -> np.ndarray:
        """Scale normalized action vector from [norm_low, norm_high] to physical bounds."""
        action_norm = np.asarray(action_norm, dtype=np.float32)
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        if action_norm.shape != action_low.shape or action_norm.shape != action_high.shape:
            raise ValueError(
                "Action denormalization shape mismatch: "
                f"action={action_norm.shape}, low={action_low.shape}, "
                f"high={action_high.shape}"
            )

        span = action_high - action_low
        if np.any(span < 0.0):
            raise ValueError(
                "Action denormalization bounds must satisfy high >= low "
                f"for all dimensions. Got low={action_low}, high={action_high}."
            )

        norm_span = norm_high - norm_low
        if norm_span <= 0.0:
            raise ValueError(
                f"Action denormalization norm span must be positive, got {norm_span}."
            )

        normalized_01 = (action_norm - norm_low) / norm_span
        scaled = action_low + normalized_01 * span
        constant_value = (action_low + action_high) / 2.0
        scaled[span == 0.0] = constant_value[span == 0.0]
        return np.clip(scaled, action_low, action_high).astype(np.float32)
