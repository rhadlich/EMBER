from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class AdapterRuntimeState:
    """Mutable runtime state carried by a concrete adapter."""

    history: dict[str, float]
    target_gen: Any


class EnvAdapter(ABC):
    """Contract for plugging environment-specific behavior into realtime pipeline."""

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
    def get_filter_state_features(self) -> list[str]:
        """Ordered filter-input feature names used to build filter schema."""

    @abstractmethod
    def get_filter_output_features(self) -> list[str]:
        """Ordered filter-output feature names used to build filter schema."""

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
    ) -> np.ndarray:
        """Map sampled actor action into filter action domain."""

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
