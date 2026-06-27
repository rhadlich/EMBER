from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.environments.env_adapter import AdapterRuntimeState, EnvAdapter


PROBE6_ADAPTER_ID = "probe6"


class Probe6Env(gym.Env):
    """
    This is the class that defines the probe 6 environment with continuous observation and state spaces.

    One continuous action, one observation, reward is a cumulative function of both action and observation.

    This tests if the environment can learn and plan. Gamma should be non-zero, something like 0.9.
    """

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        # Keep a one-dimensional constant observation so it fits the current
        # shared-memory/ONNX pipeline (which expects a non-empty state vector).

        self.delta = 1.0
        self.lambda_penalty = 2.0
        self.lower_interest = 0.0
        self.upper_interest = 1.0

        self.observation_space = spaces.Box(
            low=np.array([self.lower_interest - self.delta], dtype=np.float32),
            high=np.array([self.upper_interest + self.delta], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-0.1], dtype=np.float32),
            high=np.array([0.1], dtype=np.float32),
            dtype=np.float32,
        )
        self.reward = (lambda x: float(x))
        self.target = 0.8

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self.observation_space.sample()
        return self._get_obs(), {}

    def step(self, filtered_action_vals=None, nominal_action_vals=None):
        self._obs = np.clip(self._obs + filtered_action_vals[0], self.observation_space.low, self.observation_space.high)
        obs = self._get_obs()
        reward = self.reward(- np.abs(self._obs[0] - self.target))
        upper_penalty = np.minimum(0.0, self.upper_interest - self._obs[0])
        lower_penalty = np.minimum(0.0, self._obs[0] - self.lower_interest)
        reward += self.reward((upper_penalty + lower_penalty) * self.lambda_penalty) + 4.0
        obs["reward"] = reward
        reward_vec = np.array([reward, 0.0, 0.0], dtype=np.float32)
        terminated = False
        truncated = False
        return obs, reward, reward_vec, terminated, truncated, {}

    def _get_obs(self) -> dict[str, float]:
        return {
            "probe_obs": float(self._obs[0]),
            "probe_target": self.target,
        }


class _ConstantTargetGenerator:
    def __init__(self, value: float = 1.0):
        self._value = float(value)

    def current(self) -> float:
        return self._value

    def next(self) -> float:
        return self._value


@dataclass
class Probe6RuntimeState(AdapterRuntimeState):
    """Runtime state for the probe 6 environment."""

class Probe6Adapter(EnvAdapter):
    """Adapter that preserves current realtime probe 6 behavior."""

    @property
    def adapter_id(self) -> str:
        return "probe6"

    def build_env(self, *, reward_fn, env_kwargs: dict[str, Any]) -> gym.Env:
        return Probe6Env()

    def get_actor_state_features(self) -> list[str]:
        return ["Probe constant observation"]

    def get_filter_state_features(self) -> list[str]:
        return ["Probe constant observation"]

    def get_filter_output_features(self) -> list[str]:
        return ["Probe reward"]

    def init_runtime_state(
        self,
        *,
        env: gym.Env,
        env_seed: int | None,
    ) -> Probe6RuntimeState:
        return Probe6RuntimeState(
            history={"last_action": 0.0},
            target_gen=_ConstantTargetGenerator(1.0),
        )

    def target_current(self, runtime_state: AdapterRuntimeState) -> float:
        return float(runtime_state.target_gen.current())

    def target_next(self, runtime_state: AdapterRuntimeState) -> float:
        return float(runtime_state.target_gen.next())

    def set_target(
        self,
        *,
        env: gym.Env,
        obs: dict[str, float],
        target: float,
        runtime_state: AdapterRuntimeState,
    ) -> None:
        obs["probe_target"] = float(target)

    def obs_to_actor(
        self,
        *,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        return np.array([obs["probe_obs"]], dtype=np.float32)

    def obs_to_filter_input(
        self,
        *,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        return np.array([obs["probe_obs"]], dtype=np.float32)

    def obs_to_filter_output(self, *, obs: dict[str, float]) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def action_actor_to_filter(
        self,
        *,
        action: np.ndarray,
        action_adapter: Any,
    ) -> np.ndarray:
        action_for_filter = action.astype(np.float32, copy=True)
        return action_adapter.get_action_in_env_range(action_for_filter)

    def action_filter_to_env(
        self,
        *,
        obs: dict[str, float],
        action: np.ndarray,
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        return np.array([action[0]], dtype=np.float32).reshape(-1)

    def update_history(
        self,
        *,
        action_in_env_range: np.ndarray,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> None:
        runtime_state.history["last_action"] = float(action_in_env_range[0])

    def build_gui_message(
        self,
        *,
        topic: str,
        obs: dict[str, float],
        info: dict[str, Any],
        runtime_state: AdapterRuntimeState,
    ) -> dict[str, float | str]:
        return {
            "topic": topic,
            "probe_obs": float(obs.get("probe_obs", 1.0)),
            "target": float(obs.get("probe_target", 1.0)),
        }