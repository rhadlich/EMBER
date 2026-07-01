from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.environments.env_adapter import AdapterRuntimeState, EnvAdapter


PROBE4_ADAPTER_ID = "probe4"


class Probe4Env(gym.Env):
    """
    This is the class that defines the probe 4 environment with continuous observation and state spaces.
    This was obtained from Andy Jones's advices: https://andyljones.com/posts/rl-debugging.html

    Two actions, zero observation, one timestep long, action-dependent +1/-1 reward.

    The first env to exercise the policy! If my agent can't learn to pick the better action, there's 
    something wrong with either my advantage calculations, my policy loss or my policy update. That's three 
    things, but it's easy to work out by hand the expected values for each one and check that the values 
    produced by your actual code line up with them.
    """

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        # Keep a one-dimensional constant observation so it fits the current
        # shared-memory/ONNX pipeline (which expects a non-empty state vector).
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.reward = (lambda x: float(x))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {}

    def step(self, filtered_action_vals=None, nominal_action_vals=None):
        obs = self._get_obs()
        reward = self.reward(filtered_action_vals[0])
        reward_vec = np.array([reward, 0.0, 0.0], dtype=np.float32)
        terminated = False
        truncated = False
        return obs, reward, reward_vec, terminated, truncated, {}

    def _get_obs(self) -> dict[str, float]:
        obs = self.observation_space.sample()
        return {
            "probe_obs": float(obs[0]),
            "probe_target": 1.0,
        }


class _ConstantTargetGenerator:
    def __init__(self, value: float = 1.0):
        self._value = float(value)

    def current(self) -> float:
        return self._value

    def next(self) -> float:
        return self._value


@dataclass
class Probe4RuntimeState(AdapterRuntimeState):
    """Runtime state for the probe 4 environment."""

class Probe4Adapter(EnvAdapter):
    """Adapter that preserves current realtime probe 4 behavior."""

    @property
    def adapter_id(self) -> str:
        return "probe4"

    def build_env(self, *, reward_fn, env_kwargs: dict[str, Any]) -> gym.Env:
        return Probe4Env()

    def get_actor_state_features(self) -> list[str]:
        return ["Probe constant observation"]

    def get_actor_obs_bounds(self, *, env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([env.observation_space.low[0]], dtype=np.float32),
            np.array([env.observation_space.high[0]], dtype=np.float32),
        )

    def get_filter_state_features(self) -> list[str]:
        return ["Probe constant observation"]

    def get_filter_output_features(self) -> list[str]:
        return ["Probe reward"]

    def init_runtime_state(
        self,
        *,
        env: gym.Env,
        env_seed: int | None,
    ) -> Probe4RuntimeState:
        actor_obs_low, actor_obs_high = self.get_actor_obs_bounds(env=env)
        return Probe4RuntimeState(
            history={
                "last_action": 0.0,
                "actor_obs_low": actor_obs_low,
                "actor_obs_high": actor_obs_high,
            },
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
        history = runtime_state.history
        actor_obs_raw = np.array([obs["probe_obs"]], dtype=np.float32)
        return self.normalize_actor_obs(
            obs_vec=actor_obs_raw,
            obs_low=history["actor_obs_low"],
            obs_high=history["actor_obs_high"],
        )

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