from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.environments.env_adapter import AdapterRuntimeState, EnvAdapter


PROBE1_ADAPTER_ID = "probe1"


class Probe1Env(gym.Env):
    """
    This is the class that defines the probe 1 environment with continuous observation and state spaces.
    This was obtained from Andy Jones's advices: https://andyljones.com/posts/rl-debugging.html

    One action, one constant observation, one timestep long, +1 reward every timestep.

    This isolates the value network. If my agent can't learn that the value of the only observation it 
    ever sees it 1, there's a problem with the value loss calculation or the optimizer.

    Note: This can also be used for probe 3, just set the batch size to 2 in the setup_run.py file and
    set gamma to 0.5. If the value converges to 2 then everything is correct.
    """

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        # Keep a one-dimensional constant observation so it fits the current
        # shared-memory/ONNX pipeline (which expects a non-empty state vector).
        self.observation_space = spaces.Box(
            low=np.array([1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._obs = np.array([1.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {}

    def step(self, filtered_action_vals=None, nominal_action_vals=None):
        reward = 1.0
        reward_vec = np.array([reward, 0.0, 0.0], dtype=np.float32)
        terminated = False
        truncated = False
        return self._get_obs(), reward, reward_vec, terminated, truncated, {}

    def _get_obs(self) -> dict[str, float]:
        return {
            "probe_obs": float(self._obs[0]),
            "probe_target": 1.0,
            "probe_reward": 1.0,
        }


class _ConstantTargetGenerator:
    def __init__(self, value: float = 1.0):
        self._value = float(value)

    def current(self) -> float:
        return self._value

    def next(self) -> float:
        return self._value


@dataclass
class Probe1RuntimeState(AdapterRuntimeState):
    """Runtime state for the probe 1 environment."""

class Probe1Adapter(EnvAdapter):
    """Adapter that preserves current realtime probe 1 behavior."""

    @property
    def adapter_id(self) -> str:
        return "probe1"

    def build_env(self, *, reward_fn, env_kwargs: dict[str, Any]) -> gym.Env:
        return Probe1Env()

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
    ) -> Probe1RuntimeState:
        return Probe1RuntimeState(
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