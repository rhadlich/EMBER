from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from core.environments.engine_env import EngineEnvContinuous
from core.environments.env_adapter import AdapterRuntimeState, EnvAdapter
from core.environments.target_curve_generator import IMEPTargetCurveGenerator


ENGINE_CONTINUOUS_ADAPTER_ID = "engine_continuous"


@dataclass
class EngineRuntimeState(AdapterRuntimeState):
    """Runtime state for the engine continuous adapter."""


class EngineContinuousAdapter(EnvAdapter):
    """Adapter that preserves current realtime engine behavior."""

    @property
    def adapter_id(self) -> str:
        return ENGINE_CONTINUOUS_ADAPTER_ID

    def build_env(self, *, reward_fn, env_kwargs: dict[str, Any]) -> gym.Env:
        return EngineEnvContinuous(
            reward=reward_fn,
            predictor_weights_path=env_kwargs.get("predictor_checkpoint_path"),
            sample_data_dir=env_kwargs.get("sample_data_dir"),
        )

    def get_actor_state_features(self) -> list[str]:
        return [
            "IMEP setpoint, k",
            "IMEP setpoint, k-1",
            "IMEP actual, k-1",
            "Injection duration before IVC (ID1), k",
            "Injection duration before IVC (ID1), k-1",
            "Start of injection after IVC (SOI2), k-1",
            "Injection duration after IVC (ID2), k-1",
            # 'Pressure intake @ IVC, k-1',
            # 'CA50, k-1',
            # 'CA10 to CA90, k-1',
            # 'Net heat release, k-1',
            # 'Pressure max, k-1',
            # 'MPRR, k-1',
            # 'Moving average IMEP (20 cycles), k-1',
            # 'Skewness of moving averate IMEP (20 cycles), k-1',
        ]

    def get_actor_obs_bounds(self, *, env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(
                [
                    env.imep_env_limits[0],
                    env.imep_env_limits[0],
                    env.imep_env_limits[0],
                    env.ID1_lims[0],
                    env.ID1_lims[0],
                    env.SOI2_lims[0],
                    env.ID2_lims[0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    env.imep_env_limits[1],
                    env.imep_env_limits[1],
                    env.imep_env_limits[1],
                    env.ID1_lims[1],
                    env.ID1_lims[1],
                    env.SOI2_lims[1],
                    env.ID2_lims[1],
                ],
                dtype=np.float32,
            ),
        )

    def get_filter_state_features(self) -> list[str]:
        return [
            "IMEP actual, k-1",
            "Injection duration before IVC (ID1), k",
            "Injection duration before IVC (ID1), k-1",
            "Start of injection after IVC (SOI2), k-1",
            "Injection duration after IVC (ID2), k-1",
            # 'Pressure intake @ IVC, k-1',
            # 'CA50, k-1',
            # 'CA10 to CA90, k-1',
            # 'Net heat release, k-1',
            # 'Pressure max, k-1',
            # 'Moving average IMEP (20 cycles), k-1',
            # 'Skewness of moving averate IMEP (20 cycles), k-1',
            'MPRR, k-1',    # keep it last for easier extraction in SafetyFilter
        ]

    def get_filter_output_features(self) -> list[str]:
        return ["achieved_mprr"]

    def get_filter_action_features(self, *, env: gym.Env) -> list[str]:
        # ID1 is not filter-adjustable in the current cycle because it has already
        # been executed; the filter controls SOI2 and ID2 only.
        return ["Start of injection after IVC (SOI2), k", "Injection duration after IVC (ID2), k"]

    def init_runtime_state(
        self,
        *,
        env: gym.Env,
        env_seed: int | None,
    ) -> EngineRuntimeState:
        target_seed = int(env_seed) if env_seed is not None else None
        imep_lo, imep_hi = env.imep_sample_lims
        actor_obs_low, actor_obs_high = self.get_actor_obs_bounds(env=env)
        target_gen = IMEPTargetCurveGenerator(
            low=float(imep_lo),
            high=float(imep_hi),
            seed=target_seed,
            min_hold_len=40,
            max_hold_len=100,
            min_transition_len=60,
            max_transition_len=150,
        )
        return EngineRuntimeState(
            history={
                "previous desired imep": 0.0,
                "previous SOI2": -140.0,
                "previous ID2": 0.6,
                "previous ID1": 0.6,
                "current ID1": 0.6,
                "actor_obs_low": actor_obs_low,
                "actor_obs_high": actor_obs_high,
            },
            target_gen=target_gen,
        )

    def sync_history_from_obs(
        self,
        *,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
        env: gym.Env,
    ) -> None:
        """Align adapter lag features with the current observation after a reset."""
        history = runtime_state.history
        id1_mid = 0.5 * (env.ID1_lims[0] + env.ID1_lims[1])
        soi2_mid = 0.5 * (env.SOI2_lims[0] + env.SOI2_lims[1])
        id2_mid = 0.5 * (env.ID2_lims[0] + env.ID2_lims[1])
        history["previous desired imep"] = float(obs["achieved_imep"])
        history["previous ID1"] = float(id1_mid)
        history["current ID1"] = float(id1_mid)
        history["previous SOI2"] = float(soi2_mid)
        history["previous ID2"] = float(id2_mid)

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
        env._desired_imep = float(target)
        obs["desired_imep"] = float(target)

    def obs_to_actor(
        self,
        *,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        history = runtime_state.history
        actor_obs_raw = np.array(
            [
                obs["desired_imep"],
                history["previous desired imep"],
                obs["achieved_imep"],
                history["current ID1"],
                history["previous ID1"],
                history["previous SOI2"],
                history["previous ID2"],
            ],
            dtype=np.float32,
        )
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
        history = runtime_state.history
        return np.array(
            [
                obs["achieved_imep"],
                history["current ID1"],
                history["previous ID1"],
                history["previous SOI2"],
                history["previous ID2"],
                obs["achieved_mprr"],
            ],
            dtype=np.float32,
        )

    def obs_to_filter_output(self, *, obs: dict[str, float]) -> np.ndarray:
        return np.array([obs["achieved_mprr"]], dtype=np.float32)

    def action_actor_to_filter(
        self,
        *,
        action: np.ndarray,
        action_adapter: Any,
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        physical = action_adapter.get_action_in_env_range(action)
        # ID1 is applied with a one-step delay and is immutable at filter time.
        # Safety filter operates only on SOI2 and ID2.
        return np.array([physical[1], physical[2]], dtype=np.float32)

    def action_filter_to_env(
        self,
        *,
        obs: dict[str, float],
        action: np.ndarray,
        runtime_state: AdapterRuntimeState,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError(
                "EngineContinuousAdapter expected 2D filter action [SOI2, ID2], "
                f"got shape {action.shape}."
            )
        history = runtime_state.history
        return np.array([history["current ID1"], action[0], action[1]], dtype=np.float32)

    def update_history(
        self,
        *,
        action_in_env_range: np.ndarray,
        obs: dict[str, float],
        runtime_state: AdapterRuntimeState,
    ) -> None:
        history = runtime_state.history
        history["previous desired imep"] = obs["desired_imep"]
        history["previous ID1"] = history["current ID1"]
        history["current ID1"] = action_in_env_range[0]
        history["previous SOI2"] = action_in_env_range[1]
        history["previous ID2"] = action_in_env_range[2]

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
            "current imep": float(info["current imep"]),
            "mprr": float(info["mprr"]),
            "target": float(obs["desired_imep"]),
        }
