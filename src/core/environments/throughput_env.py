import numpy as np
import gymnasium as gym

from core.environments.engine_env import (
    EngineEnvContinuous,
    reward_fn,
)
from core.environments.target_curve_generator import IMEPTargetCurveGenerator


class ThroughputEngineEnvContinuous(gym.Env):
    """Non-realtime RLlib environment for high-throughput training.

    This env keeps the same action/observation spaces used by the realtime profile,
    but removes the minion/shared-memory/ONNX pipeline by stepping the simulator
    directly inside RLlib workers.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self._episode_step = 0
        self._max_episode_steps = int(config.get("max_episode_steps", 32))
        predictor_checkpoint_path = config.get("predictor_checkpoint_path")
        sample_data_dir = config.get("sample_data_dir")

        self._env = EngineEnvContinuous(
            reward=reward_fn,
            predictor_weights_path=predictor_checkpoint_path,
            sample_data_dir=sample_data_dir,
        )
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        # Match minion reset semantics.
        self._last_action = np.array(
            [self._env.soi_lims[0], self._env.inj_d_lims[0]],
            dtype=np.float32,
        )
        self._current_target = None

        seed = config.get("env_seed")
        seed = int(seed) if seed is not None else None
        self._target_gen = IMEPTargetCurveGenerator(
            low=float(self._env.imep_lims[0]),
            high=float(self._env.imep_lims[1]),
            seed=seed,
            min_hold_len=15,
            max_hold_len=60,
            min_transition_len=20,
            max_transition_len=90,
        )

    def reset(self, *, seed=None, options=None):
        obs_scalar, info = self._env.reset(seed=seed, options=options)
        self._episode_step = 0
        self._current_target = float(self._target_gen.current())
        self._env._desired_imep = self._current_target
        self._last_action = np.array(
            [self._env.soi_lims[0], self._env.inj_d_lims[0]],
            dtype=np.float32,
        )

        # Same 5D observation layout used in minion:
        # [soi_prev, inj_d_prev, target_prev, imep_current, target_current]
        obs = np.array(
            [
                self._last_action[0],
                self._last_action[1],
                self._current_target,
                float(info["current imep"]),
                self._current_target,
            ],
            dtype=np.float32,
        )
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._episode_step += 1

        prev_target = float(self._current_target)
        self._current_target = float(self._target_gen.next())
        self._env._desired_imep = self._current_target

        # Reuse simulator/reward path without realtime safety-filter plumbing.
        env_action = np.array([550.0, action[0], action[1]], dtype=np.float32)
        _, reward, _, _, _, info = self._env.step(
            filtered_action_vals=env_action,
            nominal_action_vals=env_action,
        )

        self._last_action = action
        obs = np.array(
            [
                self._last_action[0],
                self._last_action[1],
                prev_target,
                float(info["current imep"]),
                self._current_target,
            ],
            dtype=np.float32,
        )

        terminated = False
        truncated = self._episode_step >= self._max_episode_steps
        return obs, float(reward), terminated, truncated, info
