"""Non-realtime Gymnasium environment for high-throughput RLlib training.

Wraps any registered EnvAdapter so that the throughput profile shares the
same observation/action semantics as the realtime pipeline without the
shared-memory / ONNX / minion infrastructure.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from core.environments.engine_adapter import ENGINE_CONTINUOUS_ADAPTER_ID
from core.environments.engine_env import reward_fn as _default_reward_fn
from core.environments.env_adapter import AdapterRuntimeState, EnvAdapter
from core.environments.target_curve_generator import IMEPTargetCurveGenerator
from utils.utils import ActionAdapter


class ThroughputEngineEnvContinuous(gym.Env):
    """Non-realtime RLlib environment for high-throughput training.

    Delegates all observation mapping, action mapping, history tracking, and
    target-curve management to the configured EnvAdapter, keeping the
    throughput profile in sync with realtime semantics automatically.

    Config keys (passed via RLlib env_config dict):
        env_adapter (str): Adapter ID from the adapter registry.
            Default: ENGINE_CONTINUOUS_ADAPTER_ID.
        max_episode_steps (int): Episode truncation length. Default: 32.
        predictor_checkpoint_path (str | None): Passed to adapter.build_env.
        sample_data_dir (str | None): Passed to adapter.build_env.
        env_seed (int | None): Base RNG seed for the target curve generator.
        target_min_hold_len (int): Minimum flat-hold steps. Default: 15.
        target_max_hold_len (int): Maximum flat-hold steps. Default: 60.
        target_min_transition_len (int): Minimum transition steps. Default: 20.
        target_max_transition_len (int): Maximum transition steps. Default: 90.
    """

    metadata = {"render_modes": []}

    def _validate_actor_obs(self, actor_obs: np.ndarray) -> None:
        if not np.all(np.isfinite(actor_obs)):
            raise ValueError(f"Non-finite actor observation values: {actor_obs}")
        norm_low = float(self._adapter.ACTOR_NORM_LOW)
        norm_high = float(self._adapter.ACTOR_NORM_HIGH)
        tol = 1e-4
        if np.any(actor_obs < norm_low - tol) or np.any(actor_obs > norm_high + tol):
            raise ValueError(
                "Actor observation outside normalized bounds "
                f"[{norm_low}, {norm_high}]: {actor_obs}"
            )

    def __init__(self, config=None):
        super().__init__()
        # Lazy import of get_env_adapter avoids a circular import when
        # core/environments/__init__.py also exports ThroughputEngineEnvContinuous.
        from core.environments import get_env_adapter  # noqa: PLC0415

        config = config or {}
        self._episode_step = 0
        self._max_episode_steps = int(config.get("max_episode_steps", 32))

        adapter_id = config.get("env_adapter", ENGINE_CONTINUOUS_ADAPTER_ID)
        self._adapter = get_env_adapter(adapter_id)
        self._env = self._adapter.build_env(
            reward_fn=_default_reward_fn, env_kwargs=config
        )

        # Spaces exposed to RLlib use normalized actor obs/actions; physical
        # bounds remain on the underlying env for filter/predictor stepping.
        self.action_space = self._adapter.get_normalized_action_space(env=self._env)
        self.observation_space = self._adapter.get_normalized_actor_observation_space(
            env=self._env
        )
        self._action_adapter = ActionAdapter(self._env.action_space)

        self._env_seed = config.get("env_seed")
        self._runtime_state: AdapterRuntimeState | None = None
        self._current_raw_obs: dict = {}

        # Target curve timing — fast-cycling defaults accelerate throughput
        # training; override via env_config to match realtime distribution.
        self._target_min_hold_len = int(config.get("target_min_hold_len", 15))
        self._target_max_hold_len = int(config.get("target_max_hold_len", 60))
        self._target_min_transition_len = int(
            config.get("target_min_transition_len", 20)
        )
        self._target_max_transition_len = int(
            config.get("target_max_transition_len", 90)
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0

        raw_obs, info = self._env.reset(seed=seed, options=options)
        self._current_raw_obs = raw_obs

        effective_seed = seed if seed is not None else self._env_seed
        seed_int = int(effective_seed) if effective_seed is not None else None

        self._runtime_state = self._adapter.init_runtime_state(
            env=self._env,
            env_seed=effective_seed,
        )

        # Replace the adapter's target generator with throughput-specific
        # timing parameters, preserving the IMEP bounds from the adapter's
        # own generator.  Adapters whose generator does not use low/high
        # bounds (e.g. probe envs with constant targets) are left unchanged.
        original_gen = self._runtime_state.target_gen
        if isinstance(original_gen, IMEPTargetCurveGenerator):
            self._runtime_state.target_gen = IMEPTargetCurveGenerator(
                low=original_gen.low,
                high=original_gen.high,
                seed=seed_int,
                min_hold_len=self._target_min_hold_len,
                max_hold_len=self._target_max_hold_len,
                min_transition_len=self._target_min_transition_len,
                max_transition_len=self._target_max_transition_len,
            )

        target = self._adapter.target_current(self._runtime_state)
        self._adapter.set_target(
            env=self._env,
            obs=raw_obs,
            target=target,
            runtime_state=self._runtime_state,
        )

        actor_obs = self._adapter.obs_to_actor(
            obs=raw_obs,
            runtime_state=self._runtime_state,
        )
        self._validate_actor_obs(actor_obs)
        return actor_obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._episode_step += 1

        physical_action = EnvAdapter.denormalize_action(
            action,
            action_low=self._env.action_space.low,
            action_high=self._env.action_space.high,
        )
        effective_action = self._adapter.action_actor_to_filter(
            action=action,
            action_adapter=self._action_adapter,
            runtime_state=self._runtime_state,
        )
        env_action = self._adapter.action_filter_to_env(
            obs=self._current_raw_obs,
            action=effective_action,
            runtime_state=self._runtime_state,
        )

        raw_obs, reward, _, _, _, info = self._env.step(
            filtered_action_vals=env_action,
            nominal_action_vals=env_action,
        )
        self._current_raw_obs = raw_obs

        self._adapter.update_history(
            action_in_env_range=physical_action,
            obs=raw_obs,
            runtime_state=self._runtime_state,
        )

        target = self._adapter.target_next(self._runtime_state)
        self._adapter.set_target(
            env=self._env,
            obs=raw_obs,
            target=target,
            runtime_state=self._runtime_state,
        )

        actor_obs = self._adapter.obs_to_actor(
            obs=raw_obs,
            runtime_state=self._runtime_state,
        )
        self._validate_actor_obs(actor_obs)

        terminated = False
        truncated = self._episode_step >= self._max_episode_steps
        return actor_obs, float(reward), terminated, truncated, info
