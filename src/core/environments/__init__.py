"""Environment definitions, adapters, and utilities."""

from core.environments.engine_adapter import (
    ENGINE_CONTINUOUS_ADAPTER_ID,
    EngineContinuousAdapter,
)
from core.environments.engine_env import EngineEnvContinuous, reward_fn
from core.environments.env_adapter import EnvAdapter
from core.environments.predictor import Predictor
from core.environments.reward_typing import RewardFn


_ADAPTER_REGISTRY: dict[str, type[EnvAdapter]] = {
    ENGINE_CONTINUOUS_ADAPTER_ID: EngineContinuousAdapter,
}


def get_env_adapter(adapter_id: str) -> EnvAdapter:
    """Instantiate an environment adapter by id."""
    try:
        adapter_cls = _ADAPTER_REGISTRY[adapter_id]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise KeyError(f"Unknown env adapter '{adapter_id}'. Known adapters: {known}") from exc
    return adapter_cls()


__all__ = [
    "ENGINE_CONTINUOUS_ADAPTER_ID",
    "EngineEnvContinuous",
    "EnvAdapter",
    "Predictor",
    "RewardFn",
    "get_env_adapter",
    "reward_fn",
]
