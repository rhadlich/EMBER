"""Environment definitions, adapters, and utilities."""

from core.environments.engine_adapter import (
    ENGINE_CONTINUOUS_ADAPTER_ID,
    EngineContinuousAdapter,
)
from core.environments.engine_env import EngineEnvContinuous, reward_fn
from core.environments.env_adapter import EnvAdapter
from core.environments.predictor import Predictor
from core.environments.probe1_env import PROBE1_ADAPTER_ID, Probe1Adapter
from core.environments.probe2_env import PROBE2_ADAPTER_ID, Probe2Adapter
from core.environments.probe4_env import PROBE4_ADAPTER_ID, Probe4Adapter
from core.environments.probe5_env import PROBE5_ADAPTER_ID, Probe5Adapter
from core.environments.probe6_env import PROBE6_ADAPTER_ID, Probe6Adapter
from core.environments.reward_typing import RewardFn
from core.environments.throughput_env import ThroughputEngineEnvContinuous


_ADAPTER_REGISTRY: dict[str, type[EnvAdapter]] = {
    ENGINE_CONTINUOUS_ADAPTER_ID: EngineContinuousAdapter,
    PROBE1_ADAPTER_ID: Probe1Adapter,
    PROBE2_ADAPTER_ID: Probe2Adapter,
    PROBE4_ADAPTER_ID: Probe4Adapter,
    PROBE5_ADAPTER_ID: Probe5Adapter,
    PROBE6_ADAPTER_ID: Probe6Adapter,
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
    "PROBE1_ADAPTER_ID",
    "PROBE2_ADAPTER_ID",
    "PROBE4_ADAPTER_ID",
    "PROBE5_ADAPTER_ID",
    "PROBE6_ADAPTER_ID",
    "EngineEnvContinuous",
    "EnvAdapter",
    "Predictor",
    "RewardFn",
    "ThroughputEngineEnvContinuous",
    "get_env_adapter",
    "reward_fn",
]
