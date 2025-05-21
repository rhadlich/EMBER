from __future__ import annotations
from typing import Protocol, TypeVar, Callable

# Generic type‑vars match whatever dtypes your spaces carry
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class RewardFn(Protocol[ObsT, ActT]):
    """Callable sign‑off for any reward function you want to plug in."""
    def __call__(
        self,
        next_obs: ObsT,        # observation *after* the step
        info: dict | None = None
    ) -> float: ...
