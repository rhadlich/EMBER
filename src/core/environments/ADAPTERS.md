# Realtime Environment Adapters

Realtime training uses a shared-memory pipeline (`setup_run.py` -> `env_runner.py` -> `minion.py`).
Environment-specific logic is isolated behind the adapter contract in `env_adapter.py`.

## Current adapter

- `engine_continuous`: implemented in `engine_adapter.py` (`EngineContinuousAdapter`).

## Adapter responsibilities

Each adapter owns:

- Building the concrete Gym environment (`build_env`).
- Actor/filter feature schema declarations.
- Observation mapping for actor/filter.
- Action mapping from actor/filter domains to `env.step(...)`.
- Target/setpoint lifecycle and runtime history updates.
- GUI telemetry payload mapping.

The realtime transport and rollout plumbing remain generic in `minion.py` and `env_runner.py`.

## How to add a new probe environment

1. Implement a new environment class under `core/environments/`.
2. Add a new adapter class implementing `EnvAdapter` in a new `*_adapter.py` file.
3. Register it in `core/environments/__init__.py` by adding an id -> class entry in `_ADAPTER_REGISTRY`.
4. Run realtime with `--env-adapter <your_adapter_id>` and `--env-type continuous`.
5. If your adapter changes actor/filter feature counts, verify:
   - `setup_run.py` builds `ep_shm_properties` successfully.
   - minion rollout writes and env runner reads batches without shape mismatch.

## Notes

- Realtime path is continuous-only in this refactor.
- Throughput integration is intentionally deferred and should be wired to the same adapter contract in a follow-up.
