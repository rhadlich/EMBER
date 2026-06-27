# Realtime and Throughput Environment Adapters

Both training profiles — realtime and throughput — share the same adapter
contract defined in `env_adapter.py`.  Environment-specific logic is isolated
behind that contract so that new environments can be added without touching
the training infrastructure.

## Current adapters

| ID | Class | File |
|---|---|---|
| `engine_continuous` | `EngineContinuousAdapter` | `engine_adapter.py` |
| `probe1` | `Probe1Adapter` | `probe1_env.py` |
| `probe2` | `Probe2Adapter` | `probe2_env.py` |
| `probe4` | `Probe4Adapter` | `probe4_env.py` |
| `probe5` | `Probe5Adapter` | `probe5_env.py` |
| `probe6` | `Probe6Adapter` | `probe6_env.py` |

## Adapter responsibilities

Each adapter owns:

- Building the concrete Gym environment (`build_env`).
- Actor/filter feature schema declarations.
- Observation mapping for actor/filter.
- Action mapping from actor/filter domains to `env.step(...)`.
- Target/setpoint lifecycle and runtime history updates.
- GUI telemetry payload mapping.

The realtime transport and rollout plumbing remain generic in `minion.py` and
`env_runner.py`.  The throughput environment wrapper (`throughput_env.py`)
delegates to the adapter in the same way.

## How to add a new environment (works for both profiles)

1. Implement a new environment class under `core/environments/`.
2. Add a new adapter class implementing `EnvAdapter` in a new `*_adapter.py`
   (or alongside the env class for probes).
3. Register it in `core/environments/__init__.py` by adding an
   `id -> class` entry in `_ADAPTER_REGISTRY`.
4. **Realtime**: run with `--env-adapter <your_adapter_id> --env-type continuous`.
5. **Throughput**: run with
   `--runtime-profile throughput --env-adapter <your_adapter_id> --env-type continuous`.
6. If your adapter changes actor/filter feature counts, verify for realtime:
   - `setup_run.py` builds `ep_shm_properties` successfully.
   - minion rollout writes and env runner reads batches without shape mismatch.

## Notes

- Both profiles are continuous-action only in the current codebase.
- The throughput profile uses fast-cycling target curves by default
  (`min_hold_len=15`, `max_hold_len=60`).  Pass
  `--throughput-target-{min,max}-{hold,transition}-len` to override.
- Adapters whose target generator is not `IMEPTargetCurveGenerator` (e.g.
  probe envs with constant targets) are not affected by the throughput
  target-curve timing parameters.
