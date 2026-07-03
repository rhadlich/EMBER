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
- Actor observation normalization bounds and mapping.
- Actor action normalization bounds and policy-to-physical mapping.
- Filter action schema declaration (`get_filter_action_features`) used by
  setup/runtime/checkpoint compatibility checks.

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

## Actor Observation Normalization

- Actor-side observations are normalized inside each adapter via `obs_to_actor`.
- Each adapter defines ordered raw actor bounds with `get_actor_obs_bounds`.
- Bounds order must exactly match `get_actor_state_features` and `obs_to_actor`.
- Both realtime and throughput profiles consume the same normalized actor vectors.

## Actor Action Normalization

- RLlib, the policy network, and the actor replay buffer use actions in `[-1, 1]`.
- Physical action bounds come from the underlying `env.action_space`.
- `EnvAdapter.get_normalized_action_space()` declares the policy action space for RLlib.
- `ActionAdapter.get_action_in_env_range()` converts policy actions to physical units
  at the filter boundary (`action_actor_to_filter`).
- Filter and environment stepping never receive normalized actor actions.
- Filter **states** (`obs_to_filter_input`) are raw physical features at the adapter
  boundary. `SafetyFilter` and online filter training normalize state and action
  inputs using HDF5 stats from `--filter-sample-data-dir` (`feature_mean`,
  `feature_std`, `action_min`, `action_max`). Barrier/CBF terms (`compute_h`,
  `lie_F`) always use the physical state; model targets (`next_state` / MPRR)
  stay in raw physical units.

## Safety Filter Action Schema

- The adapter declares filter-action feature order with
  `get_filter_action_features(env=...)`.
- Realtime setup derives `filter_action_dim` from this feature list instead of
  hardcoding from `env.action_space`.
- For `engine_continuous`, filter action is now 2D and ordered as:
  1. `SOI2`
  2. `ID2`
- Engine ID1 is immutable for the current cycle at the filter boundary; the
  adapter reassembles environment actions as
  `[current_executed_ID1, filtered_SOI2, filtered_ID2]`.

### Breaking change

Prior RLlib module checkpoints are incompatible with this action representation.
Retrain from scratch after upgrading; replay-buffer actions are now in `[-1, 1]`
rather than physical units.

Legacy safety-filter checkpoints trained with 3D action vectors (ID1, SOI2, ID2)
are incompatible with the new 2D engine safety schema. Regenerate safety-filter
HDF5 stats/data and retrain/export filter checkpoints/ORT artifacts with
action order `[SOI2, ID2]`.
