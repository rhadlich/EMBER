# Configuration Guide

## Primary Configuration Surfaces

- Global/common CLI arguments: `src/configs/args.py`
- Algorithm-specific CLI/config updates: `src/configs/algorithms/*.py`
- Environment adapter registration: `src/core/environments/__init__.py`

## Runtime Profiles

`src/setup_run.py` exposes:

- `--runtime-profile realtime` (shared-memory/minion path)
- `--runtime-profile throughput` (RLlib-native throughput path)

## Important Runtime Inputs

- Dataset path flags (for predictor and safety filter statistics)
- Checkpoint load directory flags
- Algorithm choice (`--algo`)
- Environment type (`--env-type`)
- Seeding and stop criteria

## Known Portability Caveat

Some defaults are machine-specific absolute paths in scripts and argument defaults. For portable runs, explicitly pass local paths in CLI arguments.

## Safety Filter and Digital Twin

Safety-filter and digital-twin training/evaluation modules have their own arguments and expected data formats. Use each module's `--help` output before running.
