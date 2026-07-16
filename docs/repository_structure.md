# Repository Structure

## Top-Level Layout

- `src/`: core source code and runnable scripts
- `docs/`: release-facing documentation
- `scripts/`: helper shell/python scripts
- `environment.yml`, `environment-gui.yml`: reproducibility-focused environments
- `pyproject.toml`: pip packaging metadata

## Source Subpackages

- `src/core/environments/`: environment implementations and adapter interfaces
- `src/core/digital_twin/`: digital twin model architectures, training, analysis helpers
- `src/core/safety/`: safety-filter model, training, checkpoints interface
- `src/core/training/`: shared training/HPO utilities
- `src/core/rl_modules/`: custom RLlib module implementations
- `src/configs/`: argument and algorithm configuration helpers
- `src/utils/`: utility helpers

## Script-Style Modules

- `src/setup_run.py`
- `src/run_algorithm.py`
- `src/run_algorithm_throughput.py`
- `src/env_runner.py`
- `src/minion.py`
- `src/RLapp.py`

These are packaged as Python modules for installability while preserving the current repository layout.
