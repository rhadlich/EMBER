# Legacy Files

This folder contains the original files from before the repository reorganization.

These files were moved here after the migration to the new structure:
- `src/` - Core source code
- `configs/` - Configuration files
- `scripts/` - Entry point scripts
- `assets/` - Model weights and other assets

## Contents

### Root-level Python files:
- `Master.py` → Now `scripts/master.py`
- `custom_run.py` → Now `src/training/custom_run.py`
- `define_args.py` → Now `configs/args.py`
- `gymCustom.py` → Now `src/core/environments/engine_env.py`
- `logging_setup.py` → Now `src/utils/logging_setup.py`
- `minion.py` → Now `scripts/minion.py`
- `ray_primitives.py` → Now `src/utils/ray_primitives.py`
- `safety_filter.py` → Now `src/core/safety/safety_filter.py`
- `shared_memory_env_runner.py` → Now `src/training/env_runner.py`
- `utils.py` → Now `src/utils/utils.py`

### Directories:
- `algo_configs/` → Now `configs/algorithms/`
- `gym_utils/` → Now `src/core/environments/` (split into multiple files)
- `torch_rl_modules/` → Now `src/core/rl_modules/`

## Note

These files are kept for reference only. The new structure should be used going forward.
All imports and paths have been updated in the new structure.
