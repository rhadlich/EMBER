# Reproducibility Notes

## What This Public Release Supports

- Source code transparency for research methods.
- Local installation using `pip install .` and `pip install -e .`.
- Reproduction of workflows where required external files are available.

## What Is Not Included

- Trained model checkpoints/weights generated during private experiments.
- Local checkpoint directories and generated outputs.
- Institution/local absolute-path data folders.

## Environment Files

- `pyproject.toml` defines pip-install dependencies.
- `environment.yml` and `environment-gui.yml` preserve historical pinned environments.

## Practical Guidance

- Prefer explicit CLI path arguments instead of relying on script defaults.
- Record dataset versions, seeds, and command invocations for each experiment.
- Keep generated artifacts outside the tracked source tree unless intentionally archived for a release.
