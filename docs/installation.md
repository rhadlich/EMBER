# Installation

## Scope

This repository is released as research software. Installation is supported for local development and reproducibility workflows, not production deployment.

## Python Version

- Supported: Python `>=3.9`

## Install From Repository Root

```bash
python -m pip install .
```

## Editable Install

```bash
python -m pip install -e .
```

## Optional GUI Dependencies

GUI components (`src/RLapp.py`) require optional GUI packages:

```bash
python -m pip install -e ".[gui]"
```

## Verify Basic Import

```bash
python -c "import core; print('Import successful')"
```

## Alternative Reproducibility Environment

For pinned historical environments used during research development:

- `environment.yml`
- `environment-gui.yml`

These environment files are useful for reproducibility, while `pyproject.toml` is the package installation source of truth for pip installation.
