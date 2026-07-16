# EMBER

Engine Model-Based Barrier-Enhanced Reinforcement Learning (EMBER) is a research software repository for studying safe reinforcement-learning control of internal combustion engines with model-based safety filtering.

> [!IMPORTANT]
> This is an initial public **research release**. It is provided for research transparency and reproducibility and is **not** intended as production software.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXXXX)  
Replace the DOI placeholder after Zenodo creates the release record.

## Research Motivation

The repository explores how reinforcement learning can be combined with model-based constraints and safety filtering for adaptive engine control under nonlinear and safety-critical conditions.

## Main Capabilities

- RL training harness built on Ray RLlib with custom modules and runners.
- Shared-memory training/runtime pipeline (`setup_run.py`, `env_runner.py`, `minion.py`).
- Engine-focused environment adapters and safety-filter training/evaluation code.
- Digital twin model training and analysis utilities.
- Optional local GUI telemetry via ZMQ + PyQt.

## Repository Status

- Development status: alpha research codebase.
- API and repository organization may evolve.
- Includes components that require external datasets and model artifacts not shipped with this release.

## Repository Structure

```text
.
├── pyproject.toml
├── environment.yml
├── environment-gui.yml
├── docs/
├── scripts/
└── src/
    ├── setup_run.py
    ├── run_algorithm.py
    ├── run_algorithm_throughput.py
    ├── env_runner.py
    ├── minion.py
    ├── RLapp.py
    ├── configs/
    ├── core/
    │   ├── environments/
    │   ├── digital_twin/
    │   ├── safety/
    │   ├── training/
    │   └── rl_modules/
    └── utils/
```

See [`docs/repository_structure.md`](docs/repository_structure.md) for details.

## Installation

### Standard installation

```bash
python -m pip install .
```

### Editable installation

```bash
python -m pip install -e .
```

### Verify imports

```bash
python -c "import core; print('Import successful')"
```

Additional installation notes: [`docs/installation.md`](docs/installation.md).

## Dependencies and Environments

- Packaging dependencies for `pip install` are declared in `pyproject.toml`.
- Reproducibility-focused pinned environments are provided in:
  - `environment.yml` (main research environment)
  - `environment-gui.yml` (GUI extras)

## Quick Start

Run from repository root.

### Realtime profile (default)

```bash
python src/setup_run.py --algo SAC --env-type continuous --stop-iters 10
```

### Throughput profile

```bash
python src/setup_run.py --runtime-profile throughput --algo TD3 --env-type continuous --stop-iters 10
```

### Digital twin training module help

```bash
python -m core.digital_twin.train_digital_twin --help
```

More examples: [`docs/quickstart.md`](docs/quickstart.md).

## Typical Workflows

- RL training runs: realtime or throughput profile via `src/setup_run.py`.
- Safety model workflow: `core.safety.train_safety_filter` then `core.safety.evaluate_safety_filter`.
- Digital twin workflow: `core.digital_twin.train_digital_twin`.
- Post-hoc interpretability and visualization scripts for analysis.

## Configuration

Primary runtime parameters are in CLI flags and algorithm presets:

- `src/configs/args.py`
- `src/configs/algorithms/*.py`

See [`docs/configuration.md`](docs/configuration.md).

## Data Expectations and Outputs

- External HDF5 datasets are expected for digital twin and safety modules.
- Some scripts currently contain machine-specific absolute paths and require local adjustment.
- Trained model weights, checkpoints, and generated experiment outputs are excluded from release artifacts by design.

Details:

- [`docs/data.md`](docs/data.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/hardware.md`](docs/hardware.md)

## Reproducibility Notes

This public release enables code inspection and installation, and supports local execution where required external datasets and artifacts are available. Full reproduction of all prior private experiments is not guaranteed from this repository alone.

## Limitations

- Not production-hardened (error handling, API stability, CI, and testing are limited).
- Several analysis scripts are configured with local absolute paths by default.
- Some workflows depend on external datasets and model artifacts not included in the repository.

## Citation

Please cite EMBER using [`CITATION.cff`](CITATION.cff).

## Related Publications

Publication references and DOIs should be added when finalized.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

Developed as academic research software for engine-control and safe RL investigations.

## Contact and Issue Reporting

Please use GitHub issues for bug reports and questions:  
<https://github.com/rhadlich/EMBER/issues>

For private vulnerability reporting, see [`SECURITY.md`](SECURITY.md).
