# Quick Start

All commands below assume you are in the repository root.

## 1) Install

```bash
python -m pip install -e .
```

## 2) Realtime Training Profile

```bash
python src/setup_run.py --algo SAC --env-type continuous --stop-iters 10
```

## 3) Throughput Training Profile

```bash
python src/setup_run.py --runtime-profile throughput --algo TD3 --env-type continuous --stop-iters 10
```

## 4) Digital Twin Training Script

```bash
python -m core.digital_twin.train_digital_twin --help
```

## 5) Safety Filter Training Script

```bash
python -m core.safety.train_safety_filter --help
```

## Notes

- Many workflows require external HDF5 datasets and model artifacts not included in this public release.
- Some scripts include machine-specific absolute default paths and may require local CLI overrides.
- This repository is intended for research use and transparency.
