# 🔥 EMBER

**E**ngine **M**odel-Based **B**arrier-Enhanced **R**einforcement Learning

EMBER is a research framework for **safe and adaptive control of internal combustion engines** using **reinforcement learning (RL)** with **Control Barrier Functions (CBFs)**.  
It is designed to enable learning-based control while **provably enforcing safety constraints**, with a focus on **low-temperature combustion (LTC)** regimes.

---

## 🚀 Features

- Reinforcement learning for nonlinear engine control
- Safety filtering (CBF-inspired) with a learned dynamics model
- Explicit enforcement of state and input constraints
- Modular design for models, constraints, and agents
- Built on Ray + RLlib (custom training harness + custom RLModules)
- Shared-memory bridge between RLlib and an external “minion” process (low-latency rollout collection)
- Optional live telemetry to a PyQt6 GUI via ZMQ/IPC
- Discrete + continuous engine environments backed by learned surrogate model weights

---

## ⚙️ Getting Started

### Prerequisites

- **Python**: 3.9 (see `rayenv2_mac.yml`)
- **Conda**: recommended (miniconda/anaconda)
- **OS notes**:
  - The runtime uses `onnxruntime` and may leverage the **CoreML execution provider** on macOS (with CPU fallback).
  - Some options (like real-time scheduling priority) are Linux-specific and may be ignored on macOS.

### Clone the repository

```bash
git clone <REPO_URL>
cd EMBER
```

### Install dependencies (recommended: Conda)

This repo ships a pinned environment file (there is no `pyproject.toml`/`requirements.txt` at the moment).

```bash
conda env create -f rayenv2_mac.yml
conda activate rayenv2
```

Tip: run scripts from the repo root so imports like `import src...` work. If you run into import errors, use:

```bash
export PYTHONPATH=.
```

### Quickstart: run a local training loop

The primary entrypoint is `scripts/master.py`. It configures an RLlib algorithm, spins up a custom `EnvRunner`, and schedules a “minion” worker that collects rollouts via shared memory.

#### IMPALA (discrete env by default)

```bash
python scripts/master.py \
  --algo IMPALA \
  --env-type discrete \
  --stop-iters 10 \
  --local-mode
```

#### SAC (continuous env by default)

```bash
python scripts/master.py \
  --algo SAC \
  --env-type continuous \
  --stop-iters 10 \
  --local-mode
```

#### APPO (continuous env by default)

```bash
python scripts/master.py \
  --algo APPO \
  --env-type continuous \
  --stop-iters 10 \
  --local-mode
```

### Optional: enable telemetry for the GUI

Both the minion (“engine”) side and the trainer can publish telemetry over ZMQ when enabled:

```bash
python scripts/master.py \
  --algo IMPALA \
  --env-type discrete \
  --enable-zmq True
```

The GUI (`app/RLapp.py`) subscribes to:
- `ipc:///tmp/engine.ipc`
- `ipc:///tmp/training.ipc`

Note: `app/RLapp.py` is currently a research/monitoring script and may need small local path tweaks (it spawns a master process and may not point at `scripts/master.py` out of the box).

### Where to look (repo layout)

- **`scripts/`**: entrypoints (`master.py`, `minion.py`)
- **`configs/`**: CLI + algorithm presets (`configs/args.py`, `configs/algorithms/*_cfg.py`)
- **`src/training/`**: RLlib training harness and shared-memory env runner (`custom_run.py`, `env_runner.py`)
- **`src/core/environments/`**: engine environments and learned surrogate model interface (`engine_env.py`, `predictor.py`)
- **`src/core/safety/`**: learned dynamics + runtime safety filter (`safety_filter.py`)
- **`assets/`**: model weights used by the surrogate predictor (`assets/models/…`)
- **`legacy/`**: archived pre-reorg code (see `legacy/README.md`)

### Troubleshooting

- **Import errors (`ModuleNotFoundError: src...`)**: run from the repo root and/or set `PYTHONPATH=.` (see above).
- **Stale shared-memory / stuck processes**: stop the Python processes and rerun. Shared-memory segments and IPC endpoints can linger if the program is interrupted.
- **ONNXRuntime provider issues**: if CoreML isn’t available (or fails), the code should fall back to CPU execution, but performance may differ across machines.
