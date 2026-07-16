# Troubleshooting

## `ModuleNotFoundError`

- Ensure installation was done from repository root:
  - `python -m pip install -e .`
- Re-run from a clean virtual environment if import resolution is inconsistent.

## Missing Data or Checkpoint Files

- Many workflows require external HDF5 data and trained artifacts not included in this release.
- Pass explicit local paths via CLI flags instead of relying on defaults.

## Runtime Shared Memory/ZMQ Issues

- Stale processes can leave shared-memory or IPC endpoints behind.
- Stop all related Python processes and restart the run.

## GUI Issues

- Install GUI extras: `python -m pip install -e ".[gui]"`
- Ensure local IPC endpoints are accessible on your OS/runtime.

## Build/Packaging Issues

- Verify packaging with:
  - `python -m pip install .`
  - `python -m pip install -e .`
  - `python -m build`
  - `python -m pip check`
