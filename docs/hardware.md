# Hardware and System Notes

## Supported Context

This codebase is research-oriented and has been developed around local workstation workflows.

## Runtime Considerations

- Some paths use shared memory and low-latency process communication.
- ZMQ + IPC endpoints are used for optional GUI telemetry.
- Certain scheduling and CPU-affinity options are Linux-specific and may be ignored on macOS.
- ONNXRuntime provider behavior may differ by operating system/hardware.

## GPU/CPU Expectations

- Training code supports CPU and CUDA-capable setups depending on environment.
- Exact throughput and reproducibility may vary across hardware configurations.

## Not Included in This Release

- Specialized hardware integration instructions beyond what exists in code and script arguments.
