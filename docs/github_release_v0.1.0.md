# EMBER v0.1.0

## Overview

Initial public research release of EMBER (Engine Model-Based Barrier-Enhanced Reinforcement Learning), provided for research transparency and reproducibility.

## What Is Included

- Research codebase for RL training, safety filtering, and digital twin workflows.
- Packaging configuration for local pip installation from repository root.
- Release documentation, changelog, citation metadata, and release checklist.

## Installation

```bash
python -m pip install .
```

Editable install:

```bash
python -m pip install -e .
```

## Documentation

- Main project overview: `README.md`
- Detailed docs: `docs/`
- Citation metadata: `CITATION.cff`

## Reproducibility Notes

- This release focuses on code transparency and installability.
- External datasets and trained artifacts are required for many full workflows.
- The repository/API structure may evolve.

## Known Limitations

- Not yet intended for production deployment.
- Some scripts retain machine-specific default paths and require local configuration.
- Full reproduction of private historical experiments may require external, non-public assets.

## Citation

Please cite this software using `CITATION.cff`.  
Zenodo DOI details should be updated after archive creation.

## License

Released under the MIT License.
