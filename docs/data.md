# Data Requirements

## Expected Formats

The repository expects HDF5-based data inputs for multiple workflows, including:

- Digital twin model training/evaluation.
- Safety-filter training/evaluation.
- Normalization/statistics lookup for runtime components.

## Public Release Data Policy

This repository does not bundle private or institution-specific datasets in the code release.

## User Responsibilities

- Prepare local dataset directories compatible with module expectations.
- Provide explicit path arguments for dataset roots when running scripts.
- Verify split names and HDF5 schema expected by each training/evaluation module.

## Current Limitation

Some modules contain absolute default paths intended for the original research environment. These defaults should be treated as placeholders and replaced with local paths at runtime.
