# Release Checklist

## Versioning and Metadata

- [ ] Confirm release version in `pyproject.toml`.
- [ ] Set release date in `CITATION.cff`.
- [ ] Update `CITATION.cff` metadata as needed.
- [ ] Update `.zenodo.json` metadata (if used).
- [ ] Update `CHANGELOG.md`.

## Packaging and Installation

- [ ] Run `python -m pip install .` in a clean environment.
- [ ] Run `python -m pip install -e .` in a clean environment.
- [ ] Run `python -m build`.
- [ ] Run `python -m pip check`.
- [ ] Verify base import command (`python -c "import core; print('Import successful')"`).

## Documentation Checks

- [ ] Validate README commands and links.
- [ ] Validate docs links and relative paths.
- [ ] Confirm limitations and external requirements are clearly stated.
- [ ] Confirm DOI placeholder/badge is updated if DOI exists.

## Repository Hygiene

- [ ] Check for secrets/credentials/local private data.
- [ ] Check for absolute local/institution-specific paths in tracked files.
- [ ] Confirm generated outputs/checkpoints are excluded from tracked release files.
- [ ] Confirm license file is present and correct.

## Release Execution

- [ ] Create and push release tag (for example, `v0.1.0`).
- [ ] Create GitHub release notes from `docs/github_release_v0.1.0.md`.
- [ ] Verify Zenodo archive creation for the tagged release.
- [ ] Confirm version-specific DOI.
- [ ] Confirm all-versions Zenodo DOI.
