# Contributing

Thanks for your interest in EMBER.

## Project Context

EMBER is currently maintained primarily as research software. Contributions are welcome, but major architectural changes or refactors should be discussed before implementation.

## How to Report Issues

- Open an issue on GitHub with:
  - a clear description,
  - reproduction steps (if applicable),
  - environment details,
  - relevant logs/traceback snippets.

## How to Propose Changes

- Open an issue for substantial changes before opening a pull request.
- Keep pull requests scoped and reviewable.
- Preserve existing behavior unless change of behavior is explicitly intended and documented.

## Development Setup

```bash
python -m pip install -e .
```

Optional:

```bash
python -m pip install -e ".[test,docs,gui]"
```

## Research-Code Policy

- Avoid large refactors without prior discussion.
- Do not commit private datasets, local checkpoints, or generated experiment artifacts.
