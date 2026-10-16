# Contributing

Thanks for your interest in Decision4Rec. Issues and pull requests are welcome.

## Reporting a problem

Open an issue with the command you ran, the configuration files, the last lines of the stage log
(`logs/small/<stage>.log`) and your hardware. For wrong numbers, add the report file under `runs/`.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Install the development environment: `bash scripts/env.sh`.
3. Add a test under `tests/` for any change of behaviour, and run `pytest -q` (all tests must pass).
4. Keep one change per pull request and describe how you checked it.

New datasets, decision-model backends and agents are especially welcome; the guides in `docs/` show where they plug in.

## Rules for experiments

- Never train a decision model on the outputs of a commercial model whose terms forbid it (for example the TypeSafe
  terms of service for Jev).
- Never commit API keys, datasets or answers of hosted models.
- Tune on validation users only; test users are read once per configuration.
