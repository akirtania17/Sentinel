# Contributing to Sentinel

## Setup

```bash
uv sync --all-groups
uvx pre-commit install       # optional: ruff and gitleaks on every commit
```

Sentinel supports Python 3.11 and newer. Local development uses the version in `.python-version`.

## Checks

Every change must pass the same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest -m "not live"
uv build
```

## Secrets

- The TypeSafe API key is read only from the `TYPESAFE_API_KEY` environment variable.
- Never put a key in source, tests, fixtures, recorded responses, or configuration files.
- `.env` files are git-ignored; `.env.example` shows the variables without values.
- `gitleaks` runs in CI and in the optional pre-commit hook, with a rule for TypeSafe keys.

## Live tests

Tests marked `live` call the real TypeSafe API. They are skipped unless `TYPESAFE_API_KEY` is set:

```bash
TYPESAFE_API_KEY=... uv run pytest -m live
```

Normal CI never needs a key. The scheduled live workflow uses the repository secret.

## Predicates

Predicate files are code. Keep questions narrow and literal, point them at named state fields with
backticked paths, keep arithmetic and date logic in code, and back every threshold change with an
evaluation run.

## Commits

Write focused commits with a clear subject line. Do not commit generated reports or local telemetry.
