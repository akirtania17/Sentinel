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
for suite in agent_action output_verification trace_semantics; do
  uv run sentinel eval run "$suite" --offline --fail-on-regression
done
uv build
```

CI also installs the built wheel and sdist into clean environments and runs the command line from
outside the checkout.

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

Normal CI never needs a key. The `live` workflow uses the repository secret: it runs weekly and on
demand, and opens a pull request with refreshed evaluation caches when the answers changed materially
(see [docs/evals.md](docs/evals.md#weekly-live-refresh)).

## Predicates

Predicate files are code. Keep questions narrow and literal, point them at named state fields with
backticked paths, keep arithmetic and date logic in code, and back every threshold change with an
evaluation run.

## Commits

Write focused commits with a clear subject line. Do not commit generated reports or local telemetry.

## Releasing

1. Set `version` in `pyproject.toml`; `sentinel_runtime.__version__` reads it from the installed
   package metadata.
2. Optionally rehearse by running the `release` workflow by hand, which publishes to TestPyPI only.
3. Publish a GitHub Release tagged `v<version>` with the release notes. The `release` workflow
   checks that the tag matches the version, builds once, and publishes to TestPyPI and then PyPI
   through trusted publishing. There is no hand-maintained changelog; release notes live in GitHub
   Releases.
