"""Shared CLI plumbing: exit codes, consoles, error handling, input loading, runtime construction."""

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape

from sentinel_runtime._yaml import YAMLError, load_yaml
from sentinel_runtime.config import SentinelConfig, resolve_predicate_set_path
from sentinel_runtime.errors import (
    CacheMissError,
    ConfigError,
    FixtureValidationError,
    PredicateValidationError,
    ProviderError,
    StateError,
)
from sentinel_runtime.predicates import PredicateSet
from sentinel_runtime.providers.caching import CacheMode
from sentinel_runtime.runtime import Runtime
from sentinel_runtime.telemetry import TelemetrySink

EXIT_OK = 0
EXIT_USAGE = 1
"""Invalid input: predicate files, fixtures, state, configuration, or arguments."""
EXIT_PROVIDER = 2
"""The provider could not produce judgments."""
EXIT_REGRESSION = 3
"""An evaluation regressed against its baseline."""
EXIT_CACHE_MISS = 4
"""An offline run needed responses that are not in the cache."""

# Terminals get their real width; pipes and logs get a fixed, wide width so tables never truncate.
PIPE_WIDTH = 120
console = Console(highlight=False, width=None if sys.stdout.isatty() else PIPE_WIDTH)
err_console = Console(stderr=True, highlight=False, width=None if sys.stderr.isatty() else PIPE_WIDTH)


def fail(message: str, code: int = EXIT_USAGE) -> NoReturn:
    err_console.print(f"[bold red]error:[/] {escape(message)}")
    raise typer.Exit(code)


@contextmanager
def handle_errors() -> Generator[None]:
    """Turn Sentinel errors into one-line messages and documented exit codes."""
    failure: tuple[str, int] | None = None
    try:
        yield
    except CacheMissError as error:
        failure = (str(error), EXIT_CACHE_MISS)
    except ProviderError as error:
        failure = (str(error), EXIT_PROVIDER)
    except (PredicateValidationError, FixtureValidationError, StateError, ConfigError) as error:
        failure = (str(error), EXIT_USAGE)
    if failure is not None:
        fail(*failure)


def load_data_file(path: str, what: str) -> Any:
    """Read a JSON or YAML file (``-`` for stdin)."""
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read {what} {path}: {error.strerror}") from None
    try:
        return load_yaml(text)
    except YAMLError as error:
        raise ConfigError(f"{what} {path} is not valid JSON or YAML: {error}") from None


def load_predicate_set(reference: str) -> PredicateSet:
    """Load a predicate set by path or by name."""
    return PredicateSet.from_file(resolve_predicate_set_path(reference, config=SentinelConfig.load()))


def split_ids(value: str | None) -> list[str] | None:
    """Parse a comma-separated id list."""
    if value is None:
        return None
    ids = [part.strip() for part in value.split(",") if part.strip()]
    return ids or None


def cache_mode(*, offline: bool, refresh: bool) -> CacheMode:
    if offline and refresh:
        raise ConfigError("--offline and --refresh cannot be combined")
    if offline:
        return "offline"
    return "record" if refresh else "replay"


def make_runtime(
    *,
    model: str | None,
    cache: Path | None,
    offline: bool = False,
    refresh: bool = False,
    sinks: Sequence[TelemetrySink] = (),
) -> Runtime:
    """A runtime from the environment, optionally behind a response cache."""
    mode = cache_mode(offline=offline, refresh=refresh)
    if cache is None and mode != "replay":
        raise ConfigError("--offline and --refresh need --cache")
    return Runtime.from_env(
        model=model, cache=cache, cache_mode=mode, sinks=sinks, config=SentinelConfig.load()
    )
