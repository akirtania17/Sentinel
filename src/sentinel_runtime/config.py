"""Configuration: ``sentinel.toml``, model precedence, and name resolution.

``sentinel.toml`` is optional. Sentinel looks for it in the working directory and its parents::

    [runtime]
    model = "jev-1.13.0"

    [paths]
    predicates = "predicates"   # relative to the sentinel.toml directory
    evals = "evals"

    [pricing."jev-1.13.0"]
    input_per_mtok = 0.042
    output_per_mtok = 0.0
    as_of = "2026-09-17"

Model precedence, highest first: an explicit per-call model, ``Runtime(model=...)``,
``SENTINEL_MODEL``, ``TYPESAFE_DEFAULT_MODEL``, ``[runtime] model``, then ``jev-latest``.

Predicate sets and eval suites are referenced by path or by name. A reference that contains a path
separator, starts with ``.``, or ends in ``.yaml``/``.yml``/``.json`` is a path. Anything else is a
name, looked up in ``[paths]`` from ``sentinel.toml`` and then in ``./predicates`` or ``./evals``.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sentinel_runtime.errors import ConfigError

CONFIG_FILENAME = "sentinel.toml"
DEFAULT_MODEL = "jev-latest"
MODEL_ENV = "SENTINEL_MODEL"
TYPESAFE_MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
DEFAULT_PREDICATES_DIR = "predicates"
DEFAULT_EVALS_DIR = "evals"
_SUFFIXES = (".yaml", ".yml", ".json")
_CASES_FILES = ("cases.yaml", "cases.yml", "cases.json")


class PriceEntry(BaseModel):
    """Price for one model, in US dollars per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(default=0.0, ge=0)
    as_of: str | None = None


class _RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None


class _PathsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicates: str | None = None
    evals: str | None = None


class _SentinelToml(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: _RuntimeSection = Field(default_factory=_RuntimeSection)
    paths: _PathsSection = Field(default_factory=_PathsSection)
    pricing: dict[str, PriceEntry] = Field(default_factory=dict)


@dataclass(frozen=True)
class SentinelConfig:
    """Resolved configuration. ``root`` is the directory of the ``sentinel.toml`` in use, if any."""

    root: Path | None = None
    model: str | None = None
    predicates_dir: Path | None = None
    evals_dir: Path | None = None
    pricing: Mapping[str, PriceEntry] = field(default_factory=dict[str, PriceEntry])

    @classmethod
    def load(cls, start: Path | None = None) -> "SentinelConfig":
        """Load the nearest ``sentinel.toml`` at or above ``start`` (default: the working directory)."""
        path = find_config_file(start)
        return cls() if path is None else cls.from_file(path)

    @classmethod
    def from_file(cls, path: Path) -> "SentinelConfig":
        """Load a specific ``sentinel.toml``."""
        try:
            raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ConfigError(f"{path}: cannot read: {error.strerror}") from None
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: invalid TOML: {error}") from None
        try:
            parsed = _SentinelToml.model_validate(raw)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors(include_url=False, include_input=False)
            )
            raise ConfigError(f"{path}: {problems}") from None
        root = path.resolve().parent
        return cls(
            root=root,
            model=parsed.runtime.model,
            predicates_dir=root / parsed.paths.predicates if parsed.paths.predicates else None,
            evals_dir=root / parsed.paths.evals if parsed.paths.evals else None,
            pricing=dict(parsed.pricing),
        )


def find_config_file(start: Path | None = None) -> Path | None:
    """Return the nearest ``sentinel.toml`` at or above ``start``."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        path = candidate / CONFIG_FILENAME
        if path.is_file():
            return path
    return None


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def resolve_model(
    explicit: str | None = None, *, runtime_model: str | None = None, config: SentinelConfig | None = None
) -> str:
    """Apply the documented model precedence and return the model name to request."""
    for candidate in (
        explicit,
        runtime_model,
        _env(MODEL_ENV),
        _env(TYPESAFE_MODEL_ENV),
        config.model if config else None,
    ):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return DEFAULT_MODEL


def is_path_reference(reference: str | Path) -> bool:
    """Whether ``reference`` should be read as a path rather than a name."""
    if isinstance(reference, Path):
        return True
    return (
        "/" in reference
        or os.sep in reference
        or reference.startswith(".")
        or reference.lower().endswith(_SUFFIXES)
    )


def _search_dirs(configured: Path | None, default: str) -> list[Path]:
    dirs: list[Path] = []
    if configured is not None:
        dirs.append(configured)
    fallback = Path.cwd() / default
    if fallback.resolve() not in {d.resolve() for d in dirs}:
        dirs.append(fallback)
    return dirs


def resolve_predicate_set_path(
    reference: str | Path, *, relative_to: Path | None = None, config: SentinelConfig | None = None
) -> Path:
    """Resolve a predicate-set reference (path or name) to an existing file."""
    config = config if config is not None else SentinelConfig.load()
    if is_path_reference(reference):
        path = Path(reference)
        if not path.is_absolute() and relative_to is not None:
            path = relative_to / path
        if path.is_file():
            return path
        raise ConfigError(f"predicate file not found: {path}")
    name = str(reference)
    tried: list[Path] = []
    for directory in _search_dirs(config.predicates_dir, DEFAULT_PREDICATES_DIR):
        for suffix in _SUFFIXES:
            candidate = directory / f"{name}{suffix}"
            tried.append(candidate)
            if candidate.is_file():
                return candidate
    raise ConfigError(f"predicate set {name!r} not found; looked in: {', '.join(str(p) for p in tried)}")


def resolve_eval_suite_path(reference: str | Path, *, config: SentinelConfig | None = None) -> Path:
    """Resolve an eval-suite reference (a cases file, a suite directory, or a name) to a cases file."""
    config = config if config is not None else SentinelConfig.load()
    if is_path_reference(reference) or Path(reference).exists():
        path = Path(reference)
        if path.is_file():
            return path
        if path.is_dir():
            for filename in _CASES_FILES:
                if (path / filename).is_file():
                    return path / filename
            raise ConfigError(f"eval suite directory {path} has no cases.yaml")
        raise ConfigError(f"eval suite not found: {path}")
    name = str(reference)
    tried: list[Path] = []
    for directory in _search_dirs(config.evals_dir, DEFAULT_EVALS_DIR):
        for filename in _CASES_FILES:
            candidate = directory / name / filename
            tried.append(candidate)
            if candidate.is_file():
                return candidate
    raise ConfigError(f"eval suite {name!r} not found; looked in: {', '.join(str(p) for p in tried)}")
