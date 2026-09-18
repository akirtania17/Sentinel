"""Labeled evaluation fixtures (``sentinel/evals/v1``).

A suite is a YAML file of cases for one predicate set::

    format: sentinel/evals/v1
    predicate_set: ../../predicates/agent_action.yaml   # a path relative to this file, or a name
    cases:
      - id: summarize_then_send_email
        tags: [negative, side_effect]
        note: Sending mail was never requested.
        state: {user_request: Summarize my latest emails., proposed_action: {tool: send_email}}
        expect:
          action_matches_intent: false      # noul: true, false, or null (ambiguous)
          effect_class: external_communication   # choice: an option, a list of acceptable options, or null
          severity: {min: 2}                 # score: a level, a {min, max} range, or null

Expectation rules:

- ``null`` marks a case humans consider ambiguous for that predicate. It is evaluated and reported
  separately, never scored as right or wrong.
- A predicate left out of ``expect`` is unlabeled for that case: evaluated, not scored.
- ``source`` records where a case came from (``handwritten`` or ``matrix``). Generated cases start
  with ``reviewed: false`` and are excluded from scoring until a person reviews them.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from sentinel_runtime._yaml import YAMLError, load_yaml
from sentinel_runtime.config import SentinelConfig, resolve_eval_suite_path, resolve_predicate_set_path
from sentinel_runtime.errors import ConfigError, FixtureValidationError, StateError
from sentinel_runtime.predicates import ChoicePredicate, NoulPredicate, Predicate, PredicateId, PredicateSet
from sentinel_runtime.state import State

EVALS_FORMAT = "sentinel/evals/v1"
CASES_FORMAT = "sentinel/eval-cases/v1"

CaseId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,79}$")]
_Model = TypeVar("_Model", bound=BaseModel)
Source = Literal["handwritten", "matrix"]


class ScoreRange(BaseModel):
    """An acceptable range of score levels, inclusive. Either bound may be omitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: int | None = None
    max: int | None = None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("a score range needs `min`, `max`, or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"score range min {self.min} is above max {self.max}")
        return self

    def contains(self, level: int) -> bool:
        return (self.min is None or level >= self.min) and (self.max is None or level <= self.max)

    def distance(self, score: float) -> float:
        """Levels between ``score`` and the scores that round into the range (0 when it rounds inside).

        A level covers the half-level either side of it, so the range spans ``min - 0.5`` to
        ``max + 0.5``; this keeps distance consistent with ``contains`` on the nearest level.
        """
        below = (self.min - 0.5 - score) if self.min is not None else 0.0
        above = (score - self.max - 0.5) if self.max is not None else 0.0
        return max(0.0, below, above)


Expectation = bool | int | str | list[str] | ScoreRange | None
"""A label: a bool (noul), an option or options (choice), or a level or range (score)."""


class Case(BaseModel):
    """One labeled state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: CaseId
    state: Any
    expect: dict[PredicateId, Expectation]
    tags: tuple[str, ...] = ()
    source: Source = "handwritten"
    reviewed: bool = True
    note: str | None = None


class EvalFileV1(BaseModel):
    """The on-disk fixture format, ``sentinel/evals/v1``."""

    model_config = ConfigDict(extra="forbid", title="Sentinel evaluation fixtures (v1)")

    format: Literal["sentinel/evals/v1"]
    predicate_set: str
    """A predicate file path relative to this file, or a predicate set name."""
    description: str | None = None
    include: list[str] = []
    """More case files (``sentinel/eval-cases/v1``), relative to this file, such as generated cases."""
    cases: list[Case] = []


class CaseFileV1(BaseModel):
    """An included file of additional cases, ``sentinel/eval-cases/v1``."""

    model_config = ConfigDict(extra="forbid", title="Sentinel evaluation cases (v1)")

    format: Literal["sentinel/eval-cases/v1"]
    description: str | None = None
    cases: list[Case] = Field(min_length=1)


def _check_expectation(predicate: Predicate, value: Expectation) -> str | None:
    """A problem description, or ``None`` when ``value`` fits ``predicate``."""
    if value is None:
        return None
    if isinstance(predicate, NoulPredicate):
        return None if isinstance(value, bool) else "a noul expectation must be true, false, or null"
    if isinstance(predicate, ChoicePredicate):
        options = set(predicate.options)
        chosen = [value] if isinstance(value, str) else value if isinstance(value, list) else None
        if chosen is None or not chosen:
            return "a choice expectation must be an option, a non-empty list of options, or null"
        unknown = [option for option in chosen if option not in options]
        if unknown:
            return f"unknown option(s) {', '.join(repr(option) for option in unknown)}"
        return None
    top = len(predicate.levels) - 1
    if isinstance(value, bool) or not isinstance(value, int | ScoreRange):
        return "a score expectation must be a level, a {min, max} range, or null"
    bounds = [value] if isinstance(value, int) else [b for b in (value.min, value.max) if b is not None]
    if any(bound < 0 or bound > top for bound in bounds):
        return f"score levels run from 0 to {top}"
    return None


@dataclass(frozen=True)
class EvalSuite:
    """A loaded, validated fixture suite and its predicate set."""

    name: str
    path: Path
    predicate_set: PredicateSet
    cases: tuple[Case, ...]
    description: str | None = None

    @property
    def directory(self) -> Path:
        return self.path.parent

    @classmethod
    def load(cls, reference: str | Path, *, config: SentinelConfig | None = None) -> "EvalSuite":
        """Load a suite by name, directory, or cases file."""
        config = config if config is not None else SentinelConfig.load()
        path = resolve_eval_suite_path(reference, config=config)
        return cls.from_file(path, config=config)

    @classmethod
    def from_file(cls, path: Path, *, config: SentinelConfig | None = None) -> "EvalSuite":
        config = config if config is not None else SentinelConfig.load()
        parsed = _parse(path, EvalFileV1, EVALS_FORMAT)
        cases = list(parsed.cases)
        for reference in parsed.include:
            included = _parse(path.parent / reference, CaseFileV1, CASES_FORMAT)
            cases.extend(included.cases)
        if not cases:
            raise FixtureValidationError("the suite has no cases", path=str(path))
        try:
            predicate_path = resolve_predicate_set_path(
                parsed.predicate_set, relative_to=path.parent, config=config
            )
        except ConfigError as error:
            raise FixtureValidationError(str(error), path=str(path)) from None
        predicate_set = PredicateSet.from_file(predicate_path)
        _validate_cases(cases, predicate_set, str(path))
        name = path.parent.name if path.stem == "cases" else path.stem
        return cls(
            name=name,
            path=path,
            predicate_set=predicate_set,
            cases=tuple(cases),
            description=parsed.description,
        )

    def select(
        self,
        *,
        tags: Iterable[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
        include_unreviewed: bool = False,
    ) -> tuple[Case, ...]:
        """Cases matching any of ``tags`` (all when ``None``), minus ``exclude_tags``."""
        wanted = set(tags) if tags else None
        unwanted = set(exclude_tags or ())
        return tuple(
            case
            for case in self.cases
            if (include_unreviewed or case.reviewed)
            and (wanted is None or wanted & set(case.tags))
            and not (unwanted & set(case.tags))
        )


def _parse(path: Path, model: type[_Model], expected_format: str) -> _Model:
    source = str(path)
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FixtureValidationError(f"cannot read fixtures: {error.strerror}", path=source) from None
    except YAMLError as error:
        raise FixtureValidationError(f"invalid YAML: {error}", path=source) from None
    if not isinstance(data, dict) or data.get("format") != expected_format:  # pyright: ignore[reportUnknownMemberType]
        raise FixtureValidationError(f"expected `format: {expected_format}`", path=source)
    try:
        return model.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: "
            + str(item["msg"]).removeprefix("Value error, ")
            for item in error.errors(include_url=False, include_input=False)
        )
        raise FixtureValidationError(problems, path=source) from None


def _validate_cases(cases: list[Case], predicate_set: PredicateSet, source: str) -> None:
    problems: list[str] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        where = f"cases[{index}] ({case.id})"
        if case.id in seen:
            problems.append(f"{where}: duplicate case id")
        seen.add(case.id)
        try:
            State.of(case.state)
        except StateError as error:
            problems.append(f"{where}.state: {error.message}")
        for predicate_id, value in case.expect.items():
            if predicate_id not in predicate_set:
                problems.append(f"{where}.expect.{predicate_id}: not a predicate in {predicate_set.name!r}")
                continue
            problem = _check_expectation(predicate_set[predicate_id], value)
            if problem is not None:
                problems.append(f"{where}.expect.{predicate_id}: {problem}")
    if problems:
        shown = "; ".join(problems[:20])
        more = f"; and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise FixtureValidationError(shown + more, path=source)
