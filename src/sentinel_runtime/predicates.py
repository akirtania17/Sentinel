"""Semantic predicates: named, fingerprinted, typed judgments over application state.

A predicate is one narrow question a semantic model answers about a state:

- ``noul``: the probability that a statement is true.
- ``choice``: a probability distribution over named options.
- ``score``: a probability distribution over ordered levels.

Predicates live in reviewable YAML files (``sentinel/predicates/v1``) or are built in code with
:func:`noul`, :func:`choice` and :func:`score`. Both routes produce the same immutable models.

Every predicate has a ``fingerprint``: ``sha256`` over exactly the fields the model sees (kind,
question, and criteria, options or levels). Thresholds, tags, descriptions and the optional human
``version`` label do not change it, so a changed question is always distinguishable from a changed
threshold. Choice options are fingerprinted in order, because option order is part of the question.

A predicate may declare ``sees``: the state paths it is shown (see ``sentinel_runtime.views``). The
runtime sends it only that part of the state, so text elsewhere in the state cannot reach it. A view
changes the state, not the question, so it is not part of the fingerprint; every cached answer is
keyed by the exact state its request sent.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from sentinel_runtime._yaml import YAMLError, dump_yaml, load_yaml
from sentinel_runtime.errors import PredicateValidationError
from sentinel_runtime.hashing import canonical_json, digest_of
from sentinel_runtime.views import StatePath

PREDICATES_FORMAT = "sentinel/predicates/v1"

_IDENTIFIER = r"^[a-z][a-z0-9_]{0,63}$"
MAX_OPTIONS = 255
MAX_LEVELS = 10
MAX_OPTION_KEY_CHARS = 200

PredicateKind = Literal["noul", "choice", "score"]


def _check_text(value: Any, what: str) -> Any:
    """Require non-empty, JSON-representable text or structure."""
    checked: Any = value
    if isinstance(checked, str):
        if not checked.strip():
            raise ValueError(f"{what} must not be empty")
    elif isinstance(checked, dict | list):
        if not checked:
            raise ValueError(f"{what} must not be empty")
        try:
            canonical_json(checked)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{what} must be JSON-representable: {error}") from None
    return value


def _check_question(value: Any) -> Any:
    return _check_text(value, "question")


def _yaml_scalar_key(key: Any) -> Any:
    """Undo YAML scalar resolution on mapping keys that must be strings."""
    if key is True:
        return "true"
    if key is False:
        return "false"
    if isinstance(key, int | float):
        return str(key)
    return key


def _normalize_option_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_yaml_scalar_key(key): option for key, option in value.items()}  # pyright: ignore[reportUnknownVariableType]
    return value


def _check_options(value: dict[str, Any]) -> dict[str, Any]:
    for key, description in value.items():
        if not key.strip() or key != key.strip():
            raise ValueError(f"option key {key!r} must be non-empty without leading or trailing whitespace")
        if len(key) > MAX_OPTION_KEY_CHARS:
            raise ValueError(f"option key {key[:40]!r}... is longer than {MAX_OPTION_KEY_CHARS} characters")
        if description is not None:
            _check_text(description, f"description of option {key!r}")
    return value


def _check_levels(value: tuple[Any, ...]) -> tuple[Any, ...]:
    for index, level in enumerate(value):
        _check_text(level, f"level {index}")
    return value


def _check_tags(value: tuple[str, ...]) -> tuple[str, ...]:
    for tag in value:
        if not tag.strip():
            raise ValueError("tags must not be empty strings")
    return value


Text = str | dict[str, Any] | list[Any]
"""A question, criterion, option description or level: a string, or JSON structure for richer guidance."""

PredicateId = Annotated[str, StringConstraints(pattern=_IDENTIFIER)]
Question = Annotated[Text, AfterValidator(_check_question)]
Options = Annotated[
    dict[str, Text | None],
    BeforeValidator(_normalize_option_keys),
    Field(min_length=2, max_length=MAX_OPTIONS),
    AfterValidator(_check_options),
]
Levels = Annotated[
    tuple[Text, ...], Field(min_length=2, max_length=MAX_LEVELS), AfterValidator(_check_levels)
]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
Tags = Annotated[tuple[str, ...], AfterValidator(_check_tags)]


class NoulCriteria(BaseModel, frozen=True, extra="forbid"):
    """Optional descriptions of what a ``true`` and a ``false`` answer mean."""

    true: Text | None = None
    false: Text | None = None

    @model_validator(mode="before")
    @classmethod
    def _yaml_bool_keys(cls, data: Any) -> Any:
        # YAML reads the keys `true:` and `false:` as booleans.
        if isinstance(data, Mapping):
            return {_yaml_scalar_key(key): value for key, value in data.items()}  # pyright: ignore[reportUnknownVariableType]
        return data

    @model_validator(mode="after")
    def _describe_something(self) -> Self:
        if self.true is None and self.false is None:
            raise ValueError("criteria must describe `true`, `false`, or both")
        if self.true is not None:
            _check_text(self.true, "criteria.true")
        if self.false is not None:
            _check_text(self.false, "criteria.false")
        return self

    def wire(self) -> dict[str, Any]:
        """The criteria with undescribed outcomes omitted."""
        return {
            key: value for key, value in (("true", self.true), ("false", self.false)) if value is not None
        }


class _PredicateBase(BaseModel, frozen=True, extra="forbid"):
    id: PredicateId
    question: Question
    """The judgment, sent to the model. A string, or a JSON object/array for structured guidance."""
    description: str | None = None
    """Human documentation. Never sent to the model."""
    tags: Tags = ()
    version: str | None = None
    """An optional human label. The fingerprint, not this label, identifies model-facing content."""
    sees: tuple[StatePath, ...] | None = Field(default=None, min_length=1)
    """The state paths this predicate is shown; ``None`` shows the whole state."""

    def model_facing(self) -> dict[str, Any]:
        """Exactly the content the model sees: kind, question, and criteria, options or levels."""
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        """``sha256`` over the model-facing fields: kind, question, and criteria, options or levels."""
        return digest_of(self.model_facing())

    def __hash__(self) -> int:
        routing = {**self._decision_fields(), "sees": self.sees}
        return hash((type(self).__name__, self.id, self.fingerprint, canonical_json(routing)))

    def _decision_fields(self) -> dict[str, Any]:
        return {}


class NoulPredicate(_PredicateBase, frozen=True):
    """A yes/no judgment. The result is the probability that the answer is yes."""

    kind: Literal["noul"] = "noul"
    criteria: NoulCriteria | None = None
    threshold: UnitInterval | None = None
    """Default decision threshold: ``decision`` is ``probability >= threshold``."""

    def model_facing(self) -> dict[str, Any]:
        return {
            "kind": "noul",
            "question": self.question,
            "criteria": None if self.criteria is None else self.criteria.wire(),
        }

    def _decision_fields(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


class ChoicePredicate(_PredicateBase, frozen=True):
    """A selection among named options. The result is a distribution over the options."""

    kind: Literal["choice"] = "choice"
    options: Options
    """Option key to description (or ``None`` when the key speaks for itself). Order is significant."""
    min_confidence: UnitInterval | None = None
    """Default abstention threshold: ``decision`` is the choice when ``confidence >= min_confidence``."""

    def model_facing(self) -> dict[str, Any]:
        return {
            "kind": "choice",
            "question": self.question,
            "options": [[key, value] for key, value in self.options.items()],
        }

    def _decision_fields(self) -> dict[str, Any]:
        return {"min_confidence": self.min_confidence}


class ScorePredicate(_PredicateBase, frozen=True):
    """A position on ordered, described levels. The result is a distribution over the levels.

    Scores carry no default decision: measured repeat noise makes them suitable for ranking and
    informing, not for gating on their own.
    """

    kind: Literal["score"] = "score"
    levels: Levels
    """Level descriptions from lowest (level 0) to highest."""

    def model_facing(self) -> dict[str, Any]:
        return {"kind": "score", "question": self.question, "levels": list(self.levels)}


Predicate = NoulPredicate | ChoicePredicate | ScorePredicate
"""Any predicate. Usable in type hints and ``isinstance`` checks."""

_PredicateField = Annotated[Predicate, Field(discriminator="kind")]


def noul(
    id: str,
    *,
    question: Text,
    criteria: NoulCriteria | Mapping[str, Text | None] | None = None,
    threshold: float | None = None,
    description: str | None = None,
    tags: Iterable[str] = (),
    version: str | None = None,
    sees: Sequence[str] | None = None,
) -> NoulPredicate:
    """Build a yes/no predicate."""
    return NoulPredicate.model_validate(
        {
            "id": id,
            "question": question,
            "criteria": criteria,
            "threshold": threshold,
            "description": description,
            "tags": tuple(tags),
            "version": version,
            "sees": None if sees is None else tuple(sees),
        }
    )


def choice(
    id: str,
    *,
    question: Text,
    options: Mapping[str, Text | None],
    min_confidence: float | None = None,
    description: str | None = None,
    tags: Iterable[str] = (),
    version: str | None = None,
    sees: Sequence[str] | None = None,
) -> ChoicePredicate:
    """Build a choice predicate over named options (order is significant)."""
    return ChoicePredicate.model_validate(
        {
            "id": id,
            "question": question,
            "options": dict(options),
            "min_confidence": min_confidence,
            "description": description,
            "tags": tuple(tags),
            "version": version,
            "sees": None if sees is None else tuple(sees),
        }
    )


def score(
    id: str,
    *,
    question: Text,
    levels: Sequence[Text],
    description: str | None = None,
    tags: Iterable[str] = (),
    version: str | None = None,
    sees: Sequence[str] | None = None,
) -> ScorePredicate:
    """Build a score predicate over ordered levels, lowest first."""
    return ScorePredicate.model_validate(
        {
            "id": id,
            "question": question,
            "levels": tuple(levels),
            "description": description,
            "tags": tuple(tags),
            "version": version,
            "sees": None if sees is None else tuple(sees),
        }
    )


class PredicateSet(BaseModel, frozen=True, extra="forbid"):
    """An ordered, named collection of predicates evaluated together against one state."""

    name: PredicateId
    description: str | None = None
    predicates: tuple[_PredicateField, ...] = Field(min_length=1)

    @field_validator("predicates")
    @classmethod
    def _unique_ids(cls, value: tuple[Predicate, ...]) -> tuple[Predicate, ...]:
        seen: set[str] = set()
        for predicate in value:
            if predicate.id in seen:
                raise ValueError(f"duplicate predicate id {predicate.id!r}")
            seen.add(predicate.id)
        return value

    @property
    def fingerprint(self) -> str:
        """``sha256`` over the sorted ``(id, predicate fingerprint)`` pairs."""
        return digest_of(sorted([predicate.id, predicate.fingerprint] for predicate in self.predicates))

    def predicate_fingerprints(self) -> dict[str, str]:
        """Each predicate's fingerprint, keyed by id, in set order."""
        return {predicate.id: predicate.fingerprint for predicate in self.predicates}

    def ids(self) -> tuple[str, ...]:
        """Predicate ids in set order."""
        return tuple(predicate.id for predicate in self.predicates)

    def __getitem__(self, predicate_id: str) -> Predicate:
        for predicate in self.predicates:
            if predicate.id == predicate_id:
                return predicate
        raise KeyError(
            f"predicate set {self.name!r} has no predicate {predicate_id!r}; ids: {', '.join(self.ids())}"
        )

    def __contains__(self, predicate_id: object) -> bool:
        return any(predicate.id == predicate_id for predicate in self.predicates)

    def __len__(self) -> int:
        return len(self.predicates)

    def select(self, ids: Iterable[str]) -> "PredicateSet":
        """A set with only ``ids``, in this set's order. Unknown ids raise ``KeyError``."""
        wanted = list(dict.fromkeys(ids))
        for predicate_id in wanted:
            self[predicate_id]
        chosen = set(wanted)
        return PredicateSet(
            name=self.name,
            description=self.description,
            predicates=tuple(predicate for predicate in self.predicates if predicate.id in chosen),
        )

    @classmethod
    def of(cls, predicates: "PredicateSet | Iterable[Predicate]", *, name: str = "adhoc") -> "PredicateSet":
        """Return ``predicates`` as a set, wrapping a bare sequence under ``name``."""
        if isinstance(predicates, PredicateSet):
            return predicates
        return cls(name=name, predicates=tuple(predicates))

    @classmethod
    def from_yaml(cls, text: str, *, source: str = "<string>") -> "PredicateSet":
        """Parse a ``sentinel/predicates/v1`` document (YAML or JSON)."""
        try:
            data = load_yaml(text)
        except YAMLError as error:
            raise PredicateValidationError(f"invalid YAML: {error}", path=source) from None
        return PredicateFileV1.parse_data(data, source=source).to_set(source=source)

    @classmethod
    def from_file(cls, path: str | Path) -> "PredicateSet":
        """Load a ``sentinel/predicates/v1`` file (YAML or JSON)."""
        file = Path(path)
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as error:
            raise PredicateValidationError(
                f"cannot read predicate file: {error.strerror}", path=str(file)
            ) from None
        return cls.from_yaml(text, source=str(file))

    def to_yaml(self) -> str:
        """Serialize to a ``sentinel/predicates/v1`` YAML document."""
        return dump_yaml(PredicateFileV1.from_set(self).yaml_data())

    def to_file(self, path: str | Path) -> None:
        """Write this set as a ``sentinel/predicates/v1`` YAML file."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


class _BodyBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    description: str | None = None
    tags: Tags = ()
    version: str | None = None
    sees: list[StatePath] | None = Field(default=None, min_length=1)
    """The state paths this predicate is shown; overrides the file's `sees`."""


class NoulBody(_BodyBase):
    """A ``noul`` predicate as written in a predicate file."""

    kind: Literal["noul"]
    criteria: NoulCriteria | None = None
    threshold: UnitInterval | None = None


class ChoiceBody(_BodyBase):
    """A ``choice`` predicate as written in a predicate file."""

    kind: Literal["choice"]
    options: Options
    min_confidence: UnitInterval | None = None


class ScoreBody(_BodyBase):
    """A ``score`` predicate as written in a predicate file."""

    kind: Literal["score"]
    levels: Levels


_Body = Annotated[NoulBody | ChoiceBody | ScoreBody, Field(discriminator="kind")]


class PredicateFileV1(BaseModel):
    """The on-disk predicate file format, ``sentinel/predicates/v1``. Predicates are keyed by id."""

    model_config = ConfigDict(extra="forbid", title="Sentinel predicate file (v1)")

    format: Literal["sentinel/predicates/v1"]
    name: PredicateId
    description: str | None = None
    sees: list[StatePath] | None = Field(default=None, min_length=1)
    """The state paths every predicate in the file is shown, unless it declares its own."""
    predicates: dict[PredicateId, _Body] = Field(min_length=1)

    @classmethod
    def parse_data(cls, data: Any, *, source: str = "<string>") -> "PredicateFileV1":
        """Validate parsed YAML/JSON data, raising ``PredicateValidationError`` with readable paths."""
        if not isinstance(data, Mapping):
            raise PredicateValidationError("a predicate file must be a mapping", path=source)
        if data.get("format") != PREDICATES_FORMAT:  # pyright: ignore[reportUnknownMemberType]
            raise PredicateValidationError(f"expected `format: {PREDICATES_FORMAT}`", path=source)
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            raise _validation_error(error, source) from None

    def to_set(self, *, source: str = "<string>") -> PredicateSet:
        """Build the domain ``PredicateSet``."""
        predicates: list[Predicate] = []
        for predicate_id, body in self.predicates.items():
            fields = body.model_dump(exclude_none=True)
            if body.sees is None and self.sees is not None:
                fields["sees"] = list(self.sees)
            try:
                if isinstance(body, NoulBody):
                    predicates.append(NoulPredicate.model_validate({**fields, "id": predicate_id}))
                elif isinstance(body, ChoiceBody):
                    predicates.append(ChoicePredicate.model_validate({**fields, "id": predicate_id}))
                else:
                    predicates.append(ScorePredicate.model_validate({**fields, "id": predicate_id}))
            except ValidationError as error:
                raise _validation_error(error, source, prefix=("predicates", predicate_id)) from None
        return PredicateSet(name=self.name, description=self.description, predicates=tuple(predicates))

    @classmethod
    def from_set(cls, predicate_set: PredicateSet) -> "PredicateFileV1":
        """Build the file model from a domain ``PredicateSet``."""
        bodies: dict[str, Any] = {}
        views = {predicate.sees for predicate in predicate_set.predicates}
        shared = next(iter(views)) if len(views) == 1 else None
        for predicate in predicate_set.predicates:
            data = predicate.model_dump(exclude={"id"})
            if shared is not None:
                data.pop("sees", None)
            bodies[predicate.id] = data
        return cls.model_validate(
            {
                "format": PREDICATES_FORMAT,
                "name": predicate_set.name,
                "description": predicate_set.description,
                "sees": None if shared is None else list(shared),
                "predicates": bodies,
            }
        )

    def yaml_data(self) -> dict[str, Any]:
        """Plain data for YAML output, in a stable, readable field order."""
        out: dict[str, Any] = {"format": self.format, "name": self.name}
        if self.description is not None:
            out["description"] = self.description
        if self.sees is not None:
            out["sees"] = list(self.sees)
        predicates: dict[str, Any] = {}
        for predicate_id, body in self.predicates.items():
            entry: dict[str, Any] = {"kind": body.kind, "question": body.question}
            if isinstance(body, NoulBody):
                if body.criteria is not None:
                    # Boolean keys make YAML emit the natural `true:` / `false:`.
                    entry["criteria"] = {key == "true": value for key, value in body.criteria.wire().items()}
                if body.threshold is not None:
                    entry["threshold"] = body.threshold
            elif isinstance(body, ChoiceBody):
                entry["options"] = dict(body.options)
                if body.min_confidence is not None:
                    entry["min_confidence"] = body.min_confidence
            else:
                entry["levels"] = list(body.levels)
            if body.description is not None:
                entry["description"] = body.description
            if body.tags:
                entry["tags"] = list(body.tags)
            if body.version is not None:
                entry["version"] = body.version
            if body.sees is not None and body.sees != self.sees:
                entry["sees"] = list(body.sees)
            predicates[predicate_id] = entry
        out["predicates"] = predicates
        return out


class PredicateSummary(BaseModel):
    """One predicate as `sentinel predicates show --json` lists it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Literal["noul", "choice", "score"]
    fingerprint: str
    default_threshold: float | None
    """A noul's threshold or a choice's minimum confidence; scores never have one."""
    tags: list[str]
    sees: list[str] | None = None
    """The state paths the predicate is shown; absent when it sees the whole state."""


class PredicateListing(BaseModel):
    """A predicate set's ids, kinds, fingerprints and default thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid", title="Sentinel predicate listing (v1)")

    name: str
    fingerprint: str
    predicates: list[PredicateSummary]

    @classmethod
    def of(cls, predicate_set: PredicateSet) -> "PredicateListing":
        """Summarize a predicate set."""
        summaries: list[PredicateSummary] = []
        for predicate in predicate_set.predicates:
            default: float | None = None
            if isinstance(predicate, NoulPredicate):
                default = predicate.threshold
            elif isinstance(predicate, ChoicePredicate):
                default = predicate.min_confidence
            summaries.append(
                PredicateSummary(
                    id=predicate.id,
                    kind=predicate.kind,
                    fingerprint=predicate.fingerprint,
                    default_threshold=default,
                    tags=list(predicate.tags),
                    sees=None if predicate.sees is None else list(predicate.sees),
                )
            )
        return cls(name=predicate_set.name, fingerprint=predicate_set.fingerprint, predicates=summaries)


_KIND_TAGS = {"noul", "choice", "score"}


def _format_loc(loc: tuple[int | str, ...]) -> str:
    parts: list[str] = []
    for index, part in enumerate(loc):
        # Drop the discriminated-union tag pydantic inserts after a predicate id.
        if index == 2 and loc[0] == "predicates" and part in _KIND_TAGS:
            continue
        parts.append(str(part))
    return ".".join(parts)


def _validation_error(
    error: ValidationError, source: str, *, prefix: tuple[str, ...] = ()
) -> PredicateValidationError:
    problems: list[str] = []
    for item in error.errors(include_url=False, include_input=False):
        location = _format_loc((*prefix, *item["loc"]))
        message = str(item["msg"]).removeprefix("Value error, ")
        problems.append(f"{location}: {message}" if location else message)
    return PredicateValidationError("; ".join(problems), path=source)
