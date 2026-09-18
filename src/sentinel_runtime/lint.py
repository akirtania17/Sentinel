"""Authoring lints for predicate sets.

These warnings encode the known Jev 1.13 limitations: literal reading, weak arithmetic and date
comparison, sensitivity to negation, and levels or options that give the model nothing concrete to
match. They are advice, not errors; a predicate that triggers one can still be correct, and the
evaluation harness is the final word. Structured questions, options and levels are checked string by
string, whatever their shape.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from sentinel_runtime.predicates import (
    ChoicePredicate,
    NoulPredicate,
    Predicate,
    PredicateSet,
    ScorePredicate,
)
from sentinel_runtime.views import covers, segments

LintCode = Literal[
    "no-state-reference",
    "unresolved-state-path",
    "unseen-state-path",
    "arithmetic",
    "date-comparison",
    "negation",
    "vague-levels",
    "duplicate-levels",
    "no-escape-option",
    "long-question",
]

LONG_QUESTION_CHARS = 600
MIN_LEVEL_CHARS = 12

_BACKTICKED = re.compile(r"`([^`]+)`")
_PATH = re.compile(r"^[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*|\[\d+\])*$")
_SEGMENT = re.compile(r"\.?([A-Za-z_][\w-]*)|\[(\d+)\]")
_ARITHMETIC = re.compile(
    r"\b(how many|how much (?:money|time)|counting|count the|number of|in total|total of|sum of|average|"
    r"percentage|percent of|more than \d|fewer than \d|less than \d|at least \d|at most \d)\b",
    re.IGNORECASE,
)
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATES = re.compile(
    r"\b(?:(?:before|after|earlier than|later than|prior to|by)\s+(?:the\s+)?"
    rf"(?:date|deadline|due date|expir\w*|\d{{1,4}}\b|today|tomorrow|yesterday|{_MONTHS}\b)"
    r"|within \d+ (?:hour|day|week|month|year)s?|\d+ (?:hour|day|week|month|year)s? (?:ago|since|until|old)"
    r"|older than|newer than)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(not|never|no longer|fails? to|isn't|doesn't|don't|didn't|cannot|can't|won't|wasn't|aren't)\b",
    re.IGNORECASE,
)
_ESCAPE = re.compile(r"\b(other|none|unknown|unclear|n/?a|neither|no match|not applicable)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LintWarning:
    """One authoring warning."""

    code: LintCode
    predicate_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.predicate_id}: [{self.code}] {self.message}"


def _strings(value: Any) -> Iterator[str]:
    """Every string leaf in a question, criterion, option or level."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in cast("Mapping[str, Any]", value).values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in cast("list[Any]", value):
            yield from _strings(item)


def _question_strings(predicate: Predicate) -> list[str]:
    return list(_strings(predicate.question))


def _asked_text(predicate: Predicate) -> str:
    """The sentence actually asked: a structured question's `question` field, else every string.

    Examples, definitions and hints in a structured question may quote negations ("do not send yet")
    without inverting what yes means.
    """
    question = predicate.question
    if isinstance(question, Mapping) and "question" in question:
        return " ".join(_strings(cast("Mapping[str, Any]", question)["question"]))
    return " ".join(_strings(question))


def _resolves(path: str, state: Any) -> bool:
    current = state
    for name, index in _SEGMENT.findall(path):
        if name:
            if not isinstance(current, Mapping) or name not in current:
                return False
            current = cast("Mapping[str, Any]", current)[name]
        else:
            if not isinstance(current, list) or int(index) >= len(cast("list[Any]", current)):
                return False
            current = cast("list[Any]", current)[int(index)]
    return True


def _view_resolves(path: str, state: Any) -> bool:
    """Whether a `sees` path selects anything in ``state``; a list matches when any item does."""

    def walk(value: Any, keys: list[str]) -> bool:
        if not keys:
            return True
        if isinstance(value, list):
            return any(walk(item, keys) for item in cast("list[Any]", value))
        if not isinstance(value, Mapping) or keys[0] not in value:
            return False
        return walk(cast("Mapping[str, Any]", value)[keys[0]], keys[1:])

    return walk(state, segments(path))


def lint_predicate(predicate: Predicate, *, state: Any = None) -> list[LintWarning]:
    """Authoring warnings for one predicate, optionally checked against an example ``state``."""
    warnings: list[LintWarning] = []

    def warn(code: LintCode, message: str) -> None:
        warnings.append(LintWarning(code, predicate.id, message))

    question_text = " ".join(_question_strings(predicate))
    references = [match for text in _question_strings(predicate) for match in _BACKTICKED.findall(text)]
    paths = [reference for reference in references if _PATH.match(reference)]

    if isinstance(state, Mapping):
        if not paths:
            warn(
                "no-state-reference",
                "the question names no state field; point it at the part of the state it judges, "
                "e.g. `user_request`",
            )
        for path in paths:
            if not _resolves(path, state):
                warn("unresolved-state-path", f"`{path}` does not exist in the example state")
        for path in predicate.sees or ():
            if not _view_resolves(path, state):
                warn("unresolved-state-path", f"`sees` path `{path}` selects nothing in the example state")
    if predicate.sees is not None:
        for path in paths:
            if not covers(predicate.sees, path):
                warn(
                    "unseen-state-path",
                    f"the question names `{path}`, which is outside what this predicate sees (`sees`)",
                )

    if _ARITHMETIC.search(question_text):
        warn(
            "arithmetic",
            "the question asks for counting or arithmetic; compute numbers in code and ask about meaning",
        )
    if _DATES.search(question_text):
        warn(
            "date-comparison",
            "the question compares dates or durations; extract the dates and compare them in code",
        )
    if isinstance(predicate, NoulPredicate) and _NEGATION.search(_asked_text(predicate)):
        warn(
            "negation", "the question is phrased negatively; phrase it so that yes means the condition holds"
        )

    if isinstance(predicate, ScorePredicate):
        texts = [" ".join(_strings(level)) for level in predicate.levels]
        vague = [
            index
            for index, text in enumerate(texts)
            if text.strip().isdigit() or len(text.strip()) < MIN_LEVEL_CHARS
        ]
        if vague:
            listed = ", ".join(str(index) for index in vague)
            warn(
                "vague-levels",
                f"level(s) {listed} are numbers or very short; describe a concrete situation for each",
            )
        if len({text.strip().lower() for text in texts}) < len(texts):
            warn("duplicate-levels", "two or more levels have the same description")

    if isinstance(predicate, ChoicePredicate) and len(predicate.options) >= 4:
        candidates = list(predicate.options) + [
            " ".join(_strings(v)) for v in predicate.options.values() if v is not None
        ]
        if not any(_ESCAPE.search(text.replace("_", " ")) for text in candidates):
            warn(
                "no-escape-option",
                "no `other` or `none` option; a choice always picks something, even when nothing fits",
            )

    longest = max((len(text) for text in _question_strings(predicate)), default=0)
    if longest > LONG_QUESTION_CHARS:
        warn(
            "long-question",
            f"the question is {longest} characters; move supporting data into the state "
            "and keep the judgment narrow",
        )
    return warnings


def lint_predicate_set(predicate_set: PredicateSet, *, state: Any = None) -> list[LintWarning]:
    """Authoring warnings for every predicate in ``predicate_set``."""
    return [
        warning
        for predicate in predicate_set.predicates
        for warning in lint_predicate(predicate, state=state)
    ]


__all__ = ["LintCode", "LintWarning", "lint_predicate", "lint_predicate_set"]
