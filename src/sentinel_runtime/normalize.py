"""Raw provider answers to typed results.

Every answer is checked against the predicate that asked for it: the right kind, exactly the
predicate's options or levels, probabilities in ``[0, 1]`` that sum to 1 within the provider's
rounding, and a choice that is the most probable option. Anything else is a
``malformed_response`` provider error; a malformed answer is never passed off as a judgment.
"""

import math
from collections.abc import Mapping
from typing import Any

from sentinel_runtime.errors import ProviderError
from sentinel_runtime.predicates import ChoicePredicate, NoulPredicate, Predicate, ScorePredicate
from sentinel_runtime.providers.base import RawAnswer
from sentinel_runtime.results import ChoiceResult, NoulResult, Result, ScoreResult

SUM_TOLERANCE = 0.02
"""Allowed deviation of a distribution's sum from 1. TypeSafe rounds probabilities to two decimals."""

_RANGE_EPSILON = 1e-9
_DECIMALS = 6


class _Malformed(Exception):
    pass


def _probability(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Malformed(f"{what} is not a number")
    number = float(value)
    if not math.isfinite(number) or number < -_RANGE_EPSILON or number > 1 + _RANGE_EPSILON:
        raise _Malformed(f"{what} is {value}, outside [0, 1]")
    return round(min(max(number, 0.0), 1.0), _DECIMALS)


def _distribution(raw: Any, expected: list[str], what: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise _Malformed(f"{what} is missing")
    received = {str(key): value for key, value in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    missing = [key for key in expected if key not in received]
    extra = [key for key in received if key not in set(expected)]
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing {', '.join(repr(key) for key in missing[:5])}")
        if extra:
            parts.append(f"unexpected {', '.join(repr(key) for key in extra[:5])}")
        raise _Malformed(f"{what} do not match the predicate ({'; '.join(parts)})")
    distribution = {key: _probability(received[key], f"{what}[{key!r}]") for key in expected}
    total = sum(distribution.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise _Malformed(f"{what} sum to {total:.4f}, not 1")
    return distribution


def _normalize_noul(predicate: NoulPredicate, raw: RawAnswer) -> NoulResult:
    probability = _probability(raw.get("noul"), "noul")
    decision = None if predicate.threshold is None else probability >= predicate.threshold
    return NoulResult(probability=probability, threshold=predicate.threshold, decision=decision)


def _normalize_choice(predicate: ChoicePredicate, raw: RawAnswer) -> ChoiceResult:
    options = list(predicate.options)
    probabilities = _distribution(raw.get("probabilities"), options, "probabilities")
    selected = raw.get("choice")
    if not isinstance(selected, str) or selected not in probabilities:
        raise _Malformed(f"choice {selected!r} is not one of the predicate's options")
    if probabilities[selected] < max(probabilities.values()) - 1e-6:
        raise _Malformed(
            f"choice {selected!r} ({probabilities[selected]:.2f}) is not the most probable option "
            f"({max(probabilities.values()):.2f})"
        )
    confidence = _probability(raw.get("confidence"), "confidence")
    decision = None
    if predicate.min_confidence is not None and confidence >= predicate.min_confidence:
        decision = selected
    return ChoiceResult(
        choice=selected,
        probabilities=probabilities,
        confidence=confidence,
        min_confidence=predicate.min_confidence,
        decision=decision,
    )


def _normalize_score(predicate: ScorePredicate, raw: RawAnswer) -> ScoreResult:
    levels = len(predicate.levels)
    distribution = _distribution(
        raw.get("probabilities"), [str(index) for index in range(levels)], "probabilities"
    )
    value = raw.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise _Malformed("score is not a number")
    top = levels - 1
    if float(value) < -_RANGE_EPSILON or float(value) > top + _RANGE_EPSILON:
        raise _Malformed(f"score {value} is outside [0, {top}]")
    score_value = round(min(max(float(value), 0.0), float(top)), _DECIMALS)
    confidence = _probability(raw.get("confidence"), "confidence")
    return ScoreResult(
        score=score_value,
        probabilities={int(key): probability for key, probability in distribution.items()},
        confidence=confidence,
        levels=levels,
        normalized=round(score_value / top, _DECIMALS),
        nearest_level=min(int(score_value + 0.5), top),
    )


def normalize_answer(predicate: Predicate, raw: RawAnswer, *, provider: str) -> Result:
    """Validate one raw answer against its predicate and build the typed result."""
    try:
        if not isinstance(raw, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise _Malformed("answer is not an object")
        if raw.get("type") != predicate.kind:
            raise _Malformed(
                f"answer type {raw.get('type')!r} does not match predicate kind {predicate.kind!r}"
            )
        if isinstance(predicate, NoulPredicate):
            return _normalize_noul(predicate, raw)
        if isinstance(predicate, ChoicePredicate):
            return _normalize_choice(predicate, raw)
        return _normalize_score(predicate, raw)
    except _Malformed as problem:
        message = f"answer for {predicate.id!r}: {problem}"
    raise ProviderError("malformed_response", message, provider=provider)


def normalize_answers(
    predicates: tuple[Predicate, ...], answers: Mapping[str, RawAnswer], *, provider: str
) -> dict[str, Result]:
    """Normalize every predicate's answer. A missing answer is a ``malformed_response`` error."""
    missing = [predicate.id for predicate in predicates if predicate.id not in answers]
    if missing:
        raise ProviderError(
            "malformed_response", f"no answer for {', '.join(repr(i) for i in missing)}", provider=provider
        )
    return {
        predicate.id: normalize_answer(predicate, answers[predicate.id], provider=provider)
        for predicate in predicates
    }
