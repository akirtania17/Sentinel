"""An in-memory provider for tests, plus helpers that build raw answers.

``FakeProvider`` answers from a fixed mapping, from a ``respond`` callback, or with neutral
defaults, and records every request it receives. Use it to test application policy without network
access::

    provider = FakeProvider({"external_side_effect": noul_answer(0.95)})
    runtime = Runtime(provider)
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sentinel_runtime.errors import ProviderError
from sentinel_runtime.predicates import ChoicePredicate, NoulPredicate, Predicate
from sentinel_runtime.providers.base import ProviderLimits, ProviderRequest, ProviderResponse, RawAnswer

Responder = Callable[[ProviderRequest], Mapping[str, RawAnswer] | ProviderError]


def _entropy_confidence(probabilities: Sequence[float]) -> float:
    """1 minus the normalized Shannon entropy: 1 for a one-hot distribution, 0 for a uniform one."""
    if len(probabilities) <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return round(max(0.0, 1.0 - entropy / math.log(len(probabilities))), 6)


def noul_answer(probability: float) -> dict[str, Any]:
    """A raw ``noul`` answer."""
    return {"type": "noul", "noul": probability}


def choice_answer(
    probabilities: Mapping[str, float], *, choice: str | None = None, confidence: float | None = None
) -> dict[str, Any]:
    """A raw ``choice`` answer. ``choice`` defaults to the most probable option."""
    selected = choice if choice is not None else max(probabilities, key=lambda key: probabilities[key])
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": dict(probabilities),
        "confidence": confidence
        if confidence is not None
        else _entropy_confidence(list(probabilities.values())),
    }


def score_answer(probabilities: Sequence[float], *, confidence: float | None = None) -> dict[str, Any]:
    """A raw ``score`` answer from per-level probabilities, lowest level first."""
    return {
        "type": "score",
        "score": sum(index * probability for index, probability in enumerate(probabilities)),
        "probabilities": {str(index): probability for index, probability in enumerate(probabilities)},
        "confidence": confidence if confidence is not None else _entropy_confidence(list(probabilities)),
    }


def default_answer(predicate: Predicate) -> dict[str, Any]:
    """A neutral answer: 0.5 for a noul, the first option for a choice, the lowest level for a score."""
    if isinstance(predicate, NoulPredicate):
        return noul_answer(0.5)
    if isinstance(predicate, ChoicePredicate):
        options = list(predicate.options)
        return choice_answer({key: (1.0 if index == 0 else 0.0) for index, key in enumerate(options)})
    return score_answer([1.0] + [0.0] * (len(predicate.levels) - 1))


class FakeProvider:
    """A scripted provider that records its requests."""

    def __init__(
        self,
        answers: Mapping[str, RawAnswer] | None = None,
        *,
        respond: Responder | None = None,
        model_resolved: str = "fake-model-1",
        name: str = "fake",
        limits: ProviderLimits | None = None,
        latency_ms: float = 1.0,
    ) -> None:
        self.answers: dict[str, RawAnswer] = dict(answers or {})
        self.respond = respond
        self.model_resolved = model_resolved
        self.latency_ms = latency_ms
        self.requests: list[ProviderRequest] = []
        self.closed = False
        self._name = name
        self._limits = limits if limits is not None else ProviderLimits()

    @property
    def name(self) -> str:
        return self._name

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    def _answer(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if self.respond is not None:
            outcome = self.respond(request)
            if isinstance(outcome, ProviderError):
                raise outcome
            answers = dict(outcome)
        else:
            answers = {
                predicate.id: self.answers.get(predicate.id, default_answer(predicate))
                for predicate in request.predicates
            }
        return ProviderResponse(
            answers=answers,
            model_resolved=self.model_resolved,
            request_id=f"fake_{len(self.requests)}",
            input_tokens=request.estimated_input_tokens or request.state.estimated_tokens,
            output_tokens=10 * len(request.predicates),
            latency_ms=self.latency_ms,
        )

    def evaluate(self, request: ProviderRequest) -> ProviderResponse:
        return self._answer(request)

    async def aevaluate(self, request: ProviderRequest) -> ProviderResponse:
        return self._answer(request)

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True
