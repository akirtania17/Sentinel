"""Typed evaluation results.

The provider's raw answers become one typed result per predicate. The model's output is
probabilistic; a ``decision`` exists only when the predicate declares a default threshold, and it
is a convenience for application policy, never a replacement for it.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class NoulResult(BaseModel):
    """The probability that a yes/no predicate is true."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["noul"] = "noul"
    probability: float
    threshold: float | None = None
    decision: bool | None = None
    """``probability >= threshold`` when the predicate declares a threshold, else ``None``."""


class ChoiceResult(BaseModel):
    """A distribution over a choice predicate's options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["choice"] = "choice"
    choice: str
    """The option with the highest probability."""
    probabilities: dict[str, float]
    """Probability of every option, in the predicate's option order."""
    confidence: float
    """How concentrated the distribution is, from 0 (flat) to 1 (all mass on one option)."""
    min_confidence: float | None = None
    decision: str | None = None
    """``choice`` when ``confidence >= min_confidence``; ``None`` when abstaining or no threshold is set."""

    @property
    def top_probability(self) -> float:
        """The probability of the selected option."""
        return self.probabilities[self.choice]


class ScoreResult(BaseModel):
    """A distribution over a score predicate's ordered levels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["score"] = "score"
    score: float
    """Probability-weighted mean level, from 0 to ``levels - 1``. May fall between levels."""
    probabilities: dict[int, float]
    """Probability of each level, keyed by level index."""
    confidence: float
    levels: int
    """Number of levels in the predicate."""
    normalized: float
    """``score / (levels - 1)``, on 0 to 1."""
    nearest_level: int
    """``score`` rounded half up to the nearest level."""


Result = NoulResult | ChoiceResult | ScoreResult
"""Any typed result. Usable in type hints and ``isinstance`` checks."""

_ResultField = Annotated[Result, Field(discriminator="kind")]


class ProviderCall(BaseModel):
    """One request to a provider, as observed by the runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str | None
    model_requested: str
    model_resolved: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_input_tokens: int
    latency_ms: float
    predicate_ids: tuple[str, ...]
    attempts: int = 1
    replayed: bool = False
    """True when the response came from a cache rather than a live request."""
    state_digest: str | None = None
    """Digest of the state this request sent: the evaluation's state, or a predicate view of it."""


class Usage(BaseModel):
    """Aggregate usage for one evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: int
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    """Cost from the configured pricing table; ``None`` when the model has no price entry."""
    replayed: bool = False
    """True when any call in this evaluation was served from a cache."""


class Evaluation(BaseModel):
    """The typed outcome of evaluating a predicate set against one state."""

    model_config = ConfigDict(frozen=True, extra="forbid", title="Sentinel evaluation (v1)")

    evaluation_id: str
    created_at: datetime
    predicate_set: str
    predicate_set_fingerprint: str
    predicate_fingerprints: dict[str, str]
    state_digest: str
    state_chars: int
    provider: str
    model_requested: str
    model_resolved: str
    """The versioned model that answered, as reported by the provider (aliases resolve here)."""
    results: dict[str, _ResultField]
    calls: tuple[ProviderCall, ...]
    usage: Usage
    latency_ms: float

    def __getitem__(self, predicate_id: str) -> Result:
        try:
            return self.results[predicate_id]
        except KeyError:
            available = ", ".join(self.results)
            raise KeyError(f"no result for {predicate_id!r}; results: {available}") from None

    def __contains__(self, predicate_id: object) -> bool:
        return predicate_id in self.results

    def probability(self, predicate_id: str) -> float:
        """The probability of a ``noul`` result. Raises ``TypeError`` for other kinds."""
        result = self[predicate_id]
        if not isinstance(result, NoulResult):
            raise TypeError(
                f"{predicate_id!r} is a {result.kind} result; probability() applies to noul results"
            )
        return result.probability
