"""The provider boundary.

A provider answers typed questions about one state. Sentinel's narrow waist is the shape of a
System One request and response: a state, a set of predicates, and one raw answer per predicate in
the TypeSafe answer format (``{"type": "noul", "noul": 0.93}``, ``{"type": "choice", ...}``,
``{"type": "score", ...}``). Any backend that can produce those answers can sit behind this
protocol; the runtime normalizes and validates them into typed results.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sentinel_runtime.predicates import Predicate
from sentinel_runtime.state import State

RawAnswer = Mapping[str, Any]
"""One provider answer in the TypeSafe answer format, untouched."""


@dataclass(frozen=True)
class ProviderLimits:
    """Request-size limits the planner packs predicates against."""

    max_request_tokens: int = 64_000
    """Tokens allowed for the state plus all questions in one request."""
    max_state_plus_question_tokens: int = 32_000
    """Tokens allowed for the state plus the single longest question."""
    max_options: int = 255
    max_levels: int = 10
    fixed_overhead_tokens: int = 300
    """Tokens a request costs before any state or question (measured at about 280 for Jev 1.13)."""


@dataclass(frozen=True)
class ProviderRequest:
    """One request: a state and the predicates to judge it against."""

    state: State
    predicates: tuple[Predicate, ...]
    model: str
    timeout: float | None = None
    """Per-attempt HTTP timeout in seconds; ``None`` keeps the provider default."""
    estimated_input_tokens: int = 0
    """The planner's estimate for this request, recorded next to the actual count."""


@dataclass(frozen=True)
class ProviderResponse:
    """A provider's raw answers plus the metadata the runtime records."""

    answers: Mapping[str, RawAnswer]
    """Raw answers keyed by predicate id."""
    model_resolved: str
    """The versioned model that answered, as reported by the provider."""
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    replayed: bool = False
    """True when this response was served from a cache rather than a live request."""
    extra: Mapping[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class Provider(Protocol):
    """A semantic judgment backend."""

    @property
    def name(self) -> str:
        """Short identifier recorded in events and cache keys, such as ``typesafe``."""
        ...

    @property
    def limits(self) -> ProviderLimits:
        """Request-size limits for the planner."""
        ...

    def evaluate(self, request: ProviderRequest) -> ProviderResponse:
        """Answer every predicate in ``request`` in one provider call."""
        ...

    async def aevaluate(self, request: ProviderRequest) -> ProviderResponse:
        """Async form of :meth:`evaluate`."""
        ...

    def close(self) -> None:
        """Release synchronous resources."""
        ...

    async def aclose(self) -> None:
        """Release all resources, including asynchronous ones."""
        ...
