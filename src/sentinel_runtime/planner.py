"""Request planning: as few provider requests as the token budget allows.

Every predicate in an evaluation judges the same state, so the default plan is a single request.
The plan splits only when the estimated request would exceed the provider's budget, packing
predicates greedily in set order. Before anything is sent, the planner rejects a state that cannot
fit even with the longest single question.

Estimates are deliberately conservative and use 85% of each provider limit. When a provider still
reports that a request is too large, the runtime bisects that chunk and retries the halves.

Dependencies between judgments are not a planner concern: a judgment that needs another's answer
is a second evaluation, with new state, made by application code.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sentinel_runtime.errors import ProviderError
from sentinel_runtime.hashing import canonical_json
from sentinel_runtime.predicates import Predicate
from sentinel_runtime.providers.base import ProviderLimits
from sentinel_runtime.state import State
from sentinel_runtime.tokens import PER_QUESTION_TOKENS, estimate_tokens_for_chars

BUDGET_FRACTION = 0.85
"""Share of each provider limit the planner packs against, leaving room for estimation error."""


def estimate_question_tokens(predicate: Predicate) -> int:
    """Conservative token estimate for one predicate's question, criteria, options or levels."""
    return estimate_tokens_for_chars(len(canonical_json(predicate.model_facing()))) + PER_QUESTION_TOKENS


@dataclass(frozen=True)
class Chunk:
    """The predicates sent in one provider request."""

    predicates: tuple[Predicate, ...]
    estimated_input_tokens: int


@dataclass(frozen=True)
class Plan:
    """The requests needed to evaluate a predicate set against one state."""

    chunks: tuple[Chunk, ...]

    @property
    def estimated_input_tokens(self) -> int:
        return sum(chunk.estimated_input_tokens for chunk in self.chunks)


def make_chunk(state: State, predicates: Sequence[Predicate], limits: ProviderLimits) -> Chunk:
    """A chunk with its request-size estimate."""
    questions = sum(estimate_question_tokens(predicate) for predicate in predicates)
    return Chunk(tuple(predicates), limits.fixed_overhead_tokens + state.estimated_tokens + questions)


def plan_requests(
    state: State, predicates: Sequence[Predicate], limits: ProviderLimits, *, provider: str
) -> Plan:
    """Plan the requests for ``predicates`` against ``state``.

    Raises ``ProviderError(kind="context_too_large")`` without contacting the provider when the state
    plus the longest question cannot fit the provider's per-question budget.
    """
    if not predicates:
        raise ValueError("plan_requests needs at least one predicate")
    sizes = [estimate_question_tokens(predicate) for predicate in predicates]
    longest = max(sizes)
    per_question_budget = int(BUDGET_FRACTION * limits.max_state_plus_question_tokens)
    if state.estimated_tokens + longest > per_question_budget:
        raise ProviderError(
            "context_too_large",
            f"the state (about {state.estimated_tokens} tokens) plus the longest question (about {longest}) "
            f"exceeds the planning budget of {per_question_budget} tokens; send only the fields the "
            "predicates need",
            provider=provider,
            estimated_tokens=state.estimated_tokens + longest,
        )
    request_budget = BUDGET_FRACTION * limits.max_request_tokens
    base = limits.fixed_overhead_tokens + state.estimated_tokens
    chunks: list[Chunk] = []
    current: list[Predicate] = []
    current_tokens = base
    for predicate, size in zip(predicates, sizes, strict=True):
        if current and current_tokens + size > request_budget:
            chunks.append(Chunk(tuple(current), current_tokens))
            current, current_tokens = [], base
        current.append(predicate)
        current_tokens += size
    chunks.append(Chunk(tuple(current), current_tokens))
    return Plan(tuple(chunks))


def bisect(chunk: Chunk, state: State, limits: ProviderLimits) -> tuple[Chunk, Chunk]:
    """Split a chunk of two or more predicates into two halves."""
    if len(chunk.predicates) < 2:
        raise ValueError("cannot bisect a chunk with fewer than two predicates")
    middle = len(chunk.predicates) // 2
    return (
        make_chunk(state, chunk.predicates[:middle], limits),
        make_chunk(state, chunk.predicates[middle:], limits),
    )
