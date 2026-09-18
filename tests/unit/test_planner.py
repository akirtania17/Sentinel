import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinel_runtime import PredicateSet, ProviderError, State, noul
from sentinel_runtime.planner import (
    BUDGET_FRACTION,
    bisect,
    estimate_question_tokens,
    make_chunk,
    plan_requests,
)
from sentinel_runtime.providers import ProviderLimits

DEFAULT = ProviderLimits()


def _set(count: int, question_chars: int = 40) -> PredicateSet:
    return PredicateSet(
        name="s", predicates=tuple(noul(f"p{i}", question="q" * question_chars + "?") for i in range(count))
    )


def test_common_case_is_one_request() -> None:
    state = State.of({"request": "Summarize my emails.", "action": {"tool": "send_email"}})
    predicates = _set(25).predicates
    plan = plan_requests(state, predicates, DEFAULT, provider="p")
    assert len(plan.chunks) == 1
    assert plan.chunks[0].predicates == predicates
    expected = (
        DEFAULT.fixed_overhead_tokens
        + state.estimated_tokens
        + sum(map(estimate_question_tokens, predicates))
    )
    assert plan.estimated_input_tokens == expected


def test_question_estimate_grows_with_content() -> None:
    short = noul("a", question="Is it?")
    long = noul("a", question="Is it? " * 50)
    assert estimate_question_tokens(long) > estimate_question_tokens(short) >= 20


def test_rejects_state_that_cannot_fit_before_sending() -> None:
    state = State.of("x" * 90_000)
    with pytest.raises(ProviderError) as caught:
        plan_requests(state, _set(1).predicates, DEFAULT, provider="typesafe")
    error = caught.value
    assert error.kind == "context_too_large"
    assert error.provider == "typesafe"
    assert error.estimated_tokens is not None
    assert error.estimated_tokens > BUDGET_FRACTION * DEFAULT.max_state_plus_question_tokens
    assert "send only the fields the predicates need" in error.detail


def test_splits_greedily_in_order_when_over_budget() -> None:
    limits = ProviderLimits(
        max_request_tokens=1_000, max_state_plus_question_tokens=10_000, fixed_overhead_tokens=100
    )
    state = State.of("short state")
    predicates = _set(20, question_chars=300).predicates
    plan = plan_requests(state, predicates, limits, provider="p")
    assert len(plan.chunks) > 1
    assert tuple(p for chunk in plan.chunks for p in chunk.predicates) == predicates
    for chunk in plan.chunks:
        assert (
            chunk.estimated_input_tokens <= BUDGET_FRACTION * limits.max_request_tokens
            or len(chunk.predicates) == 1
        )
    assert plan == plan_requests(state, predicates, limits, provider="p")


@given(
    sizes=st.lists(st.integers(min_value=1, max_value=900), min_size=1, max_size=40),
    budget=st.integers(min_value=400, max_value=6_000),
)
def test_plans_cover_every_predicate_once_in_order(sizes: list[int], budget: int) -> None:
    limits = ProviderLimits(
        max_request_tokens=budget, max_state_plus_question_tokens=100_000, fixed_overhead_tokens=50
    )
    state = State.of("s")
    predicates = tuple(noul(f"p{i}", question="q" * size) for i, size in enumerate(sizes))
    plan = plan_requests(state, predicates, limits, provider="p")
    assert tuple(p for chunk in plan.chunks for p in chunk.predicates) == predicates
    assert all(chunk.predicates for chunk in plan.chunks)
    for chunk in plan.chunks:
        assert chunk.estimated_input_tokens <= BUDGET_FRACTION * budget or len(chunk.predicates) == 1


def test_bisect_halves_and_reestimates() -> None:
    state = State.of("s")
    chunk = make_chunk(state, _set(5).predicates, DEFAULT)
    left, right = bisect(chunk, state, DEFAULT)
    assert left.predicates + right.predicates == chunk.predicates
    assert len(left.predicates) == 2
    assert left.estimated_input_tokens + right.estimated_input_tokens > chunk.estimated_input_tokens
    with pytest.raises(ValueError, match="fewer than two"):
        bisect(make_chunk(state, _set(1).predicates, DEFAULT), state, DEFAULT)


def test_empty_predicate_list_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least one predicate"):
        plan_requests(State.of("s"), (), DEFAULT, provider="p")
