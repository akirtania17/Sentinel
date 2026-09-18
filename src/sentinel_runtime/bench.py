"""Batching benchmark: one request for every predicate versus one request per predicate.

For one state and predicate set, ``run_benchmark`` measures three strategies:

- **batched**: every predicate in a single request (what the runtime does by default);
- **sequential**: one request per predicate, one after another;
- **concurrent**: one request per predicate, all at once (bounded by ``concurrency``).

It reports wall-clock latency, input tokens and estimated cost per strategy, the largest difference
between batched and single-predicate answers, and batched throughput at the given concurrency.
Always run it live; replayed latencies are not measurements.
"""

import asyncio
import statistics
import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from sentinel_runtime.predicates import PredicateSet
from sentinel_runtime.results import ChoiceResult, Evaluation, NoulResult, Result, ScoreResult
from sentinel_runtime.runtime import Runtime
from sentinel_runtime.state import State, StateLike


class StrategyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str
    requests: int
    """Provider requests per evaluation of the whole set."""
    latency_ms: list[float]
    """Wall-clock time for the whole set, one entry per repeat."""
    median_latency_ms: float
    input_tokens: int | None
    """Input tokens for the whole set (from the first repeat)."""
    estimated_cost_usd: float | None


class AnswerDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicate: str
    kind: str
    batched: float | str
    single: float | str
    delta: float | None
    """Absolute difference for numeric answers; ``None`` when a choice picked a different option."""


class BenchReport(BaseModel):
    model_config = ConfigDict(frozen=True, title="Sentinel batching benchmark (v1)")

    predicate_set: str
    predicates: int
    state_chars: int
    model_requested: str
    model_resolved: str
    repeats: int
    concurrency: int
    strategies: list[StrategyResult]
    cost_ratio: float | None
    """Sequential cost divided by batched cost."""
    latency_ratio: float | None
    """Sequential median latency divided by batched median latency."""
    answers: list[AnswerDelta]
    max_answer_delta: float | None
    throughput_evaluations_per_second: float
    """Batched evaluations of the whole set per second at ``concurrency``."""


def _value(result: Result) -> float | str:
    if isinstance(result, NoulResult):
        return result.probability
    if isinstance(result, ChoiceResult):
        return result.choice
    assert isinstance(result, ScoreResult)
    return result.score


def _delta(batched: Result, single: Result) -> float | None:
    if isinstance(batched, ChoiceResult) and isinstance(single, ChoiceResult):
        if batched.choice != single.choice:
            return None
        return round(abs(batched.top_probability - single.top_probability), 6)
    left, right = _value(batched), _value(single)
    assert isinstance(left, float)
    assert isinstance(right, float)
    return round(abs(left - right), 6)


def _totals(evaluations: Sequence[Evaluation]) -> tuple[int, int | None, float | None]:
    requests = sum(e.usage.requests for e in evaluations)
    tokens = [e.usage.input_tokens for e in evaluations]
    costs = [e.usage.estimated_cost_usd for e in evaluations]
    total_tokens = None if any(t is None for t in tokens) else sum(t for t in tokens if t is not None)
    total_cost = None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 10)
    return requests, total_tokens, total_cost


def _strategy(name: str, latencies: list[float], evaluations: Sequence[Evaluation]) -> StrategyResult:
    requests, tokens, cost = _totals(evaluations)
    return StrategyResult(
        strategy=name,
        requests=requests,
        latency_ms=[round(value, 1) for value in latencies],
        median_latency_ms=round(statistics.median(latencies), 1),
        input_tokens=tokens,
        estimated_cost_usd=cost,
    )


async def run_benchmark(
    runtime: Runtime,
    state: StateLike | State,
    predicate_set: PredicateSet,
    *,
    repeats: int = 3,
    concurrency: int = 4,
    model: str | None = None,
) -> BenchReport:
    """Measure batched, sequential and concurrent strategies ``repeats`` times each."""
    if repeats < 1 or concurrency < 1:
        raise ValueError("repeats and concurrency must be at least 1")
    prepared = State.of(state)
    singles = [predicate_set.select([predicate.id]) for predicate in predicate_set.predicates]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(target: PredicateSet) -> Evaluation:
        async with semaphore:
            return await runtime.aevaluate(prepared, target, model=model)

    batched_latencies: list[float] = []
    sequential_latencies: list[float] = []
    concurrent_latencies: list[float] = []
    batched_first: Evaluation | None = None
    sequential_first: list[Evaluation] = []
    concurrent_first: list[Evaluation] = []
    for repeat in range(repeats):
        started = time.perf_counter()
        batched = await runtime.aevaluate(prepared, predicate_set, model=model)
        batched_latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        sequential = [await runtime.aevaluate(prepared, single, model=model) for single in singles]
        sequential_latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        concurrent = await asyncio.gather(*(one(single) for single in singles))
        concurrent_latencies.append((time.perf_counter() - started) * 1000)

        if repeat == 0:
            batched_first, sequential_first, concurrent_first = batched, sequential, list(concurrent)

    assert batched_first is not None
    started = time.perf_counter()
    burst = repeats * concurrency
    await asyncio.gather(*(one(predicate_set) for _ in range(burst)))
    throughput = burst / max(time.perf_counter() - started, 1e-9)

    answers: list[AnswerDelta] = []
    for single in sequential_first:
        for predicate_id, result in single.results.items():
            batched_result = batched_first.results[predicate_id]
            answers.append(
                AnswerDelta(
                    predicate=predicate_id,
                    kind=result.kind,
                    batched=_value(batched_result),
                    single=_value(result),
                    delta=_delta(batched_result, result),
                )
            )
    numeric = [answer.delta for answer in answers if answer.delta is not None]
    strategies = [
        _strategy("batched", batched_latencies, [batched_first]),
        _strategy("sequential", sequential_latencies, sequential_first),
        _strategy("concurrent", concurrent_latencies, concurrent_first),
    ]
    batched_cost, sequential_cost = strategies[0].estimated_cost_usd, strategies[1].estimated_cost_usd
    return BenchReport(
        predicate_set=predicate_set.name,
        predicates=len(predicate_set),
        state_chars=prepared.chars,
        model_requested=batched_first.model_requested,
        model_resolved=batched_first.model_resolved,
        repeats=repeats,
        concurrency=concurrency,
        strategies=strategies,
        cost_ratio=round(sequential_cost / batched_cost, 2)
        if batched_cost and sequential_cost is not None
        else None,
        latency_ratio=round(strategies[1].median_latency_ms / strategies[0].median_latency_ms, 2)
        if strategies[0].median_latency_ms
        else None,
        answers=answers,
        max_answer_delta=max(numeric) if numeric else None,
        throughput_evaluations_per_second=round(throughput, 2),
    )
