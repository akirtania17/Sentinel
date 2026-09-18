import asyncio
import json
from typing import Any

import pytest

from sentinel_runtime import PredicateSet, Runtime, choice, noul, score
from sentinel_runtime.bench import run_benchmark
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.providers import FakeProvider, ProviderRequest, choice_answer, noul_answer, score_answer
from tests.unit._schemas import assert_matches_schema

SET = PredicateSet(
    name="bench",
    predicates=(
        noul("a", question="Is `text` about cats?"),
        noul("b", question="Is `text` about dogs?"),
        choice("c", question="Which animal is in `text`?", options={"cat": None, "dog": None}),
        score("d", question="How cute is `text`?", levels=["Not cute at all", "Adorable beyond words"]),
    ),
)


def _respond(request: ProviderRequest) -> dict[str, Any]:
    answers: dict[str, Any] = {
        "a": noul_answer(0.9),
        "b": noul_answer(0.1),
        "c": choice_answer({"cat": 0.9, "dog": 0.1}),
        "d": score_answer([0.2, 0.8]),
    }
    return {p.id: answers[p.id] for p in request.predicates}


def test_benchmark_counts_requests_and_compares_answers() -> None:
    provider = FakeProvider(respond=_respond, model_resolved="jev-1.13.0")
    runtime = Runtime(provider, model="jev-1.13.0", config=SentinelConfig())
    report = asyncio.run(run_benchmark(runtime, {"text": "a cat " * 200}, SET, repeats=2, concurrency=2))
    strategies = {s.strategy: s for s in report.strategies}
    assert strategies["batched"].requests == 1
    assert strategies["sequential"].requests == 4
    assert strategies["concurrent"].requests == 4
    assert len(strategies["batched"].latency_ms) == 2
    assert report.cost_ratio is not None
    assert report.cost_ratio > 1
    assert report.max_answer_delta == 0.0
    assert {answer.predicate for answer in report.answers} == {"a", "b", "c", "d"}
    assert report.model_resolved == "jev-1.13.0"
    assert report.throughput_evaluations_per_second > 0
    # 2 repeats x (1 batched + 4 sequential + 4 concurrent) + 2 x 2 throughput evaluations
    assert len(provider.requests) == 2 * 9 + 4
    assert_matches_schema("bench", json.loads(report.model_dump_json()))


def test_benchmark_validates_arguments() -> None:
    runtime = Runtime(FakeProvider(respond=_respond), config=SentinelConfig())
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(run_benchmark(runtime, "text", SET, repeats=0))
