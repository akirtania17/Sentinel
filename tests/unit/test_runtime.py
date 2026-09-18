import logging
from pathlib import Path
from typing import Any

import pytest

from sentinel_runtime import (
    CallbackSink,
    ChoiceResult,
    ConfigError,
    EvaluationEvent,
    JsonlSink,
    NoulResult,
    PredicateSet,
    ProviderError,
    Runtime,
    ScoreResult,
    State,
    StateError,
    choice,
    noul,
    score,
)
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.planner import plan_requests
from sentinel_runtime.providers import (
    CachingProvider,
    FakeProvider,
    ProviderLimits,
    ProviderRequest,
    choice_answer,
    noul_answer,
    score_answer,
)
from sentinel_runtime.telemetry import read_events

SET = PredicateSet(
    name="agent",
    predicates=(
        noul("matches", question="Does `action` match `request`?", threshold=0.8),
        choice(
            "effect",
            question="What effect does `action` have?",
            options={"read": None, "write": None},
            min_confidence=0.6,
        ),
        score("severity", question="How severe is `action`?", levels=["harmless", "moderate", "severe"]),
    ),
)
STATE = {"request": "Summarize my email.", "action": {"tool": "send_email"}}
ANSWERS = {
    "matches": noul_answer(0.1),
    "effect": choice_answer({"read": 0.1, "write": 0.9}, confidence=0.8),
    "severity": score_answer([0.2, 0.5, 0.3], confidence=0.4),
}
SMALL = ProviderLimits(
    max_request_tokens=500, max_state_plus_question_tokens=10_000, fixed_overhead_tokens=100
)


class Events:
    def __init__(self) -> None:
        self.events: list[EvaluationEvent] = []

    def sink(self) -> CallbackSink:
        return CallbackSink(self.events.append)


def _runtime(provider: FakeProvider, events: Events | None = None, **kwargs: Any) -> Runtime:
    sinks = [events.sink()] if events is not None else []
    return Runtime(provider, model="test-model", sinks=sinks, **kwargs)


def _many(count: int) -> PredicateSet:
    return PredicateSet(
        name="many",
        predicates=tuple(
            noul(f"p{i}", question=f"Is statement {i} true? " + "x" * 200) for i in range(count)
        ),
    )


# ---- success --------------------------------------------------------------------------------------


def test_evaluate_returns_typed_results_from_one_request() -> None:
    provider = FakeProvider(ANSWERS, model_resolved="test-model-1.0")
    events = Events()
    evaluation = _runtime(provider, events).evaluate(STATE, SET)
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "test-model"
    assert provider.requests[0].predicates == SET.predicates
    assert list(evaluation.results) == ["matches", "effect", "severity"]
    matches = evaluation["matches"]
    assert isinstance(matches, NoulResult)
    assert matches.decision is False
    effect = evaluation["effect"]
    assert isinstance(effect, ChoiceResult)
    assert effect.decision == "write"
    assert isinstance(evaluation["severity"], ScoreResult)
    assert evaluation.model_requested == "test-model"
    assert evaluation.model_resolved == "test-model-1.0"
    assert evaluation.predicate_set == "agent"
    assert evaluation.predicate_set_fingerprint == SET.fingerprint
    assert evaluation.predicate_fingerprints == SET.predicate_fingerprints()
    assert evaluation.usage.requests == 1
    assert evaluation.calls[0].predicate_ids == SET.ids()
    assert evaluation.calls[0].estimated_input_tokens > 0
    assert evaluation.evaluation_id.startswith("ev_")
    assert len(events.events) == 1
    event = events.events[0]
    assert event.outcome == "ok"
    assert event.evaluation == evaluation
    assert event.state is None
    assert event.error is None


async def test_async_matches_sync() -> None:
    sync_eval = _runtime(FakeProvider(ANSWERS)).evaluate(STATE, SET)
    async_eval = await _runtime(FakeProvider(ANSWERS)).aevaluate(STATE, SET)
    assert async_eval.results == sync_eval.results
    assert async_eval.usage == sync_eval.usage


def test_accepts_a_bare_sequence_and_select() -> None:
    provider = FakeProvider(ANSWERS)
    evaluation = _runtime(provider).evaluate(STATE, list(SET.predicates), select=["severity", "matches"])
    assert evaluation.predicate_set == "adhoc"
    assert list(evaluation.results) == ["matches", "severity"]
    with pytest.raises(KeyError):
        _runtime(provider).evaluate(STATE, SET, select=["nope"])


def test_model_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider(ANSWERS)
    runtime = Runtime(provider, config=SentinelConfig(model="from-toml"))
    assert runtime.evaluate(STATE, SET).model_requested == "from-toml"
    monkeypatch.setenv("SENTINEL_MODEL", "from-env")
    assert runtime.evaluate(STATE, SET).model_requested == "from-env"
    assert Runtime(provider, model="from-runtime").evaluate(STATE, SET).model_requested == "from-runtime"
    assert runtime.evaluate(STATE, SET, model="per-call").model_requested == "per-call"


def test_usage_and_cost_use_the_resolved_model() -> None:
    provider = FakeProvider(ANSWERS, model_resolved="jev-1.13.0")
    evaluation = _runtime(provider).evaluate(STATE, SET)
    tokens = evaluation.usage.input_tokens
    assert tokens is not None
    assert evaluation.usage.estimated_cost_usd == pytest.approx(tokens * 0.042 / 1_000_000)
    unpriced = _runtime(FakeProvider(ANSWERS, model_resolved="unknown-model")).evaluate(STATE, SET)
    assert unpriced.usage.estimated_cost_usd is None


def test_redaction_and_event_state_opt_in() -> None:
    provider = FakeProvider(ANSWERS)
    events = Events()

    def redact(value: Any) -> Any:
        value["request"] = "[redacted]"
        return value

    runtime = _runtime(provider, events, redact=redact, include_state_in_events=True)
    runtime.evaluate(STATE, SET)
    assert provider.requests[0].state.value["request"] == "[redacted]"  # type: ignore[index]
    assert STATE["request"] == "Summarize my email."
    assert events.events[0].state == {"request": "[redacted]", "action": {"tool": "send_email"}}


def test_max_state_chars_guard() -> None:
    with pytest.raises(StateError) as caught:
        _runtime(FakeProvider(ANSWERS), max_state_chars=10).evaluate(STATE, SET)
    assert caught.value.reason == "too_large"


# ---- splitting, partial results, bisect -----------------------------------------------------------


def test_large_sets_split_and_merge_in_order() -> None:
    provider = FakeProvider(limits=SMALL)
    predicate_set = _many(8)
    evaluation = _runtime(provider).evaluate("state", predicate_set)
    assert len(provider.requests) > 1
    assert list(evaluation.results) == list(predicate_set.ids())
    assert evaluation.usage.requests == len(provider.requests)
    assert [p for call in evaluation.calls for p in call.predicate_ids] == list(predicate_set.ids())


def _fail_on(predicate_id: str, kind: str = "overloaded") -> Any:
    def respond(request: ProviderRequest) -> Any:
        if any(p.id == predicate_id for p in request.predicates):
            return ProviderError(kind, "boom", provider="fake", retryable=True)  # type: ignore[arg-type]
        return {p.id: noul_answer(0.9) for p in request.predicates}

    return respond


def test_sync_failure_carries_partial_results() -> None:
    predicate_set = _many(8)
    provider = FakeProvider(respond=_fail_on("p7"), limits=SMALL)
    events = Events()
    with pytest.raises(ProviderError) as caught:
        _runtime(provider, events).evaluate("state", predicate_set)
    error = caught.value
    assert error.kind == "overloaded"
    assert error.partial is not None
    assert "p0" in error.partial.results
    assert "p7" not in error.partial.results
    assert events.events[0].outcome == "error"
    assert events.events[0].error is not None
    assert events.events[0].error.kind == "overloaded"
    assert events.events[0].evaluation == error.partial


async def test_async_failure_waits_for_siblings() -> None:
    predicate_set = _many(8)
    provider = FakeProvider(respond=_fail_on("p0"), limits=SMALL)
    runtime = _runtime(provider)
    with pytest.raises(ProviderError) as caught:
        await runtime.aevaluate("state", predicate_set)
    partial = caught.value.partial
    assert partial is not None
    assert "p7" in partial.results
    assert "p0" not in partial.results
    expected = plan_requests(State.of("state"), predicate_set.predicates, SMALL, provider="fake")
    assert len(provider.requests) == len(expected.chunks)


def test_single_request_failure_has_no_partial() -> None:
    provider = FakeProvider(respond=_fail_on("matches", "rate_limited"))
    events = Events()
    with pytest.raises(ProviderError) as caught:
        _runtime(provider, events).evaluate(STATE, SET)
    assert caught.value.partial is None
    assert events.events[0].evaluation is None
    assert events.events[0].model_resolved is None


def _too_large_above(limit: int) -> Any:
    def respond(request: ProviderRequest) -> Any:
        if len(request.predicates) > limit:
            return ProviderError("context_too_large", "max_tokens_exceeded", provider="fake")
        return {p.id: noul_answer(0.3) for p in request.predicates}

    return respond


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_provider_rejection_bisects_until_it_fits(mode: str) -> None:
    provider = FakeProvider(respond=_too_large_above(2))
    predicate_set = _many(7)
    runtime = _runtime(provider)
    evaluation = (
        runtime.evaluate("s", predicate_set)
        if mode == "sync"
        else await runtime.aevaluate("s", predicate_set)
    )
    assert list(evaluation.results) == list(predicate_set.ids())
    assert all(len(call.predicate_ids) <= 2 for call in evaluation.calls)


def test_single_predicate_rejection_reports_the_estimate() -> None:
    provider = FakeProvider(respond=_too_large_above(0))
    with pytest.raises(ProviderError) as caught:
        _runtime(provider).evaluate("s", _many(1))
    assert caught.value.kind == "context_too_large"
    assert caught.value.estimated_tokens is not None


def test_oversized_state_fails_before_any_request() -> None:
    provider = FakeProvider(ANSWERS)
    events = Events()
    with pytest.raises(ProviderError) as caught:
        _runtime(provider, events).evaluate("x" * 120_000, SET)
    assert caught.value.kind == "context_too_large"
    assert provider.requests == []
    assert events.events[0].outcome == "error"


def test_malformed_answers_are_errors_not_results() -> None:
    provider = FakeProvider({**ANSWERS, "matches": {"type": "noul", "noul": 7}})
    with pytest.raises(ProviderError) as caught:
        _runtime(provider).evaluate(STATE, SET)
    assert caught.value.kind == "malformed_response"


def test_state_errors_emit_no_event() -> None:
    events = Events()
    with pytest.raises(StateError):
        _runtime(FakeProvider(ANSWERS), events).evaluate("", SET)
    assert events.events == []


def test_mixed_resolved_models_are_all_reported() -> None:
    counter = {"calls": 0}

    def respond(request: ProviderRequest) -> Any:
        counter["calls"] += 1
        return {p.id: noul_answer(0.5) for p in request.predicates}

    provider = FakeProvider(respond=respond, limits=SMALL)
    runtime = _runtime(provider)
    original = provider.evaluate

    def alternate(request: ProviderRequest) -> Any:
        provider.model_resolved = "m-2" if counter["calls"] % 2 else "m-1"
        return original(request)

    provider.evaluate = alternate  # type: ignore[method-assign]
    evaluation = runtime.evaluate("state", _many(6))
    assert evaluation.model_resolved == "m-1,m-2"


# ---- telemetry ------------------------------------------------------------------------------------


def test_failing_sinks_never_break_evaluation(caplog: pytest.LogCaptureFixture) -> None:
    def explode(event: EvaluationEvent) -> None:
        raise RuntimeError("sink down")

    runtime = Runtime(FakeProvider(ANSWERS), sinks=[CallbackSink(explode)])
    with caplog.at_level(logging.WARNING, logger="sentinel_runtime"):
        evaluation = runtime.evaluate(STATE, SET)
    assert evaluation.results
    assert "telemetry sink" in caplog.text
    assert "sink down" in caplog.text


def test_jsonl_sink_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events" / "sentinel.events.jsonl"
    runtime = Runtime(FakeProvider(ANSWERS), sinks=[JsonlSink(path)])
    runtime.evaluate(STATE, SET)
    with pytest.raises(ProviderError):
        Runtime(FakeProvider(respond=_fail_on("matches")), sinks=[JsonlSink(path)]).evaluate(STATE, SET)
    events = read_events(path)
    assert [event.outcome for event in events] == ["ok", "error"]
    assert "Summarize my email" not in path.read_text()


# ---- caching, probing, lifecycle ------------------------------------------------------------------


def test_replayed_evaluations_are_flagged(tmp_path: Path) -> None:
    inner = FakeProvider(ANSWERS, name="typesafe")
    runtime = Runtime(CachingProvider(inner, tmp_path / "c.jsonl"), model="m")
    assert runtime.evaluate(STATE, SET).usage.replayed is False
    replayed = runtime.evaluate(STATE, SET)
    assert replayed.usage.replayed is True
    assert replayed.calls[0].replayed is True
    runtime.close()
    assert inner.closed


def test_resolve_model_bypasses_the_cache(tmp_path: Path) -> None:
    inner = FakeProvider(model_resolved="jev-9.9.9", name="typesafe")
    runtime = Runtime(CachingProvider(inner, tmp_path / "c.jsonl"), model="jev-latest")
    assert runtime.resolve_model() == "jev-9.9.9"
    assert runtime.resolve_model() == "jev-9.9.9"
    assert len(inner.requests) == 2
    offline = Runtime(CachingProvider(None, tmp_path / "c.jsonl", mode="offline"))
    with pytest.raises(ConfigError, match="offline cache"):
        offline.resolve_model()


def test_from_env_offline_needs_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    runtime = Runtime.from_env(cache=tmp_path / "c.jsonl", cache_mode="offline", config=SentinelConfig())
    assert isinstance(runtime.provider, CachingProvider)
    with pytest.raises(ProviderError) as caught:
        Runtime.from_env(config=SentinelConfig())
    assert caught.value.kind == "authentication"


async def test_context_managers_close_the_provider() -> None:
    provider = FakeProvider(ANSWERS)
    with Runtime(provider):
        pass
    assert provider.closed
    async_provider = FakeProvider(ANSWERS)
    async with Runtime(async_provider):
        pass
    assert async_provider.closed
