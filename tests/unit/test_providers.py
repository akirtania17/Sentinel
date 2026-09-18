import asyncio
import json
from pathlib import Path

import pytest

from sentinel_runtime import CacheMissError, ConfigError, PredicateSet, ProviderError, State, noul
from sentinel_runtime.config import PriceEntry, SentinelConfig
from sentinel_runtime.pricing import PricingTable
from sentinel_runtime.providers import (
    CachingProvider,
    FakeProvider,
    ProviderRequest,
    ProviderResponse,
    cache_key,
    choice_answer,
    noul_answer,
    score_answer,
)
from sentinel_runtime.providers.fake import default_answer

SET = PredicateSet(name="s", predicates=(noul("a", question="q?"), noul("b", question="r?")))


def _request(
    state: object = "hello", predicates: PredicateSet = SET, model: str = "m-latest"
) -> ProviderRequest:
    return ProviderRequest(state=State.of(state), predicates=predicates.predicates, model=model)  # type: ignore[arg-type]


# ---- fake provider --------------------------------------------------------------------------------


def test_fake_provider_answers_and_records() -> None:
    provider = FakeProvider({"a": noul_answer(0.9)}, model_resolved="fake-2")
    response = provider.evaluate(_request())
    assert response.answers == {"a": noul_answer(0.9), "b": noul_answer(0.5)}
    assert response.model_resolved == "fake-2"
    assert len(provider.requests) == 1
    provider.close()
    assert provider.closed


async def test_fake_provider_respond_can_fail() -> None:
    provider = FakeProvider(respond=lambda _: ProviderError("overloaded", provider="fake", retryable=True))
    with pytest.raises(ProviderError, match="overloaded"):
        await provider.aevaluate(_request())


def test_answer_builders() -> None:
    assert choice_answer({"x": 0.2, "y": 0.8})["choice"] == "y"
    assert choice_answer({"x": 1.0, "y": 0.0})["confidence"] == 1.0
    assert choice_answer({"x": 0.5, "y": 0.5})["confidence"] == 0.0
    assert score_answer([0.0, 0.7, 0.3])["score"] == pytest.approx(1.3)
    assert score_answer([0.0, 0.7, 0.3])["probabilities"] == {"0": 0.0, "1": 0.7, "2": 0.3}
    assert default_answer(noul("x", question="q?")) == noul_answer(0.5)


# ---- caching provider -----------------------------------------------------------------------------


def test_cache_key_covers_model_state_ids_and_fingerprints() -> None:
    base = cache_key("typesafe", _request())
    assert cache_key("typesafe", _request()) == base
    assert cache_key("other", _request()) != base
    assert cache_key("typesafe", _request(model="m-1")) != base
    assert cache_key("typesafe", _request(state="bye")) != base
    renamed = PredicateSet(name="s", predicates=(noul("a2", question="q?"), noul("b", question="r?")))
    assert cache_key("typesafe", _request(predicates=renamed)) != base
    edited = PredicateSet(name="s", predicates=(noul("a", question="q!"), noul("b", question="r?")))
    assert cache_key("typesafe", _request(predicates=edited)) != base
    thresholds = PredicateSet(
        name="s", predicates=(noul("a", question="q?", threshold=0.3), noul("b", question="r?"))
    )
    assert cache_key("typesafe", _request(predicates=thresholds)) == base


def test_replay_records_misses_and_serves_hits(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    inner = FakeProvider({"a": noul_answer(0.8)}, name="typesafe", model_resolved="m-1")
    cache = CachingProvider(inner, path, mode="replay")
    first = cache.evaluate(_request())
    second = cache.evaluate(_request())
    assert first.replayed is False
    assert second.replayed is True
    assert second.answers == first.answers
    assert second.model_resolved == "m-1"
    assert len(inner.requests) == 1
    assert (cache.hits, cache.misses, cache.recorded) == (1, 1, 1)
    cache.close()
    assert inner.closed

    reloaded = CachingProvider(None, path, mode="offline")
    assert reloaded.size == 1
    assert bool(reloaded), "an empty or full cache is always truthy"
    assert reloaded.evaluate(_request()).answers == first.answers
    assert reloaded.models_resolved() == {"m-1"}


def test_cache_lines_never_store_state(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = CachingProvider(FakeProvider(name="typesafe"), path)
    cache.evaluate(_request(state={"secret": "STATE-MARKER"}))
    cache.close()
    text = path.read_text()
    assert "STATE-MARKER" not in text
    record = json.loads(text.splitlines()[0])
    assert set(record) == {
        "key",
        "provider",
        "model_requested",
        "model_resolved",
        "recorded_at",
        "request_id",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "answers",
    }


def test_offline_misses_raise(tmp_path: Path) -> None:
    cache = CachingProvider(None, tmp_path / "empty.jsonl", mode="offline")
    with pytest.raises(CacheMissError) as caught:
        cache.evaluate(_request())
    assert caught.value.keys == [cache_key("typesafe", _request())]
    assert "run again with TYPESAFE_API_KEY set and without offline mode" in str(caught.value)


def test_record_mode_always_calls_and_compacts(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    inner = FakeProvider({"a": noul_answer(0.1)}, name="typesafe")
    cache = CachingProvider(inner, path, mode="record")
    cache.evaluate(_request())
    inner.answers["a"] = noul_answer(0.2)
    cache.evaluate(_request())
    assert len(inner.requests) == 2
    assert len(path.read_text().splitlines()) == 2
    cache.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["answers"]["a"] == noul_answer(0.2)


def test_compaction_sorts_by_key(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = CachingProvider(FakeProvider(name="typesafe"), path)
    for state in ("one", "two", "three", "four"):
        cache.evaluate(_request(state=state))
    cache.close()
    keys = [json.loads(line)["key"] for line in path.read_text().splitlines()]
    assert keys == sorted(keys)


async def test_async_replay(tmp_path: Path) -> None:
    inner = FakeProvider(name="typesafe")
    cache = CachingProvider(inner, tmp_path / "c.jsonl")
    await cache.aevaluate(_request())
    assert (await cache.aevaluate(_request())).replayed
    await cache.aclose()
    assert inner.closed


def test_corrupt_cache_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text('{"key": "x"}\n')
    with pytest.raises(ConfigError, match=r"cache\.jsonl:1: invalid cache record"):
        CachingProvider(None, path, mode="offline")


def test_non_offline_modes_need_an_inner_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs an inner provider"):
        CachingProvider(None, tmp_path / "c.jsonl", mode="replay")


def test_cache_takes_the_inner_name_and_limits(tmp_path: Path) -> None:
    cache = CachingProvider(FakeProvider(name="typesafe"), tmp_path / "c.jsonl")
    assert cache.name == "typesafe"
    assert cache.limits.max_options == 255
    assert "mode='replay'" in repr(cache)


# ---- pricing --------------------------------------------------------------------------------------


def test_pricing_defaults_and_overrides() -> None:
    table = PricingTable()
    assert table.cost("jev-1.13.0", 1_000_000, 500) == pytest.approx(0.042)
    assert table.cost("jev-1.13.0", None, 10) is None
    assert table.cost("unknown-model", 100, 0) is None
    custom = PricingTable.from_config(
        SentinelConfig(pricing={"jev-1.13.0": PriceEntry(input_per_mtok=1.0, output_per_mtok=2.0)})
    )
    assert custom.cost("jev-1.13.0", 1_000_000, 1_000_000) == pytest.approx(3.0)
    assert "jev-1.13.0" in custom.models()


def test_prune_drops_entries_this_session_did_not_use(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    first = CachingProvider(FakeProvider(name="typesafe"), path)
    for state in ("one", "two", "three"):
        first.evaluate(_request(state=state))
    first.close()
    second = CachingProvider(FakeProvider(name="typesafe"), path)
    second.evaluate(_request(state="two"))
    second.evaluate(_request(state="four"))
    assert second.prune() == 2
    second.close()
    assert CachingProvider(None, path, mode="offline").size == 2


def test_record_mode_keeps_identical_entries(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    first = CachingProvider(FakeProvider({"a": noul_answer(0.1)}, name="typesafe"), path, mode="record")
    first.evaluate(_request())
    first.close()
    original = path.read_text()
    again = CachingProvider(FakeProvider({"a": noul_answer(0.1)}, name="typesafe"), path, mode="record")
    again.evaluate(_request())
    assert again.recorded == 0
    again.close()
    assert path.read_text() == original


class _Slow(FakeProvider):
    """A provider whose answers differ call by call and take a moment, optionally failing first."""

    def __init__(self, fail_first: bool = False) -> None:
        super().__init__()
        self.fail_first = fail_first

    async def aevaluate(self, request: ProviderRequest) -> ProviderResponse:
        await asyncio.sleep(0.01)
        if self.fail_first and not self.requests:
            self.requests.append(request)
            raise ProviderError("server", "boom", retryable=True, provider="fake")
        self.answers = {"a": noul_answer(0.1 * (len(self.requests) + 1))}
        return await super().aevaluate(request)


def test_identical_requests_in_flight_are_sent_once(tmp_path: Path) -> None:
    inner = _Slow()
    cache = CachingProvider(inner, tmp_path / "c.jsonl")

    async def three() -> list[ProviderResponse]:
        return list(await asyncio.gather(*(cache.aevaluate(_request()) for _ in range(3))))

    responses = asyncio.run(three())
    assert len(inner.requests) == 1
    assert {r.answers["a"]["noul"] for r in responses} == {0.1}
    assert asyncio.run(cache.aevaluate(_request())).replayed


def test_when_the_shared_request_fails_the_others_try_on_their_own(tmp_path: Path) -> None:
    inner = _Slow(fail_first=True)
    cache = CachingProvider(inner, tmp_path / "c.jsonl")

    async def two() -> list[ProviderResponse | BaseException]:
        return list(
            await asyncio.gather(*(cache.aevaluate(_request()) for _ in range(2)), return_exceptions=True)
        )

    first, second = asyncio.run(two())
    assert isinstance(first, ProviderError)
    assert not isinstance(second, BaseException)
    assert len(inner.requests) == 2
