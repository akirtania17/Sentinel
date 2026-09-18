"""The semantic runtime.

``Runtime.evaluate(state, predicates)`` validates the predicate set, canonicalizes the state, plans
the minimum number of provider requests (normally one), calls the provider, validates and types the
answers, emits one telemetry event, and returns an ``Evaluation``. The runtime never decides what the
application does, and a provider failure is always a ``ProviderError``, never a result.

When a plan needs several requests and one fails, the error carries the successful results in
``ProviderError.partial``. Sync evaluation runs requests in order and stops at the first failure;
async evaluation runs them concurrently and waits for all before raising.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from sentinel_runtime.config import SentinelConfig, resolve_model
from sentinel_runtime.errors import ConfigError, ProviderError
from sentinel_runtime.normalize import normalize_answers
from sentinel_runtime.planner import Chunk, Plan, bisect, plan_requests
from sentinel_runtime.predicates import Predicate, PredicateSet, noul
from sentinel_runtime.pricing import PricingTable
from sentinel_runtime.providers.base import Provider, ProviderRequest, ProviderResponse
from sentinel_runtime.providers.caching import CacheMode, CachingProvider
from sentinel_runtime.providers.typesafe import TypeSafeProvider
from sentinel_runtime.results import Evaluation, ProviderCall, Result, Usage
from sentinel_runtime.state import Redactor, State, StateLike
from sentinel_runtime.telemetry import EvaluationEvent, TelemetrySink

logger = logging.getLogger("sentinel_runtime")

_PROBE = noul("probe", question="Is this text a greeting?")


@dataclass(frozen=True)
class _Prepared:
    evaluation_id: str
    created_at: datetime
    started: float
    predicate_set: PredicateSet
    state: State
    model: str


@dataclass(frozen=True)
class _Outcome:
    call: ProviderCall
    results: dict[str, Result]


class Runtime:
    """Evaluate semantic predicates against application state."""

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        model: str | None = None,
        sinks: Sequence[TelemetrySink] = (),
        redact: Redactor | None = None,
        max_state_chars: int | None = None,
        include_state_in_events: bool = False,
        pricing: PricingTable | None = None,
        config: SentinelConfig | None = None,
    ) -> None:
        """Create a runtime.

        ``provider`` defaults to ``TypeSafeProvider()``, which reads ``TYPESAFE_API_KEY``. The runtime
        owns its provider and closes it in ``close()``. ``redact`` receives a plain-JSON copy of each
        state and returns the state to send. ``max_state_chars`` optionally rejects states larger
        than an application-chosen size before planning.
        """
        self._config = config if config is not None else SentinelConfig()
        self._provider: Provider = provider if provider is not None else TypeSafeProvider()
        self._model = model
        self._sinks = tuple(sinks)
        self._redact = redact
        self._max_state_chars = max_state_chars
        self._include_state = include_state_in_events
        self._pricing = pricing if pricing is not None else PricingTable.from_config(self._config)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        sinks: Sequence[TelemetrySink] = (),
        cache: str | Path | None = None,
        cache_mode: CacheMode = "replay",
        redact: Redactor | None = None,
        max_state_chars: int | None = None,
        include_state_in_events: bool = False,
        config: SentinelConfig | None = None,
    ) -> "Runtime":
        """A runtime configured from the environment and the nearest ``sentinel.toml``.

        With ``cache``, responses are served from and recorded to that JSONL file; with
        ``cache_mode="offline"`` no API key is needed and cache misses raise ``CacheMissError``.
        """
        resolved_config = config if config is not None else SentinelConfig.load()
        provider: Provider
        if cache is None:
            provider = TypeSafeProvider()
        else:
            inner = None if cache_mode == "offline" else TypeSafeProvider()
            provider = CachingProvider(inner, cache, mode=cache_mode)
        return cls(
            provider,
            model=model,
            sinks=sinks,
            redact=redact,
            max_state_chars=max_state_chars,
            include_state_in_events=include_state_in_events,
            config=resolved_config,
        )

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def sinks(self) -> tuple[TelemetrySink, ...]:
        return self._sinks

    def model_for(self, model: str | None = None) -> str:
        """The model a call with ``model`` would request, after applying the precedence rules."""
        return resolve_model(model, runtime_model=self._model, config=self._config)

    def __repr__(self) -> str:
        return f"Runtime(provider={self._provider!r}, model={self.model_for()!r})"

    # ---- evaluation -----------------------------------------------------------------------------

    def evaluate(
        self,
        state: StateLike | State,
        predicates: PredicateSet | Sequence[Predicate],
        *,
        model: str | None = None,
        select: Iterable[str] | None = None,
        timeout: float | None = None,
    ) -> Evaluation:
        """Evaluate ``predicates`` against ``state``.

        Raises ``StateError`` for unusable state, ``KeyError`` for unknown ``select`` ids, and
        ``ProviderError`` when judgments could not be obtained.
        """
        prepared = self._prepare(state, predicates, model, select)
        outcomes: list[_Outcome] = []
        failure: ProviderError | None = None
        try:
            for chunk in self._plan(prepared).chunks:
                outcomes.extend(self._run_chunk(prepared, chunk, timeout))
        except ProviderError as error:
            failure = error
        return self._finish(prepared, outcomes, failure)

    async def aevaluate(
        self,
        state: StateLike | State,
        predicates: PredicateSet | Sequence[Predicate],
        *,
        model: str | None = None,
        select: Iterable[str] | None = None,
        timeout: float | None = None,
    ) -> Evaluation:
        """Async form of :meth:`evaluate`. Multiple requests run concurrently."""
        prepared = self._prepare(state, predicates, model, select)
        outcomes: list[_Outcome] = []
        failure: ProviderError | None = None
        plan: Plan | None = None
        try:
            plan = self._plan(prepared)
        except ProviderError as error:
            failure = error
        if plan is not None:
            gathered = await asyncio.gather(
                *(self._arun_chunk(prepared, chunk, timeout) for chunk in plan.chunks), return_exceptions=True
            )
            unexpected: BaseException | None = None
            for item in gathered:
                if isinstance(item, ProviderError):
                    failure = failure or item
                elif isinstance(item, BaseException):
                    unexpected = unexpected or item
                else:
                    outcomes.extend(item)
            if unexpected is not None:
                raise unexpected
        return self._finish(prepared, outcomes, failure)

    def resolve_model(self, model: str | None = None) -> str:
        """Send one tiny live request and return the versioned model an alias resolves to now."""
        provider = self._provider
        if isinstance(provider, CachingProvider):
            if provider.inner is None:
                raise ConfigError(
                    "cannot resolve a model through an offline cache; a live provider is needed"
                )
            provider = provider.inner
        request = ProviderRequest(
            state=State.of("Hello there!"), predicates=(_PROBE,), model=self.model_for(model)
        )
        return provider.evaluate(request).model_resolved

    # ---- lifecycle ------------------------------------------------------------------------------

    def close(self) -> None:
        """Close the provider (and compact any response cache)."""
        self._provider.close()

    async def aclose(self) -> None:
        await self._provider.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.aclose()

    # ---- internals ------------------------------------------------------------------------------

    def _prepare(
        self,
        state: StateLike | State,
        predicates: PredicateSet | Sequence[Predicate],
        model: str | None,
        select: Iterable[str] | None,
    ) -> _Prepared:
        predicate_set = PredicateSet.of(predicates)
        if select is not None:
            predicate_set = predicate_set.select(select)
        return _Prepared(
            evaluation_id=f"ev_{uuid.uuid4().hex}",
            created_at=datetime.now(UTC),
            started=time.perf_counter(),
            predicate_set=predicate_set,
            state=State.of(state, redact=self._redact, max_chars=self._max_state_chars),
            model=self.model_for(model),
        )

    def _plan(self, prepared: _Prepared) -> Plan:
        return plan_requests(
            prepared.state,
            prepared.predicate_set.predicates,
            self._provider.limits,
            provider=self._provider.name,
        )

    def _request(self, prepared: _Prepared, chunk: Chunk, timeout: float | None) -> ProviderRequest:
        return ProviderRequest(
            state=prepared.state,
            predicates=chunk.predicates,
            model=prepared.model,
            timeout=timeout,
            estimated_input_tokens=chunk.estimated_input_tokens,
        )

    def _outcome(self, chunk: Chunk, request: ProviderRequest, response: ProviderResponse) -> _Outcome:
        results = normalize_answers(chunk.predicates, response.answers, provider=self._provider.name)
        call = ProviderCall(
            request_id=response.request_id,
            model_requested=request.model,
            model_resolved=response.model_resolved,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_input_tokens=chunk.estimated_input_tokens,
            latency_ms=response.latency_ms,
            predicate_ids=tuple(predicate.id for predicate in chunk.predicates),
            attempts=response.attempts,
            replayed=response.replayed,
        )
        return _Outcome(call=call, results=results)

    def _split_or_raise(self, prepared: _Prepared, chunk: Chunk, error: ProviderError) -> tuple[Chunk, Chunk]:
        """Bisect a chunk the provider rejected as too large, or raise when it cannot be split."""
        if error.kind == "context_too_large" and len(chunk.predicates) > 1:
            return bisect(chunk, prepared.state, self._provider.limits)
        if error.kind == "context_too_large" and error.estimated_tokens is None:
            raise error.replace(estimated_tokens=chunk.estimated_input_tokens)
        raise error

    def _run_chunk(self, prepared: _Prepared, chunk: Chunk, timeout: float | None) -> list[_Outcome]:
        request = self._request(prepared, chunk, timeout)
        failure: ProviderError | None = None
        response: ProviderResponse | None = None
        try:
            response = self._provider.evaluate(request)
        except ProviderError as error:
            failure = error
        if failure is not None:
            left, right = self._split_or_raise(prepared, chunk, failure)
            return self._run_chunk(prepared, left, timeout) + self._run_chunk(prepared, right, timeout)
        assert response is not None
        return [self._outcome(chunk, request, response)]

    async def _arun_chunk(self, prepared: _Prepared, chunk: Chunk, timeout: float | None) -> list[_Outcome]:
        request = self._request(prepared, chunk, timeout)
        failure: ProviderError | None = None
        response: ProviderResponse | None = None
        try:
            response = await self._provider.aevaluate(request)
        except ProviderError as error:
            failure = error
        if failure is not None:
            left, right = self._split_or_raise(prepared, chunk, failure)
            halves = await asyncio.gather(
                self._arun_chunk(prepared, left, timeout), self._arun_chunk(prepared, right, timeout)
            )
            return halves[0] + halves[1]
        assert response is not None
        return [self._outcome(chunk, request, response)]

    def _usage(self, calls: tuple[ProviderCall, ...]) -> Usage:
        def total(values: list[int | None]) -> int | None:
            return None if any(value is None for value in values) else sum(v for v in values if v is not None)

        costs = [
            self._pricing.cost(call.model_resolved, call.input_tokens, call.output_tokens) for call in calls
        ]
        return Usage(
            requests=len(calls),
            input_tokens=total([call.input_tokens for call in calls]),
            output_tokens=total([call.output_tokens for call in calls]),
            estimated_cost_usd=None
            if any(cost is None for cost in costs)
            else sum(c for c in costs if c is not None),
            replayed=any(call.replayed for call in calls),
        )

    def _evaluation(self, prepared: _Prepared, outcomes: list[_Outcome]) -> Evaluation:
        by_id: dict[str, Result] = {}
        for outcome in outcomes:
            by_id.update(outcome.results)
        predicate_set = prepared.predicate_set
        calls = tuple(outcome.call for outcome in outcomes)
        return Evaluation(
            evaluation_id=prepared.evaluation_id,
            created_at=prepared.created_at,
            predicate_set=predicate_set.name,
            predicate_set_fingerprint=predicate_set.fingerprint,
            predicate_fingerprints=predicate_set.predicate_fingerprints(),
            state_digest=prepared.state.digest,
            state_chars=prepared.state.chars,
            provider=self._provider.name,
            model_requested=prepared.model,
            model_resolved=",".join(sorted({call.model_resolved for call in calls})),
            results={p.id: by_id[p.id] for p in predicate_set.predicates if p.id in by_id},
            calls=calls,
            usage=self._usage(calls),
            latency_ms=round((time.perf_counter() - prepared.started) * 1000, 3),
        )

    def _finish(
        self, prepared: _Prepared, outcomes: list[_Outcome], failure: ProviderError | None
    ) -> Evaluation:
        if failure is None:
            evaluation = self._evaluation(prepared, outcomes)
            self._emit(lambda: self._event(prepared, evaluation, None))
            return evaluation
        partial = self._evaluation(prepared, outcomes) if outcomes else None
        error = failure.replace(partial=partial) if partial is not None else failure
        self._emit(lambda: self._event(prepared, partial, error))
        raise error

    def _event(
        self, prepared: _Prepared, evaluation: Evaluation | None, error: ProviderError | None
    ) -> EvaluationEvent:
        predicate_set = prepared.predicate_set
        return EvaluationEvent(
            evaluation_id=prepared.evaluation_id,
            ts=prepared.created_at,
            outcome="ok" if error is None else "error",
            predicate_set=predicate_set.name,
            predicate_set_fingerprint=predicate_set.fingerprint,
            predicate_fingerprints=predicate_set.predicate_fingerprints(),
            state_digest=prepared.state.digest,
            state_chars=prepared.state.chars,
            state=prepared.state.value if self._include_state else None,
            provider=self._provider.name,
            model_requested=prepared.model,
            model_resolved=evaluation.model_resolved if evaluation is not None else None,
            latency_ms=round((time.perf_counter() - prepared.started) * 1000, 3),
            evaluation=evaluation,
            error=error.info() if error is not None else None,
        )

    def _emit(self, build: Callable[[], EvaluationEvent]) -> None:
        if not self._sinks:
            return
        event = build()
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception as error:  # Telemetry must never break an evaluation.
                logger.warning("telemetry sink %r failed: %s: %s", sink, type(error).__name__, error)
