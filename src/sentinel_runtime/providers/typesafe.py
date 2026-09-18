"""The TypeSafe Jev provider, built on the official ``typesafe-sdk``.

The SDK handles transport and retries (408, 429, 5xx including 529, connection errors and
timeouts, honoring ``Retry-After``). This adapter maps predicates to System One questions, keeps the
provider's raw answers, records the resolved model, usage, request id and attempt count, and maps
every SDK failure to a sanitized ``ProviderError``.

Credentials: the key is read by the SDK from ``TYPESAFE_API_KEY``. This module never reads, stores or
passes the key itself, and mapped errors are raised without exception chaining so the SDK's HTTP
objects (whose request headers carry the key) are unreachable from them.
"""

import json
import os
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self, cast

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Question,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeRateLimitError,
)
from typesafe_sdk.constants import API_KEY_ENV

from sentinel_runtime.errors import MAX_DETAIL_CHARS, ProviderError, ProviderErrorKind
from sentinel_runtime.predicates import ChoicePredicate, NoulPredicate, Predicate
from sentinel_runtime.providers.base import ProviderLimits, ProviderRequest, ProviderResponse, RawAnswer

PROVIDER_NAME = "typesafe"
REQUEST_ID_HEADER = "x-typesafe-request-id"
RETRY_COUNT_HEADER = "x-typesafe-retry-count"
DEFAULT_RETRY = RetryPolicy(max_retries=2, timeout=30.0)

_RETRYABLE: frozenset[ProviderErrorKind] = frozenset(
    {"rate_limited", "overloaded", "server", "timeout", "connection"}
)


def wire_question(predicate: Predicate) -> dict[str, Any]:
    """The System One question for ``predicate``, as a plain JSON dictionary."""
    question: dict[str, Any] = {"type": predicate.kind, "instructions": predicate.question}
    if isinstance(predicate, NoulPredicate):
        if predicate.criteria is not None:
            question["criteria"] = predicate.criteria.wire()
    elif isinstance(predicate, ChoicePredicate):
        question["criteria"] = dict(predicate.options)
    else:
        question["criteria"] = list(predicate.levels)
    return question


def _error_fields(body: Any) -> tuple[str | None, str]:
    """``(error_type, sanitized detail)`` from an error body. Never copies echoed request input."""
    if isinstance(body, str):
        return None, body.strip()[:200]
    if not isinstance(body, Mapping):
        return None, ""
    fields = cast("Mapping[str, Any]", body)
    detail = fields.get("detail")
    if isinstance(detail, Mapping):
        described = cast("Mapping[str, Any]", detail)
        error_type = described.get("error_type")
        message = described.get("message")
        kind = error_type if isinstance(error_type, str) else None
        parts = [part for part in (kind, message if isinstance(message, str) else None) if part]
        return kind, ": ".join(parts)
    if isinstance(detail, str):
        return None, detail
    if isinstance(detail, list):
        problems: list[str] = []
        for entry in cast("list[Any]", detail):
            if not isinstance(entry, Mapping):
                continue
            item = cast("Mapping[str, Any]", entry)
            message = item.get("msg")
            if not isinstance(message, str):
                continue
            location = item.get("loc")
            path = ""
            if isinstance(location, list):
                path = ".".join(str(part) for part in cast("list[Any]", location) if part != "body")
            problems.append(f"{path}: {message}" if path else message)
        return None, "; ".join(problems)
    for key in ("message", "error"):
        value = fields.get(key)
        if isinstance(value, str):
            return None, value
    return None, ""


def map_error(error: TypeSafeError, *, provider: str = PROVIDER_NAME) -> ProviderError:
    """Map an SDK exception to a sanitized ``ProviderError``."""
    status: int | None = None
    request_id: str | None = None
    retry_after_ms: int | None = None
    kind: ProviderErrorKind
    if isinstance(error, TypeSafeAPITimeoutError):
        kind, detail = "timeout", "the request timed out"
    elif isinstance(error, TypeSafeAPIConnectionError):
        kind, detail = "connection", "could not reach the TypeSafe API"
    elif isinstance(error, TypeSafeAPIResponseValidationError):
        kind, detail = "malformed_response", f"invalid response data at {error.field_path!r}"
        status, request_id = error.status, error.request_id
    elif isinstance(error, TypeSafeAPIError):
        status, request_id = error.status, error.request_id
        error_type, detail = _error_fields(error.body)
        if isinstance(error, TypeSafeRateLimitError) or status == 429:
            kind = "rate_limited"
            delay = getattr(error, "retry_after_ms", None)
            retry_after_ms = round(delay) if isinstance(delay, int | float) else None
        elif status == 401:
            kind = "authentication"
        elif status == 403:
            kind = "permission"
        elif (status == 400 and error_type == "max_tokens_exceeded") or status == 413:
            kind = "context_too_large"
        elif status == 400 and error_type == "api_usage_error" and "unknown model" in detail.lower():
            kind = "unknown_model"
        elif status in (400, 404, 409, 422):
            kind = "invalid_request"
        elif status == 408:
            kind = "timeout"
        elif status == 529:
            kind = "overloaded"
        elif status >= 500:
            kind = "server"
        else:
            kind = "unknown"
    else:
        message = str(error)
        kind = "authentication" if "API key" in message else "invalid_request"
        detail = message
    return ProviderError(
        kind,
        detail[:MAX_DETAIL_CHARS],
        provider=provider,
        retryable=kind in _RETRYABLE,
        status=status,
        request_id=request_id,
        retry_after_ms=retry_after_ms,
    )


def _questions(request: ProviderRequest) -> dict[str, Question]:
    return {predicate.id: cast("Question", wire_question(predicate)) for predicate in request.predicates}


def _attempts(http: httpx2.Response) -> int:
    try:
        retries = http.request.headers.get(RETRY_COUNT_HEADER)
    except RuntimeError:
        return 1
    try:
        return int(retries) + 1 if retries is not None else 1
    except ValueError:
        return 1


def _to_response(response: SystemOneResponse, latency_ms: float) -> ProviderResponse:
    http = response.raw_http_response
    try:
        body: Any = json.loads(http.content)
    except ValueError:
        raise ProviderError(
            "malformed_response", "response body is not JSON", provider=PROVIDER_NAME
        ) from None
    answers: Any = cast("dict[str, Any]", body).get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        raise ProviderError("malformed_response", "response has no answers object", provider=PROVIDER_NAME)
    raw_answers: dict[str, RawAnswer] = {
        str(key): value for key, value in cast("dict[str, Any]", answers).items()
    }
    return ProviderResponse(
        answers=raw_answers,
        model_resolved=response.model,
        request_id=http.headers.get(REQUEST_ID_HEADER),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=round(latency_ms, 3),
        attempts=_attempts(http),
    )


class TypeSafeProvider:
    """Evaluate predicates with TypeSafe's System One API (Jev).

    Clients are created lazily, one sync and one async, and reused. Pass prebuilt ``client`` or
    ``async_client`` objects to control credentials and transport yourself; otherwise the SDK reads
    ``TYPESAFE_API_KEY`` from the environment.
    """

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx2.BaseTransport | None = None,
        async_transport: httpx2.AsyncBaseTransport | None = None,
        client: TypeSafeClient | None = None,
        async_client: AsyncTypeSafeClient | None = None,
        limits: ProviderLimits | None = None,
    ) -> None:
        if client is None and async_client is None and not os.environ.get(API_KEY_ENV, "").strip():
            raise ProviderError(
                "authentication",
                f"{API_KEY_ENV} is not set; export it or pass a configured TypeSafe client",
                provider=PROVIDER_NAME,
            )
        self._retry = retry if retry is not None else DEFAULT_RETRY
        self._timeout = timeout
        self._base_url = base_url
        self._headers = dict(headers or {})
        self._transport = transport
        self._async_transport = async_transport
        self._client = client
        self._async_client = async_client
        self._limits = limits if limits is not None else ProviderLimits()

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    def __repr__(self) -> str:
        return "TypeSafeProvider()"

    def _sync_client(self) -> TypeSafeClient:
        if self._client is None:
            self._client = TypeSafeClient(
                retry=self._retry,
                timeout=self._timeout,
                base_url=self._base_url,
                headers=self._headers,
                transport=self._transport,
            )
        return self._client

    def _async_client_instance(self) -> AsyncTypeSafeClient:
        if self._async_client is None:
            self._async_client = AsyncTypeSafeClient(
                retry=self._retry,
                timeout=self._timeout,
                base_url=self._base_url,
                headers=self._headers,
                transport=self._async_transport,
            )
        return self._async_client

    def evaluate(self, request: ProviderRequest) -> ProviderResponse:
        questions = _questions(request)
        started = time.perf_counter()
        response: SystemOneResponse | None = None
        failure: ProviderError | None = None
        try:
            response = self._sync_client().system_one(
                request.state.value, questions, model=request.model, timeout=request.timeout
            )
        except TypeSafeError as error:
            failure = map_error(error)
        # Raised outside the handler so the SDK exception is not attached as context.
        if response is None:
            raise failure if failure is not None else ProviderError("unknown", provider=PROVIDER_NAME)
        return _to_response(response, (time.perf_counter() - started) * 1000)

    async def aevaluate(self, request: ProviderRequest) -> ProviderResponse:
        questions = _questions(request)
        started = time.perf_counter()
        response: SystemOneResponse | None = None
        failure: ProviderError | None = None
        try:
            response = await self._async_client_instance().system_one(
                request.state.value, questions, model=request.model, timeout=request.timeout
            )
        except TypeSafeError as error:
            failure = map_error(error)
        if response is None:
            raise failure if failure is not None else ProviderError("unknown", provider=PROVIDER_NAME)
        return _to_response(response, (time.perf_counter() - started) * 1000)

    def list_models(self) -> list[dict[str, str]]:
        """Models and aliases available to the account (TypeSafe lists aliases, not versioned ids)."""
        listing = None
        failure: ProviderError | None = None
        try:
            listing = self._sync_client().models.list()
        except TypeSafeError as error:
            failure = map_error(error)
        if listing is None:
            raise failure if failure is not None else ProviderError("unknown", provider=PROVIDER_NAME)
        return [
            {"name": model.name, "description": model.description, "release_date": model.release_date}
            for model in listing.models
        ]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        self.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

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
