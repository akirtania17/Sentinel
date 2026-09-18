"""Sentinel's exception hierarchy.

A provider failure is never a semantic result. Every failure to obtain a judgment surfaces as a
``ProviderError`` with a mapped ``kind``, so callers can choose to fail open or fail closed.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from sentinel_runtime.results import Evaluation

ProviderErrorKind = Literal[
    "authentication",
    "permission",
    "invalid_request",
    "unknown_model",
    "context_too_large",
    "rate_limited",
    "overloaded",
    "timeout",
    "connection",
    "server",
    "malformed_response",
    "cancelled",
    "unknown",
]

StateErrorReason = Literal["empty", "not_serializable", "too_large"]

MAX_DETAIL_CHARS = 500


class SentinelError(Exception):
    """Base class for every error Sentinel raises."""


class PredicateValidationError(SentinelError):
    """A predicate, predicate set, or predicate file is invalid."""

    def __init__(self, message: str, *, path: str = "") -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        """Dotted location of the problem, such as ``predicates.effect_class.options``, or a file path."""
        self.message = message


class FixtureValidationError(SentinelError):
    """An evaluation fixture file is invalid."""

    def __init__(self, message: str, *, path: str = "") -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


class StateError(SentinelError):
    """The supplied state cannot be evaluated."""

    def __init__(self, reason: StateErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason: StateErrorReason = reason
        self.message = message


class ConfigError(SentinelError):
    """Configuration or path resolution failed."""


class CacheMissError(SentinelError):
    """An offline cache lookup missed; the listed keys must be recorded with a live provider."""

    def __init__(self, keys: list[str], case_ids: list[str] | None = None) -> None:
        self.keys = keys
        self.case_ids = case_ids or []
        labels = self.case_ids or keys
        shown = ", ".join(labels[:10])
        suffix = f" and {len(labels) - 10} more" if len(labels) > 10 else ""
        super().__init__(
            f"{len(keys)} evaluation(s) are not in the cache ({shown}{suffix}). To record them, run again "
            "with TYPESAFE_API_KEY set and without offline mode, then commit the cache."
        )


class ErrorInfo(BaseModel):
    """The sanitized, serializable description of a provider error used in events and reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ProviderErrorKind
    retryable: bool
    status: int | None = None
    request_id: str | None = None
    detail: str = ""


class ProviderError(SentinelError):
    """A provider could not produce the requested judgments.

    ``detail`` is sanitized: it carries only the provider's error type and message, truncated, and
    never the request body, state, or headers. Adapters raise it without exception chaining so the
    underlying HTTP objects (which carry credentials) are not reachable from the raised error.
    """

    def __init__(
        self,
        kind: ProviderErrorKind,
        detail: str = "",
        *,
        provider: str,
        retryable: bool = False,
        status: int | None = None,
        request_id: str | None = None,
        retry_after_ms: int | None = None,
        estimated_tokens: int | None = None,
        partial: "Evaluation | None" = None,
    ) -> None:
        self.kind: ProviderErrorKind = kind
        self.detail = detail[:MAX_DETAIL_CHARS]
        self.provider = provider
        self.retryable = retryable
        self.status = status
        self.request_id = request_id
        self.retry_after_ms = retry_after_ms
        self.estimated_tokens = estimated_tokens
        self.partial = partial
        super().__init__(self._message())

    def _message(self) -> str:
        parts = [f"{self.provider} {self.kind}"]
        if self.status is not None:
            parts.append(f"(HTTP {self.status})")
        message = " ".join(parts)
        if self.detail:
            message += f": {self.detail}"
        if self.request_id:
            message += f" [request_id={self.request_id}]"
        return message

    def info(self) -> ErrorInfo:
        """Return the serializable form of this error."""
        return ErrorInfo(
            kind=self.kind,
            retryable=self.retryable,
            status=self.status,
            request_id=self.request_id,
            detail=self.detail,
        )

    def replace(self, **changes: Any) -> "ProviderError":
        """A copy of this error with ``changes`` applied, for example ``partial=...``."""
        fields: dict[str, Any] = {
            "provider": self.provider,
            "retryable": self.retryable,
            "status": self.status,
            "request_id": self.request_id,
            "retry_after_ms": self.retry_after_ms,
            "estimated_tokens": self.estimated_tokens,
            "partial": self.partial,
        }
        kind = changes.pop("kind", self.kind)
        detail = changes.pop("detail", self.detail)
        fields.update(changes)
        return ProviderError(kind, detail, **fields)
