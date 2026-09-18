"""A response cache in front of any provider.

``CachingProvider`` stores raw provider answers in a JSON Lines file so evaluations can be replayed
offline: in tests, in examples, and in CI runs of the evaluation suite. It is not a production
cache. Each line holds one provider response::

    {"key": "sha256:...", "provider": "typesafe", "model_requested": "jev-1.13.0",
     "model_resolved": "jev-1.13.0", "recorded_at": "2026-09-18T03:55:51Z", "request_id": "req_...",
     "input_tokens": 681, "output_tokens": 129, "latency_ms": 403.0, "answers": {...}}

The key covers the provider, the requested model, the state digest, and every ``(id, fingerprint)``
pair, so editing a question, renaming a predicate or changing the state always misses. The state
itself is never stored.

Modes:

- ``replay``: serve hits; on a miss call the inner provider and record the answer.
- ``offline``: serve hits; a miss raises ``CacheMissError``. No inner provider is needed.
- ``record``: always call the inner provider and overwrite the entry.
"""

import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from sentinel_runtime.errors import CacheMissError, ConfigError
from sentinel_runtime.hashing import digest_of
from sentinel_runtime.providers.base import Provider, ProviderLimits, ProviderRequest, ProviderResponse

CacheMode = Literal["replay", "record", "offline"]


class CacheRecord(BaseModel):
    """One cached provider response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    provider: str
    model_requested: str
    model_resolved: str
    recorded_at: str
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    answers: dict[str, dict[str, Any]]

    def response(self) -> ProviderResponse:
        return ProviderResponse(
            answers=self.answers,
            model_resolved=self.model_resolved,
            request_id=self.request_id,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=self.latency_ms,
            attempts=1,
            replayed=True,
        )


def cache_key(provider: str, request: ProviderRequest) -> str:
    """The cache key for ``request`` sent to ``provider``."""
    return digest_of(
        {
            "provider": provider,
            "model_requested": request.model,
            "state_digest": request.state.digest,
            "questions": sorted([predicate.id, predicate.fingerprint] for predicate in request.predicates),
        }
    )


def _load(path: Path) -> dict[str, CacheRecord]:
    records: dict[str, CacheRecord] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = CacheRecord.model_validate_json(line)
            except ValidationError as error:
                raise ConfigError(
                    f"{path}:{number}: invalid cache record: {error.errors()[0]['msg']}"
                ) from None
            records[record.key] = record
    return records


def _line(record: CacheRecord) -> str:
    return json.dumps(
        record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


class CachingProvider:
    """Serve and record provider responses from a JSON Lines file."""

    def __init__(
        self,
        inner: Provider | None,
        path: str | Path,
        *,
        mode: CacheMode = "replay",
        name: str | None = None,
        limits: ProviderLimits | None = None,
    ) -> None:
        if inner is None and mode != "offline":
            raise ValueError(f"cache mode {mode!r} needs an inner provider; only 'offline' works without one")
        self._inner = inner
        self._path = Path(path)
        self._mode: CacheMode = mode
        self._name = name if name is not None else (inner.name if inner is not None else "typesafe")
        self._limits = (
            limits if limits is not None else (inner.limits if inner is not None else ProviderLimits())
        )
        self._records = _load(self._path)
        self._lock = threading.Lock()
        self._dirty = False
        self.hits = 0
        self.misses = 0
        self.recorded = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    @property
    def inner(self) -> Provider | None:
        """The wrapped live provider, or ``None`` for an offline cache."""
        return self._inner

    @property
    def mode(self) -> CacheMode:
        return self._mode

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self) -> str:
        return f"CachingProvider({self._inner!r}, {str(self._path)!r}, mode={self._mode!r})"

    def __len__(self) -> int:
        return len(self._records)

    def models_resolved(self) -> set[str]:
        """Every resolved model recorded in the cache."""
        return {record.model_resolved for record in self._records.values()}

    def _lookup(self, request: ProviderRequest) -> tuple[str, ProviderResponse | None]:
        key = cache_key(self._name, request)
        if self._mode != "record":
            record = self._records.get(key)
            if record is not None:
                with self._lock:
                    self.hits += 1
                return key, record.response()
        with self._lock:
            self.misses += 1
        if self._mode == "offline":
            raise CacheMissError([key])
        return key, None

    def _store(self, key: str, request: ProviderRequest, response: ProviderResponse) -> None:
        record = CacheRecord(
            key=key,
            provider=self._name,
            model_requested=request.model,
            model_resolved=response.model_resolved,
            recorded_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            request_id=response.request_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            answers={predicate_id: dict(answer) for predicate_id, answer in response.answers.items()},
        )
        with self._lock:
            self._records[key] = record
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(_line(record) + "\n")
            self._dirty = True
            self.recorded += 1

    def evaluate(self, request: ProviderRequest) -> ProviderResponse:
        key, cached = self._lookup(request)
        if cached is not None:
            return cached
        assert self._inner is not None
        response = self._inner.evaluate(request)
        self._store(key, request, response)
        return response

    async def aevaluate(self, request: ProviderRequest) -> ProviderResponse:
        key, cached = self._lookup(request)
        if cached is not None:
            return cached
        assert self._inner is not None
        response = await self._inner.aevaluate(request)
        self._store(key, request, response)
        return response

    def compact(self) -> None:
        """Rewrite the cache file with one line per key, sorted by key, for stable diffs."""
        with self._lock:
            lines = [_line(self._records[key]) for key in sorted(self._records)]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.writelines(line + "\n" for line in lines)
                Path(temporary).replace(self._path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            self._dirty = False

    def close(self) -> None:
        """Compact the file if anything was recorded, then close the inner provider."""
        if self._dirty:
            self.compact()
        if self._inner is not None:
            self._inner.close()

    async def aclose(self) -> None:
        if self._dirty:
            self.compact()
        if self._inner is not None:
            await self._inner.aclose()
