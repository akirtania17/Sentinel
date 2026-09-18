"""Evaluation events.

Every call to ``Runtime.evaluate`` emits exactly one ``EvaluationEvent`` to the configured sinks:
``outcome: ok`` with the full evaluation, or ``outcome: error`` with the sanitized error and any
partial results. Events carry the state digest and size, never the state itself, unless the runtime
is created with ``include_state_in_events=True``.

The field names map onto OpenTelemetry-style attributes (``semantic.predicate.id``,
``semantic.predicate.fingerprint``, ``semantic.result.probability``, ``semantic.provider``,
``semantic.model``, ``semantic.latency_ms``, ``semantic.input_tokens``); an exporter can be added
later without changing the schema.

A sink that raises never breaks an evaluation: the runtime logs the failure and continues.
"""

import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from sentinel_runtime.errors import ErrorInfo
from sentinel_runtime.results import Evaluation

EVENT_NAME = "sentinel.evaluation"
EVENT_SCHEMA_VERSION = 1


class EvaluationEvent(BaseModel):
    """One evaluation, successful or not (schema version 1)."""

    model_config = ConfigDict(frozen=True, extra="forbid", title="Sentinel evaluation event (v1)")

    event: Literal["sentinel.evaluation"] = "sentinel.evaluation"
    schema_version: Literal[1] = 1
    evaluation_id: str
    ts: datetime
    outcome: Literal["ok", "error"]
    predicate_set: str
    predicate_set_fingerprint: str
    predicate_fingerprints: dict[str, str]
    state_digest: str
    state_chars: int
    state: Any = None
    """The state as sent, only when the runtime opts in with ``include_state_in_events``."""
    provider: str
    model_requested: str
    model_resolved: str | None
    """The versioned model that answered; ``None`` when no request succeeded."""
    latency_ms: float
    evaluation: Evaluation | None = None
    """The full evaluation on success; the partial evaluation on a failure after some requests succeeded."""
    error: ErrorInfo | None = None

    def to_json(self) -> str:
        """Compact JSON for a log line."""
        return self.model_dump_json(exclude_none=False)


@runtime_checkable
class TelemetrySink(Protocol):
    """Receives evaluation events."""

    def emit(self, event: EvaluationEvent) -> None: ...


class JsonlSink:
    """Append one JSON line per event to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: EvaluationEvent) -> None:
        line = event.to_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def __repr__(self) -> str:
        return f"JsonlSink({str(self.path)!r})"


class LoggingSink:
    """Log each event as one JSON message through the standard ``logging`` module."""

    def __init__(self, logger: logging.Logger | None = None, *, level: int = logging.INFO) -> None:
        self.logger = logger if logger is not None else logging.getLogger("sentinel_runtime.events")
        self.level = level

    def emit(self, event: EvaluationEvent) -> None:
        if self.logger.isEnabledFor(self.level):
            self.logger.log(self.level, "%s", event.to_json())


class CallbackSink:
    """Call a function with each event."""

    def __init__(self, callback: Callable[[EvaluationEvent], None]) -> None:
        self.callback = callback

    def emit(self, event: EvaluationEvent) -> None:
        self.callback(event)


def read_events(path: str | Path) -> list[EvaluationEvent]:
    """Load events written by ``JsonlSink``."""
    events: list[EvaluationEvent] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(EvaluationEvent.model_validate(json.loads(line)))
    return events
