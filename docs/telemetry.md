# Telemetry

Every call to `Runtime.evaluate` or `aevaluate` emits exactly one `EvaluationEvent` to the runtime's
sinks: `outcome: ok` with the full evaluation, or `outcome: error` with the sanitized error and any
partial results. Events never contain the state unless the runtime is created with
`include_state_in_events=True`; they carry its digest and size instead.

```python
from sentinel_runtime import JsonlSink, LoggingSink, CallbackSink, Runtime

runtime = Runtime.from_env(sinks=[JsonlSink(".sentinel/events.jsonl"), LoggingSink()])
```

| Sink | Behavior |
| --- | --- |
| `JsonlSink(path)` | Appends one JSON line per event; creates the directory if needed |
| `LoggingSink(logger=None, level=INFO)` | Logs each event as one JSON message on `sentinel_runtime.events` |
| `CallbackSink(fn)` | Calls `fn(event)` |

A sink that raises never breaks an evaluation: the runtime logs a warning on the `sentinel_runtime`
logger and continues. `sentinel evaluate --events FILE` appends the event from the command line,
and `sentinel_runtime.telemetry.read_events(path)` loads a JSONL file back into typed events.

## Event schema (version 1)

`sentinel schema event` prints the JSON Schema. A trimmed real event:

```json
{
  "event": "sentinel.evaluation",
  "schema_version": 1,
  "evaluation_id": "ev_a82731b03977458aa4c94344128a3e19",
  "ts": "2026-09-18T05:41:41.333075Z",
  "outcome": "ok",
  "predicate_set": "agent_action",
  "predicate_set_fingerprint": "sha256:98b509ce32c17cfe...",
  "predicate_fingerprints": {"action_matches_intent": "sha256:269cd9354d6b38fa...", "...": "..."},
  "state_digest": "sha256:cb32be48358a49e3...",
  "state_chars": 229,
  "state": null,
  "provider": "typesafe",
  "model_requested": "jev-1.13.0",
  "model_resolved": "jev-1.13.0",
  "latency_ms": 463.773,
  "evaluation": {
    "results": {
      "action_matches_intent": {"kind": "noul", "probability": 0.02, "threshold": 0.5, "decision": false}
    },
    "calls": [
      {
        "request_id": "req_01a0b308ec767d39acfb1817194fed43",
        "input_tokens": 897,
        "output_tokens": 141,
        "estimated_input_tokens": 1160,
        "latency_ms": 463.106,
        "predicate_ids": ["action_matches_intent", "external_side_effect", "explicitly_authorized", "effect_class", "severity"],
        "attempts": 1,
        "replayed": false
      }
    ],
    "usage": {"requests": 1, "input_tokens": 897, "output_tokens": 141, "estimated_cost_usd": 3.7674e-05}
  },
  "error": null
}
```

On failure, `error` holds `{kind, retryable, status, request_id, detail}`. The detail is sanitized:
it carries only the provider's error type and message, never the request body, the state, or
headers. `evaluation` is `null` unless some requests of a split plan succeeded.

## Mapping to OpenTelemetry

Events are designed to become span events or log records without changing the schema. Suggested
attribute names:

| Attribute | Source |
| --- | --- |
| `semantic.predicate_set` | `predicate_set` |
| `semantic.predicate.id` | each key of `evaluation.results` |
| `semantic.predicate.fingerprint` | `predicate_fingerprints[id]` |
| `semantic.result.probability` | a noul result's `probability` |
| `semantic.result.choice` | a choice result's `choice` |
| `semantic.provider` | `provider` |
| `semantic.model` | `model_resolved` |
| `semantic.latency_ms` | `latency_ms` |
| `semantic.input_tokens` | `evaluation.usage.input_tokens` |
| `semantic.outcome` | `outcome` |

An exporter is not part of 0.1; a `CallbackSink` that records these attributes on the current span
is a few lines of code.
