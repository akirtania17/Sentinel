# Architecture

Sentinel evaluates semantic predicates against application state and returns typed results. It
never decides what the application does.

```text
             application code (owns control flow and policy)
                                 |
             state + PredicateSet (YAML file or Python objects)
                                 |
                                 v
 +-------------------------------------------------------------------------+
 | Runtime                                                                 |
 |  1 validate     predicate set, ids, options, levels                    |
 |  2 canonicalize State: plain-JSON deep copy, optional redaction,       |
 |                 sha256 digest, size estimate                            |
 |  3 plan         one request unless the token budget forces a split;    |
 |                 oversized states rejected before any network call      |
 |  4 call         Provider: TypeSafeProvider (typesafe-sdk)               |
 |                          CachingProvider(inner, file, mode)             |
 |                          FakeProvider (tests)                           |
 |  5 normalize    raw answers -> NoulResult | ChoiceResult | ScoreResult  |
 |  6 emit         one EvaluationEvent per evaluation -> sinks             |
 +-------------------------------------------------------------------------+
                                 |
                                 v
        Evaluation: results, model_resolved, usage, latency, fingerprints
                                 |
                                 v
          application policy: allow / confirm / deny / retry / escalate

 Beside the runtime:
   sentinel CLI      evaluate, predicates, eval, bench, models, schema
   evaluation suites fixtures -> Runtime (with CachingProvider) -> metrics, sweeps,
                     calibration, baseline comparison, stability
```

## Modules

| Module | Responsibility |
| --- | --- |
| `predicates` | Predicate kinds, predicate sets, the `sentinel/predicates/v1` file format, fingerprints |
| `state` | Canonical, immutable-by-convention state with digests and size estimates |
| `planner` | Packing predicates into requests under the provider's token budget |
| `providers.base` | The `Provider` protocol and request/response types |
| `providers.typesafe` | The Jev adapter over `typesafe-sdk`, including error mapping and redaction |
| `providers.caching` | Response cache used by tests, examples and evaluation suites |
| `normalize` | Validating raw answers against their predicates and building typed results |
| `runtime` | Orchestration, partial results, model precedence, telemetry |
| `telemetry` | `EvaluationEvent` v1 and sinks |
| `evals` | Fixtures, runner, metrics, reports, comparison, stability |
| `lint` | Authoring warnings for predicates |
| `cli` | The `sentinel` command |

## Boundaries

- **Provider answer, typed result, decision.** A provider returns raw answers in the System One
  format. The runtime validates them and returns typed results. A `decision` field appears only
  when a predicate declares a default threshold, and it is a convenience; application policy decides
  what happens.
- **Errors are never results.** Every failure to obtain a judgment is a `ProviderError` with a
  `kind` (`authentication`, `rate_limited`, `timeout`, `context_too_large`, `malformed_response`,
  and so on), so callers choose to fail open or fail closed. When a split plan fails part-way, the
  error carries the successful results in `partial`.
- **State is read-only.** The runtime converts the caller's state to a plain-JSON copy before
  redacting, hashing and sending it.

## Batching and splitting

All predicates for one state go in one request by default. The planner estimates tokens
conservatively (3.2 characters per token plus a fixed overhead per question and per request) and
packs against 85% of the provider's limits. It splits only when the estimate exceeds the request
budget, and it rejects a state that cannot fit with its longest question before sending anything.
If the provider still rejects a request as too large, the runtime halves the chunk and retries.
Every call records its estimate next to the actual token count.

A judgment that depends on another judgment's answer is not a planner concern: it is a second
evaluation, with new state, made by application code.

## Fingerprints and versions

A predicate's fingerprint is `sha256` over its model-facing content: kind, question, and criteria,
options (in order) or levels. Thresholds, tags and descriptions do not change it. The set
fingerprint covers every `(id, fingerprint)` pair. Evaluations, events, cache keys and evaluation
baselines all carry fingerprints, so an edited question can never be compared silently against
answers to the old wording.

Every response records the versioned model that answered (`model_resolved`), even when an alias such
as `jev-latest` was requested. Model precedence, highest first: the per-call `model`,
`Runtime(model=...)`, `SENTINEL_MODEL`, `TYPESAFE_DEFAULT_MODEL`, `[runtime] model` in
`sentinel.toml`, then `jev-latest`.

## Adding a provider

Implement the `Provider` protocol: a `name`, `limits`, `evaluate` and `aevaluate` that take a
`ProviderRequest` and return raw answers in the System One answer format (`{"type": "noul",
"noul": 0.93}`, and the choice and score equivalents), plus `close` and `aclose`. Normalization,
planning, caching, telemetry and evaluation work unchanged. The evaluation harness compares
providers the same way it compares model versions.
