# Sentinel

**Sentinel is a semantic runtime that lets software use meaning as a programmable condition.**

Most software can only branch on structured facts: `latency_ms > 500`, `user.role != "admin"`. Many
of the conditions that matter are about meaning instead. Does this tool call match what the user
asked for? Is this answer supported by the context? Is the agent drifting from its task? Sentinel
turns those questions into **semantic predicates**: named, versioned, reviewable judgments that a
fast calibrated model evaluates against your application state, returning typed probabilities your
code branches on.

```python
from sentinel_runtime import PredicateSet, Runtime

AGENT_ACTION = PredicateSet.from_file("predicates/agent_action.yaml")
runtime = Runtime.from_env(model="jev-1.13.0")

ev = runtime.evaluate(
    {
        "user_request": "Summarize my latest emails.",
        "proposed_action": {"tool": "send_email", "arguments": {"to": "boss@example.com"}},
    },
    AGENT_ACTION,
)

if ev.probability("external_side_effect") >= 0.8 and ev.probability("explicitly_authorized") < 0.8:
    ask_the_user_to_confirm()
```

> Status: early development, not yet released. See [Roadmap](#roadmap).

## Try it

```bash
uv sync
uv run python examples/agent_action/main.py      # replays recorded responses; no API key needed

export TYPESAFE_API_KEY=...                        # then evaluate live
uv run sentinel evaluate examples/agent_action/states/bad.json -p agent_action
uv run sentinel predicates validate agent_action --state examples/agent_action/states/bad.json
```

`sentinel evaluate` prints each predicate's result, the versioned model that answered, and the
request's token usage, latency and estimated cost. Add `--json` for the full typed evaluation.

## How it works

- **Predicates** live in YAML files you review like code. Each is a `noul` (probability that a
  statement is true), a `choice` (distribution over named options), or a `score` (distribution over
  ordered levels).
- **One request per state.** Every predicate that judges the same state is sent to the model in a
  single request. Cost scales with the state, not with the number of predicates.
- **Typed results, not prose.** You get probabilities, distributions, and confidence, plus the
  resolved model version, token usage, latency, and a fingerprint of every predicate.
- **Code owns control flow.** Sentinel never decides what your software does. Provider failures are
  errors, never a silent `false`, so you choose to fail open or fail closed.
- **Measure before you trust.** An evaluation harness runs predicates against labeled fixtures and
  reports precision, recall, threshold curves, and calibration, so thresholds are empirical.

The first backend is [TypeSafe](https://docs.typesafe.ai)'s Jev, a System One model that evaluates
many typed questions against one state in parallel and returns calibrated probabilities.

## Predicate files

```yaml
format: sentinel/predicates/v1
name: agent_action
predicates:
  explicitly_authorized:
    kind: noul
    question: >-
      Does `user_request` explicitly or clearly authorize this specific `proposed_action`,
      including its target?
    criteria:
      true: The request names or clearly implies this exact action and target.
      false: The request does not mention or imply this action, or its target differs.
    threshold: 0.8
```

Every predicate has a fingerprint over exactly what the model sees, so a changed question is always
distinguishable from a changed threshold in logs and evaluation reports.

See [docs/predicates.md](docs/predicates.md) for how to write predicates that calibrate well.

## Measure before you trust

```bash
uv run sentinel eval run agent_action          # 157 labeled cases; replays the committed cache
uv run sentinel eval run agent_action --details --markdown report.md
```

The report gives precision, recall and F1 at each predicate's threshold, a threshold sweep, ROC-AUC,
calibration, per-tag slices and every miss. CI replays the suite on every change and fails when a
predicate regresses against its accepted baseline. See [docs/evals.md](docs/evals.md) for the
workflow and the current results.

## Security posture

Semantic checks **supplement** identity and permission checks; they never replace them. A model can
be wrong and can be influenced by adversarial content. Use Sentinel predicates alongside
deterministic controls, explicit confirmation, and traditional authorization.

## Install

```bash
pip install sentinel-runtime     # not yet published
```

The CLI is named `sentinel`. HashiCorp also ships a `sentinel` binary for its policy language; if
both are installed, the one earlier on your `PATH` wins.

## Roadmap

1. Core schema: predicates, state, results (done)
2. Provider layer: TypeSafe Jev adapter and response cache (done)
3. Runtime: batching planner, typed evaluation, telemetry events (done)
4. CLI and the agent tool-call verification demo (done)
5. Evaluation and calibration harness with a labeled fixture suite (done)
6. Benchmarks, output-verification and trace demos, documentation
7. Release 0.1.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
