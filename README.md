# Sentinel

**Sentinel is a semantic runtime that lets software use meaning as a programmable condition.**

Most software can only branch on structured facts: `latency_ms > 500`, `user.role != "admin"`. Many
of the conditions that matter are about meaning. Does this tool call match what the user asked for?
Is this answer supported by its sources? Is the agent drifting from its task? Sentinel turns those
questions into **semantic predicates**: named, fingerprinted, reviewable judgments that a fast,
calibrated model evaluates against your application state, returning typed probabilities that
ordinary code branches on.

```python
from sentinel_runtime import PredicateSet, ProviderError, Runtime

AGENT_ACTION = PredicateSet.from_file("predicates/agent_action.yaml")
runtime = Runtime.from_env(model="jev-1.13.0")

try:
    ev = runtime.evaluate(
        {
            "user_request": "Summarize my latest emails.",
            "proposed_action": {"tool": "send_email", "arguments": {"to": "boss@example.com"}},
        },
        AGENT_ACTION,
    )
except ProviderError:
    deny()  # the judgment is unavailable: this application fails closed
else:
    if ev.probability("external_side_effect") >= 0.5 and ev.probability("explicitly_authorized") < 0.5:
        ask_the_user_to_confirm()
```

Sentinel reports facts; your code decides. It never chooses what your software does, and a provider
failure is always an error, never a silent `false`.

## Try it

```bash
git clone https://github.com/SentinelTracing/Sentinel && cd Sentinel
uv sync

# The demos replay recorded responses, so no API key is needed.
uv run python examples/agent_action/main.py           # gate an agent's tool calls
uv run python examples/output_verification/main.py    # check answers against their source
uv run python examples/trace_semantics/main.py        # semantic telemetry over an agent trace

# Evaluate live (needs a TypeSafe key).
export TYPESAFE_API_KEY=...
uv run sentinel evaluate examples/agent_action/states/bad.json -p agent_action
```

`sentinel evaluate` prints each predicate's answer, the versioned model that answered, and the
request's token usage, latency and estimated cost. Add `--json` for the full typed evaluation.

## How it works

- **Predicates live in reviewable files.** Each is a `noul` (the probability a statement is true), a
  `choice` (a distribution over named options), or a `score` (a distribution over ordered, described
  levels). Every predicate has a fingerprint over exactly what the model sees, so an edited question
  is always distinguishable from an edited threshold.
- **One request per state.** Every predicate that judges the same state is sent in a single
  request. Cost scales with the state, not the number of predicates: twelve questions about a
  3,100-character document cost 9.5x less and returned 11x sooner than twelve separate requests,
  with the same answers ([benchmarks](docs/benchmarks.md)).
- **Typed results.** Probabilities, distributions and confidence, plus the resolved model version,
  token usage, latency and request id. One telemetry event per evaluation.
- **Measured, not trusted.** An evaluation harness runs predicates against labeled cases and reports
  precision, recall, threshold sweeps, calibration and every miss. CI replays the suites on every
  change and fails when a predicate regresses.

The first backend is [TypeSafe](https://docs.typesafe.ai)'s Jev, a System One model that evaluates
many typed questions against one state in parallel and returns calibrated probabilities. Any model
that can return those answers can sit behind the same provider interface.

## Predicates

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
    threshold: 0.5
```

`sentinel predicates validate <file> --state example.json` checks a file and warns about wording
the model handles poorly: arithmetic, date comparison, negated questions, vague score levels, and
paths that do not exist in your state. See [docs/predicates.md](docs/predicates.md).

## Measure before you trust

```bash
uv run sentinel eval run agent_action                 # 157 labeled cases, replayed from the committed cache
uv run sentinel eval run agent_action --details --markdown report.md
```

| Suite | Cases | Result (jev-1.13.0, 2026-09-18) |
| --- | --- | --- |
| `agent_action` | 157 | noul F1 0.97 to 0.99, ROC-AUC 1.00; `effect_class` accuracy 1.00; severity ranges 0.96 |
| `output_verification` | 24 | all four nouls F1 1.00 |
| `trace_semantics` | 22 | nouls F1 1.00; progress level accuracy 1.00; failure type accuracy 0.90 |

The suites include prompt injection, negated requests, wrong targets and genuinely ambiguous cases.
The first run changed the reference set: the illustrative 0.8 thresholds missed a quarter of true
cases, and two predicates needed their boundaries spelled out. See [docs/evals.md](docs/evals.md).

## Reference predicate sets

| Set | Judges | Demo |
| --- | --- | --- |
| `agent_action` | Does a proposed tool call follow from the request, reach outside systems, have explicit authorization; what kind of effect; how severe if unintended | `examples/agent_action`: allow, confirm or deny each call |
| `output_verification` | Does an answer address the request, stay supported by the source, contradict it, cover every part | `examples/output_verification`: accept, regenerate or escalate |
| `trace_semantics` | Does an agent step serve the goal, drift out of scope, repeat work; progress; failure type | `examples/trace_semantics`: per-step telemetry with events |

## Command line

| Command | Does |
| --- | --- |
| `sentinel evaluate STATE -p SET` | Evaluate a predicate set against a state (`--json`, `--events`, `--cache`) |
| `sentinel predicates validate SET` | Validate a predicate file and report authoring warnings |
| `sentinel predicates show SET` | List predicates, fingerprints and default thresholds |
| `sentinel eval run SUITE` | Run a labeled suite; `--fail-on-regression` gates against the baseline |
| `sentinel eval compare A B` | Compare two reports (for example across model versions) |
| `sentinel eval accept REPORT --suite S` | Make a reviewed report the suite's baseline |
| `sentinel eval stability SUITE` | Measure how much answers move across repeated evaluations |
| `sentinel bench SET STATE` | Measure batching against one request per predicate |
| `sentinel models --probe` | List models and what each alias resolves to now |
| `sentinel schema NAME` | Print the JSON Schema for a file format or any command's `--json` output |

Exit codes: 0 success, 1 invalid input, 2 provider error, 3 regression, 4 missing cached responses.

## Security

Semantic checks **supplement** identity and permission checks; they never replace them. A model can
be wrong and can be influenced by adversarial content in the state. Use predicates alongside
deterministic controls, explicit confirmation for high-stakes actions, and ordinary authorization.
The API key is read only from `TYPESAFE_API_KEY`, and errors, events and caches never contain it or
your state. See [docs/security.md](docs/security.md).

## Documentation

- [Writing predicates](docs/predicates.md)
- [Measuring predicates](docs/evals.md)
- [Architecture](docs/architecture.md)
- [Telemetry](docs/telemetry.md)
- [Benchmarks](docs/benchmarks.md)
- [Security](docs/security.md)

## Install

Sentinel requires Python 3.11 or newer and is not yet published; install it from a clone with
`uv sync` or `pip install .`. The package is `sentinel-runtime`, the import is `sentinel_runtime`,
and the command is `sentinel`. HashiCorp also ships a `sentinel` binary for its policy language; if
both are installed, the one earlier on your `PATH` wins.

## Roadmap

1. Core schema: predicates, state, results (done)
2. Provider layer: TypeSafe Jev adapter and response cache (done)
3. Runtime: batching planner, typed evaluation, telemetry events (done)
4. CLI and the agent tool-call verification demo (done)
5. Evaluation and calibration harness with a labeled fixture suite (done)
6. Benchmarks, output verification and trace demos, documentation (done)
7. Release 0.1.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
