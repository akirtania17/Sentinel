# Agent field test

Real Claude Code agents work on real tasks in a sandbox project while the Sentinel hook judges every
tool call. Results and conclusions are in [docs/field-test.md](../../docs/field-test.md).

| File | Purpose |
| --- | --- |
| `sandbox/` | The project the agents work in; its team tools only log. `dot-env` becomes `.env` per run |
| `tasks.yaml` | Tasks, the actions each request authorizes, and the agent setups |
| `run.py` | Runs every setup and task as a headless Claude Code session with the hook |
| `labels.yaml` | Manual labels for the calls the rules send to review |
| `analyze.py` | Labels calls, scores policies, and reports outcomes, cost and latency |
| `replay.py` | Re-judges every recorded call with and without context |
| `counterfactual.py` | Places harmful actions into real sessions and measures detection |

```bash
# Needs the claude CLI and a file that exports TYPESAFE_API_KEY (the agents never see it).
export SENTINEL_KEY_FILE=~/typesafe.env
uv run python experiments/agent_field_test/run.py --out results
uv run python experiments/agent_field_test/analyze.py results --review     # label what the rules cannot
(. "$SENTINEL_KEY_FILE" && uv run python experiments/agent_field_test/replay.py results)
(. "$SENTINEL_KEY_FILE" && uv run python experiments/agent_field_test/counterfactual.py results)
uv run python experiments/agent_field_test/analyze.py results --replay results/replay.jsonl
```

A full run is 187 agent sessions (billed as Claude Code usage) and about $0.40 of TypeSafe usage.
Labels in `labels.yaml` are keyed by tool call ids, so a new run needs its own review.
