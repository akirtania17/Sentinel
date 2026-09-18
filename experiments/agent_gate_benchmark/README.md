# Agent gate benchmark

Scenarios, labels and tools for stress-testing gates that decide whether an AI agent's proposed
action should proceed. Results and conclusions are in [docs/stress-test.md](../../docs/stress-test.md).

| File | Purpose |
| --- | --- |
| `SCENARIOS.md` | The scenario format, labeling rules and authoring guide |
| `scenarios_dev.yaml`, `scenarios_test.yaml` | Development and held-out sets as their authors wrote them |
| `audit_dev.yaml`, `audit_test.yaml` | Blind labels from an independent auditor |
| `scenarios_*_audited.yaml` | The sets used for scoring: auditor disagreements marked ambiguous |
| `scenarios.py` | Loader and validator |
| `audit.py` | Strips labels for an auditor and compares the two sets of labels |
| `bench.py` | Runs gates over a set and reports by category; `--repeats` measures stability |
| `cascade.py` | Sentinel first, an LLM judge for uncertain cases, from stored decisions |
| `pad.py` | Adds irrelevant tool output to every scenario |
| `llamafirewall.py` | Runs Sentinel on Meta's LlamaFirewall AlignmentCheck evaluation set |
| `attack.py` | Harness for adaptive attacks that rewrite tool output |
| `candidates/` | Intermediate predicate sets tried on the development set |

The shipped predicates are `examples/claude_code/intent.yaml` and `context.yaml`, with the
decision rule `decide_for_coding_agent` in `examples/claude_code/hook.py` (`focused3` in
`bench.py` makes the same decisions; a test checks that).

```bash
B=experiments/agent_gate_benchmark
python3 $B/scenarios.py $B/scenarios_test_audited.yaml
uv run python $B/bench.py $B/scenarios_test_audited.yaml --out results --gates allow_all,readonly_allowlist,sentinel \
  --predicates examples/claude_code/context.yaml --split-intent examples/claude_code/intent.yaml --policy focused3
uv run python $B/bench.py $B/scenarios_test_audited.yaml --out results --gates llm_haiku,llm_sonnet   # needs the claude CLI
```

The Sentinel gates need `TYPESAFE_API_KEY`. The LLM judges run through the `claude` CLI and are
billed as Claude usage. Attack transcripts are not published; `docs/stress-test.md` describes the
strategies.
