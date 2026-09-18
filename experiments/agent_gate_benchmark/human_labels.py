"""Human labels on a sample of the held-out scenarios, to check the labels Claude wrote.

Every label in the benchmark so far comes from Claude subagents (authors and blind auditors). This
draws a stratified, shuffled sample, shows it to a person without any label or gate decision, and
compares their answers with the authors' labels and with each gate.

    uv run python experiments/agent_gate_benchmark/human_labels.py sample SCENARIOS.yaml OUT_DIR [--per-category 3]
    uv run python experiments/agent_gate_benchmark/human_labels.py compare OUT_DIR ANSWERS.json RESULTS_DIR

`sample` writes `items.json` (what the labeling page shows: opaque ids, no scenario or action names,
which hint at the answer) and `key.json` (id to scenario action; keep it away from the labeler).
ANSWERS.json maps item id to {"answer": "allow" | "stop" | "unsure", "note": ...}. RESULTS_DIR holds
bench.py result files for the gates to compare.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "examples" / "claude_code"))

from scenarios import Case, load  # noqa: E402

RESULT_CHARS = 1500


def shown(case: Case) -> dict[str, Any]:
    """The moment as the labeler sees it."""
    state = case.state
    steps = [
        {
            "tool": step.get("tool"),
            "arguments": step.get("arguments"),
            "result": str(step.get("result", ""))[:RESULT_CHARS],
        }
        for step in state.get("recent_steps", [])
    ]
    return {
        "domain": case.scenario.get("domain", ""),
        "conversation": state.get("conversation", []),
        "user_request": state["user_request"],
        "project_instructions": state.get("project_instructions", ""),
        "recent_steps": steps,
        "proposed_action": state["proposed_action"],
    }


def command_sample(scenarios: Path, out: Path, per_category: int, ambiguous: int) -> None:
    cases = load(scenarios)
    rng = random.Random(20260918)  # noqa: S311 - a reproducible sample, not a secret
    by_category: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        by_category["ambiguous" if case.expected is None else case.category].append(case)
    chosen: list[Case] = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda c: c.key)
        chosen += rng.sample(pool, min(len(pool), ambiguous if category == "ambiguous" else per_category))
    rng.shuffle(chosen)
    items = [{"id": f"h{i + 1:02d}", **shown(case)} for i, case in enumerate(chosen)]
    key = {
        f"h{i + 1:02d}": {"key": case.key, "category": case.category, "expected": case.expected}
        for i, case in enumerate(chosen)
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(items)} items: {dict(Counter(v['category'] for v in key.values()))}")


def command_compare(out: Path, answers_path: Path, results: Path) -> None:
    key: dict[str, dict[str, Any]] = json.loads((out / "key.json").read_text())
    answers: dict[str, dict[str, Any]] = json.loads(answers_path.read_text())
    gates: dict[str, dict[str, str]] = {}
    for path in sorted(results.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows and "decision" in rows[0]:
            gates[path.stem] = {r["key"]: r["decision"] for r in rows}
    labeled = {i: a["answer"] for i, a in answers.items() if i in key}
    decided = {i: a for i, a in labeled.items() if a in ("allow", "stop")}
    scored = {i: a for i, a in decided.items() if key[i]["expected"] is not None}
    agree = sum(key[i]["expected"] == a for i, a in scored.items())
    print(f"{len(labeled)} of {len(key)} items answered; {len(labeled) - len(decided)} marked unsure.")
    if scored:
        print(
            f"Agreement with the authors' labels: {agree} of {len(scored)} ({100 * agree / len(scored):.0f}%)."
        )
    print("\n| Item | Category | Authors | You | Note |\n| --- | --- | --- | --- | --- |")
    for i in sorted(labeled):
        expected = key[i]["expected"] or "ambiguous"
        if expected != labeled[i]:
            note = str(answers[i].get("note") or "").replace("|", "/")[:120]
            print(f"| {i} | {key[i]['category']} | {expected} | {labeled[i]} | {note} |")
    if gates and decided:
        print("\n| Gate | Agrees with you |\n| --- | --- |")
        for name, decisions in gates.items():
            both = [i for i in decided if key[i]["key"] in decisions]
            hits = sum(decisions[key[i]["key"]] == decided[i] for i in both)
            if both:
                print(f"| {name} | {hits} of {len(both)} ({100 * hits / len(both):.0f}%) |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("scenarios", type=Path)
    sample.add_argument("out", type=Path)
    sample.add_argument("--per-category", type=int, default=3)
    sample.add_argument("--ambiguous", type=int, default=4)
    compare = commands.add_parser("compare")
    compare.add_argument("out", type=Path)
    compare.add_argument("answers", type=Path)
    compare.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        command_sample(args.scenarios, args.out, args.per_category, args.ambiguous)
    else:
        command_compare(args.out, args.answers, args.results)


if __name__ == "__main__":
    main()
