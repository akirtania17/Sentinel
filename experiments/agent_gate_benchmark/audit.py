"""Independent label audit for benchmark scenarios.

    python3 experiments/agent_gate_benchmark/audit.py strip SCENARIOS.yaml UNLABELED.yaml
    python3 experiments/agent_gate_benchmark/audit.py compare SCENARIOS.yaml AUDIT.yaml [--write AGREED.yaml]

`strip` removes every label, note and category, so an auditor sees only what a gate sees. The
auditor writes AUDIT.yaml, a mapping from `scenario_id/action_id` to `{effect, authorized}`.
`compare` reports agreement on the expected decision and, with `--write`, writes a copy of the
scenarios where every action whose decision the auditor disagrees with is marked `ambiguous`
(`authorized: null`), so benchmark scores only count labels two independent reviewers share.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from scenarios import expected_decision, load, problems

LABEL_FIELDS = ("category", "effect", "authorized", "note")


def strip(source: Path, target: Path) -> None:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    for scenario in data["scenarios"]:
        scenario.pop("domain", None)
        for action in scenario["actions"]:
            for key in LABEL_FIELDS:
                action.pop(key, None)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")


def compare(source: Path, audit_path: Path, write: Path | None) -> None:
    cases = {case.key: case for case in load(source)}
    audit: dict[str, dict[str, Any]] = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    missing = sorted(set(cases) - set(audit))
    if missing:
        raise SystemExit(f"audit is missing {len(missing)} actions, for example {missing[:3]}")
    outcome: Counter[str] = Counter()
    disagreements: list[str] = []
    for key, case in cases.items():
        theirs = expected_decision(
            {"effect": audit[key]["effect"], "authorized": audit[key].get("authorized")}
        )
        ours = case.expected
        if ours is None or theirs is None:
            outcome["ambiguous for either"] += 1
        elif ours == theirs:
            outcome["agree"] += 1
        else:
            outcome["disagree"] += 1
            disagreements.append(f"  {key} ({case.category}): author {ours}, auditor {theirs}")
    decided = outcome["agree"] + outcome["disagree"]
    print(
        f"{len(cases)} actions: {dict(outcome)}; agreement {outcome['agree'] / decided:.1%} of decided pairs"
    )
    print("\n".join(disagreements))
    if write is None:
        return
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    for scenario in data["scenarios"]:
        for action in scenario["actions"]:
            key = f"{scenario['id']}/{action['id']}"
            theirs = expected_decision(
                {"effect": audit[key]["effect"], "authorized": audit[key].get("authorized")}
            )
            if theirs != expected_decision(action) or theirs is None:
                action["audit"] = "disagreed" if theirs is not None else "auditor found ambiguous"
                if action["effect"] == "none":  # the auditor saw an effect the author did not
                    action["effect"] = audit[key]["effect"]
                action["original_category"] = action["category"]
                action["category"] = "ambiguous"
                action["authorized"] = None
    assert not problems(data), problems(data)[:5]
    write.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    print(f"wrote {write}")


def main() -> None:
    command, *rest = sys.argv[1:]
    if command == "strip":
        strip(Path(rest[0]), Path(rest[1]))
    elif command == "compare":
        write = Path(rest[rest.index("--write") + 1]) if "--write" in rest else None
        compare(Path(rest[0]), Path(rest[1]), write)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
