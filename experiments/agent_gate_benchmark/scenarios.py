"""Load and validate agent-gate benchmark scenario files (see SCENARIOS.md).

    python3 experiments/agent_gate_benchmark/scenarios.py FILE.yaml [FILE.yaml ...]

Prints a summary per file and exits non-zero on any problem.
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FORMAT = "sentinel-bench/agent-gate/v1"
ALLOW = {"safe_step", "authorized_effect", "implied_authorization", "benign_lookalike"}
STOP = {
    "wrong_target",
    "scope_creep",
    "condition_unmet",
    "injection_followed",
    "gate_injection",
    "obfuscated",
    "exfiltration",
    "drift",
}
CATEGORIES = ALLOW | STOP | {"ambiguous"}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


@dataclass(frozen=True)
class Case:
    """One candidate action in one scenario, with its expected decision."""

    key: str  # scenario_id/action_id
    scenario: dict[str, Any]
    action: dict[str, Any]

    @property
    def category(self) -> str:
        return self.action["category"]

    @property
    def expected(self) -> str | None:
        """allow, stop, or None for ambiguous."""
        return expected_decision(self.action)

    @property
    def state(self) -> dict[str, Any]:
        """What a gate is shown: the request, the project's instructions, recent steps, the action."""
        return {
            "conversation": self.scenario.get("conversation", []),
            "user_request": self.scenario["user_request"],
            "proposed_action": self.action["action"],
            "project_instructions": self.scenario.get("project_instructions", ""),
            "recent_steps": self.scenario.get("recent_steps", []),
        }


def expected_decision(action: dict[str, Any]) -> str | None:
    if action["effect"] == "none":
        return "allow"
    authorized = action.get("authorized")
    if authorized is None:
        return None
    return "allow" if authorized else "stop"


def problems(data: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        return [f"top level must be a mapping with format: {FORMAT}"]
    seen: set[str] = set()
    for index, scenario in enumerate(data.get("scenarios") or []):
        where = f"scenarios[{index}]"
        sid = scenario.get("id", "")
        if not IDENTIFIER.match(str(sid)):
            out.append(f"{where}: bad id {sid!r}")
        if sid in seen:
            out.append(f"{where}: duplicate id {sid}")
        seen.add(sid)
        if not str(scenario.get("user_request", "")).strip():
            out.append(f"{sid}: user_request is empty")
        for turn in scenario.get("conversation") or []:
            if (
                not isinstance(turn, dict)
                or turn.get("role") not in ("user", "assistant")
                or not turn.get("text")
            ):
                out.append(f"{sid}: each conversation turn needs role user or assistant, and text")
        steps = scenario.get("recent_steps") or []
        if len(steps) > 6:
            out.append(f"{sid}: more than 6 recent_steps")
        for step in steps:
            if not {"tool", "arguments"} <= set(step) or not isinstance(step.get("tool"), str):
                out.append(f"{sid}: each recent step needs a tool name and arguments")
        action_ids: set[str] = set()
        for action in scenario.get("actions") or []:
            aid = action.get("id", "")
            label = f"{sid}/{aid}"
            if not IDENTIFIER.match(str(aid)) or aid in action_ids:
                out.append(f"{label}: bad or duplicate action id")
            action_ids.add(aid)
            category = action.get("category")
            if category not in CATEGORIES:
                out.append(f"{label}: unknown category {category!r}")
                continue
            proposed = action.get("action")
            if not isinstance(proposed, dict) or "tool" not in proposed or "arguments" not in proposed:
                out.append(f"{label}: action needs tool and arguments")
            effect = action.get("effect")
            if effect not in ("none", "external", "destructive"):
                out.append(f"{label}: effect must be none, external or destructive")
                continue
            if effect != "none" and "authorized" not in action:
                out.append(f"{label}: authorized is required when effect is {effect}")
            decision = expected_decision(action)
            if category in ALLOW and decision != "allow":
                out.append(f"{label}: category {category} expects allow, labels give {decision}")
            if category in STOP and decision != "stop":
                out.append(f"{label}: category {category} expects stop, labels give {decision}")
            if category == "ambiguous" and decision is not None:
                out.append(f"{label}: ambiguous actions need effect other than none and authorized: null")
        if not action_ids:
            out.append(f"{sid}: no actions")
    return out


def load(path: Path) -> list[Case]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    found = problems(data)
    if found:
        raise ValueError(f"{path}: " + "; ".join(found[:20]))
    return [
        Case(f"{scenario['id']}/{action['id']}", scenario, action)
        for scenario in data["scenarios"]
        for action in scenario["actions"]
    ]


def main() -> None:
    failed = False
    for name in sys.argv[1:]:
        path = Path(name)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        found = problems(data)
        if found:
            failed = True
            print(f"{path}: {len(found)} problems")
            for problem in found[:50]:
                print(f"  {problem}")
            continue
        cases = load(path)
        decisions = Counter(case.expected or "ambiguous" for case in cases)
        categories = Counter(case.category for case in cases)
        domains = Counter(case.scenario.get("domain", "?") for case in cases)
        print(f"{path}: {len({c.scenario['id'] for c in cases})} scenarios, {len(cases)} actions")
        print(f"  expected: {dict(decisions)}")
        print(f"  categories: {dict(sorted(categories.items()))}")
        print(f"  domains: {dict(sorted(domains.items()))}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
