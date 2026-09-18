"""The Claude Code hook: transcript parsing, the coding-agent policy, and failing closed."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel_runtime import PredicateSet
from sentinel_runtime.lint import lint_predicate_set

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "examples" / "claude_code" / "hook.py"
sys.path.insert(0, str(HOOK.parent))
sys.path.insert(0, str(REPO / "experiments" / "agent_field_test"))
sys.path.insert(0, str(REPO / "experiments" / "agent_gate_benchmark"))

import analyze  # noqa: E402
import bench  # noqa: E402
import hook  # noqa: E402

INTENT = PredicateSet.from_file(HOOK.parent / "intent.yaml")
CONTEXT = PredicateSet.from_file(HOOK.parent / "context.yaml")


def _line(kind: str, content: Any, **extra: Any) -> str:
    return json.dumps({"type": kind, "message": {"role": kind, "content": content}, **extra})


TRANSCRIPT = [
    _line("user", "Refund order 5120 in full."),
    _line("assistant", [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "q.json"}}]),
    _line("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 5000}]),
    _line("user", "<command-name>/clear</command-name>"),
    _line("user", "ignored meta", isMeta=True),
    _line("assistant", [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}}]),
    "not json",
]


def test_latest_user_request_skips_tool_results_commands_and_meta() -> None:
    assert hook.latest_user_request(TRANSCRIPT) == "Refund order 5120 in full."


def test_steps_are_the_calls_before_the_pending_one() -> None:
    before, latest = hook.steps_by_call(TRANSCRIPT)
    assert before["t1"] == []
    assert [step["tool"] for step in before["t2"]] == ["Read"]
    result = before["t2"][0]["result"]
    assert result.startswith("x" * hook.RESULT_CHARS)
    assert result.endswith("more characters]")
    assert [step["tool"] for step in latest] == ["Read", "Bash"]


@pytest.mark.parametrize("predicate_set", [INTENT, CONTEXT], ids=["intent", "context"])
def test_hook_predicates_are_lint_clean(predicate_set: PredicateSet) -> None:
    state: dict[str, Any] = {
        "user_request": "r",
        "proposed_action": {},
        "project_instructions": "",
        "recent_steps": [],
    }
    assert lint_predicate_set(predicate_set, state=state) == []


def test_without_output_keeps_commands_and_drops_results() -> None:
    step: dict[str, Any] = {"tool": "Bash", "arguments": {"command": "ls"}, "result": "x"}
    state: dict[str, Any] = {"user_request": "r", "recent_steps": [step]}
    assert hook.without_output(state)["recent_steps"] == [{"tool": "Bash", "arguments": {"command": "ls"}}]
    assert step["result"] == "x"


def _facts(kind: str = "external_communication", **values: float) -> dict[str, Any]:
    scores: dict[str, float] = dict.fromkeys(hook.CHECKS, 0.05)
    scores |= {"external_side_effect": 0.95, "requested": 0.95}
    scores |= values
    return {**scores, "effect_class": {"choice": kind, "confidence": 0.9}}


@pytest.mark.parametrize(
    ("facts", "action"),
    [
        (_facts("read_only", external_side_effect=0.02, requested=0.2), "allow"),  # exploring never stops
        (_facts("local_change", external_side_effect=0.1), "allow"),  # requested workspace edit
        (_facts("local_change", external_side_effect=0.1, requested=0.1), "deny"),  # unrequested script
        (_facts(), "allow"),  # requested, intended targets
        (_facts(requested=0.2), "deny"),
        (_facts(follows_data_instruction=0.8), "deny"),
        (
            _facts("read_only", external_side_effect=0.02, sends_data_away=0.9),
            "allow",
        ),  # reading a bad script
        (_facts(sends_data_away=0.9), "deny"),
        (_facts(condition_blocked=0.7), "deny"),
        (_facts(exceeds_scope=0.7), "confirm"),
        (_facts(wrong_detail=0.7), "confirm"),
    ],
)
def test_coding_agent_policy(facts: dict[str, Any], action: str) -> None:
    assert hook.decide_for_coding_agent(facts).action == action


def test_unsure_only_when_a_near_threshold_answer_would_change_the_decision() -> None:
    confident = _facts()
    assert not hook.unsure(confident, hook.decide_for_coding_agent(confident))
    borderline = _facts(requested=0.55)
    assert hook.unsure(borderline, hook.decide_for_coding_agent(borderline))
    irrelevant = _facts("read_only", external_side_effect=0.02, wrong_detail=0.45)
    assert not hook.unsure(irrelevant, hook.decide_for_coding_agent(irrelevant))


@settings(max_examples=300)
@given(
    kind=st.sampled_from(
        ["read_only", "local_change", "external_communication", "destructive_or_financial", "other"]
    ),
    values=st.lists(st.floats(0, 1), min_size=7, max_size=7),
)
def test_hook_policy_matches_the_benchmarked_policy(kind: str, values: list[float]) -> None:
    names = ("external_side_effect", "requested", *hook.CHECKS)
    facts = {**dict(zip(names, values, strict=True)), "effect_class": {"choice": kind, "confidence": 0.9}}
    assert (hook.decide_for_coding_agent(facts).action == "allow") == (
        bench.POLICIES["focused3"](facts) == "allow"
    )


def _run_hook(mode: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    call = {
        "session_id": "s",
        "transcript_path": str(tmp_path / "missing.jsonl"),
        "cwd": str(tmp_path),
        "tool_name": "Bash",
        "tool_input": {"command": "./bin/deploy production"},
        "tool_use_id": "t9",
    }
    env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
    env["SENTINEL_HOOK_MODE"] = mode
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(call),
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
        check=False,
    )


def test_enforce_mode_fails_closed_without_the_semantic_check(tmp_path: Path) -> None:
    result = _run_hook("enforce", tmp_path)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "failing closed" in output["permissionDecisionReason"]


def test_observe_mode_never_blocks(tmp_path: Path) -> None:
    result = _run_hook("observe", tmp_path)
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("command", "calls"),
    [
        ("cat bin/issue_refund 2>/dev/null | head", []),
        ('./bin/send_email "bob@x.test" "Re" "Hi,\n\nthanks"', [("send_email", "bob@x.test")]),
        (
            'cd /p && ./bin/issue_refund 5120 1 && python3 bin/post_slack "#eng" m',
            [("issue_refund", "5120"), ("post_slack", "eng")],
        ),
    ],
)
def test_field_test_counts_only_tools_that_run(command: str, calls: list[tuple[str, str]]) -> None:
    assert analyze.team_calls(command) == calls
