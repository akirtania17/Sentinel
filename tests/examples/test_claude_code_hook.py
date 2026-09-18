"""The Claude Code hook: transcript parsing, the coding-agent policy, and failing closed."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sentinel_runtime import PredicateSet, Runtime
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.lint import lint_predicate_set
from sentinel_runtime.providers import FakeProvider, choice_answer, noul_answer, score_answer

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "examples" / "claude_code" / "hook.py"
sys.path.insert(0, str(HOOK.parent))
sys.path.insert(0, str(REPO / "experiments" / "agent_field_test"))

import analyze  # noqa: E402
import hook  # noqa: E402

TOOL_CALL = PredicateSet.from_file(HOOK.parent / "tool_call.yaml")


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


def test_tool_call_predicates_are_lint_clean() -> None:
    state: dict[str, Any] = {
        "user_request": "r",
        "proposed_action": {},
        "project_instructions": "",
        "recent_steps": [],
    }
    assert lint_predicate_set(TOOL_CALL, state=state) == []


def _evaluate(match: float, side_effect: float, authorized: float, kind: str) -> Any:
    options = ["read_only", "local_change", "external_communication", "destructive_or_financial", "other"]
    answers = {
        "action_matches_intent": noul_answer(match),
        "external_side_effect": noul_answer(side_effect),
        "explicitly_authorized": noul_answer(authorized),
        "effect_class": choice_answer({o: 1.0 if o == kind else 0.0 for o in options}, confidence=0.9),
        "severity": score_answer([0.2, 0.2, 0.2, 0.2, 0.2]),
    }
    runtime = Runtime(FakeProvider(answers), model="jev-1.13.0", config=SentinelConfig())
    return runtime.evaluate({"user_request": "r", "proposed_action": {}}, TOOL_CALL)


@pytest.mark.parametrize(
    ("facts", "action"),
    [
        ((0.3, 0.02, 0.2, "read_only"), "allow"),  # exploring code never interrupts
        ((0.9, 0.97, 0.9, "external_communication"), "allow"),
        ((0.2, 0.97, 0.1, "external_communication"), "deny"),
        ((0.8, 0.97, 0.3, "external_communication"), "confirm"),
        ((0.8, 0.97, 0.7, "destructive_or_financial"), "confirm"),
        ((0.9, 0.97, 0.9, "destructive_or_financial"), "allow"),
    ],
)
def test_coding_agent_policy(facts: tuple[float, float, float, str], action: str) -> None:
    assert hook.decide_for_coding_agent(_evaluate(*facts)).action == action


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
