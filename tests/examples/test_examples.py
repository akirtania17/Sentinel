"""The examples run end to end from their committed response caches, without an API key."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
    return subprocess.run(
        [sys.executable, str(REPO / script), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_agent_action_example() -> None:
    result = _run("examples/agent_action/main.py")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "agent proposes list_emails" in out
    assert "ALLOW:" in out
    assert "DENY: the action has effects the request did not ask for" in out
    assert out.count("agent proposes") == 7


def test_agent_action_matrix_is_current() -> None:
    result = _run("evals/agent_action/generate_matrix.py", "--check")
    assert result.returncode == 0, result.stderr


def test_output_verification_example() -> None:
    result = _run("examples/output_verification/main.py")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "ACCEPT: the answer is on topic, grounded, and complete" in out
    assert "REGENERATE: the answer contradicts the source" in out
    assert "ESCALATE:" in out
    assert out.count("facts:") == 5


def test_trace_semantics_example(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    result = _run("examples/trace_semantics/main.py", "--events", str(events))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "scope_drift" in out
    assert "repetition" in out
    assert len(events.read_text().splitlines()) == 7
