"""The examples run end to end from their committed response caches, without an API key."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
    return subprocess.run(
        [sys.executable, str(REPO / script)],
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
