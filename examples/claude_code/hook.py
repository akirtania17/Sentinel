"""Gate Claude Code's tool calls with Sentinel, as a PreToolUse hook.

Claude Code runs this script before every tool call, with the call on stdin. The script judges the
proposed call against the user's latest request, decides with an explicit policy, and records one
JSON line per call.

What Sentinel is shown (`SENTINEL_HOOK_CONTEXT`):

- `on` (default): the request, the proposed call, the project's `CLAUDE.md` (which documents its
  commands), and the agent's last six steps with their results, judged with `tool_call.yaml`.
- `off`: only the request and the proposed call, judged with `predicates/agent_action.yaml`. In the
  field test this interrupted a third of legitimate calls: without context, Sentinel cannot tell
  what `./bin/issue_refund 5120 189.00` does or which orders the agent found double-charged.

Policy (`SENTINEL_HOOK_POLICY`):

- `effects` (default): intervene only on actions with effects outside the workspace or destructive
  ones. Reading, searching, testing and editing the workspace always proceed.
- `strict`: `examples/agent_action/gate.py`, which also questions any call that does not clearly
  follow from the request. Fine for a single proposed action; too noisy for an agent exploring code.

Mode (`SENTINEL_HOOK_MODE`):

- `observe` (default): record the decision and let every call through.
- `enforce`: deny calls the policy denies, and ask the user about calls it wants confirmed. Allowed
  calls still go through Claude Code's own permission checks; the hook never grants permission. If
  the semantic check is unavailable or the hook fails, the call is denied (fail closed).

Settings (for example `.claude/settings.json`):

    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "timeout": 30,
      "command": "/path/to/Sentinel/.venv/bin/python /path/to/Sentinel/examples/claude_code/hook.py"}]}]}}

Environment: `TYPESAFE_API_KEY` (required), the three settings above, and `SENTINEL_HOOK_LOG` (the
JSONL record file; default `.sentinel/claude-code-hook.jsonl` under the session's working directory).
"""

import time

STARTED = time.perf_counter()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "examples" / "agent_action"))

from gate import Decision, decide  # noqa: E402

from sentinel_runtime import (  # noqa: E402
    ChoiceResult,
    Evaluation,
    PredicateSet,
    ProviderError,
    Runtime,
    ScoreResult,
)
from sentinel_runtime.config import SentinelConfig  # noqa: E402

IMPORTED = time.perf_counter()

MODEL = "jev-1.13.0"
PREDICATES = {"on": HERE / "tool_call.yaml", "off": REPO / "predicates" / "agent_action.yaml"}
MAX_CHARS = 4000
"""Longer string arguments (such as a file being written) are clipped before evaluation."""
RECENT_STEPS = 6
RESULT_CHARS = 1500
INSTRUCTION_CHARS = 6000


def clip(value: Any, limit: int = MAX_CHARS) -> Any:
    """Shorten long strings anywhere in a value."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more characters]"
    if isinstance(value, dict):
        return {key: clip(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [clip(item, limit) for item in value]
    return value


def _entries(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _content(entry: dict[str, Any]) -> Any:
    message = entry.get("message")
    return message.get("content") if isinstance(message, dict) else None


def _blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    content = _content(entry)
    return [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []


def latest_user_request(lines: list[str]) -> str:
    """The text of the most recent message the user typed."""
    request = ""
    for entry in _entries(lines):
        if entry.get("type") != "user" or entry.get("isMeta"):
            continue
        content = _content(entry)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(str(b.get("text", "")) for b in _blocks(entry) if b.get("type") == "text")
        else:
            continue
        text = text.strip()
        if text and not text.startswith(("<command-", "<local-command")):
            request = text
    return request


def steps_by_call(lines: list[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """For each tool call in a transcript, the steps before it; and the last steps overall."""
    calls: list[dict[str, Any]] = []
    results: dict[str, str] = {}
    for entry in _entries(lines):
        for block in _blocks(entry):
            if entry.get("type") == "assistant" and block.get("type") == "tool_use":
                calls.append(
                    {"id": block["id"], "tool": block["name"], "arguments": clip(block.get("input", {}))}
                )
            elif entry.get("type") == "user" and block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                results[block["tool_use_id"]] = clip(str(content or ""), RESULT_CHARS)

    def step(call: dict[str, Any]) -> dict[str, Any]:
        return {"tool": call["tool"], "arguments": call["arguments"], "result": results.get(call["id"], "")}

    before = {c["id"]: [step(p) for p in calls[max(0, i - RECENT_STEPS) : i]] for i, c in enumerate(calls)}
    return before, [step(c) for c in calls[-RECENT_STEPS:]]


def decide_for_coding_agent(evaluation: Evaluation) -> Decision:
    """Intervene only on effects outside the workspace and on destructive actions."""
    matches = evaluation.probability("action_matches_intent")
    side_effect = evaluation.probability("external_side_effect")
    authorized = evaluation.probability("explicitly_authorized")
    effect = evaluation["effect_class"]
    high_risk = isinstance(effect, ChoiceResult) and effect.choice == "destructive_or_financial"
    if side_effect >= 0.5 and matches < 0.5:
        return Decision("deny", "the action has effects the request did not ask for")
    if side_effect >= 0.5 and authorized < 0.5:
        return Decision("confirm", "external effect without explicit authorization of this exact action")
    if high_risk and authorized < 0.8:
        return Decision("confirm", "destructive or financial action without clear explicit authorization")
    return Decision("allow", "no unrequested external or destructive effect")


def facts(evaluation: Evaluation) -> dict[str, Any]:
    """The judgments behind a decision, for the record."""
    out: dict[str, Any] = {}
    for predicate_id, result in evaluation.results.items():
        if isinstance(result, ChoiceResult):
            out[predicate_id] = {"choice": result.choice, "confidence": result.confidence}
        elif isinstance(result, ScoreResult):
            out[predicate_id] = result.score
        else:
            out[predicate_id] = result.probability
    return out


def main() -> None:
    call = json.load(sys.stdin)
    mode = os.environ.get("SENTINEL_HOOK_MODE", "observe")
    context = os.environ.get("SENTINEL_HOOK_CONTEXT", "on")
    policy = os.environ.get("SENTINEL_HOOK_POLICY", "effects")
    cwd = Path(call.get("cwd") or ".")
    log = Path(os.environ.get("SENTINEL_HOOK_LOG") or cwd / ".sentinel" / "claude-code-hook.jsonl")

    try:
        lines = Path(call.get("transcript_path", "")).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    user_request = latest_user_request(lines)
    action = {"tool": call.get("tool_name"), "arguments": clip(call.get("tool_input", {}))}
    state: dict[str, Any] = {"user_request": user_request, "proposed_action": action}
    if context == "on":
        instructions = cwd / "CLAUDE.md"
        state["project_instructions"] = (
            clip(instructions.read_text(encoding="utf-8"), INSTRUCTION_CHARS) if instructions.exists() else ""
        )
        before, latest = steps_by_call(lines)
        # The pending call may already be in the transcript; show the steps that came before it.
        state["recent_steps"] = before.get(str(call.get("tool_use_id")), latest)

    evaluation: Evaluation | None = None
    with Runtime.from_env(model=MODEL, config=SentinelConfig()) as runtime:
        prepared = time.perf_counter()
        try:
            evaluation = runtime.evaluate(state, PredicateSet.from_file(PREDICATES[context]))
        except ProviderError as error:
            decision = Decision("deny", f"semantic check unavailable ({error.kind}); failing closed")
        else:
            decision = decide_for_coding_agent(evaluation) if policy == "effects" else decide(evaluation)
    finished = time.perf_counter()

    record = {
        "ts": datetime.now(UTC).isoformat(),
        "mode": mode,
        "context": context,
        "policy": policy,
        "session_id": call.get("session_id"),
        "tool_use_id": call.get("tool_use_id"),
        "user_request": user_request,
        "proposed_action": action,
        "decision": decision.action,
        "reason": decision.reason,
        "facts": None if evaluation is None else facts(evaluation),
        "request_id": None if evaluation is None else evaluation.calls[0].request_id,
        "input_tokens": None if evaluation is None else evaluation.usage.input_tokens,
        "evaluation_ms": None if evaluation is None else round(evaluation.latency_ms, 1),
        "import_ms": round((IMPORTED - STARTED) * 1000, 1),
        "setup_ms": round((prepared - IMPORTED) * 1000, 1),
        "hook_ms": round((finished - STARTED) * 1000, 1),
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    if mode == "enforce" and decision.action != "allow":
        print(json.dumps(verdict(decision)))


def verdict(decision: Decision) -> dict[str, Any]:
    """The PreToolUse output: deny outright, or ask the user to confirm."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny" if decision.action == "deny" else "ask",
            "permissionDecisionReason": f"Sentinel: {decision.reason}",
        }
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Claude Code lets a call through when a hook crashes, so enforce mode denies instead.
        if os.environ.get("SENTINEL_HOOK_MODE", "observe") == "enforce":
            reason = f"semantic check failed ({type(error).__name__}); failing closed"
            print(json.dumps(verdict(Decision("deny", reason))))
        else:
            raise
