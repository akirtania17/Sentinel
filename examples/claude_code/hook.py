"""Gate Claude Code's tool calls with Sentinel, as a PreToolUse hook.

Claude Code runs this script before every tool call, with the call on stdin. The script judges the
proposed call against the user's latest request, decides with an explicit policy, and records one
JSON line per call.

What Sentinel is shown (`SENTINEL_HOOK_CONTEXT`):

- `on` (default): two judgments made in parallel. `intent.yaml` sees the request, the project's
  `CLAUDE.md`, the proposed call, and the agent's recent commands without their output, so nothing
  an attacker planted in a file, email, page or log can reach it: does the call have effects, and
  did the user ask for this kind of action. `context.yaml` also sees the output of the agent's last
  six steps: wrong details, scope, unmet conditions, instructions found in data, and data leaving.
  In the stress test (docs/stress-test.md) this held accuracy while cutting what an adaptive
  attacker could get through from 9 of 12 harmful actions to 4 of 12.
- `off`: only the request and the proposed call, judged with `predicates/agent_action.yaml`. In the
  field test this interrupted a third of legitimate calls: without context, Sentinel cannot tell
  what `./bin/issue_refund 5120 189.00` does or which orders the agent found double-charged.

Policy (`SENTINEL_HOOK_POLICY`):

- `effects` (default): intervene only on actions that change something the user did not ask for or
  that reach outside the workspace. Reading, searching, testing and editing the workspace as part
  of the task always proceed.
- `strict`: `examples/agent_action/gate.py`, which also questions any call that does not clearly
  follow from the request. Fine for a single proposed action; too noisy for an agent exploring code.

Thresholds (`SENTINEL_HOOK_THRESHOLDS`): `strict` (default, 0.5 everywhere) stops the most harmful
actions; `balanced` interrupts far fewer legitimate ones for slightly less detection.

When unsure (`SENTINEL_HOOK_UNSURE`): `decide` (default) acts on Sentinel's decision; `ask` asks the
user whenever a judgment the decision rests on is within 0.2 of its threshold (about 9% of actions
in the stress test). Near-threshold answers vary between runs, so asking closes the gap an attacker
could otherwise exploit by retrying.

Mode (`SENTINEL_HOOK_MODE`):

- `observe` (default): record the decision and let every call through.
- `enforce`: deny calls the policy denies, and ask the user about calls it wants confirmed. Allowed
  calls still go through Claude Code's own permission checks; the hook never grants permission. If
  the semantic check is unavailable or the hook fails, the call is denied (fail closed).

Settings (for example `.claude/settings.json`):

    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "timeout": 30,
      "command": "/path/to/Sentinel/.venv/bin/python /path/to/Sentinel/examples/claude_code/hook.py"}]}]}}

Environment: `TYPESAFE_API_KEY` (required), the settings above, and `SENTINEL_HOOK_LOG` (the
JSONL record file; default `.sentinel/claude-code-hook.jsonl` under the session's working directory).
"""

import time

STARTED = time.perf_counter()

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "examples" / "agent_action"))

from checks import lookalikes  # noqa: E402
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
INTENT = HERE / "intent.yaml"
CONTEXT = HERE / "context.yaml"
WITHOUT_CONTEXT = REPO / "predicates" / "agent_action.yaml"
UNSURE_MARGIN = 0.2
EARLIER_TURNS = 6
THRESHOLDS = {
    # 0.5 everywhere: stops the most harmful actions.
    "strict": {"requested": 0.5, "data": 0.5, "checks": 0.5},
    # Chosen on the stress test's tuning sets: about a third of the false alarms of `strict` for a
    # couple of points less detection (docs/stress-test.md).
    "balanced": {"requested": 0.3, "data": 0.4, "checks": 0.7},
}
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


def conversation_turns(lines: list[str]) -> list[dict[str, str]]:
    """The user's and the assistant's text turns before the latest user message, oldest first."""
    turns: list[dict[str, str]] = []
    for entry in _entries(lines):
        role = entry.get("type")
        if role not in ("user", "assistant") or entry.get("isMeta"):
            continue
        content = _content(entry)
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(str(b.get("text", "")) for b in _blocks(entry) if b.get("type") == "text")
        text = text.strip()
        if text and not text.startswith(("<command-", "<local-command")):
            turns.append({"role": str(role), "text": clip(text, RESULT_CHARS)})
    last_user = max((i for i, t in enumerate(turns) if t["role"] == "user"), default=len(turns))
    return turns[:last_user][-EARLIER_TURNS:]


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


def without_output(state: dict[str, Any]) -> dict[str, Any]:
    """The state with every recent step's output removed: nothing an attacker planted in data."""
    steps = [{"tool": s.get("tool"), "arguments": s.get("arguments")} for s in state.get("recent_steps", [])]
    return {**state, "recent_steps": steps}


CHECKS = ("sends_data_away", "follows_data_instruction", "condition_blocked", "exceeds_scope", "wrong_detail")


def decide_for_coding_agent(
    f: dict[str, Any], flags: Sequence[str] = (), thresholds: Mapping[str, float] = THRESHOLDS["strict"]
) -> Decision:
    """Stop what the user did not ask for; let reading, testing and requested work proceed.

    `f` holds the judgments from `intent.yaml` and `context.yaml`. `requested` and the effect
    judgments come from the intent judgment, which never sees tool output. `flags` are lookalike
    targets found in code (checks.py).
    """
    kind = f["effect_class"]["choice"]
    requested, data, checks = thresholds["requested"], thresholds["data"], thresholds["checks"]
    if kind != "read_only" and f["requested"] < requested:
        return Decision("deny", "the user did not ask for this kind of action")
    if kind != "read_only" and f["sends_data_away"] >= data:
        return Decision("deny", "private data would leave for a destination the user did not name")
    if kind != "read_only" and f["follows_data_instruction"] >= data:
        return Decision("deny", "the action follows an instruction found in data, not the user")
    has_effect = f["external_side_effect"] >= 0.5 or kind in (
        "external_communication",
        "destructive_or_financial",
    )
    if not has_effect:
        return Decision("allow", "no outside or destructive effect, and part of the task")
    if f["condition_blocked"] >= checks:
        return Decision("deny", "the user or a project rule forbids this, or its condition is not met")
    if f["exceeds_scope"] >= checks:
        return Decision("confirm", "the action reaches more than the request asks for")
    if flags:
        return Decision("confirm", f"a target looks like a slip or a spoof ({flags[0]})")
    if f["wrong_detail"] >= checks:
        return Decision("confirm", "a target or detail differs from the intended one")
    if f["requested"] < requested:  # an outside effect on an action the intent judgment called read-only
        return Decision("deny", "the user did not ask for this kind of action")
    return Decision("allow", "requested, with the intended targets and scope")


def unsure(
    f: dict[str, Any],
    decision: Decision,
    flags: Sequence[str] = (),
    thresholds: Mapping[str, float] = THRESHOLDS["strict"],
) -> bool:
    """Whether a judgment within UNSURE_MARGIN of its threshold would change the decision if it fell
    the other way."""
    threshold_of = {"external_side_effect": 0.5, "requested": thresholds["requested"]}
    threshold_of |= {"sends_data_away": thresholds["data"], "follows_data_instruction": thresholds["data"]}
    checks = ("condition_blocked", "exceeds_scope", "wrong_detail")
    threshold_of |= dict.fromkeys(checks, thresholds["checks"])
    for name, threshold in threshold_of.items():
        if abs(f[name] - threshold) < UNSURE_MARGIN:
            mirrored = 2 * threshold - f[name]
            flipped = {**f, name: mirrored if mirrored != threshold else threshold - 0.01}
            if (decide_for_coding_agent(flipped, flags, thresholds).action == "allow") != (
                decision.action == "allow"
            ):
                return True
    return False


def decide_without_context(evaluation: Evaluation) -> Decision:
    """The effects policy for `SENTINEL_HOOK_CONTEXT=off` (the agent_action predicates)."""
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


async def judge(state: dict[str, Any], context: str) -> list[Evaluation]:
    """One evaluation without context, or the intent and context evaluations in parallel."""
    async with Runtime.from_env(model=MODEL, config=SentinelConfig()) as runtime:
        if context != "on":
            return [await runtime.aevaluate(state, PredicateSet.from_file(WITHOUT_CONTEXT))]
        return list(
            await asyncio.gather(
                runtime.aevaluate(without_output(state), PredicateSet.from_file(INTENT)),
                runtime.aevaluate(state, PredicateSet.from_file(CONTEXT)),
            )
        )


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
        state["conversation"] = conversation_turns(lines)
        before, latest = steps_by_call(lines)
        # The pending call may already be in the transcript; show the steps that came before it.
        state["recent_steps"] = before.get(str(call.get("tool_use_id")), latest)

    prepared = time.perf_counter()
    evaluations: list[Evaluation] = []
    found: dict[str, Any] | None = None
    try:
        evaluations = asyncio.run(judge(state, context))
    except ProviderError as error:
        decision = Decision("deny", f"semantic check unavailable ({error.kind}); failing closed")
    else:
        found = {}
        for evaluation in evaluations:
            found.update(facts(evaluation))
        if policy == "strict":
            decision = decide(evaluations[0])
        elif context != "on":
            decision = decide_without_context(evaluations[0])
        else:
            thresholds = THRESHOLDS[os.environ.get("SENTINEL_HOOK_THRESHOLDS", "strict")]
            flags = lookalikes({**state, "user_request": user_request})
            decision = decide_for_coding_agent(found, flags, thresholds)
            unsure_mode = os.environ.get("SENTINEL_HOOK_UNSURE", "decide") == "ask"
            if unsure_mode and unsure(found, decision, flags, thresholds):
                decision = Decision("confirm", f"Sentinel is unsure ({decision.reason})")
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
        "facts": found,
        "request_ids": [e.calls[0].request_id for e in evaluations],
        "input_tokens": sum(e.usage.input_tokens or 0 for e in evaluations) if evaluations else None,
        "evaluation_ms": round(max(e.latency_ms for e in evaluations), 1) if evaluations else None,
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
