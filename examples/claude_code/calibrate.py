"""Recommend hook thresholds from the questions you answered.

Every time the hook asks about a call in enforce mode, your answer is a label from exactly the cases
Sentinel was least sure of. Thresholds tuned on one benchmark did not carry over to another in the
stress test (docs/stress-test.md), so the thresholds that fit your work are best learned from your
own answers.

A question counts as approved when the call ran afterwards (install the hook for PostToolUse too, see
hook_client.py) and as declined when the session transcript records that you rejected it. Anything
else is left out rather than guessed.

The recommendation only ever loosens the two noisy settings, `requested` (did the user ask for this
kind of action) and `checks` (wrong detail, wider scope, unmet condition), and only as far as every
question you declined would still be asked. The checks for injected instructions and data leaving
(`data`) are never learned from approvals, and the decision rule applies them first, so every call
they deny stays denied: approving one injected action must not teach the gate to allow the next.
Nothing is applied automatically; the script prints the setting to use.

    python examples/claude_code/calibrate.py [LOG.jsonl ...] [--min-answers 20]

With no log given it reads `.sentinel/claude-code-hook.jsonl` in the current directory.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hook import THRESHOLDS, decide_for_coding_agent, thresholds_from  # noqa: E402

REJECTED = "The user doesn't want to proceed with this tool use"
REQUESTED_FLOOR = 0.2
CHECKS_CEILING = 0.8
STEP = 0.05

Answer = Literal["approved", "declined"]


@dataclass(frozen=True)
class Record:
    """One judged call: its judgments, code checks, what the hook did, and the user's answer."""

    facts: Mapping[str, Any]
    flags: Sequence[str] = ()
    named: bool = False
    decision: str = "allow"
    answer: Answer | None = None


@dataclass
class Recommendation:
    thresholds: dict[str, float]
    approved: int
    declined: int
    approved_still_asked: int
    approved_asked_before: int
    enough: bool
    notes: list[str] = field(default_factory=list)

    @property
    def setting(self) -> str:
        values = ",".join(f"{key}={self.thresholds[key]:g}" for key in ("requested", "data", "checks"))
        return f"SENTINEL_HOOK_THRESHOLDS={values}"


def stops(record: Record, thresholds: Mapping[str, float]) -> bool:
    return decide_for_coding_agent(record.facts, record.flags, thresholds, record.named).action != "allow"


def _grid(low: float, high: float) -> list[float]:
    count = round((high - low) / STEP)
    return [round(low + i * STEP, 2) for i in range(count + 1)]


def recommend(
    records: Sequence[Record], current: Mapping[str, float] = THRESHOLDS["strict"], min_answers: int = 20
) -> Recommendation:
    """The `requested` and `checks` that ask the fewest of the approved questions while still stopping
    every declined one, preferring the smallest change among equally good ones."""
    answered = [r for r in records if r.answer is not None]
    approved = [r for r in answered if r.answer == "approved"]
    declined = [r for r in answered if r.answer == "declined"]
    best = dict(current)
    best_asked = sum(stops(r, current) for r in approved)
    before = best_asked
    for requested in _grid(REQUESTED_FLOOR, current["requested"]):
        for checks in _grid(current["checks"], CHECKS_CEILING):
            candidate = {**current, "requested": requested, "checks": checks}
            if not all(stops(r, candidate) for r in declined):
                continue
            asked = sum(stops(r, candidate) for r in approved)
            change = (current["requested"] - requested) + (checks - current["checks"])
            best_change = (current["requested"] - best["requested"]) + (best["checks"] - current["checks"])
            if asked < best_asked or (asked == best_asked and change < best_change):
                best, best_asked = candidate, asked
    notes: list[str] = []
    enough = len(answered) >= min_answers and bool(declined)
    if not enough:
        notes.append(
            f"only {len(answered)} answered questions ({len(declined)} declined); keep the current "
            f"thresholds until there are at least {min_answers}, including some you declined"
        )
        best, best_asked = dict(current), before
    return Recommendation(best, len(approved), len(declined), best_asked, before, enough, notes)


def _rejections(transcript: Path) -> set[str]:
    """Tool use ids the transcript records the user rejecting."""
    rejected: set[str] = set()
    try:
        lines = transcript.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rejected
    for line in lines:
        if REJECTED not in line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = entry.get("message", {}).get("content")
        for block in content if isinstance(content, list) else []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and REJECTED in str(block.get("content"))
            ):
                rejected.add(str(block.get("tool_use_id")))
    return rejected


def load(paths: Iterable[Path]) -> list[Record]:
    """Judged calls from hook logs, each with the user's answer when it can be known."""
    entries: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    ran = {e.get("tool_use_id") for e in entries if e.get("event") == "ran"}
    transcripts = {e["transcript_path"] for e in entries if e.get("transcript_path")}
    rejected = set().union(*(_rejections(Path(t)) for t in transcripts)) if transcripts else set()
    records: list[Record] = []
    for e in entries:
        if e.get("event", "decision") != "decision" or not e.get("facts") or "effect_class" not in e["facts"]:
            continue
        if e.get("policy", "effects") != "effects" or e.get("context", "on") != "on":
            continue
        answer: Answer | None = None
        if e.get("mode") == "enforce" and e.get("decision") == "confirm":
            if e.get("tool_use_id") in ran:
                answer = "approved"
            elif e.get("tool_use_id") in rejected:
                answer = "declined"
        records.append(
            Record(
                e["facts"],
                tuple(e.get("flags") or ()),
                bool(e.get("named_destinations")),
                e["decision"],
                answer,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", type=Path, nargs="*", default=[Path(".sentinel/claude-code-hook.jsonl")])
    parser.add_argument("--min-answers", type=int, default=20)
    parser.add_argument(
        "--current", default="strict", help="the thresholds in use: a preset or key=value list"
    )
    args = parser.parse_args()
    records = load(args.logs)
    result = recommend(records, thresholds_from(args.current), args.min_answers)
    print(f"{len(records)} judged calls; {result.approved} questions approved, {result.declined} declined.")
    for note in result.notes:
        print(f"Not enough evidence: {note}.")
    if result.enough:
        skipped = result.approved_asked_before - result.approved_still_asked
        print(
            f"With these thresholds the hook would have skipped {skipped} of the "
            f"{result.approved_asked_before} questions you approved and still asked every one you "
            f"declined. Injected instructions and data leaving are judged as before:\n\n    {result.setting}"
        )


if __name__ == "__main__":
    main()
