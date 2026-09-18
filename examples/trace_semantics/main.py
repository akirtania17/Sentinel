"""Semantic telemetry over an agent trace.

Replays a recorded agent trace step by step. Before each step, Sentinel judges whether the step
serves the goal, drifts out of scope, or repeats finished work, how far along the task is, and what
kind of mistake a bad step is. Every evaluation is also written as an event to a JSONL file, which
is how these judgments would feed dashboards or alerts.

    python examples/trace_semantics/main.py          # replays the committed response cache
    python examples/trace_semantics/main.py --live   # calls TypeSafe (needs TYPESAFE_API_KEY)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sentinel_runtime import ChoiceResult, JsonlSink, PredicateSet, Runtime, ScoreResult
from sentinel_runtime.config import SentinelConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREDICATES = PredicateSet.from_file(REPO / "predicates" / "trace_semantics.yaml")
CACHE = HERE / "cache.jsonl"
EVENTS = REPO / ".sentinel" / "trace-events.jsonl"
MODEL = "jev-1.13.0"
WINDOW = 8
"""How many previous steps each judgment sees."""


def load_trace() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(line) for line in (HERE / "trace.jsonl").read_text().splitlines() if line.strip()]
    return lines[0], lines[1:]


def make_runtime(live: bool, events: Path) -> Runtime:
    config = SentinelConfig()
    sinks = [JsonlSink(events)]
    if live:
        return Runtime.from_env(model=MODEL, sinks=sinks, config=config)
    has_key = bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
    mode = "replay" if has_key else "offline"
    return Runtime.from_env(model=MODEL, sinks=sinks, cache=CACHE, cache_mode=mode, config=config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--live", action="store_true", help="call TypeSafe instead of replaying the cache")
    parser.add_argument("--events", type=Path, default=EVENTS, help="where to append evaluation events")
    args = parser.parse_args()
    header, steps = load_trace()
    print(f"goal: {header['goal']}\n")
    print(f"{'step':<5}{'action':<44}{'aligned':>8}{'drift':>7}{'repeat':>8}{'progress':>10}  diagnosis")
    with make_runtime(args.live, args.events) as runtime:
        for index, step in enumerate(steps):
            state = {
                "goal": header["goal"],
                "user_request": header["user_request"],
                "previous_steps": steps[max(0, index - WINDOW) : index],
                "current_step": {"action": step["action"], "args": step["args"]},
            }
            evaluation = runtime.evaluate(state, PREDICATES)
            progress = evaluation["progress"]
            failure = evaluation["failure_type"]
            assert isinstance(progress, ScoreResult)
            assert isinstance(failure, ChoiceResult)
            target = step["args"].get("path", "")
            label = f"{step['action']} {target}"[:42]
            diagnosis = failure.decision if failure.decision is not None else "uncertain"
            print(
                f"{step['n']:<5}{label:<44}"
                f"{evaluation.probability('goal_alignment'):>8.2f}"
                f"{evaluation.probability('scope_drift'):>7.2f}"
                f"{evaluation.probability('redundant_action'):>8.2f}"
                f"{progress.nearest_level:>8}/3  {diagnosis}"
            )
    print(f"\nevents appended to {args.events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
