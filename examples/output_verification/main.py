"""Verify model answers against their source before showing them to a user.

Each state holds a user request, the source context an answering model was given, and one candidate
answer. Sentinel judges whether the answer addresses the request, stays faithful to the context, and
covers every part; the policy below accepts it, regenerates it, or escalates to a stronger model.

    python examples/output_verification/main.py          # replays the committed response cache
    python examples/output_verification/main.py --live   # calls TypeSafe (needs TYPESAFE_API_KEY)
"""

import argparse
import json
import os
from pathlib import Path

from sentinel_runtime import Evaluation, PredicateSet, Runtime
from sentinel_runtime.config import SentinelConfig

HERE = Path(__file__).resolve().parent
PREDICATES = PredicateSet.from_file(HERE.parents[1] / "predicates" / "output_verification.yaml")
CACHE = HERE / "cache.jsonl"
MODEL = "jev-1.13.0"
ORDER = ["faithful", "incomplete", "unsupported_detail", "contradicts", "off_topic"]

# Thresholds measured with `sentinel eval run output_verification` (see docs/evals.md).
SUPPORTED = 0.5
CONTRADICTS = 0.5
ANSWERS = 0.4
COMPLETE = 0.5


def decide(evaluation: Evaluation) -> tuple[str, str]:
    """Map the semantic facts to what the application does with the answer."""
    if evaluation.probability("contradicts_context") >= CONTRADICTS:
        return "regenerate", "the answer contradicts the source"
    if evaluation.probability("answers_request") < ANSWERS:
        return "regenerate", "the answer does not address the request"
    if evaluation.probability("supported_by_context") < SUPPORTED:
        return "regenerate", "the answer makes claims the source does not support"
    if evaluation.probability("complete") < COMPLETE:
        return "escalate", "the answer leaves part of the request unanswered; retry with a stronger model"
    return "accept", "the answer is on topic, grounded, and complete"


def make_runtime(live: bool) -> Runtime:
    config = SentinelConfig()
    if live:
        return Runtime.from_env(model=MODEL, config=config)
    has_key = bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
    return Runtime.from_env(
        model=MODEL, cache=CACHE, cache_mode="replay" if has_key else "offline", config=config
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--live", action="store_true", help="call TypeSafe instead of replaying the cache")
    live = parser.parse_args().live
    states = [HERE / "states" / f"{name}.json" for name in ORDER]
    first = json.loads(states[0].read_text())
    print(f"request: {first['user_request']}")
    with make_runtime(live) as runtime:
        for path in states:
            state = json.loads(path.read_text())
            evaluation = runtime.evaluate(state, PREDICATES)
            action, reason = decide(evaluation)
            facts = ", ".join(f"{pid} {evaluation.probability(pid):.2f}" for pid in PREDICATES.ids())
            output = state["model_output"]
            print(f"\n{path.stem}: {output[:90]}{'...' if len(output) > 90 else ''}")
            print(f"  facts: {facts}")
            print(f"  {action.upper()}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
