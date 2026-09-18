"""Re-judge every recorded tool call offline under different gate inputs, to compare them fairly.

The agents' tool calls are fixed; only what Sentinel is shown changes. For each call the script
rebuilds, from the session transcript, what the agent had seen just before it:

- `bare`: the user request and the proposed call only, with the reference predicates (what the first
  version of the hook saw).
- `context`: adds `project_instructions` (the project's CLAUDE.md, which documents its commands) and
  `recent_steps` (the agent's last tool calls with their results), with the reference predicates.
- `context_worded`: the same state with `agent_action_context.yaml`, whose questions say how to use
  that context.

    uv run python experiments/agent_field_test/replay.py RESULTS_DIR

Writes RESULTS_DIR/replay.jsonl (one line per call and variant). Responses are cached in
RESULTS_DIR/replay-cache.jsonl, so re-running costs nothing.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "examples" / "agent_action"))
sys.path.insert(0, str(REPO / "examples" / "claude_code"))

from gate import decide  # noqa: E402
from hook import facts, steps_by_call  # noqa: E402

from sentinel_runtime import PredicateSet, ProviderError, Runtime  # noqa: E402
from sentinel_runtime.config import SentinelConfig  # noqa: E402

MODEL = "jev-1.13.0"
VARIANTS = {
    "bare": REPO / "predicates" / "agent_action.yaml",
    "context": REPO / "predicates" / "agent_action.yaml",
    "context_worded": REPO / "examples" / "claude_code" / "tool_call.yaml",
}


def steps_before(stream: Path) -> dict[str, list[dict[str, Any]]]:
    """For each tool_use_id, the agent's preceding steps with their results, as the hook builds them."""
    return steps_by_call(stream.read_text().splitlines())[0]


async def main_async(results: Path, conditions: set[str] | None) -> None:
    instructions = (HERE / "sandbox" / "CLAUDE.md").read_text()
    sets = {name: PredicateSet.from_file(path) for name, path in VARIANTS.items()}
    jobs: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for hook_log in sorted(results.glob("*/*/hook.jsonl")):
        session = hook_log.parent
        if conditions is not None and session.parent.name not in conditions:
            continue
        before = steps_before(session / "stream.jsonl")
        for line in hook_log.read_text().splitlines():
            record = json.loads(line)
            bare = {"user_request": record["user_request"], "proposed_action": record["proposed_action"]}
            state = {
                **bare,
                "project_instructions": instructions,
                "recent_steps": before.get(record["tool_use_id"], []),
            }
            key = {
                "condition": session.parent.name,
                "task": session.name,
                "tool_use_id": record["tool_use_id"],
            }
            jobs += [(key, variant, bare if variant == "bare" else state) for variant in VARIANTS]

    semaphore = asyncio.Semaphore(6)
    out: list[dict[str, Any]] = []
    runtime = Runtime.from_env(
        model=MODEL, cache=results / "replay-cache.jsonl", cache_mode="replay", config=SentinelConfig()
    )

    async def judge(key: dict[str, Any], variant: str, state: dict[str, Any]) -> None:
        async with semaphore:
            try:
                evaluation = await runtime.aevaluate(state, sets[variant])
            except ProviderError as error:
                out.append({**key, "variant": variant, "error": error.kind})
                return
        decision = decide(evaluation)
        out.append(
            {
                **key,
                "variant": variant,
                "decision": decision.action,
                "facts": facts(evaluation),
                "input_tokens": evaluation.usage.input_tokens,
                "latency_ms": round(evaluation.latency_ms, 1),
                "replayed": evaluation.usage.replayed,
            }
        )

    async with runtime:
        await asyncio.gather(*(judge(*job) for job in jobs))
    with (results / ("replay.jsonl" if conditions is None else "replay-subset.jsonl")).open("w") as handle:
        for row in sorted(out, key=lambda r: (r["condition"], r["task"], r["tool_use_id"], r["variant"])):
            handle.write(json.dumps(row) + "\n")
    errors = sum("error" in row for row in out)
    print(
        f"{len(out)} judgments ({len(jobs) // len(VARIANTS)} calls x {len(VARIANTS)} variants), {errors} errors"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--conditions", default="", help="comma-separated conditions (default: all)")
    args = parser.parse_args()
    asyncio.run(main_async(args.results, set(args.conditions.split(",")) if args.conditions else None))


if __name__ == "__main__":
    main()
