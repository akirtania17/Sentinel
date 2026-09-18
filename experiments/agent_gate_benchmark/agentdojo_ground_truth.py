"""Export each AgentDojo injection task's goal and ground-truth calls, for agentdojo.py.

uv run --no-project --with agentdojo python experiments/agent_gate_benchmark/agentdojo_ground_truth.py OUT.json
"""

import json
import sys

from agentdojo.task_suite.load_suites import get_suite  # pyright: ignore[reportMissingImports]


def main() -> None:
    out: dict[str, dict[str, object]] = {}
    for name in ("banking", "slack", "travel", "workspace"):
        suite = get_suite("v1", name)
        environment = suite.load_and_inject_default_environment({})
        for task_id, task in suite.injection_tasks.items():
            calls = task.ground_truth(environment.model_copy(deep=True))
            out[f"{name}/{task_id}"] = {
                "goal": task.GOAL,
                "calls": [{"function": c.function, "args": dict(c.args)} for c in calls],
            }
    with open(sys.argv[1], "w") as handle:
        json.dump(out, handle, indent=1, default=str)
    print(f"{len(out)} injection tasks")


if __name__ == "__main__":
    main()
