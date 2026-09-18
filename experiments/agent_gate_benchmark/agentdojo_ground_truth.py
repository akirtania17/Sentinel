"""Export AgentDojo's task definitions for agentdojo.py and agentdojo_contract.py.

    uv run --no-project --with agentdojo python experiments/agent_gate_benchmark/agentdojo_ground_truth.py \
        INJECTIONS.json [SUITES.json]

INJECTIONS.json maps `suite/injection_task_N` to the injection's goal and ground-truth calls.
SUITES.json maps each suite to its tools (name, description, parameters) and its user tasks (prompt and
the functions of the ground-truth solution).
"""

import json
import sys
from typing import Any

from agentdojo.task_suite.load_suites import get_suite  # pyright: ignore[reportMissingImports]

SUITES = ("banking", "slack", "travel", "workspace")


def main() -> None:
    injections: dict[str, dict[str, Any]] = {}
    suites: dict[str, dict[str, Any]] = {}
    for name in SUITES:
        suite = get_suite("v1", name)
        environment = suite.load_and_inject_default_environment({})
        for task_id, task in suite.injection_tasks.items():
            calls = task.ground_truth(environment.model_copy(deep=True))
            injections[f"{name}/{task_id}"] = {
                "goal": task.GOAL,
                "calls": [{"function": c.function, "args": dict(c.args)} for c in calls],
            }
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": sorted(tool.parameters.model_json_schema().get("properties", {})),
            }
            for tool in suite.tools
        ]
        tasks = {
            task_id: {
                "prompt": task.PROMPT,
                "functions": sorted(
                    {c.function for c in task.ground_truth(environment.model_copy(deep=True))}
                ),
            }
            for task_id, task in suite.user_tasks.items()
        }
        suites[name] = {"tools": tools, "user_tasks": tasks}
    with open(sys.argv[1], "w") as handle:
        json.dump(injections, handle, indent=1, default=str)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as handle:
            json.dump(suites, handle, indent=1, default=str)
    print(
        f"{len(injections)} injection tasks, {sum(len(s['user_tasks']) for s in suites.values())} user tasks"
    )


if __name__ == "__main__":
    main()
