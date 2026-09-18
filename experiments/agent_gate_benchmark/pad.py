"""Make a copy of a scenario file with long, irrelevant tool output added to every scenario.

    python3 experiments/agent_gate_benchmark/pad.py SCENARIOS.yaml OUT.yaml TOKENS

The padding is a realistic dependency listing and build log, prepended to the output of each
scenario's oldest step (or added as a step when there are none), so the deciding facts stay where
they were and only the amount of irrelevant text changes. It contains no instructions.
"""

import random
import sys
from pathlib import Path

import yaml

CHARS_PER_TOKEN = 3.2
PACKAGES = [
    "react", "react-dom", "lodash", "axios", "express", "webpack", "babel-loader", "eslint", "jest",
    "typescript", "zod", "date-fns", "uuid", "chalk", "commander", "yargs", "dotenv", "cors", "helmet",
    "pino", "semver", "glob", "rimraf", "prettier", "vite", "rollup", "postcss", "tailwindcss",
]  # fmt: skip


def padding(tokens: int, seed: str) -> str:
    rng = random.Random(seed)  # noqa: S311 - reproducible filler text, not security
    lines = ["$ npm ls --all (earlier output)", "acme-app@4.12.0 /srv/app"]
    target = int(tokens * CHARS_PER_TOKEN)
    size = 0
    depth_prefix = ["├── ", "│ ├── ", "│ │ ├── ", "│ │ │ └── "]
    while size < target:
        if rng.random() < 0.7:
            line = f"{rng.choice(depth_prefix)}{rng.choice(PACKAGES)}@{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 20)}"
        else:
            line = (
                f"[{rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}] webpack compiled "
                f"chunk {rng.randint(100, 999)} ({rng.randint(1, 900)} KiB) in {rng.randint(20, 900)} ms"
            )
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines) + "\n"


def main() -> None:
    source, target, tokens = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    for scenario in data["scenarios"]:
        text = padding(tokens, scenario["id"])
        steps = scenario.get("recent_steps") or []
        if steps:
            steps[0]["result"] = text + str(steps[0].get("result", ""))
        else:
            scenario["recent_steps"] = [
                {"tool": "Bash", "arguments": {"command": "npm ls --all"}, "result": text}
            ]
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
