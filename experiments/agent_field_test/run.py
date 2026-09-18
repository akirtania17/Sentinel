"""Run real Claude Code agents on the field-test tasks with the Sentinel hook on every tool call.

Each (condition, task) session gets a fresh copy of the sandbox project. Claude Code runs headless
with no MCP servers, no user settings, only the Bash, Read, Edit, Write, Glob and Grep tools, and a
short allowlist of shell commands; anything else is denied. The hook records every proposed tool
call, including denied ones, and the sandbox's team tools record every effect that actually ran.

    uv run python experiments/agent_field_test/run.py --out RESULTS_DIR [--conditions a,b] [--tasks x,y]

The hook reads the TypeSafe key from the file named by `SENTINEL_KEY_FILE` (a shell file that
exports TYPESAFE_API_KEY); the agents themselves never get the key in their environment.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
HOOK = REPO / "examples" / "claude_code" / "hook.py"
TIMEOUT_S = 420


def allowed_tools(run_dir: Path) -> list[str]:
    tools = ["Read(./**)", "Glob", "Grep", "Edit(./**)", "Write(./**)"]
    for prefix in ("./bin/", "bin/", f"{run_dir}/bin/", "python3 bin/", "python3 ./bin/"):
        tools.append(f"Bash({prefix}*)")
    tools += ["Bash(python3 -m unittest*)", "Bash(ls*)", "Bash(git status*)", "Bash(git diff*)"]
    return tools


def run_session(
    out: Path, condition: str, config: dict[str, Any], task: dict[str, Any], spec: dict[str, Any]
) -> str:
    session_dir = out / condition / task["id"]
    if (session_dir / "done.json").exists():
        return f"{condition}/{task['id']}: already done"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    project = session_dir / "project"
    shutil.copytree(HERE / "sandbox", project)
    shutil.move(project / "dot-env", project / ".env")
    hook_log = session_dir / "hook.jsonl"

    key_file = os.environ["SENTINEL_KEY_FILE"]
    hook_command = (
        f". {shlex.quote(key_file)} && SENTINEL_HOOK_MODE={config['mode']} "
        f"SENTINEL_HOOK_CONTEXT={config.get('context', 'off')} SENTINEL_HOOK_POLICY={config.get('policy', 'strict')} "
        f"SENTINEL_HOOK_LOG={shlex.quote(str(hook_log))} exec {shlex.quote(str(PYTHON))} {shlex.quote(str(HOOK))}"
    )
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {"type": "command", "command": f"bash -c {shlex.quote(hook_command)}", "timeout": 60}
                    ],
                }
            ]
        }
    }
    settings_path = session_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))

    command = [
        "claude", "-p", task["prompt"],
        "--model", config["model"],
        "--setting-sources", "project",
        "--settings", str(settings_path),
        "--strict-mcp-config",
        "--tools", "Bash", "Read", "Edit", "Write", "Glob", "Grep",
        "--permission-mode", "dontAsk",
        "--allowedTools", *allowed_tools(project),
        "--output-format", "stream-json", "--verbose",
    ]  # fmt: skip
    if config.get("compromised"):
        command += ["--append-system-prompt", spec["compromised_prompt"]]
    if config.get("system"):
        command += ["--append-system-prompt", spec[config["system"]]]
    env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
    with (session_dir / "stream.jsonl").open("w") as stream, (session_dir / "stderr.txt").open("w") as err:
        try:
            completed = subprocess.run(
                command, cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=err,
                timeout=TIMEOUT_S, check=False,
            )  # fmt: skip
            status = completed.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    effects = project / ".effects.log"
    done = {
        "condition": condition,
        "task": task["id"],
        "status": status,
        "effects": [json.loads(line) for line in effects.read_text().splitlines()]
        if effects.exists()
        else [],
    }
    (session_dir / "done.json").write_text(json.dumps(done, indent=2))
    calls = len(hook_log.read_text().splitlines()) if hook_log.exists() else 0
    return f"{condition}/{task['id']}: exit {status}, {calls} tool calls, {len(done['effects'])} effects"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    if "SENTINEL_KEY_FILE" not in os.environ:
        sys.exit("set SENTINEL_KEY_FILE to a shell file that exports TYPESAFE_API_KEY")

    spec = yaml.safe_load((HERE / "tasks.yaml").read_text())
    conditions = spec["conditions"]
    chosen_conditions = args.conditions.split(",") if args.conditions else list(conditions)
    chosen_tasks = set(args.tasks.split(",")) if args.tasks else None
    jobs = [
        (condition, task)
        for condition in chosen_conditions
        for task in spec["tasks"]
        if chosen_tasks is None or task["id"] in chosen_tasks
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_session, args.out, c, conditions[c], t, spec) for c, t in jobs]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
