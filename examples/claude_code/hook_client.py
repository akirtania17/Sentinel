"""The Claude Code hook command: hands each call to the resident helper, or runs hook.py itself.

Starting Python and importing Sentinel takes about a third of a second on every tool call, and each
call opens a new connection to Jev. This client uses only the standard library, so it starts in a
few milliseconds, and passes the call to a helper process (helper.py) that keeps everything loaded
and connected. When no helper is running it starts one for the next call and runs hook.py for this
one, so the first call of a session costs what it always did.

Use it exactly like hook.py (same stdin, output and SENTINEL_HOOK_* settings). With Sentinel cloned
at /opt/sentinel:

    {"hooks": {
      "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "timeout": 30,
        "command": "/opt/sentinel/.venv/bin/python /opt/sentinel/examples/claude_code/hook_client.py"}]}],
      "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "timeout": 30,
        "command": "/opt/sentinel/.venv/bin/python /opt/sentinel/examples/claude_code/hook_client.py"}]}]}}

The PostToolUse entry is optional; it records which of the hook's questions you approved, for
calibrate.py. `SENTINEL_HOOK_HELPER=off` always runs hook.py directly.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOOK = HERE / "hook.py"
TIMEOUT_S = 28.0
"""Below the hook timeout Claude Code is given, so a stuck helper fails closed instead of timing out."""


def watched() -> list[Path]:
    """Every file whose change must start a fresh helper."""
    files = [HERE / name for name in ("hook.py", "helper.py", "checks.py", "intent.yaml", "context.yaml")]
    files += [REPO / "examples" / "agent_action" / "gate.py", REPO / "predicates" / "agent_action.yaml"]
    package = REPO / "src" / "sentinel_runtime"
    if package.is_dir():
        files += sorted(package.rglob("*.py"))
    return files


def socket_path() -> Path:
    """The helper's socket for this code, interpreter and API key."""
    digest = hashlib.sha256(sys.executable.encode())
    for file in watched():
        try:
            status = file.stat()
        except OSError:
            continue
        digest.update(f"{file}:{status.st_mtime_ns}:{status.st_size}".encode())
    digest.update(hashlib.sha256(os.environ.get("TYPESAFE_API_KEY", "").encode()).digest())
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".cache" / "sentinel") / "sentinel-hook"
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    return base / f"{digest.hexdigest()[:16]}.sock"


def in_process(raw: bytes) -> int:
    completed = subprocess.run([sys.executable, str(HOOK)], input=raw, capture_output=True, check=False)
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


def start_helper(path: Path) -> None:
    subprocess.Popen(
        [sys.executable, str(HOOK), "--serve", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def deny(reason: str) -> str:
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Sentinel: {reason}; failing closed",
            }
        }
    )


def main() -> int:
    raw = sys.stdin.buffer.read()
    if os.environ.get("SENTINEL_HOOK_HELPER", "on") == "off" or not hasattr(socket, "AF_UNIX"):
        return in_process(raw)
    path = socket_path()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(TIMEOUT_S)
    try:
        connection.connect(str(path))
    except OSError:
        connection.close()
        start_helper(path)
        return in_process(raw)
    env = {key: value for key, value in os.environ.items() if key.startswith("SENTINEL_HOOK_")}
    enforce = env.get("SENTINEL_HOOK_MODE", "observe") == "enforce"
    try:
        with connection:
            connection.sendall(json.dumps({"call": json.loads(raw), "env": env}).encode())
            connection.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while chunk := connection.recv(65536):
                chunks.append(chunk)
        reply = json.loads(b"".join(chunks))
    except (OSError, ValueError) as error:
        if enforce:
            print(deny(f"the hook helper failed ({type(error).__name__})"))
            return 0
        raise
    if reply.get("stdout"):
        print(reply["stdout"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
