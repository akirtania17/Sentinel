"""The hook's resident helper: keeps the hook's code loaded and one connection to Jev open.

Started by `hook_client.py` the first time it finds no helper, the helper listens on a Unix socket
that only its user can open, and judges each call it receives exactly as `hook.py` would. It exits,
removing its socket, after `SENTINEL_HOOK_IDLE` seconds without a call (default 1800) or on SIGTERM.
Its socket name includes a digest of the hook's code and of the API key, so after any change a new
helper starts and the old one idles out.
"""

import asyncio
import contextlib
import json
import os
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from sentinel_runtime import TypeSafeProvider
from sentinel_runtime.providers import Provider

Handle = Callable[[Mapping[str, Any], Mapping[str, str], Provider, str], Awaitable[str | None]]
FailClosed = Callable[[BaseException, Mapping[str, str]], str | None]


def serve(
    path: Path, handle: Handle, fail_closed: FailClosed, provider: Callable[[], Provider] = TypeSafeProvider
) -> None:
    """Serve hook calls on the Unix socket at `path` until idle, sharing one `provider()`."""
    asyncio.run(_serve(path, handle, fail_closed, provider))


async def _running(path: Path) -> bool:
    try:
        _, writer = await asyncio.open_unix_connection(str(path))
    except OSError:
        return False
    writer.close()
    return True


async def _serve(
    path: Path, handle: Handle, fail_closed: FailClosed, provider: Callable[[], Provider]
) -> None:
    if await _running(path):
        return  # another helper already serves this socket
    idle = float(os.environ.get("SENTINEL_HOOK_IDLE", "1800"))
    inner = provider()
    active = 0
    last = time.monotonic()

    async def client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal active, last
        active += 1
        try:
            message = json.loads(await reader.read())
            env: dict[str, str] = message["env"]
            try:
                reply: dict[str, Any] = {"stdout": await handle(message["call"], env, inner, "helper")}
            except Exception as error:
                reply = {"stdout": fail_closed(error, env), "error": type(error).__name__}
            writer.write(json.dumps(reply).encode())
            await writer.drain()
        finally:
            writer.close()
            active -= 1
            last = time.monotonic()

    previous = os.umask(0o177)  # the socket is created readable and writable by its owner only
    try:
        server = await asyncio.start_unix_server(client, path=str(path))
    finally:
        os.umask(previous)
    stop = asyncio.Event()
    if threading.current_thread() is threading.main_thread():  # always, except in tests
        for number in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(number, stop.set)
    try:
        while not stop.is_set() and (active or time.monotonic() - last < idle):
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=min(idle, 30))
    finally:
        server.close()
        path.unlink(missing_ok=True)
        await inner.aclose()
