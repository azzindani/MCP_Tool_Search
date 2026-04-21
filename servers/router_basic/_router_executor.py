"""JIT server launch, MCP call proxy, and TTL process pool management."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from contextlib import AsyncExitStack
from typing import Any

from ._router_helpers import REAP_INTERVAL, TTL_SECONDS, get_db_path

# TTL pool: server_name -> {session, stack, last_used}
_server_pool: dict[str, dict[str, Any]] = {}
_pool_lock = threading.Lock()

# Persistent background event loop for async MCP sessions
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None


def _get_or_start_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    if _bg_loop is not None and _bg_loop.is_running():
        return _bg_loop
    _bg_loop = asyncio.new_event_loop()
    _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
    _bg_thread.start()
    asyncio.run_coroutine_threadsafe(_start_reaper(), _bg_loop)
    return _bg_loop


async def _start_reaper() -> None:
    while True:
        await asyncio.sleep(REAP_INTERVAL)
        now = time.monotonic()
        expired = []
        with _pool_lock:
            for name, entry in _server_pool.items():
                if now - entry["last_used"] > TTL_SECONDS:
                    expired.append(name)
        for name in expired:
            await _close_pool_entry(name)


async def _close_pool_entry(server_name: str) -> None:
    with _pool_lock:
        entry = _server_pool.pop(server_name, None)
    if entry and entry.get("stack"):
        try:
            await entry["stack"].aclose()
        except Exception:
            pass


async def _get_or_launch_session(server_name: str, server_cwd: str, launch_command: str) -> Any:
    """Return a live MCP ClientSession, launching the server if needed."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with _pool_lock:
        entry = _server_pool.get(server_name)
        if entry:
            entry["last_used"] = time.monotonic()
            return entry["session"]

    parts = launch_command.split()
    server_params = StdioServerParameters(
        command=parts[0],
        args=parts[1:],
        cwd=server_cwd,
    )

    stack = AsyncExitStack()
    read, write = await stack.enter_async_context(stdio_client(server_params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    with _pool_lock:
        if server_name not in _server_pool:
            _server_pool[server_name] = {
                "session": session,
                "stack": stack,
                "last_used": time.monotonic(),
            }
        else:
            # Concurrent call already launched; close our duplicate
            await stack.aclose()
            return _server_pool[server_name]["session"]

    return session


def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _record_usage(server_name: str, tool_name: str, success: bool) -> None:
    """Record a tool call in tool_usage (best-effort, never raises)."""
    try:
        from datetime import UTC, datetime

        conn = sqlite3.connect(str(get_db_path()))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                server_name TEXT NOT NULL,
                tool_name   TEXT NOT NULL,
                called_at   TEXT NOT NULL,
                success     INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute(
            "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?, ?, ?, ?)",
            (server_name, tool_name, datetime.now(UTC).isoformat(), int(success)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


async def _execute_tool_async(server_name: str, tool_name: str, arguments: str) -> dict:
    from shared.progress import fail, info, ok

    progress: list[dict] = []
    db_path = get_db_path()

    if not db_path.exists():
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": "Index not found.",
            "hint": "Call list_servers()",
            "progress": [fail("Index not found")],
            "token_estimate": 50,
        }

    conn = _get_db_conn()
    row = conn.execute(
        "SELECT server_cwd, launch_command FROM servers WHERE server_name = ?",
        (server_name,),
    ).fetchone()
    conn.close()

    if not row:
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": f"Server '{server_name}' not found in index.",
            "hint": "Call list_servers()",
            "progress": [fail(f"Unknown server: {server_name}")],
            "token_estimate": 50,
        }

    try:
        args_dict = json.loads(arguments)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": f"Invalid JSON in arguments: {e}",
            "hint": "arguments must be valid JSON string",
            "progress": [fail("JSON parse error")],
            "token_estimate": 50,
        }

    server_cwd = row["server_cwd"]
    launch_command = row["launch_command"]
    progress.append(info(f"Connecting to {server_name}"))
    t_start = time.monotonic()

    try:
        session = await asyncio.wait_for(
            _get_or_launch_session(server_name, server_cwd, launch_command),
            timeout=60.0,
        )
    except TimeoutError:
        await _close_pool_entry(server_name)
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": "Server startup timed out.",
            "hint": "Check uv sync in server directory",
            "progress": [fail("Server startup timeout")],
            "token_estimate": 50,
        }
    except Exception as e:
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": f"Server startup failed: {e}",
            "hint": "Check uv sync in server directory",
            "progress": [fail(f"Startup error: {e}")],
            "token_estimate": 50,
        }

    progress.append(info(f"Calling {tool_name} on {server_name}"))

    try:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, args_dict),
            timeout=120.0,
        )
    except TimeoutError:
        await _close_pool_entry(server_name)
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": "Tool call timed out.",
            "hint": "Operation timed out",
            "progress": [fail("Tool call timeout")],
            "token_estimate": 50,
        }
    except Exception as e:
        await _close_pool_entry(server_name)
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": f"Tool call failed: {e}",
            "hint": "Call search_tools()",
            "progress": [fail(f"Tool error: {e}")],
            "token_estimate": 50,
        }

    elapsed = round(time.monotonic() - t_start, 3)
    progress.append(ok(f"Completed in {elapsed}s"))
    _record_usage(server_name, tool_name, True)

    result_data: Any = {}
    if hasattr(result, "content"):
        for item in result.content:
            if hasattr(item, "text"):
                try:
                    result_data = json.loads(item.text)
                except Exception:
                    result_data = {"text": item.text}
                break
    else:
        result_data = {"raw": str(result)}

    return {
        "success": True,
        "op": "execute_tool",
        "server_name": server_name,
        "tool_name": tool_name,
        "result": result_data,
        "elapsed_seconds": elapsed,
        "progress": progress,
        "token_estimate": len(json.dumps(result_data, default=str)) // 4,
    }


def execute_tool(server_name: str, tool_name: str, arguments: str) -> dict:
    """Proxy a tool call to a child MCP server via JIT launch."""
    loop = _get_or_start_loop()
    future = asyncio.run_coroutine_threadsafe(
        _execute_tool_async(server_name, tool_name, arguments),
        loop,
    )
    try:
        return future.result(timeout=130)
    except TimeoutError:
        return {
            "success": False,
            "op": "execute_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "error": "Overall timeout exceeded.",
            "hint": "Operation timed out",
            "progress": [],
            "token_estimate": 50,
        }


def shutdown_pool() -> None:
    """Close all pooled MCP sessions. Call on router shutdown."""
    loop = _get_or_start_loop()
    with _pool_lock:
        names = list(_server_pool.keys())
    futures = [asyncio.run_coroutine_threadsafe(_close_pool_entry(name), loop) for name in names]
    for f in futures:
        try:
            f.result(timeout=5)
        except Exception:
            pass
