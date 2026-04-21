"""Tests for _router_executor: JIT launch, tool call proxy, TTL pool, errors."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import servers.router_basic._router_executor as executor_mod
from servers.router_basic._router_executor import execute_tool, shutdown_pool

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_A_DIR = FIXTURES_DIR / "mock_server_a"
MOCK_B_DIR = FIXTURES_DIR / "mock_server_b"


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "registry.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE servers (
            server_name TEXT PRIMARY KEY, repo_name TEXT NOT NULL,
            server_cwd TEXT NOT NULL, launch_command TEXT NOT NULL,
            launch_style TEXT NOT NULL DEFAULT 'standalone',
            tool_count INTEGER NOT NULL, indexed_at TEXT NOT NULL
        );
        CREATE TABLE tools (
            tool_id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name TEXT NOT NULL, tool_name TEXT NOT NULL,
            description TEXT NOT NULL, json_schema TEXT NOT NULL,
            enriched_text TEXT NOT NULL,
            UNIQUE(server_name, tool_name)
        );
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    # Use sys.executable so the mock servers run in the current venv (no uv needed)
    conn.execute(
        "INSERT INTO servers VALUES (?, 'mock_repo', ?, ?, 'standalone', 3, '2026-01-01')",
        ("mock_server_a", str(MOCK_A_DIR), f"{sys.executable} server.py"),
    )
    conn.execute(
        "INSERT INTO servers VALUES (?, 'mock_repo', ?, ?, 'standalone', 2, '2026-01-01')",
        ("mock_server_b", str(MOCK_B_DIR), f"{sys.executable} server.py"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture(autouse=True)
def reset_executor_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level state and patch DB path before each test."""
    shutdown_pool()
    executor_mod._server_pool.clear()
    executor_mod._bg_loop = None
    executor_mod._bg_thread = None

    db_path = _make_db(tmp_path)
    monkeypatch.setattr("servers.router_basic._router_executor.get_db_path", lambda: db_path)

    yield

    shutdown_pool()
    executor_mod._server_pool.clear()


class TestExecuteToolErrors:
    def test_error_for_unknown_server(self) -> None:
        result = execute_tool("nonexistent_server", "some_tool", "{}")
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "hint" in result

    def test_error_for_invalid_json_arguments(self) -> None:
        result = execute_tool("mock_server_a", "tool_alpha", "not-valid-json")
        assert result["success"] is False
        assert "JSON" in result["error"] or "json" in result["error"].lower()
        assert result["hint"] == "arguments must be valid JSON string"

    def test_error_for_missing_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "servers.router_basic._router_executor.get_db_path",
            lambda: tmp_path / "no.db",
        )
        result = execute_tool("any_server", "any_tool", "{}")
        assert result["success"] is False
        assert "Index not found" in result["error"]

    def test_error_for_bad_server_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        db_path = bad_dir / "registry.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE servers (server_name TEXT PRIMARY KEY, repo_name TEXT NOT NULL,
            server_cwd TEXT NOT NULL, launch_command TEXT NOT NULL,
            launch_style TEXT NOT NULL DEFAULT 'standalone',
            tool_count INTEGER NOT NULL, indexed_at TEXT NOT NULL);
            CREATE TABLE tools (tool_id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name TEXT NOT NULL, tool_name TEXT NOT NULL, description TEXT NOT NULL,
            json_schema TEXT NOT NULL, enriched_text TEXT NOT NULL,
            UNIQUE(server_name, tool_name));
            CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        conn.execute(
            "INSERT INTO servers VALUES ('bad_srv', 'r', '/nonexistent', "
            "'python nonexistent_script.py', 'standalone', 0, '2026-01-01')"
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr("servers.router_basic._router_executor.get_db_path", lambda: db_path)
        result = execute_tool("bad_srv", "tool", "{}")
        assert result["success"] is False


class TestExecuteToolSuccess:
    def test_successful_call_to_tool_alpha(self) -> None:
        result = execute_tool("mock_server_a", "tool_alpha", '{"data": "hello"}')
        assert result["success"] is True
        assert "result" in result
        assert "elapsed_seconds" in result

    def test_result_contains_child_tool_output(self) -> None:
        result = execute_tool("mock_server_a", "tool_alpha", '{"data": "world"}')
        assert result["success"] is True
        assert "alpha:world" in str(result["result"])

    def test_token_estimate_present(self) -> None:
        result = execute_tool("mock_server_a", "tool_gamma", "{}")
        assert result["success"] is True
        assert "token_estimate" in result
        assert isinstance(result["token_estimate"], int)

    def test_progress_array_present(self) -> None:
        result = execute_tool("mock_server_a", "tool_gamma", "{}")
        assert result["success"] is True
        assert isinstance(result["progress"], list)
        assert len(result["progress"]) > 0


class TestTtlPool:
    def test_server_added_to_pool_after_call(self) -> None:
        execute_tool("mock_server_a", "tool_gamma", "{}")
        assert "mock_server_a" in executor_mod._server_pool

    def test_second_call_reuses_pool_entry(self) -> None:
        execute_tool("mock_server_a", "tool_alpha", '{"data": "a"}')
        pool_size_1 = len(executor_mod._server_pool)

        execute_tool("mock_server_a", "tool_alpha", '{"data": "b"}')
        pool_size_2 = len(executor_mod._server_pool)

        assert pool_size_1 == 1
        assert pool_size_2 == 1  # not duplicated

    def test_different_servers_have_separate_pool_entries(self) -> None:
        execute_tool("mock_server_a", "tool_gamma", "{}")
        execute_tool("mock_server_b", "tool_delta", '{"x": 1.0, "y": 2.0}')
        assert len(executor_mod._server_pool) == 2
        assert "mock_server_a" in executor_mod._server_pool
        assert "mock_server_b" in executor_mod._server_pool
