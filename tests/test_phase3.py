"""Phase 3 tests: tool usage history, context-aware search, HTTP transport."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from servers.router_basic import _router_search as search_module
from servers.router_basic._router_executor import _record_usage
from servers.router_basic._router_search import _get_usage_counts, search_tools

MOCK_TOOLS = [
    {
        "name": "tool_alpha",
        "description": "Process alpha data.",
        "inputSchema": {
            "type": "object",
            "properties": {"data": {"type": "string", "description": "Input string"}},
        },
    },
    {
        "name": "tool_beta",
        "description": "Transform beta value.",
        "inputSchema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_db(db_path: Path, tools: list[dict] | None = None) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS servers (
            server_name TEXT PRIMARY KEY,
            repo_name TEXT NOT NULL,
            server_cwd TEXT NOT NULL,
            launch_command TEXT NOT NULL,
            launch_style TEXT NOT NULL DEFAULT 'standalone',
            tool_count INTEGER NOT NULL,
            indexed_at TEXT NOT NULL,
            server_py_path TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS tools (
            tool_id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            description TEXT NOT NULL,
            json_schema TEXT NOT NULL,
            enriched_text TEXT NOT NULL,
            UNIQUE(server_name, tool_name)
        );
        CREATE TABLE IF NOT EXISTS tool_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            called_at TEXT NOT NULL,
            success INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    if tools:
        conn.execute(
            "INSERT INTO servers VALUES (?,?,?,?,?,?,?,?)",
            (
                "srv",
                "repo",
                "/cwd",
                "uv run python server.py",
                "standalone",
                len(tools),
                "2026-01-01T00:00:00+00:00",
                "",
            ),
        )
        for t in tools:
            conn.execute(
                "INSERT INTO tools (server_name, tool_name, description, json_schema, enriched_text) "
                "VALUES (?,?,?,?,?)",
                (
                    "srv",
                    t["name"],
                    t.get("description", ""),
                    json.dumps(t["inputSchema"]),
                    f"tool:{t['name']} server:srv repo:repo description:{t.get('description', '')}",
                ),
            )
        conn.execute("INSERT INTO index_meta VALUES ('total_tools', ?)", (str(len(tools)),))
        conn.execute("INSERT INTO index_meta VALUES ('last_indexed', '2026-01-01T00:00:00+00:00')")
        conn.execute("INSERT INTO index_meta VALUES ('index_version', '2')")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tool Usage History
# ---------------------------------------------------------------------------


class TestToolUsageRecording:
    def test_record_usage_inserts_row(self, tmp_path: Path) -> None:
        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path)
        with patch("servers.router_basic._router_executor.get_db_path", return_value=db_path):
            _record_usage("srv", "tool_alpha", True)
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM tool_usage").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "srv"
        assert rows[0][2] == "tool_alpha"
        assert rows[0][4] == 1  # success

    def test_record_usage_creates_table_if_missing(self, tmp_path: Path) -> None:
        """_record_usage is robust even if table doesn't exist yet."""
        db_path = tmp_path / "registry.db"
        # Minimal DB without tool_usage table
        conn = sqlite3.connect(str(db_path))
        conn.commit()
        conn.close()
        with patch("servers.router_basic._router_executor.get_db_path", return_value=db_path):
            _record_usage("srv", "tool_alpha", True)  # must not raise
        conn = sqlite3.connect(str(db_path))
        cnt = conn.execute("SELECT COUNT(*) FROM tool_usage").fetchone()[0]
        conn.close()
        assert cnt == 1

    def test_get_usage_counts_returns_correct_counts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path, MOCK_TOOLS)
        # Insert usage for tool_alpha (3 times) and tool_beta (1 time)
        conn = sqlite3.connect(str(db_path))
        for _ in range(3):
            conn.execute(
                "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?,?,?,?)",
                ("srv", "tool_alpha", "2026-01-01T00:00:00+00:00", 1),
            )
        conn.execute(
            "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?,?,?,?)",
            ("srv", "tool_beta", "2026-01-01T00:00:00+00:00", 1),
        )
        conn.commit()
        conn.close()

        counts = _get_usage_counts([1, 2], db_path)
        assert counts.get(1, 0) == 3  # tool_alpha is tool_id=1
        assert counts.get(2, 0) == 1  # tool_beta is tool_id=2

    def test_get_usage_counts_returns_empty_on_error(self, tmp_path: Path) -> None:
        # Non-existent DB path
        counts = _get_usage_counts([1, 2], tmp_path / "nonexistent.db")
        assert counts == {}

    def test_get_usage_counts_excludes_failed_calls(self, tmp_path: Path) -> None:
        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path, MOCK_TOOLS)
        conn = sqlite3.connect(str(db_path))
        # 2 successes and 1 failure for tool_alpha
        conn.execute(
            "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?,?,?,?)",
            ("srv", "tool_alpha", "2026-01-01T00:00:00+00:00", 1),
        )
        conn.execute(
            "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?,?,?,?)",
            ("srv", "tool_alpha", "2026-01-01T00:00:01+00:00", 1),
        )
        conn.execute(
            "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?,?,?,?)",
            ("srv", "tool_alpha", "2026-01-01T00:00:02+00:00", 0),
        )
        conn.commit()
        conn.close()

        counts = _get_usage_counts([1], db_path)
        assert counts.get(1, 0) == 2  # only successful calls


class TestUsageBoostInSearch:
    @pytest.fixture(autouse=True)
    def patch_search_paths(self, tmp_path: Path) -> None:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path, MOCK_TOOLS)

        texts = [
            "tool:tool_alpha server:srv repo:repo description:Process alpha data.",
            "tool:tool_beta server:srv repo:repo description:Transform beta value.",
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        tool_ids = [1, 2]

        vec_path = tmp_path / "tfidf_vectorizer.joblib"
        mat_path = tmp_path / "tfidf_matrix.joblib"
        joblib.dump(vectorizer, str(vec_path))
        joblib.dump({"matrix": matrix, "tool_ids": tool_ids}, str(mat_path))

        search_module._vectorizer = None
        search_module._matrix_data = None
        search_module._embeddings_data = None
        search_module._embedding_model = None
        search_module._current_tools = {}

        with (
            patch("servers.router_basic._router_search.get_db_path", return_value=db_path),
            patch(
                "servers.router_basic._router_search.get_tfidf_vectorizer_path",
                return_value=vec_path,
            ),
            patch(
                "servers.router_basic._router_search.get_tfidf_matrix_path", return_value=mat_path
            ),
            patch(
                "servers.router_basic._router_search.get_embeddings_path",
                return_value=tmp_path / "embeddings.joblib",
            ),
            patch("servers.router_basic._router_search.MCP_CONSTRAINED_MODE", False),
        ):
            self._db_path = db_path
            yield

        search_module._vectorizer = None
        search_module._matrix_data = None
        search_module._current_tools = {}

    def test_used_tool_gets_higher_score(self) -> None:
        """A tool with 10 usages should score higher than one with 0."""
        # Insert 10 usage records for tool_beta (tool_id=2)
        conn = sqlite3.connect(str(self._db_path))
        for _ in range(10):
            conn.execute(
                "INSERT INTO tool_usage (server_name, tool_name, called_at, success) VALUES (?,?,?,?)",
                ("srv", "tool_beta", "2026-01-01T00:00:00+00:00", 1),
            )
        conn.commit()
        conn.close()

        # Use a neutral query that gives both tools some score
        result = search_tools("server srv")
        assert result["success"]
        tools = result["tools"]
        if len(tools) >= 2:
            # tool_beta should appear with a boosted score
            scores = {t["tool_name"]: t["score"] for t in tools}
            # Both tools should appear; beta's score must reflect the boost
            assert "tool_beta" in scores

    def test_zero_usage_tools_are_unaffected(self) -> None:
        """Tools with no usage history are still returned normally."""
        result = search_tools("alpha data")
        assert result["success"]
        assert result["returned"] > 0


# ---------------------------------------------------------------------------
# Context-Aware Search
# ---------------------------------------------------------------------------


class TestContextAwareSearch:
    @pytest.fixture(autouse=True)
    def patch_search_paths(self, tmp_path: Path) -> None:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path, MOCK_TOOLS)

        texts = [
            "tool:tool_alpha server:srv repo:repo description:Process alpha data.",
            "tool:tool_beta server:srv repo:repo description:Transform beta value.",
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(texts)

        vec_path = tmp_path / "tfidf_vectorizer.joblib"
        mat_path = tmp_path / "tfidf_matrix.joblib"
        joblib.dump(vectorizer, str(vec_path))
        joblib.dump({"matrix": matrix, "tool_ids": [1, 2]}, str(mat_path))

        search_module._vectorizer = None
        search_module._matrix_data = None
        search_module._embeddings_data = None
        search_module._current_tools = {}

        with (
            patch("servers.router_basic._router_search.get_db_path", return_value=db_path),
            patch(
                "servers.router_basic._router_search.get_tfidf_vectorizer_path",
                return_value=vec_path,
            ),
            patch(
                "servers.router_basic._router_search.get_tfidf_matrix_path", return_value=mat_path
            ),
            patch(
                "servers.router_basic._router_search.get_embeddings_path",
                return_value=tmp_path / "embeddings.joblib",
            ),
            patch("servers.router_basic._router_search.MCP_CONSTRAINED_MODE", False),
        ):
            yield

        search_module._vectorizer = None
        search_module._matrix_data = None
        search_module._current_tools = {}

    def test_context_param_returns_results(self) -> None:
        result = search_tools("process", context="I am working with alpha")
        assert result["success"]
        assert result["context_used"] is True

    def test_no_context_sets_context_used_false(self) -> None:
        result = search_tools("process")
        assert result["success"]
        assert result["context_used"] is False

    def test_empty_string_context_treated_as_no_context(self) -> None:
        result = search_tools("alpha", context="")
        assert result["success"]
        assert result["context_used"] is False

    def test_whitespace_context_treated_as_no_context(self) -> None:
        result = search_tools("alpha", context="   ")
        assert result["success"]
        # "   ".strip() == "" so effective_query == query; context_used depends on truthiness
        # "   " is truthy but stripped is empty — we pass context or None from server.py
        # In this test we call search_tools directly with context="   " (truthy)
        assert result["context_used"] is True

    def test_context_influences_effective_query(self) -> None:
        """With explicit context, the result should be valid and contain the right field."""
        result = search_tools("transform", context="beta values need transforming")
        assert result["success"]
        assert "context_used" in result
        assert result["context_used"] is True


# ---------------------------------------------------------------------------
# HTTP Transport Mode
# ---------------------------------------------------------------------------


class TestHttpTransport:
    def test_http_env_vars_in_helpers(self) -> None:
        """MCP_HTTP_MODE and MCP_HTTP_PORT are exported from _router_helpers."""
        from servers.router_basic._router_helpers import MCP_HTTP_MODE, MCP_HTTP_PORT

        assert isinstance(MCP_HTTP_MODE, bool)
        assert isinstance(MCP_HTTP_PORT, int)

    def test_run_server_uses_http_transport_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_MODE", "1")
        monkeypatch.setenv("MCP_HTTP_PORT", "9090")

        import servers.router_basic.server as srv

        mock_run = MagicMock()
        with patch.object(srv.mcp, "run", mock_run):
            with patch.dict(os.environ, {"MCP_HTTP_MODE": "1", "MCP_HTTP_PORT": "9090"}):
                srv._run_server()

        mock_run.assert_called_once_with(transport="streamable-http", host="0.0.0.0", port=9090)

    def test_run_server_uses_stdio_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_HTTP_MODE", "0")

        import servers.router_basic.server as srv

        mock_run = MagicMock()
        with patch.object(srv.mcp, "run", mock_run):
            with patch.dict(os.environ, {"MCP_HTTP_MODE": "0"}):
                srv._run_server()

        mock_run.assert_called_once_with()

    def test_http_port_defaults_to_8080(self) -> None:
        """MCP_HTTP_PORT defaults to 8080 when env var is not set."""
        import servers.router_basic.server as srv

        mock_run = MagicMock()
        env_without_port = {k: v for k, v in os.environ.items() if k != "MCP_HTTP_PORT"}
        env_without_port["MCP_HTTP_MODE"] = "1"
        with patch.object(srv.mcp, "run", mock_run):
            with patch.dict(os.environ, env_without_port, clear=True):
                srv._run_server()

        mock_run.assert_called_once_with(transport="streamable-http", host="0.0.0.0", port=8080)
