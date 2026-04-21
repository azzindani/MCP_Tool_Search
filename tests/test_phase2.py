"""Phase 2 tests: server_py_path, staleness detection, semantic index, hybrid search."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from servers.router_basic import _router_search as search_module
from servers.router_basic._router_indexer import (
    _build_semantic_index,
    _migrate_schema,
    discover_servers,
    list_servers,
    reindex,
)
from servers.router_basic._router_search import search_tools

FIXTURES_DIR = Path(__file__).parent / "fixtures"

MOCK_TOOLS = [
    {
        "name": "tool_alpha",
        "description": "Process alpha data.",
        "inputSchema": {
            "type": "object",
            "properties": {"data": {"type": "string", "description": "Input string"}},
            "required": ["data"],
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
    """Create a minimal registry.db with optional tools."""
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
                "INSERT INTO tools (server_name, tool_name, description, json_schema, enriched_text) VALUES (?,?,?,?,?)",
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
# discover_servers includes server_py_path
# ---------------------------------------------------------------------------


class TestDiscoverServersPyPath:
    def test_standalone_has_server_py_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "SomeRepo"
        srv = repo / "servers" / "srv_basic"
        srv.mkdir(parents=True)
        (srv / "server.py").write_text("# stub")
        (srv / "pyproject.toml").write_text("[project]\nname='x'\n")

        results = discover_servers(tmp_path)
        assert len(results) == 1
        assert results[0]["server_py_path"] == str(srv / "server.py")

    def test_module_style_has_server_py_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "SomeRepo"
        repo.mkdir(parents=True)
        (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
        srv = repo / "servers" / "srv_mod"
        srv.mkdir(parents=True)
        (srv / "server.py").write_text("# stub")
        # No pyproject.toml in srv — module style

        results = discover_servers(tmp_path)
        assert len(results) == 1
        assert results[0]["server_py_path"] == str(srv / "server.py")
        assert results[0]["launch_style"] == "module"


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestMigrateSchema:
    def test_adds_column_to_old_schema(self, tmp_path: Path) -> None:
        """Old DB without server_py_path column gets migrated."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE servers (
                server_name TEXT PRIMARY KEY,
                repo_name TEXT NOT NULL,
                server_cwd TEXT NOT NULL,
                launch_command TEXT NOT NULL,
                launch_style TEXT NOT NULL DEFAULT 'standalone',
                tool_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            );
        """)
        conn.commit()
        _migrate_schema(conn)
        # Should succeed — column now exists
        conn.execute("SELECT server_py_path FROM servers")
        conn.close()

    def test_idempotent_on_new_schema(self, tmp_path: Path) -> None:
        """Running migrate twice doesn't raise."""
        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path)
        conn = sqlite3.connect(str(db_path))
        _migrate_schema(conn)
        _migrate_schema(conn)
        conn.close()


# ---------------------------------------------------------------------------
# reindex stores server_py_path in DB
# ---------------------------------------------------------------------------


class TestReindexServerPyPath:
    def test_stores_server_py_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "MockRepo"
        srv = repo / "servers" / "mock_srv"
        srv.mkdir(parents=True)
        server_py = srv / "server.py"
        server_py.write_text("# stub")
        (srv / "pyproject.toml").write_text("[project]\nname='x'\n")

        fake_tools = [
            {
                "name": "t1",
                "description": "desc1",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

        with (
            patch(
                "servers.router_basic._router_indexer.get_mcp_base_dir",
                return_value=tmp_path,
            ),
            patch(
                "servers.router_basic._router_indexer.get_db_path",
                return_value=tmp_path / "registry.db",
            ),
            patch(
                "servers.router_basic._router_indexer.get_router_dir",
                return_value=tmp_path,
            ),
            patch(
                "servers.router_basic._router_indexer.get_tfidf_vectorizer_path",
                return_value=tmp_path / "tfidf_vectorizer.joblib",
            ),
            patch(
                "servers.router_basic._router_indexer.get_tfidf_matrix_path",
                return_value=tmp_path / "tfidf_matrix.joblib",
            ),
            patch(
                "servers.router_basic._router_indexer.get_embeddings_path",
                return_value=tmp_path / "embeddings.joblib",
            ),
            patch(
                "servers.router_basic._router_indexer._collect_tools_from_server",
                return_value=fake_tools,
            ),
            patch(
                "servers.router_basic._router_indexer._build_semantic_index",
                return_value=False,
            ),
        ):
            result = reindex(base_dir=tmp_path)

        assert result["success"]
        conn = sqlite3.connect(str(tmp_path / "registry.db"))
        row = conn.execute(
            "SELECT server_py_path FROM servers WHERE server_name='mock_srv'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == str(server_py)


# ---------------------------------------------------------------------------
# list_servers staleness detection
# ---------------------------------------------------------------------------


class TestListServersStale:
    def _make_db_with_server_py(self, db_path: Path, server_py: Path, indexed_at: str) -> None:
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
                enriched_text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO servers VALUES (?,?,?,?,?,?,?,?)",
            (
                "srv",
                "repo",
                "/cwd",
                "uv run python server.py",
                "standalone",
                1,
                indexed_at,
                str(server_py),
            ),
        )
        conn.execute("INSERT INTO index_meta VALUES ('total_tools', '1')")
        conn.execute("INSERT INTO index_meta VALUES ('last_indexed', ?)", (indexed_at,))
        conn.execute("INSERT INTO index_meta VALUES ('index_version', '2')")
        conn.commit()
        conn.close()

    def test_stale_when_file_modified_after_index(self, tmp_path: Path) -> None:
        server_py = tmp_path / "server.py"
        server_py.write_text("# stub")
        # Use a timestamp well in the past so file mtime is definitely newer
        indexed_at = "2020-01-01T00:00:00+00:00"
        db_path = tmp_path / "registry.db"
        self._make_db_with_server_py(db_path, server_py, indexed_at)

        with patch(
            "servers.router_basic._router_indexer.get_db_path",
            return_value=db_path,
        ):
            result = list_servers()

        assert result["success"]
        assert "srv" in result["stale_servers"]
        assert result["servers"][0]["stale"] is True
        assert "reindex_hint" in result

    def test_fresh_when_file_not_modified(self, tmp_path: Path) -> None:
        server_py = tmp_path / "server.py"
        server_py.write_text("# stub")
        # Use far-future indexed_at so file looks fresh
        indexed_at = "2099-01-01T00:00:00+00:00"
        db_path = tmp_path / "registry.db"
        self._make_db_with_server_py(db_path, server_py, indexed_at)

        with patch(
            "servers.router_basic._router_indexer.get_db_path",
            return_value=db_path,
        ):
            result = list_servers()

        assert result["success"]
        assert result["stale_servers"] == []
        assert result["servers"][0]["stale"] is False
        assert "reindex_hint" not in result


# ---------------------------------------------------------------------------
# _build_semantic_index — graceful skip without sentence-transformers
# ---------------------------------------------------------------------------


class TestBuildSemanticIndex:
    def test_returns_false_without_sentence_transformers(self, tmp_path: Path) -> None:
        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path, MOCK_TOOLS)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        import sys

        # Temporarily hide sentence_transformers
        orig = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            result = _build_semantic_index(conn)
        finally:
            if orig is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = orig
        conn.close()
        assert result is False

    def test_returns_true_and_saves_file_with_sentence_transformers(self, tmp_path: Path) -> None:
        pytest.importorskip("sentence_transformers")
        db_path = tmp_path / "registry.db"
        _make_mock_db(db_path, MOCK_TOOLS)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        with (
            patch(
                "servers.router_basic._router_indexer.get_embeddings_path",
                return_value=tmp_path / "embeddings.joblib",
            ),
            patch(
                "servers.router_basic._router_indexer.get_router_dir",
                return_value=tmp_path,
            ),
        ):
            result = _build_semantic_index(conn)

        conn.close()
        assert result is True
        assert (tmp_path / "embeddings.joblib").exists()


# ---------------------------------------------------------------------------
# search_tools retrieval_mode field
# ---------------------------------------------------------------------------


class TestSearchRetrievalMode:
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
            patch("servers.router_basic._router_search.is_constrained_mode", return_value=False),
        ):
            yield

        search_module._vectorizer = None
        search_module._matrix_data = None
        search_module._embeddings_data = None
        search_module._embedding_model = None
        search_module._current_tools = {}

    def test_retrieval_mode_tfidf_when_no_embeddings(self) -> None:
        result = search_tools("alpha data")
        assert result["success"]
        assert result["retrieval_mode"] == "tfidf"

    def test_retrieval_mode_hybrid_when_embeddings_present(self, tmp_path: Path) -> None:
        pytest.importorskip("sentence_transformers")
        import joblib
        import numpy as np

        # Create fake embeddings matching the 2 tools
        embeddings = np.random.rand(2, 384).astype("float32")
        tool_ids = [1, 2]
        emb_path = tmp_path / "embeddings.joblib"
        joblib.dump({"embeddings": embeddings, "tool_ids": tool_ids}, str(emb_path))

        with patch(
            "servers.router_basic._router_search.get_embeddings_path",
            return_value=emb_path,
        ):
            search_module._embeddings_data = None
            result = search_tools("alpha data")

        assert result["success"]
        assert result["retrieval_mode"] == "hybrid"

    def test_no_match_includes_retrieval_mode(self) -> None:
        result = search_tools("zzzznonexistent")
        assert "retrieval_mode" in result
