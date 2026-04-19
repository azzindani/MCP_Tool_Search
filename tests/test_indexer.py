"""Tests for _router_indexer: discovery, enriched text, SQLite population."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from servers.router_basic._router_indexer import (
    _build_enriched_text,
    discover_servers,
    reindex,
)

MOCK_TOOLS_A = [
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
            "properties": {
                "value": {"type": "integer"},
                "label": {"type": "string", "description": "Label text"},
            },
        },
    },
    {
        "name": "tool_gamma",
        "description": "Return gamma status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

MOCK_TOOLS_B = [
    {
        "name": "tool_delta",
        "description": "Delta processing.",
        "inputSchema": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        },
    },
    {
        "name": "tool_epsilon",
        "description": "Epsilon aggregation.",
        "inputSchema": {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        },
    },
]


def _make_standalone_server(base: Path, repo_name: str, server_name: str) -> Path:
    server_dir = base / repo_name / "servers" / server_name
    server_dir.mkdir(parents=True)
    (server_dir / "server.py").write_text("# mock")
    (server_dir / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    return server_dir


def _make_module_server(base: Path, repo_name: str, server_name: str) -> Path:
    repo_dir = base / repo_name
    server_dir = repo_dir / "servers" / server_name
    server_dir.mkdir(parents=True)
    (repo_dir / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    (server_dir / "server.py").write_text("# mock")
    return server_dir


class TestDiscoverServers:
    def test_discovers_standalone_server(self, tmp_path: Path) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        results = discover_servers(tmp_path)
        assert len(results) == 1
        assert results[0]["server_name"] == "server_a"
        assert results[0]["launch_style"] == "standalone"
        assert results[0]["launch_command"] == "uv run python server.py"

    def test_discovers_module_style_server(self, tmp_path: Path) -> None:
        _make_module_server(tmp_path, "repo_a", "server_a")
        results = discover_servers(tmp_path)
        assert len(results) == 1
        assert results[0]["launch_style"] == "module"
        assert "python -m" in results[0]["launch_command"]

    def test_skips_router_itself(self, tmp_path: Path) -> None:
        _make_standalone_server(tmp_path, "MCP_Tool_Search", "router_basic")
        results = discover_servers(tmp_path)
        assert results == []

    def test_skips_repo_without_servers_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "bare_repo").mkdir()
        results = discover_servers(tmp_path)
        assert results == []

    def test_skips_server_dir_without_server_py(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "repo_a" / "servers" / "empty_server"
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
        results = discover_servers(tmp_path)
        assert results == []

    def test_returns_empty_for_nonexistent_base(self, tmp_path: Path) -> None:
        results = discover_servers(tmp_path / "nonexistent")
        assert results == []

    def test_discovers_multiple_repos(self, tmp_path: Path) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        _make_standalone_server(tmp_path, "repo_b", "server_b")
        results = discover_servers(tmp_path)
        assert len(results) == 2

    def test_results_sorted_by_repo_then_server(self, tmp_path: Path) -> None:
        _make_standalone_server(tmp_path, "zzz_repo", "zzz_server")
        _make_standalone_server(tmp_path, "aaa_repo", "aaa_server")
        results = discover_servers(tmp_path)
        names = [r["server_name"] for r in results]
        assert names == sorted(names)

    def test_server_cwd_is_server_dir_for_standalone(self, tmp_path: Path) -> None:
        server_dir = _make_standalone_server(tmp_path, "repo_a", "server_a")
        results = discover_servers(tmp_path)
        assert results[0]["server_cwd"] == str(server_dir)

    def test_server_cwd_is_repo_root_for_module(self, tmp_path: Path) -> None:
        _make_module_server(tmp_path, "repo_a", "server_a")
        results = discover_servers(tmp_path)
        assert results[0]["server_cwd"] == str(tmp_path / "repo_a")


class TestBuildEnrichedText:
    def test_contains_tool_name(self) -> None:
        tool = {"name": "my_tool", "description": "does stuff", "inputSchema": {"properties": {}}}
        text = _build_enriched_text("srv", "repo", tool)
        assert "tool:my_tool" in text

    def test_contains_server_and_repo(self) -> None:
        tool = {"name": "t", "description": "d", "inputSchema": {"properties": {}}}
        text = _build_enriched_text("my_server", "my_repo", tool)
        assert "server:my_server" in text
        assert "repo:my_repo" in text

    def test_contains_description(self) -> None:
        tool = {"name": "t", "description": "train model on CSV", "inputSchema": {"properties": {}}}
        text = _build_enriched_text("srv", "repo", tool)
        assert "train model on CSV" in text

    def test_contains_param_names(self) -> None:
        tool = {
            "name": "t",
            "description": "d",
            "inputSchema": {"properties": {"alpha": {}, "beta": {"description": "the beta"}}},
        }
        text = _build_enriched_text("srv", "repo", tool)
        assert "alpha" in text
        assert "beta" in text

    def test_contains_param_descriptions(self) -> None:
        tool = {
            "name": "t",
            "description": "d",
            "inputSchema": {"properties": {"p": {"description": "special description"}}},
        }
        text = _build_enriched_text("srv", "repo", tool)
        assert "special description" in text

    def test_handles_missing_properties(self) -> None:
        tool = {"name": "t", "description": "d", "inputSchema": {}}
        text = _build_enriched_text("srv", "repo", tool)
        assert "tool:t" in text


class TestReindex:
    def _patch_paths(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = tmp_path / "registry.db"
        monkeypatch.setattr("servers.router_basic._router_indexer.get_db_path", lambda: db_path)
        monkeypatch.setattr("servers.router_basic._router_indexer.get_router_dir", lambda: tmp_path)
        monkeypatch.setattr(
            "servers.router_basic._router_indexer.get_tfidf_vectorizer_path",
            lambda: tmp_path / "vec.joblib",
        )
        monkeypatch.setattr(
            "servers.router_basic._router_indexer.get_tfidf_matrix_path",
            lambda: tmp_path / "mat.joblib",
        )

    def _mock_collect(self, server_info: dict) -> list[dict]:
        return MOCK_TOOLS_A if "server_a" in server_info["server_name"] else MOCK_TOOLS_B

    def test_reindex_populates_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        self._patch_paths(monkeypatch, tmp_path)

        async def mock_collect(info: dict) -> list[dict]:
            return MOCK_TOOLS_A

        monkeypatch.setattr(
            "servers.router_basic._router_indexer._collect_tools_from_server",
            mock_collect,
        )

        result = reindex(base_dir=tmp_path)
        assert result["success"] is True
        assert result["total_tools"] == len(MOCK_TOOLS_A)
        assert "server_a" in result["servers_indexed"]

    def test_reindex_with_two_servers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        _make_standalone_server(tmp_path, "repo_b", "server_b")
        self._patch_paths(monkeypatch, tmp_path)

        async def mock_collect(info: dict) -> list[dict]:
            if "server_a" in info["server_name"]:
                return MOCK_TOOLS_A
            return MOCK_TOOLS_B

        monkeypatch.setattr(
            "servers.router_basic._router_indexer._collect_tools_from_server",
            mock_collect,
        )

        result = reindex(base_dir=tmp_path)
        assert result["success"] is True
        assert result["total_tools"] == len(MOCK_TOOLS_A) + len(MOCK_TOOLS_B)

    def test_reindex_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        self._patch_paths(monkeypatch, tmp_path)

        async def mock_collect(info: dict) -> list[dict]:
            return MOCK_TOOLS_A[:1]

        monkeypatch.setattr(
            "servers.router_basic._router_indexer._collect_tools_from_server",
            mock_collect,
        )

        r1 = reindex(base_dir=tmp_path)
        r2 = reindex(base_dir=tmp_path)
        assert r1["total_tools"] == r2["total_tools"] == 1

    def test_enriched_text_stored_in_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        self._patch_paths(monkeypatch, tmp_path)

        async def mock_collect(info: dict) -> list[dict]:
            return MOCK_TOOLS_A[:1]

        monkeypatch.setattr(
            "servers.router_basic._router_indexer._collect_tools_from_server",
            mock_collect,
        )

        reindex(base_dir=tmp_path)
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT enriched_text FROM tools").fetchone()
        conn.close()
        assert row is not None
        assert "tool:tool_alpha" in row[0]

    def test_no_servers_returns_success_with_zero_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_paths(monkeypatch, tmp_path)
        result = reindex(base_dir=tmp_path)
        assert result["success"] is True
        assert result["total_tools"] == 0

    def test_tfidf_artifacts_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_standalone_server(tmp_path, "repo_a", "server_a")
        self._patch_paths(monkeypatch, tmp_path)

        async def mock_collect(info: dict) -> list[dict]:
            return MOCK_TOOLS_A

        monkeypatch.setattr(
            "servers.router_basic._router_indexer._collect_tools_from_server",
            mock_collect,
        )

        reindex(base_dir=tmp_path)
        assert (tmp_path / "vec.joblib").exists()
        assert (tmp_path / "mat.joblib").exists()
