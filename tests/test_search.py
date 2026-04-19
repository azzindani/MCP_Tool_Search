"""Tests for _router_search: TF-IDF search, ranking, result structure."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import servers.router_basic._router_search as search_mod
from servers.router_basic._router_search import search_tools

_TOOLS_DATA = [
    (
        "ml_basic",
        "MCP_ML",
        "train_classifier",
        "Train a classifier on CSV data. Returns accuracy, F1, model path.",
        '{"type": "object", "properties": {"file_path": {"type": "string", "description": "Path to CSV"}, "algorithm": {"type": "string", "description": "lr svm rf"}}}',
        "tool:train_classifier server:ml_basic repo:MCP_ML description:Train classifier CSV parameters:file_path algorithm param_details:Path to CSV lr svm rf",
    ),
    (
        "ml_basic",
        "MCP_ML",
        "predict",
        "Run predictions with a trained model.",
        '{"type": "object", "properties": {"model_path": {"type": "string"}}}',
        "tool:predict server:ml_basic repo:MCP_ML description:Run predictions trained model parameters:model_path",
    ),
    (
        "data_basic",
        "MCP_Data",
        "load_csv",
        "Load CSV file into a pandas dataframe.",
        '{"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}}',
        "tool:load_csv server:data_basic repo:MCP_Data description:Load CSV pandas dataframe parameters:path",
    ),
    (
        "data_basic",
        "MCP_Data",
        "filter_rows",
        "Filter dataframe rows by column condition.",
        '{"type": "object", "properties": {"column": {"type": "string"}, "value": {}}}',
        "tool:filter_rows server:data_basic repo:MCP_Data description:Filter dataframe rows column condition parameters:column value",
    ),
]


def _build_index(tmp_path: Path) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "registry.db"
    vec_path = tmp_path / "tfidf_vectorizer.joblib"
    mat_path = tmp_path / "tfidf_matrix.joblib"

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
    conn.execute(
        "INSERT INTO servers VALUES ('ml_basic','MCP_ML','/tmp','uv run python server.py','standalone',2,'2026-01-01')"
    )
    conn.execute(
        "INSERT INTO servers VALUES ('data_basic','MCP_Data','/tmp','uv run python server.py','standalone',2,'2026-01-01')"
    )
    for server_name, repo_name, tool_name, desc, schema, enriched in _TOOLS_DATA:
        conn.execute(
            "INSERT INTO tools (server_name, tool_name, description, json_schema, enriched_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (server_name, tool_name, desc, schema, enriched),
        )
    conn.execute("INSERT INTO index_meta VALUES ('total_tools', '4')")
    conn.commit()
    conn.close()

    texts = [row[5] for row in _TOOLS_DATA]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    tool_ids = list(range(1, len(_TOOLS_DATA) + 1))

    joblib.dump(vectorizer, str(vec_path))
    joblib.dump({"matrix": matrix, "tool_ids": tool_ids}, str(mat_path))

    return db_path, vec_path, mat_path


@pytest.fixture(autouse=True)
def patch_search_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path, vec_path, mat_path = _build_index(tmp_path)
    monkeypatch.setattr("servers.router_basic._router_search.get_db_path", lambda: db_path)
    monkeypatch.setattr(
        "servers.router_basic._router_search.get_tfidf_vectorizer_path", lambda: vec_path
    )
    monkeypatch.setattr(
        "servers.router_basic._router_search.get_tfidf_matrix_path", lambda: mat_path
    )
    search_mod._vectorizer = None
    search_mod._matrix_data = None
    search_mod._current_tools = {}


class TestSearchTools:
    def test_finds_relevant_tool_for_classifier_query(self) -> None:
        result = search_tools("train classifier machine learning")
        assert result["success"] is True
        assert result["returned"] > 0
        assert result["tools"][0]["tool_name"] == "train_classifier"

    def test_finds_csv_tool_for_data_query(self) -> None:
        result = search_tools("load csv file dataframe")
        assert result["success"] is True
        names = [t["tool_name"] for t in result["tools"]]
        assert "load_csv" in names

    def test_returns_full_json_schema(self) -> None:
        result = search_tools("train classifier")
        assert result["returned"] > 0
        tool = result["tools"][0]
        assert "json_schema" in tool
        assert isinstance(tool["json_schema"], dict)

    def test_results_sorted_by_score_descending(self) -> None:
        result = search_tools("classifier training csv")
        scores = [t["score"] for t in result["tools"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self) -> None:
        result = search_tools("data tool", top_n=2)
        assert result["returned"] <= 2

    def test_no_match_returns_success_with_empty_tools(self) -> None:
        result = search_tools("zzz_xyzzy_definitely_no_match_qqqq")
        assert result["success"] is True
        assert result["returned"] == 0
        assert result["tools"] == []

    def test_result_contains_server_name_and_repo(self) -> None:
        result = search_tools("train classifier")
        tool = result["tools"][0]
        assert "server_name" in tool
        assert "repo_name" in tool

    def test_token_estimate_present_and_positive(self) -> None:
        result = search_tools("classifier")
        assert "token_estimate" in result
        assert isinstance(result["token_estimate"], int)
        assert result["token_estimate"] >= 0

    def test_cache_hint_present_on_results(self) -> None:
        result = search_tools("classifier")
        assert "cache_hint" in result

    def test_current_tools_cache_populated(self) -> None:
        search_tools("train classifier")
        assert len(search_mod._current_tools) > 0
        assert any("train_classifier" in k for k in search_mod._current_tools)

    def test_total_indexed_reported(self) -> None:
        result = search_tools("anything")
        assert "total_indexed" in result

    def test_no_index_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "servers.router_basic._router_search.get_db_path",
            lambda: tmp_path / "no.db",
        )
        monkeypatch.setattr(
            "servers.router_basic._router_search.get_tfidf_vectorizer_path",
            lambda: tmp_path / "no.joblib",
        )
        monkeypatch.setattr(
            "servers.router_basic._router_search.get_tfidf_matrix_path",
            lambda: tmp_path / "no.joblib",
        )
        result = search_tools("anything")
        assert result["success"] is False
        assert "error" in result
