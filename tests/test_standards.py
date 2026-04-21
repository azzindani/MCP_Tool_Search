"""Standards compliance tests: docstrings, progress schema, shared utilities."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# §11 — Docstring ≤ 80 characters for all MCP tools
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = ["search_tools", "execute_tool", "list_servers", "reindex_servers"]


def test_all_tool_docstrings_under_80_chars() -> None:
    import servers.router_basic.server as srv

    for name in TOOL_FUNCTIONS:
        fn = getattr(srv, name)
        doc = fn.__doc__ or ""
        assert len(doc) <= 80, f"{name}: docstring is {len(doc)} chars (max 80): {doc!r}"


# ---------------------------------------------------------------------------
# §22 — Progress schema: icon + msg + detail fields
# ---------------------------------------------------------------------------


def test_progress_ok_has_correct_schema() -> None:
    from shared.progress import ok

    result = ok("done")
    assert result == {"icon": "✔", "msg": "done", "detail": ""}


def test_progress_fail_has_correct_schema() -> None:
    from shared.progress import fail

    result = fail("error", "details here")
    assert result == {"icon": "✗", "msg": "error", "detail": "details here"}


def test_progress_info_has_correct_schema() -> None:
    from shared.progress import info

    result = info("note")
    assert result["icon"] == "ℹ"
    assert result["msg"] == "note"
    assert "detail" in result


def test_progress_warn_has_correct_schema() -> None:
    from shared.progress import warn

    result = warn("caution")
    assert result["icon"] == "⚠"
    assert "detail" in result


def test_progress_undo_has_correct_schema() -> None:
    from shared.progress import undo

    result = undo("reverted")
    assert result["icon"] == "↶"
    assert "detail" in result


# ---------------------------------------------------------------------------
# §20 — platform_utils reads at call time (not import time)
# ---------------------------------------------------------------------------


def test_is_constrained_mode_reads_at_call_time(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from shared.platform_utils import is_constrained_mode

    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "0")
    assert is_constrained_mode() is False
    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
    assert is_constrained_mode() is True


def test_get_max_rows_respects_constrained_mode(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from shared.platform_utils import get_max_rows

    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
    assert get_max_rows() == 10
    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "0")
    assert get_max_rows() == 20


def test_get_max_results_respects_constrained_mode(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from shared.platform_utils import get_max_results

    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
    assert get_max_results() == 10
    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "0")
    assert get_max_results() == 50


# ---------------------------------------------------------------------------
# §18 — file_utils.resolve_path() security
# ---------------------------------------------------------------------------


def test_resolve_path_accepts_home_subpath(tmp_path: Path) -> None:
    from unittest.mock import patch

    from shared.file_utils import resolve_path

    target = tmp_path / "subdir" / "file.csv"
    target.parent.mkdir(parents=True)
    target.touch()

    # Patch Path.home() so tmp_path is treated as home for this test
    with patch.object(Path, "home", return_value=tmp_path):
        result = resolve_path(str(target))
    assert result == target.resolve()


def test_resolve_path_rejects_outside_home() -> None:
    import pytest

    from shared.file_utils import resolve_path

    with pytest.raises(ValueError, match="outside allowed directory"):
        resolve_path("/etc/passwd")


def test_resolve_path_extension_filter(tmp_path: Path) -> None:
    from unittest.mock import patch

    import pytest

    from shared.file_utils import resolve_path

    f = tmp_path / "file.txt"
    f.touch()
    with patch.object(Path, "home", return_value=tmp_path):
        with pytest.raises(ValueError, match="not in allowed set"):
            resolve_path(str(f), allowed_extensions=(".csv", ".db"))


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    from shared.file_utils import atomic_write_text

    target = tmp_path / "out.txt"
    atomic_write_text(str(target), "hello world")
    assert target.read_text() == "hello world"


def test_atomic_write_text_overwrites(tmp_path: Path) -> None:
    from shared.file_utils import atomic_write_text

    target = tmp_path / "out.txt"
    target.write_text("old")
    atomic_write_text(str(target), "new")
    assert target.read_text() == "new"


# ---------------------------------------------------------------------------
# §12 — ToolAnnotations: all four flags present on every tool
# ---------------------------------------------------------------------------


def test_all_tools_have_complete_annotations() -> None:
    import servers.router_basic.server as srv

    for name in TOOL_FUNCTIONS:
        fn = getattr(srv, name)
        # Annotations are passed to FastMCP; verify the function exists and is decorated
        assert callable(fn), f"{name} should be callable"
        # Check the mcp tool registry if available
    # Verify by checking that ToolAnnotations were used with all 4 fields
    # (structural test: server.py must import and use all 4 fields)
    import inspect

    src = inspect.getsource(srv)
    assert "destructiveHint" in src, "destructiveHint missing from server.py"
    assert "idempotentHint" in src, "idempotentHint missing from server.py"
    assert "readOnlyHint" in src, "readOnlyHint missing from server.py"
    assert "openWorldHint" in src, "openWorldHint missing from server.py"


# ---------------------------------------------------------------------------
# §16 — Return value contract: token_estimate + truncated in search_tools
# ---------------------------------------------------------------------------


def test_search_tools_has_truncated_field(tmp_path: Path) -> None:
    import sqlite3
    from unittest.mock import patch

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    from servers.router_basic import _router_search as sm
    from servers.router_basic._router_search import search_tools

    db_path = tmp_path / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE servers (server_name TEXT PRIMARY KEY, repo_name TEXT NOT NULL,
            server_cwd TEXT NOT NULL, launch_command TEXT NOT NULL,
            launch_style TEXT NOT NULL DEFAULT 'standalone', tool_count INTEGER NOT NULL,
            indexed_at TEXT NOT NULL, server_py_path TEXT NOT NULL DEFAULT '');
        CREATE TABLE tools (tool_id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name TEXT NOT NULL, tool_name TEXT NOT NULL, description TEXT NOT NULL,
            json_schema TEXT NOT NULL, enriched_text TEXT NOT NULL,
            UNIQUE(server_name, tool_name));
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO servers VALUES (?,?,?,?,?,?,?,?)",
        (
            "s",
            "r",
            "/c",
            "uv run python server.py",
            "standalone",
            1,
            "2026-01-01T00:00:00+00:00",
            "",
        ),
    )
    conn.execute(
        "INSERT INTO tools (server_name,tool_name,description,json_schema,enriched_text)"
        " VALUES (?,?,?,?,?)",
        ("s", "t1", "desc", "{}", "tool:t1 server:s repo:r description:desc"),
    )
    conn.commit()
    conn.close()

    texts = ["tool:t1 server:s repo:r description:desc"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    vec_path = tmp_path / "tfidf_vectorizer.joblib"
    mat_path = tmp_path / "tfidf_matrix.joblib"
    joblib.dump(vec, str(vec_path))
    joblib.dump({"matrix": mat, "tool_ids": [1]}, str(mat_path))

    sm._vectorizer = None
    sm._matrix_data = None
    sm._current_tools = {}

    with (
        patch("servers.router_basic._router_search.get_db_path", return_value=db_path),
        patch(
            "servers.router_basic._router_search.get_tfidf_vectorizer_path", return_value=vec_path
        ),
        patch("servers.router_basic._router_search.get_tfidf_matrix_path", return_value=mat_path),
        patch(
            "servers.router_basic._router_search.get_embeddings_path",
            return_value=tmp_path / "e.joblib",
        ),
        patch("servers.router_basic._router_search.is_constrained_mode", return_value=False),
    ):
        result = search_tools("desc")

    sm._vectorizer = None
    sm._matrix_data = None
    sm._current_tools = {}

    assert "truncated" in result
    assert isinstance(result["truncated"], bool)
    assert "token_estimate" in result
    assert isinstance(result["token_estimate"], int)
    assert result["token_estimate"] > 0
