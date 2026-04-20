"""TF-IDF search over the indexed tool database."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ._router_helpers import (
    CONSTRAINED_TOP_N,
    DEFAULT_TOP_N,
    MCP_CONSTRAINED_MODE,
    get_db_path,
    get_tfidf_matrix_path,
    get_tfidf_vectorizer_path,
)

# In-memory cache of the last search results, keyed by "server_name/tool_name"
_current_tools: dict[str, dict] = {}

# Lazy-loaded TF-IDF artifacts (reset to None to force reload)
_vectorizer: Any = None
_matrix_data: dict[str, Any] | None = None


def _load_tfidf() -> tuple[Any, Any, list[int]]:
    global _vectorizer, _matrix_data
    if _vectorizer is None or _matrix_data is None:
        import joblib

        _vectorizer = joblib.load(str(get_tfidf_vectorizer_path()))
        _matrix_data = joblib.load(str(get_tfidf_matrix_path()))
    data = _matrix_data
    assert data is not None
    return _vectorizer, data["matrix"], data["tool_ids"]


def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def search_tools(query: str, top_n: int | None = None) -> dict:
    """Search indexed tools by TF-IDF cosine similarity. Returns top-N results."""
    from shared.progress import fail, ok, warn

    if top_n is None:
        top_n = CONSTRAINED_TOP_N if MCP_CONSTRAINED_MODE else DEFAULT_TOP_N

    db_path = get_db_path()
    vec_path = get_tfidf_vectorizer_path()

    if not db_path.exists() or not vec_path.exists():
        return {
            "success": False,
            "op": "search_tools",
            "query": query,
            "error": "Index not found. Restart the router to rebuild.",
            "tools": [],
            "returned": 0,
            "total_indexed": 0,
            "progress": [fail("Index not found")],
            "token_estimate": 50,
        }

    try:
        vectorizer, matrix, tool_ids = _load_tfidf()
    except Exception as e:
        return {
            "success": False,
            "op": "search_tools",
            "query": query,
            "error": f"Failed to load TF-IDF index: {e}",
            "tools": [],
            "returned": 0,
            "total_indexed": 0,
            "progress": [fail(f"TF-IDF load error: {e}")],
            "token_estimate": 50,
        }

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_n]
    top_tool_ids = [tool_ids[i] for i in top_indices if scores[i] > 0]
    top_scores = [float(scores[i]) for i in top_indices if scores[i] > 0]

    if not top_tool_ids:
        return {
            "success": True,
            "op": "search_tools",
            "query": query,
            "returned": 0,
            "total_indexed": len(tool_ids),
            "tools": [],
            "cache_hint": "No matches found. Try different keywords.",
            "progress": [warn("No matching tools found")],
            "token_estimate": 50,
        }

    conn = _get_db_conn()
    placeholders = ",".join("?" * len(top_tool_ids))
    rows = conn.execute(
        f"SELECT t.tool_id, t.server_name, t.tool_name, t.description, t.json_schema, "
        f"s.repo_name FROM tools t JOIN servers s ON t.server_name = s.server_name "
        f"WHERE t.tool_id IN ({placeholders})",
        top_tool_ids,
    ).fetchall()
    total_indexed = conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
    conn.close()

    id_to_row = {r["tool_id"]: r for r in rows}
    tools = []
    global _current_tools
    _current_tools = {}

    for tool_id, score in zip(top_tool_ids, top_scores):
        row = id_to_row.get(tool_id)
        if not row:
            continue
        schema = json.loads(row["json_schema"])
        entry: dict[str, Any] = {
            "server_name": row["server_name"],
            "repo_name": row["repo_name"],
            "tool_name": row["tool_name"],
            "description": row["description"],
            "score": round(score, 4),
            "json_schema": schema,
        }
        tools.append(entry)
        _current_tools[f"{row['server_name']}/{row['tool_name']}"] = entry

    return {
        "success": True,
        "op": "search_tools",
        "query": query,
        "returned": len(tools),
        "total_indexed": total_indexed,
        "tools": tools,
        "cache_hint": "Call execute_tool with server_name and tool_name from above.",
        "progress": [ok(f"Found {len(tools)} matching tools")],
        "token_estimate": len(json.dumps(tools)) // 4,
    }
