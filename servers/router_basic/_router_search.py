"""TF-IDF search over the indexed tool database."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from shared.platform_utils import is_constrained_mode

from ._router_helpers import (
    CONSTRAINED_TOP_N,
    DEFAULT_TOP_N,
    EMBEDDING_MODEL_NAME,
    USAGE_BOOST_FACTOR,
    get_db_path,
    get_embeddings_path,
    get_tfidf_matrix_path,
    get_tfidf_vectorizer_path,
)

# In-memory cache of the last search results, keyed by "server_name/tool_name"
_current_tools: dict[str, dict] = {}

# Lazy-loaded TF-IDF artifacts (reset to None to force reload)
_vectorizer: Any = None
_matrix_data: dict[str, Any] | None = None

# Lazy-loaded semantic artifacts
_embeddings_data: dict[str, Any] | None = None
_embedding_model: Any = None


def _load_tfidf() -> tuple[Any, Any, list[int]]:
    global _vectorizer, _matrix_data
    if _vectorizer is None or _matrix_data is None:
        import joblib

        _vectorizer = joblib.load(str(get_tfidf_vectorizer_path()))
        _matrix_data = joblib.load(str(get_tfidf_matrix_path()))
    data = _matrix_data
    assert data is not None
    return _vectorizer, data["matrix"], data["tool_ids"]


def _load_embeddings() -> tuple[Any, list[int]] | None:
    global _embeddings_data
    emb_path = get_embeddings_path()
    if not emb_path.exists():
        return None
    if _embeddings_data is None:
        import joblib

        _embeddings_data = joblib.load(str(emb_path))
    data = _embeddings_data
    assert data is not None
    return data["embeddings"], data["tool_ids"]


def _get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _min_max_normalize(scores: Any) -> Any:
    import numpy as np

    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def _get_usage_counts(tool_ids: list[int], db_path: Path) -> dict[int, int]:
    """Fetch successful call counts per tool_id. Returns {} on any error."""
    if not tool_ids:
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ph = ",".join("?" * len(tool_ids))
        rows = conn.execute(
            f"SELECT t.tool_id, COUNT(u.id) AS cnt "
            f"FROM tools t LEFT JOIN tool_usage u "
            f"ON t.server_name = u.server_name AND t.tool_name = u.tool_name "
            f"WHERE t.tool_id IN ({ph}) AND (u.success IS NULL OR u.success = 1) "
            f"GROUP BY t.tool_id",
            tool_ids,
        ).fetchall()
        conn.close()
        return {r["tool_id"]: r["cnt"] for r in rows}
    except Exception:
        return {}


def search_tools(query: str, top_n: int | None = None, context: str | None = None) -> dict:
    """Search indexed tools by TF-IDF cosine similarity. Returns top-N results."""
    from shared.progress import fail, ok, warn

    constrained = is_constrained_mode()
    if top_n is None:
        top_n = CONSTRAINED_TOP_N if constrained else DEFAULT_TOP_N

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
            "truncated": False,
            "progress": [fail("Index not found")],
            "token_estimate": 50,
        }

    try:
        vectorizer, matrix, tfidf_tool_ids = _load_tfidf()
    except Exception as e:
        return {
            "success": False,
            "op": "search_tools",
            "query": query,
            "error": f"Failed to load TF-IDF index: {e}",
            "tools": [],
            "returned": 0,
            "total_indexed": 0,
            "truncated": False,
            "progress": [fail(f"TF-IDF load error: {e}")],
            "token_estimate": 50,
        }

    # Build effective query from optional context
    effective_query = f"{context.strip()} {query.strip()}" if context else query

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = vectorizer.transform([effective_query])
    tfidf_scores = cosine_similarity(query_vec, matrix).flatten()

    retrieval_mode = "tfidf"
    final_scores = tfidf_scores
    final_tool_ids = tfidf_tool_ids

    # Attempt hybrid retrieval when embeddings are available and not constrained
    if not constrained:
        try:
            emb_result = _load_embeddings()
            if emb_result is not None:
                embeddings, emb_tool_ids = emb_result
                model = _get_embedding_model()
                query_emb = model.encode([effective_query], convert_to_numpy=True)
                sem_scores = cosine_similarity(query_emb, embeddings).flatten()

                if emb_tool_ids == tfidf_tool_ids:
                    norm_tfidf = _min_max_normalize(tfidf_scores)
                    norm_sem = _min_max_normalize(sem_scores)
                    final_scores = 0.5 * norm_tfidf + 0.5 * norm_sem
                    retrieval_mode = "hybrid"
        except Exception:
            pass  # fall back to TF-IDF silently

    # Apply usage boost (re-ranks tools used successfully in the past)
    try:
        nonzero_mask = final_scores > 0
        if nonzero_mask.any():
            nonzero_idx = list(np.where(nonzero_mask)[0])
            nonzero_ids = [final_tool_ids[i] for i in nonzero_idx]
            usage_counts = _get_usage_counts(nonzero_ids, db_path)
            for idx, tid in zip(nonzero_idx, nonzero_ids):
                cnt = usage_counts.get(tid, 0)
                if cnt > 0:
                    final_scores[idx] *= 1.0 + math.log1p(cnt) * USAGE_BOOST_FACTOR
    except Exception:
        pass

    # Count all matching results before top-N cut for truncated flag
    nonzero_count = int((final_scores > 0).sum())
    top_indices = np.argsort(final_scores)[::-1][:top_n]
    top_tool_ids = [final_tool_ids[i] for i in top_indices if final_scores[i] > 0]
    top_scores = [float(final_scores[i]) for i in top_indices if final_scores[i] > 0]

    if not top_tool_ids:
        response: dict[str, Any] = {
            "success": True,
            "op": "search_tools",
            "query": query,
            "returned": 0,
            "total_indexed": len(final_tool_ids),
            "tools": [],
            "retrieval_mode": retrieval_mode,
            "context_used": bool(context),
            "truncated": False,
            "cache_hint": "No matches found. Try different keywords.",
            "progress": [warn("No matching tools found")],
        }
        response["token_estimate"] = len(json.dumps(response, default=str)) // 4
        return response

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

    response = {
        "success": True,
        "op": "search_tools",
        "query": query,
        "returned": len(tools),
        "total_indexed": total_indexed,
        "tools": tools,
        "retrieval_mode": retrieval_mode,
        "context_used": bool(context),
        "truncated": nonzero_count > top_n,
        "cache_hint": "Call execute_tool with server_name and tool_name from above.",
        "progress": [ok(f"Found {len(tools)} matching tools")],
    }
    response["token_estimate"] = len(json.dumps(response, default=str)) // 4
    return response
