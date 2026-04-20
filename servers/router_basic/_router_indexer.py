"""Server discovery, MCP handshake, SQLite indexing, and list_servers."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC
from pathlib import Path

from ._router_helpers import (
    EMBEDDING_MODEL_NAME,
    REPO_NAME,
    get_db_path,
    get_embeddings_path,
    get_mcp_base_dir,
    get_router_dir,
    get_tfidf_matrix_path,
    get_tfidf_vectorizer_path,
)


def _get_db_conn() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS servers (
            server_name     TEXT PRIMARY KEY,
            repo_name       TEXT NOT NULL,
            server_cwd      TEXT NOT NULL,
            launch_command  TEXT NOT NULL,
            launch_style    TEXT NOT NULL DEFAULT 'standalone',
            tool_count      INTEGER NOT NULL,
            indexed_at      TEXT NOT NULL,
            server_py_path  TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS tools (
            tool_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name     TEXT NOT NULL REFERENCES servers(server_name),
            tool_name       TEXT NOT NULL,
            description     TEXT NOT NULL,
            json_schema     TEXT NOT NULL,
            enriched_text   TEXT NOT NULL,
            UNIQUE(server_name, tool_name)
        );
        CREATE TABLE IF NOT EXISTS index_meta (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        );
    """)
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add new columns to existing DBs without recreating tables."""
    try:
        conn.execute("ALTER TABLE servers ADD COLUMN server_py_path TEXT NOT NULL DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def discover_servers(base_dir: Path) -> list[dict]:
    """Scan base_dir for valid sibling MCP server repos."""
    siblings: list[dict] = []
    if not base_dir.is_dir():
        return siblings

    for repo_dir in sorted(base_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        if repo_dir.name == REPO_NAME:
            continue
        servers_dir = repo_dir / "servers"
        if not servers_dir.is_dir():
            continue

        for server_dir in sorted(servers_dir.iterdir()):
            if not server_dir.is_dir():
                continue
            server_py = server_dir / "server.py"
            pyproject = server_dir / "pyproject.toml"
            if server_py.exists() and pyproject.exists():
                siblings.append(
                    {
                        "repo_name": repo_dir.name,
                        "server_name": server_dir.name,
                        "server_cwd": str(server_dir),
                        "launch_style": "standalone",
                        "launch_command": "uv run python server.py",
                        "server_py_path": str(server_py),
                    }
                )

        root_pyproject = repo_dir / "pyproject.toml"
        if root_pyproject.exists():
            for server_dir in sorted(servers_dir.iterdir()):
                if not server_dir.is_dir():
                    continue
                server_py = server_dir / "server.py"
                server_pyproject = server_dir / "pyproject.toml"
                if server_py.exists() and not server_pyproject.exists():
                    module_path = f"servers.{server_dir.name}.server"
                    siblings.append(
                        {
                            "repo_name": repo_dir.name,
                            "server_name": server_dir.name,
                            "server_cwd": str(repo_dir),
                            "launch_style": "module",
                            "launch_command": f"uv run python -m {module_path}",
                            "server_py_path": str(server_py),
                        }
                    )

    return siblings


def _build_enriched_text(server_name: str, repo_name: str, tool: dict) -> str:
    params = tool.get("inputSchema", {}).get("properties", {}) or {}
    param_names = " ".join(params.keys())
    param_descs = " ".join(p.get("description", "") for p in params.values() if isinstance(p, dict))
    return (
        f"tool:{tool['name']} "
        f"server:{server_name} "
        f"repo:{repo_name} "
        f"description:{tool.get('description', '')} "
        f"parameters:{param_names} "
        f"param_details:{param_descs}"
    )


async def _collect_tools_from_server(server_info: dict) -> list[dict]:
    """Start a child MCP server via stdio, collect its tools, then shut down."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cwd = server_info["server_cwd"]
    cmd = server_info["launch_command"]
    parts = cmd.split()

    server_params = StdioServerParameters(
        command=parts[0],
        args=parts[1:],
        cwd=cwd,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]


def _build_tfidf_index(conn: sqlite3.Connection) -> None:
    """Build and persist the TF-IDF index from enriched_text in SQLite."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = conn.execute("SELECT tool_id, enriched_text FROM tools ORDER BY tool_id").fetchall()
    if not rows:
        return

    tool_ids = [r["tool_id"] for r in rows]
    texts = [r["enriched_text"] for r in rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)

    router_dir = get_router_dir()
    router_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, str(get_tfidf_vectorizer_path()))
    joblib.dump({"matrix": matrix, "tool_ids": tool_ids}, str(get_tfidf_matrix_path()))


def _build_semantic_index(conn: sqlite3.Connection) -> bool:
    """Build and persist semantic embeddings. Returns True if successful."""
    try:
        import joblib
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError:
        return False

    rows = conn.execute("SELECT tool_id, enriched_text FROM tools ORDER BY tool_id").fetchall()
    if not rows:
        return False

    tool_ids = [r["tool_id"] for r in rows]
    texts = [r["enriched_text"] for r in rows]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    router_dir = get_router_dir()
    router_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"embeddings": embeddings, "tool_ids": tool_ids}, str(get_embeddings_path()))
    return True


def reindex(base_dir: Path | None = None) -> dict:
    """Discover sibling servers, collect tools, rebuild SQLite + TF-IDF index."""
    from datetime import datetime

    from shared.progress import fail, info, ok, warn

    if base_dir is None:
        base_dir = get_mcp_base_dir()

    progress = []
    progress.append(info(f"Scanning {base_dir} for MCP servers"))

    servers = discover_servers(base_dir)
    if not servers:
        progress.append(warn("No sibling MCP servers found"))

    conn = _get_db_conn()
    _init_schema(conn)
    _migrate_schema(conn)

    conn.execute("DELETE FROM tools")
    conn.execute("DELETE FROM servers")
    conn.execute("DELETE FROM index_meta")
    conn.commit()

    total_tools = 0
    indexed_servers: list[str] = []

    for server_info in servers:
        name = server_info["server_name"]
        progress.append(info(f"Indexing {name} ({server_info['repo_name']})"))

        try:
            tools = asyncio.run(_collect_tools_from_server(server_info))
        except Exception as e:
            progress.append(fail(f"Failed to collect tools from {name}: {e}"))
            continue

        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO servers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                server_info["repo_name"],
                server_info["server_cwd"],
                server_info["launch_command"],
                server_info["launch_style"],
                len(tools),
                now,
                server_info.get("server_py_path", ""),
            ),
        )

        for tool in tools:
            enriched = _build_enriched_text(name, server_info["repo_name"], tool)
            schema = tool["inputSchema"]
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
            elif hasattr(schema, "dict"):
                schema = schema.dict()  # type: ignore[union-attr]
            conn.execute(
                "INSERT OR REPLACE INTO tools "
                "(server_name, tool_name, description, json_schema, enriched_text) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, tool["name"], tool.get("description", ""), json.dumps(schema), enriched),
            )

        conn.commit()
        total_tools += len(tools)
        indexed_servers.append(name)
        progress.append(ok(f"Indexed {len(tools)} tools from {name}"))

    if total_tools > 0:
        _build_tfidf_index(conn)
        progress.append(ok(f"TF-IDF index built for {total_tools} tools"))

        if _build_semantic_index(conn):
            progress.append(ok("Semantic embeddings built"))
        else:
            progress.append(
                info("Semantic embeddings skipped (sentence-transformers not installed)")
            )

    now_str = datetime.now(UTC).isoformat()
    conn.execute("INSERT OR REPLACE INTO index_meta VALUES ('total_tools', ?)", (str(total_tools),))
    conn.execute("INSERT OR REPLACE INTO index_meta VALUES ('last_indexed', ?)", (now_str,))
    conn.execute("INSERT OR REPLACE INTO index_meta VALUES ('index_version', '2')")
    conn.commit()
    conn.close()

    progress.append(ok(f"Reindex complete: {len(indexed_servers)} servers, {total_tools} tools"))
    return {
        "success": True,
        "op": "reindex",
        "servers_indexed": indexed_servers,
        "total_tools": total_tools,
        "progress": progress,
    }


def list_servers() -> dict:
    """List all indexed MCP servers and their tool counts."""
    from datetime import datetime

    from shared.progress import fail, ok

    db_path = get_db_path()
    if not db_path.exists():
        return {
            "success": False,
            "op": "list_servers",
            "error": "Index not found. Restart the router to trigger reindexing.",
            "servers": [],
            "total_servers": 0,
            "total_tools": 0,
            "index_age_hours": None,
            "reindex_hint": "Restart the router server to rebuild the index.",
            "progress": [fail("registry.db not found")],
            "token_estimate": 50,
        }

    conn = _get_db_conn()
    _migrate_schema(conn)
    rows = conn.execute(
        "SELECT server_name, repo_name, tool_count, indexed_at, server_py_path "
        "FROM servers ORDER BY server_name"
    ).fetchall()
    meta = {
        r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM index_meta").fetchall()
    }
    conn.close()

    last_indexed = meta.get("last_indexed")
    index_age_hours = None
    if last_indexed:
        try:
            last_dt = datetime.fromisoformat(last_indexed)
            now = datetime.now(UTC)
            index_age_hours = round((now - last_dt).total_seconds() / 3600, 2)
        except Exception:
            pass

    stale_servers: list[str] = []
    servers = []
    for r in rows:
        server_py_path = r["server_py_path"]
        is_stale = False
        if server_py_path:
            try:
                mtime = Path(server_py_path).stat().st_mtime
                indexed_at_dt = datetime.fromisoformat(r["indexed_at"])
                if mtime > indexed_at_dt.timestamp():
                    is_stale = True
                    stale_servers.append(r["server_name"])
            except Exception:
                pass

        servers.append(
            {
                "server_name": r["server_name"],
                "repo_name": r["repo_name"],
                "tool_count": r["tool_count"],
                "indexed_at": r["indexed_at"],
                "stale": is_stale,
            }
        )

    total_tools = int(meta.get("total_tools", sum(s["tool_count"] for s in servers)))

    result: dict = {
        "success": True,
        "op": "list_servers",
        "servers": servers,
        "total_servers": len(servers),
        "total_tools": total_tools,
        "index_age_hours": index_age_hours,
        "stale_servers": stale_servers,
        "progress": [ok(f"Found {len(servers)} servers, {total_tools} tools")],
        "token_estimate": len(json.dumps(servers)) // 4,
    }
    if stale_servers:
        result["reindex_hint"] = (
            f"{len(stale_servers)} server(s) have changed since indexing. Call reindex_servers()."
        )
    return result
