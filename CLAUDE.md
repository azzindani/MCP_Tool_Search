# CLAUDE.md — MCP Tool Router

## Project Overview

MCP Tool Router is a **meta-MCP server** that solves the context window overflow
problem when running many MCP servers with local LLMs. Instead of loading all tool
schemas from all servers into the LLM's context (which can exceed 50K+ tokens for
100+ tools), the router is the **only server toggled ON** in the AI client. It
discovers, indexes, and selectively retrieves tool schemas on demand — injecting
only a small relevant subset per inference turn.

**One-line description:** A self-hosted MCP tool router that indexes sibling MCP
servers and selectively retrieves tool schemas via RAG, enabling local LLMs to
access 500+ tools without exceeding context limits.

### The Problem

| Scenario | Tool schemas in context | Tokens consumed |
|---|---|---|
| MCP_Data_Analyst (7 servers) | 84 schemas | ~25K–42K tokens |
| MCP_Machine_Learning (3 servers) | 35 schemas | ~10K–17K tokens |
| Both projects loaded | 119 schemas | ~35K–60K tokens |
| 10 MCP projects (future) | 500+ schemas | 150K+ tokens |

A local 9B model on 8 GB VRAM has ~10K–12K effective context tokens. Loading even
one project's full tool set can exhaust the entire context window before the user
types anything.

### The Solution

The Tool Router exposes **3 tools** to the LLM (~800 tokens of schema). At runtime
it uses RAG to retrieve the top-N most relevant tool schemas from a local database
and returns them as structured text. The LLM then calls `execute_tool` to proxy
the actual operation to the correct child server — launched just-in-time via `uv run`
and killed after the call completes.

```
User message → LLM calls search_tools("train random forest")
            → Router returns top-20 tool schemas as text
            → LLM reads schemas, selects train_classifier
            → LLM calls execute_tool("ml_basic", "train_classifier", {...})
            → Router: cd ~/.mcp_servers/MCP_Machine_Learning/servers/ml_basic
            → Router: uv run python server.py (stdio MCP)
            → Router: tools/call train_classifier → result
            → Router: kills child process (or caches with TTL)
            → Result returned to LLM
```

### Hard Constraints

1. **Self-hosted execution.** No cloud APIs, no API keys, no paid services. The
   RAG index, embeddings, and vector search all run locally. Internet access is
   permitted only at install time (clone repos, download embedding model).

2. **CPU-only execution.** The router and its RAG pipeline run entirely on CPU.
   GPU is reserved for the local LLM. The embedding model must be lightweight
   enough for CPU inference (<100ms per query).

3. **Sibling directory convention.** The router lives at
   `~/.mcp_servers/MCP_Tool_Router/` alongside sibling MCP server repos like
   `~/.mcp_servers/MCP_Data_Analyst/` and `~/.mcp_servers/MCP_Machine_Learning/`.
   It discovers siblings by scanning `~/.mcp_servers/*/`.

4. **Zero modification to child servers.** The router never modifies sibling repos.
   It connects to them via stdio MCP protocol — the same way LM Studio would.

5. **Cross-platform.** Windows, macOS, and Linux. pathlib everywhere. POSIX sh for
   install scripts. PowerShell for Windows mcp.json entries. No platform-specific
   code paths in engine logic.

---

## Goals

### Phase 1 — Core Router (MVP)

- [ ] Discover sibling MCP servers in `~/.mcp_servers/*/`
- [ ] Index all tools from all servers into a local SQLite database
- [ ] Build a TF-IDF search index over enriched tool descriptions
- [ ] Expose 3 tools: `search_tools`, `execute_tool`, `list_servers`
- [ ] JIT server launch via `uv run` with stdio MCP transport
- [ ] TTL-based process pool (keep server alive for 60s after last call)
- [ ] Tool result caching (last search results persist in memory)
- [ ] Cross-platform CI (Ubuntu, macOS, Windows)
- [ ] README with standard mcp.json entries (Windows + macOS/Linux)

### Phase 2 — Enhanced Retrieval

- [ ] Optional sentence-transformer embeddings (`all-MiniLM-L6-v2`) for semantic search
- [ ] Hybrid retrieval: TF-IDF + semantic with score fusion
- [ ] Re-index command (tool) for when new servers are installed
- [ ] Index versioning — detect when sibling servers have been updated

### Phase 3 — Advanced Features

- [ ] Tool usage history — boost frequently used tools in rankings
- [ ] Multi-turn context awareness — use conversation history to refine search
- [ ] HTTP transport mode for remote access

---

## Repository Structure

```
MCP_Tool_Router/
│
├── shared/                          # Ring-2 utilities (no MCP imports)
│   ├── __init__.py
│   ├── platform_utils.py           # MCP_CONSTRAINED_MODE, get_max_rows()
│   ├── progress.py                 # ok/fail/info/warn/undo helpers
│   └── file_utils.py               # resolve_path(), path helpers
│
├── servers/
│   └── router_basic/               # single-tier server (the router itself)
│       ├── __init__.py
│       ├── server.py               # FastMCP wrapper — 3 tools, thin
│       ├── engine.py               # thin router re-exports from sub-modules
│       ├── _router_helpers.py      # shared constants, platform detection, utilities
│       ├── _router_indexer.py      # server discovery, MCP handshake, SQLite indexing
│       ├── _router_search.py       # TF-IDF / semantic search over tool index
│       ├── _router_executor.py     # JIT server launch, MCP call proxy, TTL pool
│       └── pyproject.toml
│
├── tests/
│   ├── fixtures/                   # mock server.py stubs for testing
│   │   ├── mock_server_a/
│   │   │   ├── server.py           # fake MCP server with 3 tools
│   │   │   └── pyproject.toml
│   │   └── mock_server_b/
│   │       ├── server.py           # fake MCP server with 2 tools
│   │       └── pyproject.toml
│   ├── conftest.py
│   ├── test_indexer.py
│   ├── test_search.py
│   └── test_executor.py
│
├── install/
│   ├── install.sh                  # POSIX sh — Linux / macOS
│   ├── install.bat                 # Windows CMD
│   └── mcp_config_writer.py        # writes to AI client config files
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
├── pyproject.toml                  # root workspace
├── uv.lock
├── .python-version                 # 3.12
├── .gitattributes
├── .editorconfig
├── CLAUDE.md
├── STANDARDS.md                    # symlink or copy of Standards repo
└── README.md
```

### Why Single-Tier

The Tool Router is a **single-tier** project (no basic/medium/advanced split).
It exposes exactly 3 tools. There is no domain complexity that warrants tier
separation. The three-tier split from STANDARDS.md §7 applies to domain servers
with many tools — the router is a meta-server with a fixed, minimal tool surface.

---

## Architecture

### Engine / Server Separation (STANDARDS §14)

**`server.py`** — thin MCP wrapper, zero domain logic:

```python
from mcp.server.fastmcp import FastMCP
from . import engine

mcp = FastMCP("tool-router")

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def search_tools(query: str, top_n: int = 20) -> dict:
    """Search indexed tools by query. Returns top-N tool schemas."""
    return engine.search_tools(query, top_n)

@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def execute_tool(server_name: str, tool_name: str, arguments: str) -> dict:
    """Execute a tool on a child MCP server via JIT launch."""
    return engine.execute_tool(server_name, tool_name, arguments)

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def list_servers() -> dict:
    """List all indexed MCP servers and their tool counts."""
    return engine.list_servers()
```

**`engine.py`** — thin router re-exporting from sub-modules:

```python
"""tool-router engine — zero MCP imports."""

from ._router_search import search_tools
from ._router_executor import execute_tool
from ._router_indexer import list_servers, reindex

__all__ = ["search_tools", "execute_tool", "list_servers", "reindex"]
```

### Sub-Module Responsibilities

| Module | Responsibility | Heavy deps |
|---|---|---|
| `_router_helpers.py` | Constants, DB path, platform detection, progress helpers | None |
| `_router_indexer.py` | Discover servers, MCP handshake, populate SQLite, list_servers | `mcp` (client) |
| `_router_search.py` | Load TF-IDF index, embed query, retrieve top-N tools | `scikit-learn` (lazy) |
| `_router_executor.py` | JIT launch child server, MCP tools/call, TTL pool management | `mcp` (client) |

All heavy imports are **lazy** (inside function body, not at module top level)
per STANDARDS §15.

---

## Data Model

### SQLite Database: `~/.mcp_servers/MCP_Tool_Router/registry.db`

The database is the single source of truth for all indexed tools. Built at install
time, queried at runtime. No child servers need to be running for search to work.

**Table: `servers`**

```sql
CREATE TABLE servers (
    server_name     TEXT PRIMARY KEY,       -- e.g. "data_basic", "ml_advanced"
    repo_name       TEXT NOT NULL,           -- e.g. "MCP_Data_Analyst"
    server_cwd      TEXT NOT NULL,           -- absolute path to server directory
    launch_command  TEXT NOT NULL,           -- e.g. "uv run python server.py"
    launch_style    TEXT NOT NULL DEFAULT 'standalone',  -- "standalone" or "module"
    tool_count      INTEGER NOT NULL,
    indexed_at      TEXT NOT NULL            -- ISO 8601 UTC timestamp
);
```

**Table: `tools`**

```sql
CREATE TABLE tools (
    tool_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    server_name     TEXT NOT NULL REFERENCES servers(server_name),
    tool_name       TEXT NOT NULL,           -- e.g. "train_classifier"
    description     TEXT NOT NULL,           -- tool docstring (≤80 chars)
    json_schema     TEXT NOT NULL,           -- full JSON schema as string
    enriched_text   TEXT NOT NULL,           -- name + desc + param names for RAG
    UNIQUE(server_name, tool_name)
);
```

**Table: `index_meta`**

```sql
CREATE TABLE index_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
-- Stores: "tfidf_path", "total_tools", "last_indexed", "index_version"
```

### Enriched Text for RAG

Each tool gets an enriched text document combining its name, description, server
context, and parameter names. This is what the TF-IDF/embedding index operates on.

```python
def _build_enriched_text(server_name: str, repo_name: str, tool: dict) -> str:
    """Build enriched document for RAG indexing."""
    params = tool.get("inputSchema", {}).get("properties", {})
    param_names = " ".join(params.keys())
    param_descs = " ".join(
        p.get("description", "") for p in params.values()
    )
    return (
        f"tool:{tool['name']} "
        f"server:{server_name} "
        f"repo:{repo_name} "
        f"description:{tool.get('description', '')} "
        f"parameters:{param_names} "
        f"param_details:{param_descs}"
    )
```

Example output for `train_classifier`:
```
tool:train_classifier server:ml_basic repo:MCP_Machine_Learning
description:Train classifier on CSV. Returns accuracy, F1, model path.
parameters:file_path target_column algorithm test_size
param_details:Path to CSV file Column to predict lr svm rf dtc knn nb xgb ...
```

---

## Indexing Process

### When Indexing Happens

1. **At install time** — the router's mcp.json launch command includes an indexing
   step that runs before the server starts.
2. **On explicit re-index** — the LLM can call `list_servers()` which includes a
   `reindex_hint` when the DB is stale or missing.
3. **Never at query time** — search and execute always read from the existing DB.

### Discovery Logic

The indexer scans `~/.mcp_servers/*/` for sibling MCP server repos. A valid
sibling must meet ALL of these criteria:

1. Directory is not the router itself (`MCP_Tool_Router`)
2. Contains a `servers/` subdirectory
3. Contains at least one `server.py` file under `servers/*/`
4. The parent of `server.py` contains a `pyproject.toml`

```python
def discover_servers(base_dir: Path) -> list[dict]:
    """Scan ~/.mcp_servers/ for valid MCP server repos."""
    siblings = []
    for repo_dir in sorted(base_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        if repo_dir.name == "MCP_Tool_Router":
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
                siblings.append({
                    "repo_name": repo_dir.name,
                    "server_name": server_dir.name,
                    "server_cwd": str(server_dir),
                    "launch_style": "standalone",
                    "launch_command": "uv run python server.py",
                })
        # Also check for module-style servers (run from repo root)
        root_pyproject = repo_dir / "pyproject.toml"
        if root_pyproject.exists():
            for server_dir in sorted(servers_dir.iterdir()):
                if not server_dir.is_dir():
                    continue
                server_py = server_dir / "server.py"
                server_pyproject = server_dir / "pyproject.toml"
                # Module-style: has server.py but no own pyproject.toml
                # OR: both exist but repo root also has pyproject.toml
                if server_py.exists() and not server_pyproject.exists():
                    module_path = f"servers.{server_dir.name}.server"
                    siblings.append({
                        "repo_name": repo_dir.name,
                        "server_name": server_dir.name,
                        "server_cwd": str(repo_dir),
                        "launch_style": "module",
                        "launch_command": f"uv run python -m {module_path}",
                    })
    return siblings
```

### MCP Handshake for Tool Collection

For each discovered server, the indexer:

1. Runs `uv sync --quiet` in the server's cwd (ensures deps are cached)
2. Starts the server via subprocess (stdio transport)
3. Performs MCP protocol handshake (`initialize` → `initialized`)
4. Calls `tools/list` to collect all tool schemas
5. Stores each tool in SQLite with enriched text
6. Sends MCP shutdown and kills the process

```python
async def _collect_tools_from_server(server_info: dict) -> list[dict]:
    """Start a child MCP server, collect tools, shut down."""
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cwd = server_info["server_cwd"]
    cmd = server_info["launch_command"]

    # Parse command into executable + args
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
```

### Index Building

After all tools are collected, the indexer:

1. Inserts all servers and tools into SQLite
2. Builds a TF-IDF matrix over the `enriched_text` column
3. Persists the TF-IDF vectorizer and matrix to disk using `joblib`
4. Records metadata in `index_meta` table

```
~/.mcp_servers/MCP_Tool_Router/
├── registry.db              # SQLite database
├── tfidf_vectorizer.joblib  # fitted TF-IDF vectorizer
└── tfidf_matrix.joblib      # sparse matrix of tool embeddings
```

---

## Tool Design

### Tool 1: `search_tools`

```
search_tools(query: str, top_n: int = 20) -> dict
```

**Purpose:** Search the indexed tool database using the user's query. Returns
top-N matching tools with their full JSON schemas, ready for the LLM to select
and call via `execute_tool`.

**Annotations:** `readOnlyHint=True`, `destructiveHint=False`, `idempotentHint=True`, `openWorldHint=False`

**Docstring:** `"""Search indexed tools by query. Returns top-N tool schemas."""`
(55 chars)

**Flow:**

1. Load TF-IDF vectorizer and matrix from disk (cached in memory after first load)
2. Vectorize the query string
3. Compute cosine similarity against all tool vectors
4. Sort by score descending, take top-N
5. For each match, fetch full JSON schema from SQLite
6. Store results in in-memory cache (`_current_tools`)
7. Return structured result

**Return value:**

```python
{
    "success": True,
    "op": "search_tools",
    "query": "train random forest",
    "returned": 20,
    "total_indexed": 119,
    "tools": [
        {
            "server_name": "ml_basic",
            "repo_name": "MCP_Machine_Learning",
            "tool_name": "train_classifier",
            "description": "Train classifier on CSV. Returns accuracy, F1, model path.",
            "score": 0.87,
            "json_schema": { ... }  # full inputSchema
        },
        ...
    ],
    "cache_hint": "Call execute_tool with server_name and tool_name from above.",
    "progress": [...],
    "token_estimate": ...
}
```

**Token budget concern:** 20 tool schemas at ~300 tokens each = ~6K tokens. This
is acceptable for 8 GB VRAM (leaves ~4K–6K for user message + response). For
constrained mode, reduce default `top_n` to 10.

### Tool 2: `execute_tool`

```
execute_tool(server_name: str, tool_name: str, arguments: str) -> dict
```

**Purpose:** Proxy a tool call to a child MCP server. The router launches the
server JIT via `uv run`, performs the MCP handshake, calls the specified tool,
returns the result, and manages the server process lifecycle.

**Why `arguments` is `str` not `dict`:** Local LLMs (especially ≤9B) are more
reliable at generating a JSON string than a nested dict parameter. The engine
parses the JSON string internally with error handling.

**Annotations:** `readOnlyHint=False`, `destructiveHint=False`, `idempotentHint=False`, `openWorldHint=False`

**Docstring:** `"""Execute a tool on a child MCP server via JIT launch."""`
(55 chars)

**Flow:**

1. Look up `server_name` in SQLite → get `server_cwd`, `launch_command`
2. Parse `arguments` from JSON string → dict
3. Check TTL pool — if server process is still alive, reuse it
4. If not in pool: start server via subprocess, MCP handshake
5. Call `tools/call` with `tool_name` and parsed arguments
6. Add/refresh server in TTL pool (60-second TTL)
7. Return the tool's result wrapped in router metadata

**Return value:**

```python
{
    "success": True,
    "op": "execute_tool",
    "server_name": "ml_basic",
    "tool_name": "train_classifier",
    "result": { ... },   # the child tool's full return dict
    "elapsed_seconds": 3.2,
    "progress": [...],
    "token_estimate": ...
}
```

**Error cases:**

- Server not found in registry → error dict with `hint: "Call list_servers()"`
- Tool not found on server → error dict with `hint: "Call search_tools()"`
- JSON parse failure on arguments → error dict with `hint: "arguments must be valid JSON string"`
- Server startup failure → error dict with `hint: "Check uv sync in server directory"`
- Server timeout (>120s) → error dict with `hint: "Operation timed out"`

### Tool 3: `list_servers`

```
list_servers() -> dict
```

**Purpose:** List all indexed MCP servers with their tool counts, repo names,
and index freshness. Used by the LLM to understand what's available and by
the user to verify the router is configured correctly.

**Annotations:** `readOnlyHint=True`, `destructiveHint=False`, `idempotentHint=True`, `openWorldHint=False`

**Docstring:** `"""List all indexed MCP servers and their tool counts."""`
(52 chars)

**Return value:**

```python
{
    "success": True,
    "op": "list_servers",
    "servers": [
        {
            "server_name": "data_basic",
            "repo_name": "MCP_Data_Analyst",
            "tool_count": 9,
            "indexed_at": "2026-04-15T10:00:00Z"
        },
        ...
    ],
    "total_servers": 10,
    "total_tools": 119,
    "index_age_hours": 2.5,
    "progress": [...],
    "token_estimate": ...
}
```

---

## Runtime Behavior

### TTL Process Pool

The router maintains an in-memory pool of running child server processes.
When `execute_tool` is called:

1. Check if `server_name` has a live process in the pool
2. If alive and last-used < TTL (60s default): reuse the session
3. If expired or not present: start new process, handshake, add to pool
4. After the call: update `last_used` timestamp
5. A background reaper task checks every 15s and kills expired processes

```python
# In-memory pool structure
_server_pool: dict[str, {
    "process": subprocess.Popen,
    "session": ClientSession,
    "read": ...,
    "write": ...,
    "last_used": float,  # time.monotonic()
}]

TTL_SECONDS = 60
REAP_INTERVAL = 15
```

This means: if the LLM calls `execute_tool("ml_basic", ...)` three times in a
row, only the first call pays the ~2s startup cost. The second and third calls
reuse the running process.

### Tool Result Cache

After `search_tools` returns, the results are stored in `_current_tools` (a
module-level dict). The LLM can reference these results in subsequent turns
without re-searching. The cache is replaced (not appended) on each new
`search_tools` call.

```python
_current_tools: dict[str, dict] = {}
# Key: "{server_name}/{tool_name}"
# Value: full tool info dict from search results
```

This cache is informational — the LLM uses it to avoid re-searching when it
already knows which tool it needs. It does NOT change which tools are "callable"
— `execute_tool` can call any indexed tool regardless of cache state.

### Startup Sequence

When the router's `server.py` starts (via mcp.json):

1. Check if `registry.db` exists at the expected path
2. If missing or empty: run full indexing (discover + handshake + build index)
3. If present: load TF-IDF artifacts into memory, verify DB integrity
4. Start the MCP server in stdio mode
5. Ready to serve `search_tools`, `execute_tool`, `list_servers`

The mcp.json launch command includes indexing as part of the bootstrap:

```
uv sync --quiet; uv run python -c "from servers.router_basic._router_indexer import reindex; reindex()" ; uv run python -m servers.router_basic.server
```

This ensures the index is always fresh on startup. The indexing step is idempotent
— if the DB is already up-to-date, it completes in <1s.

---

## Cross-Platform Support (STANDARDS §28)

### Path Handling

All paths use `pathlib.Path`. The base directory is:

```python
from pathlib import Path

def get_mcp_base_dir() -> Path:
    """Return ~/.mcp_servers/ on all platforms."""
    return Path.home() / ".mcp_servers"
```

- Windows: `C:\Users\{user}\.mcp_servers\`
- macOS: `/Users/{user}/.mcp_servers/`
- Linux: `/home/{user}/.mcp_servers/`

### Launch Command Platform Differences

When the indexer discovers a server, it records the launch command as a
platform-neutral string (`uv run python server.py`). The executor resolves
`uv` to the correct binary at runtime using `shutil.which("uv")`.

For module-style servers:
- Standalone: `uv run python server.py` (cwd = server dir)
- Module: `uv run python -m servers.{name}.server` (cwd = repo root)

### mcp.json Entries

**Windows (PowerShell):**

```json
{
  "mcpServers": {
    "tool_router": {
      "command": "powershell",
      "args": [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Tool_Router'; $g = Join-Path $d '.git'; if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; git clone https://github.com/azzindani/MCP_Tool_Router.git $d --quiet } else { Set-Location $d; git fetch origin --quiet; git reset --hard FETCH_HEAD --quiet }; Set-Location $d; uv sync --quiet; uv run python -c \"from servers.router_basic._router_indexer import reindex; reindex()\"; Set-Location (Join-Path $d 'servers\\router_basic'); uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

**macOS / Linux (bash):**

```json
{
  "mcpServers": {
    "tool_router": {
      "command": "bash",
      "args": [
        "-c",
        "d=\"$HOME/.mcp_servers/MCP_Tool_Router\"; if [ ! -d \"$d/.git\" ]; then rm -rf \"$d\"; git clone https://github.com/azzindani/MCP_Tool_Router.git \"$d\" --quiet; else cd \"$d\" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; fi; cd \"$d\"; uv sync --quiet; uv run python -c 'from servers.router_basic._router_indexer import reindex; reindex()'; cd \"$d/servers/router_basic\"; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

### Line Endings

```
# .gitattributes
* text=auto eol=lf
*.bat text eol=crlf
*.cmd text eol=crlf
```

---

## Dependencies

### Runtime Dependencies

```toml
[project]
requires-python = "==3.12.*"
dependencies = [
    "fastmcp>=2.0,<3.0",
    "mcp>=1.0,<2.0",           # MCP client SDK for handshake + tool calls
    "scikit-learn>=1.5",        # TF-IDF vectorizer + cosine similarity
    "joblib>=1.4",              # persist TF-IDF artifacts to disk
]
```

All MIT / BSD / Apache 2.0 licensed. No GPL. No cloud SDKs.

**Why scikit-learn for TF-IDF instead of a lighter library:** scikit-learn's
`TfidfVectorizer` is battle-tested, handles tokenization + n-grams + IDF in one
call, and is already a transitive dependency of both sibling projects. No new
dependency footprint.

**Phase 2 optional dependency (not in Phase 1):**

```toml
[project.optional-dependencies]
semantic = [
    "sentence-transformers>=3.0",  # ~80MB model for semantic search
]
```

### Dev Dependencies

```toml
[dependency-groups]
dev = [
    "pytest>=9.0",
    "ruff>=0.9",
    "pyright>=1.1",
]
```

### Prohibited

- `torch` / `transformers` in Phase 1 (too heavy for a routing layer)
- Any cloud embedding API (OpenAI, Cohere, etc.)
- `chromadb` / `qdrant` / `weaviate` (overkill — SQLite + FAISS or numpy suffice)
- `pip` anywhere in install or launch commands

---

## Testing Standards (STANDARDS §27)

### Test Strategy

Tests import `engine.py` directly. Never spin up the router as an MCP server
in tests. Child server interactions are tested with mock server fixtures.

### Mock Server Fixtures

```
tests/fixtures/mock_server_a/
├── server.py         # FastMCP server with 3 dummy tools
└── pyproject.toml    # minimal deps (fastmcp only)

tests/fixtures/mock_server_b/
├── server.py         # FastMCP server with 2 dummy tools
└── pyproject.toml
```

Mock servers return static dicts from their tools. They exist solely to test
the indexer's MCP handshake and the executor's JIT launch.

### Test Files

| File | What it tests |
|---|---|
| `test_indexer.py` | Discovery logic, MCP handshake, SQLite population, enriched text |
| `test_search.py` | TF-IDF index building, query matching, top-N retrieval, score ordering |
| `test_executor.py` | JIT launch, tool call proxying, TTL pool, timeout handling |

### Required Test Cases

**Indexer:**
- Discovers valid sibling repos in a tmp directory
- Skips the router's own directory
- Skips directories without `servers/` subdirectory
- Handles both standalone and module-style servers
- Stores correct enriched text in SQLite
- Re-index is idempotent (run twice, same result)

**Search:**
- Returns top-N results sorted by score descending
- Returns full JSON schema for each match
- Returns empty list with hint when no matches
- Constrained mode reduces default top_n
- TF-IDF artifacts load from disk correctly
- `token_estimate` is present and reasonable

**Executor:**
- Successfully calls a tool on a mock server
- Returns child tool's result wrapped in router metadata
- Returns error dict for unknown server_name
- Returns error dict for invalid JSON in arguments
- Returns error dict when server startup fails
- Respects timeout (kills server after limit)
- TTL pool: second call reuses existing process

### Coverage Targets

| Module | Target |
|---|---|
| `shared/` | 100% |
| `_router_helpers.py` | 100% |
| `_router_indexer.py` | ≥ 90% |
| `_router_search.py` | ≥ 90% |
| `_router_executor.py` | ≥ 85% (process management is harder to test) |

---

## CI/CD (STANDARDS §34)

### CI Workflow

```yaml
name: CI
on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]

jobs:
  test:
    name: Test (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-22.04, macos-latest, windows-latest]
    env:
      MCP_CONSTRAINED_MODE: "1"
      PYTHONPATH: "."
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright servers/ shared/
      - run: uv run python -m pytest tests/ -q --tb=short
```

No `brew install libomp` needed — the router has no C++ dependencies.

---

## What to Never Do (STANDARDS §36 + Router-Specific)

All 27 prohibitions from STANDARDS §36 apply. Additional router-specific rules:

1. **Never modify a sibling MCP server's files.** The router is read-only with
   respect to sibling repos. It connects via MCP protocol, never imports their
   code directly.

2. **Never import engine code from sibling servers.** The router talks to siblings
   only via MCP stdio transport. No `from MCP_Data_Analyst.servers.data_basic
   import engine`.

3. **Never cache child server tool results across router restarts.** The SQLite
   DB stores tool *schemas*, not tool *results*. Results are ephemeral.

4. **Never leave orphan child processes.** The TTL reaper must be robust. On
   router shutdown, all pooled child processes must be terminated.

5. **Never run indexing at query time.** If the DB is missing, return an error
   dict with a hint to restart the router (which triggers re-indexing).

6. **Never hardcode server names or tool names.** Everything is discovered
   dynamically from the filesystem.

7. **Never send user data to the indexer.** The indexer processes tool *schemas*
   (metadata), not user data. User data flows only through `execute_tool` →
   child server.

---

## Standards Deviations

This project deviates from STANDARDS.md in documented ways:

| Standard | Deviation | Justification |
|---|---|---|
| §7 Three-tier split | Single tier only | Router has exactly 3 tools — no domain complexity |
| §9 Four-tool pattern | Not applicable | Router is a meta-server, not a data/state tool |
| §13 Patch protocol | Not applicable | Router doesn't modify structured data |
| §19 Version control | No snapshots | Router doesn't write user data files |
| §25 Receipt log | No receipts | Router proxies calls — child servers log their own receipts |
| §26 Output generation | No HTML output | Router returns JSON only — visualization is child server's job |

All other standards (engine/server split, error handling, progress arrays,
token_estimate, constrained mode, security, testing, CI/CD, naming, cross-platform)
are followed exactly.

---

## Implementation Sequence

### Week 1: Foundation
- [ ] Repository setup (pyproject.toml, shared/, .github/)
- [ ] `_router_helpers.py` — constants, DB path, platform utils
- [ ] `_router_indexer.py` — discovery logic (filesystem scan only, no MCP yet)
- [ ] Tests for discovery with mock directory structures

### Week 2: MCP Client Integration
- [ ] `_router_indexer.py` — MCP handshake + tools/list collection
- [ ] SQLite schema creation and population
- [ ] Mock server fixtures for testing
- [ ] Tests for full indexing pipeline

### Week 3: Search
- [ ] `_router_search.py` — TF-IDF index building from SQLite
- [ ] `_router_search.py` — query vectorization + cosine similarity
- [ ] `search_tools` engine function with caching
- [ ] Tests for search accuracy and ranking

### Week 4: Executor
- [ ] `_router_executor.py` — JIT server launch + MCP call proxy
- [ ] TTL pool with background reaper
- [ ] `execute_tool` engine function
- [ ] Tests for execution, timeout, error handling

### Week 5: Integration + Polish
- [ ] `server.py` — thin MCP wrapper for all 3 tools
- [ ] `engine.py` — thin router re-exports
- [ ] End-to-end test: index → search → execute on mock servers
- [ ] CI/CD on all 3 platforms
- [ ] README with mcp.json entries
- [ ] Manual test with LM Studio + real sibling servers

---

*This CLAUDE.md follows STANDARDS.md v5.1 from
https://github.com/azzindani/Standards/blob/main/local_mcp/STANDARDS.md.
Where this document conflicts with STANDARDS.md, this document takes precedence
for this project.*
