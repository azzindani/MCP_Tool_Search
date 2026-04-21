# MCP Tool Search

A self-hosted MCP tool router that indexes sibling MCP servers and selectively retrieves tool schemas via TF-IDF RAG — enabling local LLMs to access 500+ tools without exceeding context limits.

## The Problem

Loading all tool schemas from multiple MCP servers exhausts the context window of local LLMs before the user types anything:

| Scenario | Tool schemas | Tokens consumed |
|---|---|---|
| MCP_Data_Analyst (7 servers) | 84 schemas | ~25K–42K tokens |
| MCP_Machine_Learning (3 servers) | 35 schemas | ~10K–17K tokens |
| Both projects loaded | 119 schemas | ~35K–60K tokens |

A 9B model on 8 GB VRAM has ~10K–12K usable context tokens. One project's full tool set can exhaust it entirely.

## The Solution

MCP Tool Search exposes **3 tools** (~800 tokens of schema). At runtime it uses TF-IDF RAG to retrieve the top-N most relevant tool schemas and proxies calls to the correct child server via JIT launch.

```
User message → LLM calls search_tools("train random forest")
            → Router returns top-20 tool schemas as text
            → LLM selects train_classifier, calls execute_tool(...)
            → Router: launches child server via uv run (stdio MCP)
            → Router: proxies tools/call → result → LLM
```

## Tools

| Tool | Description |
|---|---|
| `search_tools(query, top_n=20)` | TF-IDF search over indexed tool schemas |
| `execute_tool(server_name, tool_name, arguments)` | Proxy a call to a child MCP server |
| `list_servers()` | List all indexed servers and tool counts |

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Sibling MCP servers cloned into `~/.mcp_servers/`

### Quick Install

**macOS / Linux:**
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/azzindani/MCP_Tool_Search/main/install/install.sh)
```

**Windows:**
```cmd
curl -fsSL https://raw.githubusercontent.com/azzindani/MCP_Tool_Search/main/install/install.bat -o install.bat && install.bat
```

## mcp.json Configuration

### macOS / Linux

```json
{
  "mcpServers": {
    "tool_router": {
      "command": "bash",
      "args": [
        "-c",
        "d=\"$HOME/.mcp_servers/MCP_Tool_Search\"; if [ ! -d \"$d/.git\" ]; then rm -rf \"$d\"; git clone https://github.com/azzindani/MCP_Tool_Search.git \"$d\" --quiet; else cd \"$d\" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; fi; cd \"$d\"; uv sync --quiet; uv run python -c 'from servers.router_basic._router_indexer import reindex; reindex()'; uv run python -m servers.router_basic.server"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

### Windows (PowerShell)

```json
{
  "mcpServers": {
    "tool_router": {
      "command": "powershell",
      "args": [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Tool_Search'; $g = Join-Path $d '.git'; if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; git clone https://github.com/azzindani/MCP_Tool_Search.git $d --quiet } else { Set-Location $d; git fetch origin --quiet; git reset --hard FETCH_HEAD --quiet }; Set-Location $d; uv sync --quiet; uv run python -c \"from servers.router_basic._router_indexer import reindex; reindex()\"; uv run python -m servers.router_basic.server"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

Set `MCP_CONSTRAINED_MODE=1` to reduce `search_tools` default from 20 to 10 results (recommended for models with <8K context).

## Sibling Server Layout

The router auto-discovers sibling repos under `~/.mcp_servers/`:

```
~/.mcp_servers/
├── MCP_Tool_Search/        ← this repo (excluded from discovery)
├── MCP_Data_Analyst/
│   └── servers/
│       └── data_basic/
│           ├── server.py
│           └── pyproject.toml
└── MCP_Machine_Learning/
    └── servers/
        └── ml_basic/
            ├── server.py
            └── pyproject.toml
```

## Architecture

- **`_router_indexer`** — filesystem discovery + MCP handshake + SQLite indexing
- **`_router_search`** — TF-IDF cosine similarity search (scikit-learn, CPU-only)
- **`_router_executor`** — JIT server launch + TTL process pool (60s keepalive)
- **`engine.py`** — zero-MCP-import re-exports
- **`server.py`** — thin FastMCP wrapper

Index stored at `~/.mcp_servers/MCP_Tool_Search/registry.db` with TF-IDF artifacts alongside. Rebuilt on every router startup.

## Development

```bash
git clone https://github.com/azzindani/MCP_Tool_Search.git
cd MCP_Tool_Search
uv sync
PYTHONPATH=. uv run python -m pytest tests/ -q
```

## Requirements

- Python 3.12
- CPU-only (GPU reserved for local LLM)
- No cloud APIs or API keys
