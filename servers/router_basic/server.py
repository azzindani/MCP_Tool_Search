"""Tool Router MCP server — exposes search_tools, execute_tool, list_servers."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow standalone launch: python server.py (not just python -m servers.router_basic.server)
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from . import engine  # noqa: E402

mcp = FastMCP("tool-router")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def search_tools(query: str, top_n: int = 20, context: str = "") -> dict:
    """Search indexed tools by query. Returns top-N tool schemas."""
    return engine.search_tools(query, top_n, context=context or None)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
def execute_tool(server_name: str, tool_name: str, arguments: str) -> dict:
    """Execute a tool on a child MCP server via JIT launch."""
    return engine.execute_tool(server_name, tool_name, arguments)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def list_servers() -> dict:
    """List all indexed MCP servers and their tool counts."""
    return engine.list_servers()


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
def reindex_servers() -> dict:
    """Re-discover and re-index all sibling MCP servers."""
    return engine.reindex()


def _run_server() -> None:
    """Start the MCP server. Uses HTTP transport when MCP_HTTP_MODE=1."""
    import os

    if os.environ.get("MCP_HTTP_MODE", "0") == "1":
        port = int(os.environ.get("MCP_HTTP_PORT", "8080"))
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)  # type: ignore[call-arg]
    else:
        mcp.run()


if __name__ == "__main__":
    _run_server()
