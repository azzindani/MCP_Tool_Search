"""Tool-router engine — zero MCP imports. Re-exports from sub-modules."""

from ._router_executor import execute_tool, shutdown_pool
from ._router_indexer import list_servers, reindex
from ._router_search import search_tools

__all__ = ["search_tools", "execute_tool", "list_servers", "reindex", "shutdown_pool"]
