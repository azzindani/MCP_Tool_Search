"""Mock MCP server B — 2 tools for testing the router."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mock_server_b")


@mcp.tool()
def tool_delta(x: float, y: float) -> dict:
    """Delta processing: compute sum and product of two numbers."""
    return {"sum": x + y, "product": x * y}


@mcp.tool()
def tool_epsilon(items: list) -> dict:
    """Epsilon aggregation: count and join list items."""
    return {"count": len(items), "joined": ",".join(str(i) for i in items)}


if __name__ == "__main__":
    mcp.run()
