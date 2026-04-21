"""Mock MCP server A — 3 tools for testing the router."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mock_server_a")


@mcp.tool()
def tool_alpha(data: str) -> dict:
    """Process alpha data and return transformed result."""
    return {"result": f"alpha:{data}"}


@mcp.tool()
def tool_beta(value: int, label: str) -> dict:
    """Transform beta value with label. Doubles the value."""
    return {"result": value * 2, "label": label}


@mcp.tool()
def tool_gamma() -> dict:
    """Return gamma status check result."""
    return {"status": "ok", "server": "mock_server_a"}


if __name__ == "__main__":
    mcp.run()
