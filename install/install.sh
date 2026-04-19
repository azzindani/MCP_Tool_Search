#!/usr/bin/env sh
# Install MCP Tool Search into ~/.mcp_servers/MCP_Tool_Search/
set -e

REPO_URL="https://github.com/azzindani/MCP_Tool_Search.git"
INSTALL_DIR="$HOME/.mcp_servers/MCP_Tool_Search"

echo "[mcp-tool-search] Installing to $INSTALL_DIR"

if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[mcp-tool-search] Updating existing installation"
    cd "$INSTALL_DIR"
    git fetch origin --quiet
    git reset --hard FETCH_HEAD --quiet
else
    echo "[mcp-tool-search] Cloning repository"
    rm -rf "$INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR" --quiet
    cd "$INSTALL_DIR"
fi

echo "[mcp-tool-search] Installing dependencies"
uv sync --quiet

echo "[mcp-tool-search] Building tool index"
uv run python -c "from servers.router_basic._router_indexer import reindex; reindex()"

echo "[mcp-tool-search] Installation complete"
echo ""
echo "Add to your mcp.json (macOS/Linux):"
cat <<'EOF'
{
  "mcpServers": {
    "tool_router": {
      "command": "bash",
      "args": [
        "-c",
        "d=\"$HOME/.mcp_servers/MCP_Tool_Search\"; cd \"$d\" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; uv sync --quiet; uv run python -c 'from servers.router_basic._router_indexer import reindex; reindex()'; uv run python -m servers.router_basic.server"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
EOF
