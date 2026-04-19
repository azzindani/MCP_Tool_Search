@echo off
REM Install MCP Tool Search into %USERPROFILE%\.mcp_servers\MCP_Tool_Search\
setlocal

set REPO_URL=https://github.com/azzindani/MCP_Tool_Search.git
set INSTALL_DIR=%USERPROFILE%\.mcp_servers\MCP_Tool_Search

echo [mcp-tool-search] Installing to %INSTALL_DIR%

if exist "%INSTALL_DIR%\.git" (
    echo [mcp-tool-search] Updating existing installation
    cd /d "%INSTALL_DIR%"
    git fetch origin --quiet
    git reset --hard FETCH_HEAD --quiet
) else (
    echo [mcp-tool-search] Cloning repository
    if exist "%INSTALL_DIR%" rmdir /s /q "%INSTALL_DIR%"
    git clone "%REPO_URL%" "%INSTALL_DIR%" --quiet
    cd /d "%INSTALL_DIR%"
)

echo [mcp-tool-search] Installing dependencies
uv sync --quiet

echo [mcp-tool-search] Building tool index
uv run python -c "from servers.router_basic._router_indexer import reindex; reindex()"

echo [mcp-tool-search] Installation complete
echo.
echo Add to your mcp.json (Windows) - see README.md for the PowerShell entry.

endlocal
