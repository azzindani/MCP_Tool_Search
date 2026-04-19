"""Write the tool-router entry to AI client mcp.json config files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _get_lm_studio_config_path() -> Path | None:
    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Roaming" / "LM Studio"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "LM Studio"
    else:
        base = Path.home() / ".config" / "LM Studio"
    candidate = base / "mcp.json"
    return candidate if base.is_dir() else None


def _build_entry_posix() -> dict:
    return {
        "command": "bash",
        "args": [
            "-c",
            (
                'd="$HOME/.mcp_servers/MCP_Tool_Search"; '
                'if [ ! -d "$d/.git" ]; then rm -rf "$d"; '
                'git clone https://github.com/azzindani/MCP_Tool_Search.git "$d" --quiet; '
                'else cd "$d" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; fi; '
                'cd "$d"; uv sync --quiet; '
                "uv run python -c 'from servers.router_basic._router_indexer import reindex; reindex()'; "
                "uv run python -m servers.router_basic.server"
            ),
        ],
        "env": {"MCP_CONSTRAINED_MODE": "0"},
        "timeout": 600000,
    }


def _build_entry_windows() -> dict:
    return {
        "command": "powershell",
        "args": [
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            (
                "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Tool_Search'; "
                "$g = Join-Path $d '.git'; "
                "if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; "
                "git clone https://github.com/azzindani/MCP_Tool_Search.git $d --quiet } "
                "else { Set-Location $d; git fetch origin --quiet; git reset --hard FETCH_HEAD --quiet }; "
                "Set-Location $d; uv sync --quiet; "
                'uv run python -c "from servers.router_basic._router_indexer import reindex; reindex()"; '
                "uv run python -m servers.router_basic.server"
            ),
        ],
        "env": {"MCP_CONSTRAINED_MODE": "0"},
        "timeout": 600000,
    }


def write_config(config_path: Path | None = None) -> None:
    if config_path is None:
        config_path = _get_lm_studio_config_path()
    if config_path is None:
        print("Could not find AI client config directory. Please add the entry manually.")
        _print_manual_entry()
        return

    entry = _build_entry_windows() if sys.platform == "win32" else _build_entry_posix()

    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing.setdefault("mcpServers", {})["tool_router"] = entry
    config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Written tool_router entry to {config_path}")


def _print_manual_entry() -> None:
    if sys.platform == "win32":
        entry = _build_entry_windows()
    else:
        entry = _build_entry_posix()
    print(json.dumps({"mcpServers": {"tool_router": entry}}, indent=2))


if __name__ == "__main__":
    write_config()
