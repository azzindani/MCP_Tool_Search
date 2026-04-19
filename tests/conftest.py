"""Shared fixtures for MCP Tool Search tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_SERVER_A_DIR = FIXTURES_DIR / "mock_server_a"
MOCK_SERVER_B_DIR = FIXTURES_DIR / "mock_server_b"


@pytest.fixture
def mock_server_a_info() -> dict:
    """Server info for mock_server_a using sys.executable (no uv needed)."""
    return {
        "repo_name": "mock_repo",
        "server_name": "mock_server_a",
        "server_cwd": str(MOCK_SERVER_A_DIR),
        "launch_style": "standalone",
        "launch_command": f"{sys.executable} server.py",
    }


@pytest.fixture
def mock_server_b_info() -> dict:
    """Server info for mock_server_b using sys.executable (no uv needed)."""
    return {
        "repo_name": "mock_repo",
        "server_name": "mock_server_b",
        "server_cwd": str(MOCK_SERVER_B_DIR),
        "launch_style": "standalone",
        "launch_command": f"{sys.executable} server.py",
    }


@pytest.fixture
def tmp_mcp_dir(tmp_path: Path) -> Path:
    """Create a temp MCP base directory with two mock repos."""
    for repo_name, server_name, fixture_dir in [
        ("mock_repo_a", "mock_server_a", MOCK_SERVER_A_DIR),
        ("mock_repo_b", "mock_server_b", MOCK_SERVER_B_DIR),
    ]:
        server_dir = tmp_path / repo_name / "servers" / server_name
        server_dir.mkdir(parents=True)
        (server_dir / "server.py").write_text((fixture_dir / "server.py").read_text())
        (server_dir / "pyproject.toml").write_text((fixture_dir / "pyproject.toml").read_text())
    return tmp_path
