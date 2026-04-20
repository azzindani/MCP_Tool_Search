"""Hardware-awareness and constraint helpers. All reads happen at call time."""

from __future__ import annotations

import os


def is_constrained_mode() -> bool:
    """Return True if MCP_CONSTRAINED_MODE=1 is set in the environment."""
    return os.environ.get("MCP_CONSTRAINED_MODE", "0") == "1"


def get_max_rows() -> int:
    return 10 if is_constrained_mode() else 20


def get_max_results() -> int:
    return 10 if is_constrained_mode() else 50
