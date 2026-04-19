import os

MCP_CONSTRAINED_MODE: bool = os.environ.get("MCP_CONSTRAINED_MODE", "0") == "1"


def get_max_rows() -> int:
    return 10 if MCP_CONSTRAINED_MODE else 20
