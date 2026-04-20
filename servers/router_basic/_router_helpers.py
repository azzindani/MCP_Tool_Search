"""Shared constants, DB paths, and platform detection for the tool router."""

from __future__ import annotations

import os
from pathlib import Path

REPO_NAME = "MCP_Tool_Search"
SERVER_NAME = "router_basic"
DB_FILENAME = "registry.db"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
TFIDF_MATRIX_FILENAME = "tfidf_matrix.joblib"
EMBEDDINGS_FILENAME = "embeddings.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TTL_SECONDS = 60
REAP_INTERVAL = 15
DEFAULT_TOP_N = 20
CONSTRAINED_TOP_N = 10
USAGE_BOOST_FACTOR: float = 0.2

MCP_CONSTRAINED_MODE: bool = os.environ.get("MCP_CONSTRAINED_MODE", "0") == "1"
MCP_HTTP_MODE: bool = os.environ.get("MCP_HTTP_MODE", "0") == "1"
MCP_HTTP_PORT: int = int(os.environ.get("MCP_HTTP_PORT", "8080"))


def get_mcp_base_dir() -> Path:
    return Path.home() / ".mcp_servers"


def get_router_dir() -> Path:
    return get_mcp_base_dir() / REPO_NAME


def get_db_path() -> Path:
    return get_router_dir() / DB_FILENAME


def get_tfidf_vectorizer_path() -> Path:
    return get_router_dir() / TFIDF_VECTORIZER_FILENAME


def get_tfidf_matrix_path() -> Path:
    return get_router_dir() / TFIDF_MATRIX_FILENAME


def get_embeddings_path() -> Path:
    return get_router_dir() / EMBEDDINGS_FILENAME


def estimate_tokens(obj: object) -> int:
    import json

    return len(json.dumps(obj, default=str)) // 4
