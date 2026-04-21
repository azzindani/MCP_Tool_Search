"""Shared file utilities: path validation, atomic writes."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def resolve_path(file_path: str | Path, allowed_extensions: tuple[str, ...] = ()) -> Path:
    """Resolve and validate a path. Raises ValueError if outside home or wrong extension."""
    path = Path(file_path).expanduser().resolve()
    try:
        path.relative_to(Path.home().resolve())
    except ValueError:
        raise ValueError(f"Path outside allowed directory: {file_path}")
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Extension {path.suffix!r} not in allowed set {allowed_extensions}")
    return path


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to path atomically via a temp file + rename."""
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=target.suffix)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        shutil.move(tmp, str(target))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
