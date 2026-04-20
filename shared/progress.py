"""Progress helper functions for tool response arrays."""

from __future__ import annotations


def ok(msg: str, detail: str = "") -> dict:
    return {"icon": "✔", "msg": msg, "detail": detail}


def fail(msg: str, detail: str = "") -> dict:
    return {"icon": "✗", "msg": msg, "detail": detail}


def info(msg: str, detail: str = "") -> dict:
    return {"icon": "ℹ", "msg": msg, "detail": detail}


def warn(msg: str, detail: str = "") -> dict:
    return {"icon": "⚠", "msg": msg, "detail": detail}


def undo(msg: str, detail: str = "") -> dict:
    return {"icon": "↶", "msg": msg, "detail": detail}
