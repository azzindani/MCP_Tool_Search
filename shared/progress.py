def ok(msg: str) -> dict:
    return {"level": "ok", "msg": msg}


def fail(msg: str) -> dict:
    return {"level": "fail", "msg": msg}


def info(msg: str) -> dict:
    return {"level": "info", "msg": msg}


def warn(msg: str) -> dict:
    return {"level": "warn", "msg": msg}


def undo(msg: str) -> dict:
    return {"level": "undo", "msg": msg}
