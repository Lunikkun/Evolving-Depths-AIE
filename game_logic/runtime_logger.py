from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

_LOGGER: Optional[logging.Logger] = None
_SESSION_DIR: Optional[str] = None
_STRUCTURED_LOG_PATH: Optional[str] = None

def get_session_dir() -> str:
    global _SESSION_DIR
    if _SESSION_DIR is None:
        session_name = f"session_{int(time.time())}"
        _SESSION_DIR = os.path.join("logs", session_name)
        os.makedirs(_SESSION_DIR, exist_ok=True)
    return _SESSION_DIR


def get_structured_log_path() -> str:
    global _STRUCTURED_LOG_PATH
    if _STRUCTURED_LOG_PATH is None:
        _STRUCTURED_LOG_PATH = os.path.join(get_session_dir(), "events.jsonl")
    return _STRUCTURED_LOG_PATH

def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    session_dir = get_session_dir()
    logger = logging.getLogger("evolving_depths")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_file = os.path.join(session_dir, "runtime_debug.log")
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def log_event(message: str) -> None:
    get_logger().info(message)


def log_structured_event(event_type: str, **payload: object) -> None:
    event = {
        "ts_unix": round(time.time(), 3),
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    with open(get_structured_log_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    get_logger().info("%s | %s", event_type, json.dumps(payload, ensure_ascii=True, sort_keys=True))