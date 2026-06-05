from __future__ import annotations

import logging
import os
import time
from typing import Optional

_LOGGER: Optional[logging.Logger] = None
_SESSION_DIR: Optional[str] = None

def get_session_dir() -> str:
    """Crea e restituisce una cartella di log univoca per questa sessione."""
    global _SESSION_DIR
    if _SESSION_DIR is None:
        # Crea un nome basato sul timestamp per evitare sovrascritture ma mantenerli ordinati
        session_name = f"session_{int(time.time())}"
        _SESSION_DIR = os.path.join("logs", session_name)
        os.makedirs(_SESSION_DIR, exist_ok=True)
    return _SESSION_DIR

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