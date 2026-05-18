from __future__ import annotations

import logging
import os
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("evolving_depths")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler("logs/runtime_debug.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def log_event(message: str) -> None:
    get_logger().info(message)
