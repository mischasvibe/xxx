from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "ai_trading_bot", log_level: int = logging.INFO) -> logging.Logger:
    global _LOGGER

    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_dir = Path("ai_trading_bot/data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "trading_bot.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _LOGGER = logger
    return logger
