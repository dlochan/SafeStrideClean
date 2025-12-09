from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module- or application-level logger.

    Configures a simple formatter with timestamps and levels on first use.
    Subsequent calls reuse the existing configuration.
    """

    logger_name = name or "safestride.vnext"
    logger = logging.getLogger(logger_name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger
