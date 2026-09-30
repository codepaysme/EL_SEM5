"""Logging utilities for MetaHPO-C."""

import logging
import sys
from typing import Union


def get_logger(name: str = "metahpo_c", level: Union[int, str] = logging.INFO) -> logging.Logger:
    """Configure and return a standardized console logger.

    Parameters
    ----------
    name : str, default="metahpo_c"
        Name of the logger hierarchy.
    level : int or str, default=logging.INFO
        Logging severity level (e.g. logging.INFO or "INFO").

    Returns
    -------
    logging.Logger
        Configured logger instance with a formatted StreamHandler.
    """
    logger = logging.getLogger(name)

    # Set level if string representation provided
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
