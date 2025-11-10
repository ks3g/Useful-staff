"""Simple logging utility."""

import logging
import sys


def get_logger(name: str = "rag_system") -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
