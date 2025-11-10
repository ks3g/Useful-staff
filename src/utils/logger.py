"""Logging utilities for RAG system."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> None:
    """Set up logger with file and console output.

    Args:
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string
    """
    # Remove default logger
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console logger
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )

    # Add file logger if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )

    logger.info(f"Logger initialized with level: {level}")


def get_logger():
    """Get the configured logger instance.

    Returns:
        Logger instance
    """
    return logger
