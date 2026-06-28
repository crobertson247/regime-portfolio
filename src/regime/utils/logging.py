"""
Logging configuration for the regime portfolio data pipeline.

Provides structured logging via Python's logging module.
All pipeline operations are logged for reproducibility and debugging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str | Path] = None,
    base_path: Optional[Path] = None,
) -> None:
    """
    Configure logging for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format string for log messages.
        log_file: Optional path to log file (relative to base_path if provided).
        base_path: Base path for relative log file paths.
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger for the regime package
    root_logger = logging.getLogger("regime")
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        if base_path is not None:
            log_path = base_path / log_path

        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__ from the calling module).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if name not in _loggers:
        # Create logger as child of regime root logger
        if name.startswith("regime"):
            logger = logging.getLogger(name)
        else:
            logger = logging.getLogger(f"regime.{name}")
        _loggers[name] = logger

    return _loggers[name]


class LogContext:
    """
    Context manager for logging operation start/end with timing.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, "data ingestion"):
        ...     # do work
        ...     pass
        # Logs: "Starting data ingestion..."
        # Logs: "Completed data ingestion in X.XX seconds"
    """

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        """
        Initialize log context.

        Args:
            logger: Logger instance to use.
            operation: Description of the operation being performed.
            level: Logging level for messages.
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self._start_time: Optional[float] = None

    def __enter__(self) -> "LogContext":
        """Log operation start."""
        import time

        self._start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting {self.operation}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Log operation completion or failure."""
        import time

        elapsed = time.perf_counter() - self._start_time if self._start_time else 0

        if exc_type is not None:
            self.logger.error(f"Failed {self.operation} after {elapsed:.2f}s: {exc_val}")
            return False

        self.logger.log(self.level, f"Completed {self.operation} in {elapsed:.2f}s")
        return False
