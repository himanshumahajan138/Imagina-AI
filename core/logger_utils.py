import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "app",
    level: int = logging.INFO,
    log_file: str | None = None,
    log_to_console: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up and configure a logger with console and/or file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is used
        log_to_console: Whether to log to console
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create default logger instance
logger = setup_logger(
    name="app", level=logging.INFO, log_file="logs/app.log", log_to_console=True
)


# Convenience functions
def set_level(level: int):
    """Set logging level for the default logger."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_file_handler(log_file: str, level: int = logging.INFO):
    """Add a file handler to the default logger."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
