import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Dict, Any, Callable
from contextlib import contextmanager


class EasyLogger:
    """
    A simple wrapper around Python's logging library that provides an easy interface
    for configuration and usage.
    """

    # Standard logging levels as class attributes for easy access
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # Log format presets
    FORMAT_SIMPLE = "%(message)s"
    FORMAT_STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    FORMAT_DETAILED = (
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )

    def __init__(
        self,
        name: str = None,
        level: int = INFO,
        format_string: str = None,
        log_to_console: bool = True,
        log_to_file: Union[bool, str] = False,
        file_mode: str = "a",
    ):
        """
        Initialize a new EasyLogger instance.

        Args:
            name: Name of the logger. If None, uses the root logger.
            level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            format_string: Custom format string for log messages. If None, uses the standard format.
            log_to_console: Whether to log to the console.
            log_to_file: If True, logs to a file with the same name as the logger.
                         If a string, uses that as the filename.
            file_mode: File mode for the log file ('a' for append, 'w' for overwrite).
        """
        self.name = name or "root"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers = []

        # Set default format if none provided
        if format_string is None:
            format_string = self.FORMAT_STANDARD

        self.formatter = logging.Formatter(format_string)

        # Add console handler if requested
        if log_to_console:
            self._add_console_handler()

        # Add file handler if requested
        if log_to_file:
            filename = (
                log_to_file if isinstance(log_to_file, str) else f"{self.name}.log"
            )
            self._add_file_handler(filename, file_mode)

    def _add_console_handler(self):
        """Add a handler that logs to the console."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self, filename: str, file_mode: str = "a"):
        """Add a handler that logs to a file."""
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        file_handler = logging.FileHandler(filename, mode=file_mode)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def add_rotating_file_handler(
        self, filename: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
    ):
        """
        Add a size-based rotating file handler.

        Args:
            filename: Path to the log file.
            max_bytes: Maximum size of the log file before rotating (default: 10MB).
            backup_count: Number of backup files to keep (default: 5).
        """
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        handler = RotatingFileHandler(
            filename, maxBytes=max_bytes, backupCount=backup_count
        )
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def add_timed_rotating_file_handler(
        self, filename: str, when: str = "midnight", backup_count: int = 7
    ):
        """
        Add a time-based rotating file handler.

        Args:
            filename: Path to the log file.
            when: When to rotate the log file ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight').
                  Default is 'midnight'.
            backup_count: Number of backup files to keep (default: 7).
        """
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        handler = TimedRotatingFileHandler(
            filename, when=when, backupCount=backup_count
        )
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def set_level(self, level: int):
        """Set the logging level for this logger."""
        self.logger.setLevel(level)

    def debug(self, message: Any, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: Any, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: Any, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: Any, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: Any, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: Any, *args, **kwargs):
        """Log an exception message with traceback."""
        self.logger.exception(message, *args, **kwargs)

    @contextmanager
    def silence(self):
        """Temporarily silence the logger."""
        previous_level = self.logger.level
        self.logger.setLevel(logging.CRITICAL + 1)  # Higher than any standard level
        try:
            yield
        finally:
            self.logger.setLevel(previous_level)

    def get_native_logger(self):
        """Get the underlying Python logger instance."""
        return self.logger

    @classmethod
    def configure_root_logger(
        cls,
        level: int = INFO,
        format_string: str = None,
        log_to_console: bool = True,
        log_to_file: Union[bool, str] = False,
    ):
        """
        Configure the root logger with the given settings.

        Returns an EasyLogger instance wrapped around the root logger.
        """
        return cls(
            name=None,
            level=level,
            format_string=format_string,
            log_to_console=log_to_console,
            log_to_file=log_to_file,
        )


# Example usage
if __name__ == "__main__":
    # Basic usage with default settings
    logger = EasyLogger(name="my_app")
    logger.info("This is an info message")
    logger.error("This is an error message")

    # Debug logger with rotating file
    debug_logger = EasyLogger(
        name="debug_logger",
        level=EasyLogger.DEBUG,
        format_string=EasyLogger.FORMAT_DETAILED,
        log_to_file="logs/debug.log",
    )
    debug_logger.debug("This debug message goes to both console and file")
    debug_logger.add_rotating_file_handler("logs/rotating_debug.log")

    # Configure a logger for a specific module
    api_logger = EasyLogger(
        name="api",
        level=EasyLogger.INFO,
        format_string="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        log_to_console=True,
        log_to_file="logs/api.log",
    )
    api_logger.info("API initialized")

    # Using the silence context manager
    with api_logger.silence():
        api_logger.info("This won't be logged")
    api_logger.info("But this will be logged")

    # Configure the root logger
    root_logger = EasyLogger.configure_root_logger(
        level=EasyLogger.WARNING, log_to_file="logs/root.log"
    )
    root_logger.warning("This is a warning from the root logger")

    # Log an exception
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Caught an exception")
