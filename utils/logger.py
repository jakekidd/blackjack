import os
import sys
import logging
from datetime import datetime
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class Logger:
    def __init__(self, session_id: str, log_dir="data/logs", log_to_console=True, level=LogLevel.INFO, renderer=None):
        """
        Initialize the Logger instance.

        Args:
            session_id (str): Unique session identifier for log file naming.
            log_dir (str): Directory where log files will be stored.
            log_to_console (bool): Whether to print logs to the console.
            level (LogLevel): Minimum logging level to print to console.
        """
        self.session_id = session_id
        self.log_dir = log_dir
        self.log_to_console = log_to_console
        self.console_level = level

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up log file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"log_{session_id}_{timestamp}.txt")

        # Initialize logger
        self.logger = logging.getLogger(session_id)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(file_handler)

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._map_log_level_to_logging_level(level))
            console_handler.setFormatter(self._get_console_formatter())
            self.logger.addHandler(console_handler)

        self.renderer = renderer

    def _get_formatter(self):
        """Return the log file formatter."""
        return logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def _get_console_formatter(self):
        """Return the console log formatter with colors."""
        try:
            from colorama import Fore, Style, init
            init(autoreset=True)
            
            class ColoredFormatter(logging.Formatter):
                COLORS = {
                    LogLevel.DEBUG.value: Fore.BLUE,
                    LogLevel.INFO.value: Fore.GREEN,
                    LogLevel.WARN.value: Fore.YELLOW,
                    LogLevel.ERROR.value: Fore.RED,
                }

                def format(self, record):
                    color = self.COLORS.get(record.levelname, "")
                    reset = Style.RESET_ALL
                    record.msg = f"{color}{record.msg}{reset}"
                    return super().format(record)

            return ColoredFormatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        except ImportError:
            # If colorama is not installed, fall back to default
            return self._get_formatter()

    def _map_log_level_to_logging_level(self, level: LogLevel):
        """Map custom LogLevel to the logging library's levels."""
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }[level]

    def log(self, level: LogLevel, message: str):
        """
        Log a message with the specified log level.

        Args:
            level (LogLevel): Logging level (DEBUG, INFO, WARNING, ERROR).
            message (str): Message to log.
        """
        if self.renderer and level != LogLevel.DEBUG:
            self.renderer.log(message)

        if level == LogLevel.DEBUG:
            self.logger.debug(message)
        elif level == LogLevel.INFO:
            self.logger.info(message)
        elif level == LogLevel.WARN:
            self.logger.warning(message)
        elif level == LogLevel.ERROR:
            self.logger.error(message)

    def log_multiline(self, level: LogLevel, messages: list):
        """
        Log multiple lines with the specified log level.

        Args:
            level (LogLevel): Logging level (DEBUG, INFO, WARNING, ERROR).
            messages (list): List of strings to log.
        """
        for message in messages:
            self.log(level, message)

# Example usage
if __name__ == "__main__":
    logger = Logger(session_id="1234567890", level=LogLevel.WARN)
    logger.log(LogLevel.INFO, "This is an info message.")
    logger.log(LogLevel.WARNING, "This is a warning message.")
    logger.log(LogLevel.ERROR, "This is an error message.")
    logger.log_multiline(LogLevel.DEBUG, ["Line 1", "Line 2", "Line 3"])
