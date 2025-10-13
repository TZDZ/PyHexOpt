import contextlib
import logging
from pathlib import Path

import yaml

setting_file = Path(__file__).parent.parent / "settings.yaml"
with setting_file.open("r") as f:
    params = yaml.safe_load(f)
LOGGER_NAME = params.get("logger_name", "pyhexopt")


def configure_logger(level=logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)

    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(CustomFormatter())
        root.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(fh)

    return root


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s - %(asctime)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {  # noqa: RUF012
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LogCaptureHandler(logging.Handler):
    """Custom logging handler to capture log messages in memory."""

    def __init__(self, max_lines=10):
        super().__init__()
        self.log_buffer = []  # List to store log messages
        self.max_lines = max_lines

    def emit(self, record):
        """Store the formatted log message in the buffer."""
        self.log_buffer.append(self.format(record))
        # Keep only the last `max_lines` messages
        if len(self.log_buffer) > self.max_lines:
            self.log_buffer.pop(0)

    def get_logs(self):
        """Retrieve the captured log messages."""
        return "\n".join(self.log_buffer)


@contextlib.contextmanager
def suppress_logging(logger, level=logging.WARNING):
    previous = logger.level
    previous_handler_levels = [h.level for h in logger.handlers]
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)
        for h, prev_level in zip(logger.handlers, previous_handler_levels):
            h.setLevel(prev_level)


@contextlib.contextmanager
def global_suppress_logging(level=logging.CRITICAL):
    prev = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(prev)


logger = configure_logger()
logger.propagate = False
