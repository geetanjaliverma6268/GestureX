"""Event-based logging.

The app logs *events* (camera ready, gesture fired, action executed, control
locked, warnings) rather than per-frame chatter. A small ``ThrottledLogger``
helper is available for anything that would otherwise repeat every frame.
"""

import logging
import os
import time

_CONFIGURED = False
_LOGGER_NAME = "touchless"


def setup_logging(level="INFO", log_file=None, root=None):
    """Configure the root ``touchless`` logger once. Safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        try:
            if root and not os.path.isabs(log_file):
                log_file = os.path.join(root, log_file)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            logger.warning("Could not open log file %s", log_file)

    _CONFIGURED = True
    return logger


def get_logger(name=None):
    if not _CONFIGURED:
        setup_logging()
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


class ThrottledLogger:
    """Emit the same message at most once per ``interval`` seconds."""

    def __init__(self, logger, interval=2.0):
        self._logger = logger
        self._interval = interval
        self._last = {}

    def warn(self, key, msg, *args):
        now = time.time()
        if now - self._last.get(key, -1e9) >= self._interval:
            self._last[key] = now
            self._logger.warning(msg, *args)
