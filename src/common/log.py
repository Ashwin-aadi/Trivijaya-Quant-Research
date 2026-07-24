"""Console logging with a consistent, timestamped format.

Kept as its own module so every script configures logging the same way. Timestamps are UTC on
purpose: runs get compared across machines and a local-time log makes that needlessly painful.
"""

from __future__ import annotations

import logging
import time

_CONFIGURED = False
_FORMAT = "%(asctime)s UTC | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stream handler on the root logger. Idempotent.

    Calling this more than once (e.g. from several entry points in one process) must not stack
    duplicate handlers, hence the module-level guard.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    formatter = logging.Formatter(_FORMAT)
    formatter.converter = time.gmtime  # emit UTC, not local time
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root handler on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
