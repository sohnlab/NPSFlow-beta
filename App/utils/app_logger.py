"""Centralized logging for NPSView.

Usage in any module:
    from utils.app_logger import logger
    logger.info("message")      # shown in UI log panel
    logger.debug("message")     # shown only in debug mode
    logger.error("message")     # shown in UI log panel

Call connect_to_app(app) once at startup to route messages to the UI.
"""

import logging

logger = logging.getLogger("npsview")
logger.setLevel(logging.DEBUG)

# Null handler by default (no terminal output)
logger.addHandler(logging.NullHandler())


class _AppLogHandler(logging.Handler):
    """Routes log records to NPSWorkflowApp.log / log_debug."""

    def __init__(self, app):
        super().__init__()
        self._app = app

    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.WARNING:
                self._app.log(msg)
            elif record.levelno >= logging.INFO:
                self._app.log(msg)
            else:
                self._app.log_debug(msg)
        except Exception:
            pass


def connect_to_app(app):
    """Connect the logger to the UI log panel."""
    handler = _AppLogHandler(app)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
