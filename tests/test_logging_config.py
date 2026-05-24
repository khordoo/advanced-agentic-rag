from __future__ import annotations

import logging

from rag_core.core.config import setup_logging


def test_setup_logging_preserves_existing_handlers_and_is_idempotent(
    monkeypatch,
) -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    existing_handler = logging.NullHandler()

    try:
        root_logger.handlers = [existing_handler]
        root_logger.setLevel(logging.WARNING)
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        setup_logging()
        setup_logging()

        rag_handlers = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, "_rag_json_stdout_handler", False)
        ]
        assert existing_handler in root_logger.handlers
        assert len(rag_handlers) == 1
        assert root_logger.level == logging.INFO
        assert rag_handlers[0].level == logging.INFO
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
