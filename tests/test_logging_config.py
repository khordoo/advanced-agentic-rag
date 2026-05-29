from __future__ import annotations

import logging

from rag_core.core.config import setup_logging
from rag_core.core.telemetry import (
    FAILED_VERIFICATION_ATTRIBUTE,
    FAILED_VERIFICATION_VALUE,
)


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


def test_mark_failed_verification_attempt_sets_recording_span(monkeypatch) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    class RecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    span = RecordingSpan()
    monkeypatch.setattr(
        "rag_core.core.telemetry.trace.get_current_span",
        lambda: span,
    )

    mark_failed_verification_attempt()

    assert span.attributes == {
        FAILED_VERIFICATION_ATTRIBUTE: FAILED_VERIFICATION_VALUE,
    }


def test_mark_failed_verification_attempt_skips_non_recording_span(
    monkeypatch,
) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    class NonRecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def is_recording(self) -> bool:
            return False

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    span = NonRecordingSpan()
    monkeypatch.setattr(
        "rag_core.core.telemetry.trace.get_current_span",
        lambda: span,
    )

    mark_failed_verification_attempt()

    assert span.attributes == {}


def test_mark_failed_verification_attempt_never_raises(monkeypatch) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    def raise_on_span():
        raise RuntimeError("otel unavailable")

    monkeypatch.setattr(
        "rag_core.core.telemetry.trace.get_current_span",
        raise_on_span,
    )

    mark_failed_verification_attempt()
