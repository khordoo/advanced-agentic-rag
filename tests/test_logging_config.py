from __future__ import annotations

import logging
import os

from rag_core.core.config import setup_logging
from rag_core.core.telemetry import (
    FAILED_VERIFICATION_ATTRIBUTE,
    FAILED_VERIFICATION_SPAN_NAME,
    FAILED_VERIFICATION_VALUE,
    VERIFICATION_GROUNDED_ATTRIBUTE,
    VERIFICATION_REASON_ATTRIBUTE,
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


def test_mark_failed_verification_attempt_creates_marker_span(monkeypatch) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    class RecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    class RecordingTracer:
        def __init__(self) -> None:
            self.span = RecordingSpan()
            self.span_name = None

        def start_as_current_span(self, name: str, context=None):
            self.span_name = name
            return self

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, tb):
            return False

    tracer = RecordingTracer()
    monkeypatch.setattr("rag_core.core.telemetry.tracer", tracer)

    mark_failed_verification_attempt(
        "unsupported claim",
        question="What is the revenue?",
        answer="Revenue is $1B",
        contexts=["doc-a", "doc-b"],
    )

    assert tracer.span_name == FAILED_VERIFICATION_SPAN_NAME
    assert tracer.span.attributes == {
        FAILED_VERIFICATION_ATTRIBUTE: FAILED_VERIFICATION_VALUE,
        VERIFICATION_GROUNDED_ATTRIBUTE: False,
        VERIFICATION_REASON_ATTRIBUTE: "unsupported claim",
        "question": "What is the revenue?",
        "answer": "Revenue is $1B",
        "contexts": ["doc-a", "doc-b"],
    }


def test_mark_failed_verification_attempt_allows_missing_reason(monkeypatch) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    class RecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    class RecordingTracer:
        def __init__(self) -> None:
            self.span = RecordingSpan()

        def start_as_current_span(self, name: str, context=None):
            return self

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, tb):
            return False

    tracer = RecordingTracer()
    monkeypatch.setattr("rag_core.core.telemetry.tracer", tracer)

    mark_failed_verification_attempt()

    assert tracer.span.attributes == {
        FAILED_VERIFICATION_ATTRIBUTE: FAILED_VERIFICATION_VALUE,
        VERIFICATION_GROUNDED_ATTRIBUTE: False,
    }
    # question/answer/contexts must NOT appear when omitted
    assert "question" not in tracer.span.attributes
    assert "answer" not in tracer.span.attributes
    assert "contexts" not in tracer.span.attributes


def test_mark_failed_verification_attempt_never_raises(monkeypatch) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    class FailingTracer:
        def start_as_current_span(self, name: str):
            raise RuntimeError("otel unavailable")

    monkeypatch.setattr("rag_core.core.telemetry.tracer", FailingTracer())

    mark_failed_verification_attempt()


def test_mark_failed_verification_attempt_no_longer_depends_on_current_span(
    monkeypatch,
) -> None:
    from rag_core.core.telemetry import mark_failed_verification_attempt

    class RecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    class RecordingTracer:
        def __init__(self) -> None:
            self.span = RecordingSpan()

        def start_as_current_span(self, name: str):
            return self

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, tb):
            return False

    span = RecordingSpan()
    monkeypatch.setattr(
        "rag_core.core.telemetry.trace.get_current_span",
        lambda: span,
    )
    monkeypatch.setattr("rag_core.core.telemetry.tracer", RecordingTracer())

    mark_failed_verification_attempt()

    assert span.attributes == {}


def test_phoenix_project_defaults_to_eval_project(monkeypatch) -> None:
    from rag_core.core.telemetry import _resolve_phoenix_project_name

    monkeypatch.delenv("PHOENIX_PROJECT_NAME", raising=False)
    monkeypatch.setenv("EVAL_PHOENIX_PROJECT", "advanced-rag-eval")

    assert _resolve_phoenix_project_name() == "advanced-rag-eval"


def test_phoenix_project_name_overrides_eval_project(monkeypatch) -> None:
    from rag_core.core.telemetry import _resolve_phoenix_project_name

    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "advanced-rag-runtime")
    monkeypatch.setenv("EVAL_PHOENIX_PROJECT", "advanced-rag-eval")

    assert _resolve_phoenix_project_name() == "advanced-rag-runtime"


def test_phoenix_project_header_preserves_existing_headers(monkeypatch) -> None:
    from rag_core.core.telemetry import _ensure_phoenix_project_header

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=token")

    _ensure_phoenix_project_header("advanced-rag-eval")

    assert (
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        == "authorization=token,x-project-name=advanced-rag-eval"
    )
