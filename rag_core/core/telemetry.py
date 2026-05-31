import logging
import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_instrumented = False
FAILED_VERIFICATION_SPAN_NAME = "agent.verification.rejected"
FAILED_VERIFICATION_ATTRIBUTE = "custom.verification_status"
FAILED_VERIFICATION_VALUE = "failed"
VERIFICATION_REASON_ATTRIBUTE = "custom.verification_reason"
VERIFICATION_GROUNDED_ATTRIBUTE = "custom.verification_grounded"
PHOENIX_PROJECT_RESOURCE_ATTRIBUTE = "openinference.project.name"
PHOENIX_PROJECT_HEADER = "x-project-name"


def init_observability(service_name: str = "advanced-rag") -> None:
    """Configure OpenInference LangChain tracing with an OTLP exporter.

    The destination is intentionally controlled by standard OpenTelemetry
    environment variables. Point OTEL_EXPORTER_OTLP_TRACES_ENDPOINT at Phoenix,
    Langfuse's OTLP endpoint, or any compatible collector.
    """
    global _instrumented
    if _instrumented:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    phoenix_project = _resolve_phoenix_project_name()

    try:
        if endpoint:
            os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
            if phoenix_project:
                _ensure_phoenix_project_header(phoenix_project)
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": service_name,
                        **(
                            {PHOENIX_PROJECT_RESOURCE_ATTRIBUTE: phoenix_project}
                            if phoenix_project
                            else {}
                        ),
                    }
                )
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            logger.info(
                "OTLP tracing enabled for service=%s endpoint=%s project=%s",
                service_name,
                endpoint,
                phoenix_project or "default",
            )
        else:
            logger.info("OTLP tracing endpoint is not configured; instrumentation is local only.")

        LangChainInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
        _instrumented = True
    except Exception as exc:
        logger.warning("OpenInference instrumentation failed: %s", exc)


def _resolve_phoenix_project_name() -> str | None:
    """Resolve the Phoenix project for runtime traces."""
    return os.getenv("PHOENIX_PROJECT_NAME") or os.getenv("EVAL_PHOENIX_PROJECT")


def _ensure_phoenix_project_header(project_name: str) -> None:
    """Route OTLP HTTP exports to a Phoenix project without clobbering headers."""
    existing = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    header = f"{PHOENIX_PROJECT_HEADER}={project_name}"
    if not existing:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = header
        return
    if f"{PHOENIX_PROJECT_HEADER}=" in existing:
        return
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"{existing},{header}"


def mark_failed_verification_attempt(
    reason: str | None = None,
    *,
    question: str | None = None,
    answer: str | None = None,
    contexts: list[str] | None = None,
) -> None:
    """Create a marker span so CE drift jobs can find verifier rejections.

    The ``question``, ``answer``, and ``contexts`` fields are set as
    top-level span attributes so that ``_extract_trace_example()`` in the
    drift-detection flow can turn marker spans directly into eval examples
    without crawling the trace tree.

    Telemetry must never affect generation, so unexpected OpenTelemetry
    failures are logged at debug level and swallowed.
    """
    try:
        with tracer.start_as_current_span(FAILED_VERIFICATION_SPAN_NAME) as span:
            span.set_attribute(
                FAILED_VERIFICATION_ATTRIBUTE,
                FAILED_VERIFICATION_VALUE,
            )
            span.set_attribute(VERIFICATION_GROUNDED_ATTRIBUTE, False)
            if reason:
                span.set_attribute(VERIFICATION_REASON_ATTRIBUTE, reason)
            if question:
                span.set_attribute("question", question)
            if answer:
                span.set_attribute("answer", answer)
            if contexts:
                span.set_attribute("contexts", contexts)
    except Exception as exc:
        logger.debug("Failed to mark verifier failure span: %s", exc)
