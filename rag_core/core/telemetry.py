import logging
import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_instrumented = False
FAILED_VERIFICATION_ATTRIBUTE = "custom.verification_status"
FAILED_VERIFICATION_VALUE = "failed"


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

    try:
        if endpoint:
            os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
            provider = TracerProvider(
                resource=Resource.create({"service.name": service_name})
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            logger.info("OTLP tracing enabled for service=%s endpoint=%s", service_name, endpoint)
        else:
            logger.info("OTLP tracing endpoint is not configured; instrumentation is local only.")

        LangChainInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
        _instrumented = True
    except Exception as exc:
        logger.warning("OpenInference instrumentation failed: %s", exc)


def mark_failed_verification_attempt() -> None:
    """Annotate the active span so CE drift jobs can find verifier failures.

    Telemetry must never affect generation, so unexpected OpenTelemetry
    failures are logged at debug level and swallowed.
    """
    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(
                FAILED_VERIFICATION_ATTRIBUTE,
                FAILED_VERIFICATION_VALUE,
            )
    except Exception as exc:
        logger.debug("Failed to mark verifier failure span: %s", exc)
