"""
OpenTelemetry Tracing for QuantMindX

Provides distributed tracing across the QuantMindX trading platform:
- FastAPI request tracing
- Database query tracing
- Trading operation tracing
- Trace context propagation for WebSocket/SSE

Usage:
    from src.monitoring.tracing import init_tracing, get_tracer

    # Initialize on app startup
    init_tracing(app)

    # Get tracer for custom spans
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("custom.attribute", "value")
"""

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: Optional[trace.Tracer] = None
_initialized = False


def get_trace_exporter_endpoint() -> str:
    """Get OTLP exporter endpoint from environment."""
    return os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        os.getenv("GRAFANA_TEMPO_OTLP_ENDPOINT", "http://localhost:4317")
    )


def get_service_name() -> str:
    """Get service name for tracing from environment."""
    return os.getenv("OTEL_SERVICE_NAME", "quantmind-api")


def get_deployment_environment() -> str:
    """Get deployment environment."""
    return os.getenv("NODE_ENVIRONMENT", os.getenv("QUANTMIND_ENV", "production"))


def init_tracing(app=None, service_name: Optional[str] = None) -> None:
    """
    Initialize OpenTelemetry tracing.

    Args:
        app: FastAPI application instance for auto-instrumentation
        service_name: Custom service name (default: from OTEL_SERVICE_NAME env)
    """
    global _tracer, _initialized

    if _initialized:
        logger.warning("Tracing already initialized")
        return

    service_name = service_name or get_service_name()
    environment = get_deployment_environment()

    # Create resource with service metadata
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: os.getenv("QUANTMIND_VERSION", "1.0.0"),
        DEPLOYMENT_ENVIRONMENT: environment,
        "quantmind.node_role": os.getenv("NODE_ROLE", "local"),
    })

    # Set up tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure OTLP exporter
    otlp_endpoint = get_trace_exporter_endpoint()
    if otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=otlp_endpoint.startswith("http://"),
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            logger.info(f"OTLP trace exporter configured: {otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Set global propagator to W3C Trace Context
    set_global_textmap(TraceContextTextMapPropagator())

    # Get tracer
    _tracer = trace.get_tracer(__name__)

    # Auto-instrument FastAPI if app provided
    if app is not None:
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                excluded_urls="health,metrics,/metrics",
            )
            logger.info("FastAPI auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")

    _initialized = True
    logger.info(f"Tracing initialized for service={service_name}, env={environment}")


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually __name__)

    Returns:
        Tracer instance
    """
    global _tracer

    if _tracer is None:
        # Fallback to default tracer if not initialized
        return trace.get_tracer(name)

    return _tracer


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a hex string.

    Returns:
        Trace ID or None if no active span
    """
    span = trace.get_current_span()
    if span is None or not span.get_span_context().is_valid:
        return None

    trace_id = span.get_span_context().trace_id
    return format(trace_id, '032x')


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID as a hex string.

    Returns:
        Span ID or None if no active span
    """
    span = trace.get_current_span()
    if span is None or not span.get_span_context().is_valid:
        return None

    span_id = span.get_span_context().span_id
    return format(span_id, '016x')


def create_trading_span(
    operation: str,
    symbol: Optional[str] = None,
    action: Optional[str] = None,
    mode: Optional[str] = None,
    bot_id: Optional[str] = None,
    strategy: Optional[str] = None,
    regime: Optional[str] = None,
):
    """
    Create a span for trading operations with standard attributes.

    Args:
        operation: Operation name (e.g., "execute_trade", "close_position")
        symbol: Trading symbol (e.g., "EURUSD")
        action: Trade action ("BUY" or "SELL")
        mode: Trading mode ("live", "demo", "paper")
        bot_id: EA identifier
        strategy: Strategy name
        regime: Current market regime

    Returns:
        Span context manager
    """
    tracer = get_tracer("quantmind.trading")

    span_name = f"trading.{operation}"
    span = tracer.start_span(span_name)

    # Set trading attributes
    if symbol:
        span.set_attribute("trading.symbol", symbol)
    if action:
        span.set_attribute("trading.action", action)
    if mode:
        span.set_attribute("trading.mode", mode)
    if bot_id:
        span.set_attribute("trading.bot_id", bot_id)
    if strategy:
        span.set_attribute("trading.strategy", strategy)
    if regime:
        span.set_attribute("trading.regime", regime)

    return span


def add_trace_context_to_log(record):
    """
    Add trace context to log records for correlation.

    Use with logging.JSONFormatter or similar.

    Args:
        record: Log record to modify in place
    """
    trace_id = get_current_trace_id()
    span_id = get_current_span_id()

    if trace_id:
        record.trace_id = trace_id
    if span_id:
        record.span_id = span_id


class QuantMindSpanProcessor:
    """
    Custom span processor that adds QuantMind-specific context.

    Can be used to enrich spans with current trading context
    (e.g., active bot, current regime) from thread-local state.
    """

    def __init__(self, span_processor):
        self.processor = span_processor

    def on_start(self, span):
        """Add custom attributes on span start."""
        # Add node role if not already set
        if not span.is_recording():
            return

        resource = span.resource
        if resource and not span.attributes.get("quantmind.node_role"):
            span.set_attribute(
                "quantmind.node_role",
                resource.attributes.get("quantmind.node_role", "unknown")
            )

        # Add any thread-local trading context here
        # (e.g., current_bot_id from contextvars)

        self.processor.on_start(span)

    def on_end(self, span):
        """Process span on end."""
        self.processor.on_end(span)

    def shutdown(self):
        """Shutdown the processor."""
        self.processor.shutdown()


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled and configured."""
    return _initialized and _tracer is not None
