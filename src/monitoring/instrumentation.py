"""
Custom Instrumentation for QuantMindX

Provides specialized instrumentation for trading operations, including:
- Strategy router spans
- Trade execution spans
- MT5 bridge spans
- Agent/Department spans
- WebSocket/SSE trace context handling

Usage:
    from src.monitoring.instrumentation import (
        trace_strategy_routing,
        trace_trade_execution,
        trace_mt5_operation,
        trace_agent_task,
    )

    # Context manager for trading spans
    with trace_trade_execution("EURUSD", "BUY", "live") as span:
        # Execute trade
        result = execute_trade(...)
        span.set_attribute("trade.result", result)
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.monitoring.tracing import get_tracer, get_current_trace_id

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Router Instrumentation
# =============================================================================

def trace_strategy_routing(symbol: str, regime: str):
    """
    Context manager for strategy routing spans.

    Args:
        symbol: Trading symbol being evaluated
        regime: Current market regime

    Yields:
        Span for the routing operation
    """
    tracer = get_tracer("quantmind.router")

    with tracer.start_as_current_span("router.evaluate_strategy") as span:
        span.set_attribute("router.symbol", symbol)
        span.set_attribute("router.regime", regime)
        span.set_attribute("router.operation", "strategy_selection")

        yield span


def trace_regime_detection(symbol: str, timeframe: str):
    """
    Context manager for regime detection spans.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe being analyzed

    Yields:
        Span for the regime detection
    """
    tracer = get_tracer("quantmind.hmm")

    with tracer.start_as_current_span("hmm.detect_regime") as span:
        span.set_attribute("hmm.symbol", symbol)
        span.set_attribute("hmm.timeframe", timeframe)
        span.set_attribute("hmm.operation", "regime_classification")

        yield span


# =============================================================================
# Trade Execution Instrumentation
# =============================================================================

@contextmanager
def trace_trade_execution(symbol: str, action: str, mode: str):
    """
    Context manager for trade execution spans.

    Args:
        symbol: Trading symbol
        action: BUY or SELL
        mode: live, demo, or paper

    Yields:
        Span for the trade execution
    """
    tracer = get_tracer("quantmind.trading")

    with tracer.start_as_current_span(f"trade.execute.{symbol}") as span:
        span.set_attribute("trading.symbol", symbol)
        span.set_attribute("trading.action", action)
        span.set_attribute("trading.mode", mode)
        span.set_attribute("trading.operation", "execute")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def trace_position_close(trade_id: str, symbol: str, mode: str):
    """
    Context manager for position close spans.

    Args:
        trade_id: Trade/ticket ID
        symbol: Trading symbol
        mode: Trading mode

    Yields:
        Span for the close operation
    """
    tracer = get_tracer("quantmind.trading")

    with tracer.start_as_current_span(f"trade.close.{trade_id}") as span:
        span.set_attribute("trading.trade_id", trade_id)
        span.set_attribute("trading.symbol", symbol)
        span.set_attribute("trading.mode", mode)
        span.set_attribute("trading.operation", "close")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# =============================================================================
# MT5 Bridge Instrumentation
# =============================================================================

@contextmanager
def trace_mt5_operation(operation: str, symbol: Optional[str] = None):
    """
    Context manager for MT5 bridge operations.

    Args:
        operation: Operation type (connect, disconnect, trade, account_info)
        symbol: Optional trading symbol

    Yields:
        Span for the MT5 operation
    """
    tracer = get_tracer("quantmind.mt5")

    span_name = f"mt5.{operation}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("mt5.operation", operation)
        if symbol:
            span.set_attribute("mt5.symbol", symbol)
        span.set_attribute("mt5.component", "bridge")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# =============================================================================
# Agent/Department Instrumentation
# =============================================================================

@contextmanager
def trace_agent_task(agent_id: str, task_type: str):
    """
    Context manager for agent task spans.

    Args:
        agent_id: Agent identifier
        task_type: Type of task (research, development, trading, risk)

    Yields:
        Span for the agent task
    """
    tracer = get_tracer("quantmind.agents")

    with tracer.start_as_current_span(f"agent.task.{task_type}") as span:
        span.set_attribute("agent.id", agent_id)
        span.set_attribute("agent.task_type", task_type)
        span.set_attribute("agent.operation", "task_execution")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def trace_department_operation(department: str, operation: str):
    """
    Context manager for department-level operations.

    Args:
        department: Department name (research, development, trading, risk, portfolio)
        operation: Operation being performed

    Yields:
        Span for the department operation
    """
    tracer = get_tracer("quantmind.departments")

    with tracer.start_as_current_span(f"department.{department}.{operation}") as span:
        span.set_attribute("department.name", department)
        span.set_attribute("department.operation", operation)

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# =============================================================================
# Workflow/Flow Instrumentation
# =============================================================================

@contextmanager
def trace_workflow_stage(stage_name: str, workflow_id: str):
    """
    Context manager for Alpha Forge workflow stages.

    Args:
        stage_name: Name of the workflow stage
        workflow_id: Unique workflow identifier

    Yields:
        Span for the workflow stage
    """
    tracer = get_tracer("quantmind.workflow")

    with tracer.start_as_current_span(f"workflow.{stage_name}") as span:
        span.set_attribute("workflow.id", workflow_id)
        span.set_attribute("workflow.stage", stage_name)
        span.set_attribute("workflow.type", "alpha_forge")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# =============================================================================
# WebSocket/SSE Context Handling
# =============================================================================

def inject_trace_context_to_websocket(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject current trace context into WebSocket connection headers.

    Args:
        headers: WebSocket connection headers

    Returns:
        Updated headers with trace context
    """
    from opentelemetry.propagate import inject

    carrier = {}
    inject(carrier)
    headers.update(carrier)

    return headers


def extract_trace_context_from_websocket(headers: Dict[str, str]):
    """
    Extract trace context from WebSocket connection headers.

    Args:
        headers: WebSocket connection headers containing trace context

    Returns:
        Extracted context or None
    """
    from opentelemetry.propagate import extract
    from opentelemetry.trace import SpanKind

    context = extract(headers)
    return context


def create_websocket_span(
    websocket_id: str,
    operation: str,
    parent_context=None
):
    """
    Create a span for a WebSocket connection.

    Args:
        websocket_id: Unique WebSocket connection ID
        operation: Operation being performed
        parent_context: Optional parent context from headers

    Returns:
        Started span
    """
    tracer = get_tracer("quantmind.websocket")

    kind = SpanKind.SERVER
    if parent_context:
        ctx = trace.set_span_in_context(
            trace.get_current_span() if parent_context else None,
            parent_context
        )
        span = tracer.start_span(
            f"websocket.{operation}",
            kind=kind,
            context=ctx
        )
    else:
        span = tracer.start_span(f"websocket.{operation}", kind=kind)

    span.set_attribute("websocket.id", websocket_id)
    span.set_attribute("websocket.operation", operation)
    span.set_attribute("websocket.type", "realtime")

    return span


# =============================================================================
# Decorators for Function Instrumentation
# =============================================================================

def traced_function(func: Callable = None, *, operation_name: str = None):
    """
    Decorator to trace a function.

    Args:
        func: Function to trace
        operation_name: Optional custom operation name

    Usage:
        @traced_function
        def my_function(arg1, arg2):
            ...

        @traced_function(operation_name="custom.name")
        def my_function(arg1, arg2):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(fn.__module__)
            span_name = operation_name or f"{fn.__module__}.{fn.__name__}"

            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes (limited set)
                if args:
                    span.set_attribute("function.args_count", len(args))
                if kwargs:
                    span.set_attribute("function.kwargs_keys", list(kwargs.keys()))

                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


# =============================================================================
# SSE (Server-Sent Events) Instrumentation
# =============================================================================

@contextmanager
def trace_sse_stream(stream_id: str, event_type: str):
    """
    Context manager for SSE stream spans.

    Args:
        stream_id: Unique stream identifier
        event_type: Type of events being streamed

    Yields:
        Span for the SSE stream
    """
    tracer = get_tracer("quantmind.sse")

    with tracer.start_as_current_span(f"sse.stream.{stream_id}") as span:
        span.set_attribute("sse.stream_id", stream_id)
        span.set_attribute("sse.event_type", event_type)
        span.set_attribute("sse.type", "server_sent_events")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def add_sse_trace_headers(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add trace context to SSE event data.

    Args:
        event_data: SSE event data dictionary

    Returns:
        Updated event data with trace headers
    """
    trace_id = get_current_trace_id()

    if trace_id:
        event_data["trace_id"] = trace_id

    return event_data


# =============================================================================
# Database Instrumentation
# =============================================================================

@contextmanager
def trace_db_query(operation: str, table: str):
    """
    Context manager for database query spans.

    Args:
        operation: Query operation (select, insert, update, delete)
        table: Database table name

    Yields:
        Span for the database operation
    """
    tracer = get_tracer("quantmind.database")

    with tracer.start_as_current_span(f"db.{operation}.{table}") as span:
        span.set_attribute("db.operation", operation)
        span.set_attribute("db.table", table)
        span.set_attribute("db.system", "postgresql")

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
