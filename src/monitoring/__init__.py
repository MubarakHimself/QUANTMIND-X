"""
QuantMindX Monitoring Module

This module provides centralized observability infrastructure for the
QuantMindX trading system, integrating with Grafana Cloud for metrics,
log aggregation, and distributed tracing.

Components:
- prometheus_exporter: Prometheus metrics collection and exposition
- grafana_cloud_pusher: Push metrics to Grafana Cloud remote write endpoint
- tracing: OpenTelemetry distributed tracing with Grafana Tempo
- instrumentation: Custom spans for trading operations

Architecture:
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │  API Server     │     │  Strategy Router │     │   MT5 Bridge    │
    └────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
             │                       │                        │
             └───────────────────────┼────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     prometheus_exporter        │
                    │   (Metrics Collection)         │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │  HTTP /metrics       │      │  grafana_cloud_pusher│
        │  (Prometheus Scrape) │      │  (Remote Write)      │
        └──────────────────────┘      └──────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                    OpenTelemetry Tracing                        │
    │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐  │
    │  │  FastAPI     │───▶│  OTLP        │───▶│  Grafana Tempo    │  │
    │  │  AutoInstr   │    │  Collector   │    │  (Trace Storage)  │  │
    │  └─────────────┘    └──────────────┘    └──────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from src.monitoring import start_metrics_server, track_api_request

    # Start metrics server on startup
    start_metrics_server(port=9090)

    # Track API requests
    track_api_request('GET', '/api/status', 200, 0.025)

    # Track trading operations
    from src.monitoring import track_trade, update_chaos_score
    track_trade('EURUSD', 'BUY', 'live', pnl=150.0)
    update_chaos_score(0.35)

    # Initialize tracing (call on startup)
    from src.monitoring import init_tracing
    init_tracing(app)
"""

# Core functions for metrics server
from src.monitoring.prometheus_exporter import (
    start_metrics_server,
    get_metrics,
)

# Helper functions for tracking events
from src.monitoring.prometheus_exporter import (
    track_api_request,
    track_trade,
    track_mt5_operation,
    track_regime_change,
    update_chaos_score,
    update_kelly_fraction,
    update_mt5_status,
    update_active_eas,
    track_shutdown,
    track_db_query,
    track_websocket_message,
)

# Metric objects for direct access
from src.monitoring.prometheus_exporter import (
    api_requests_total,
    api_request_duration_seconds,
    api_errors_total,
    trades_executed_total,
    trade_profit_loss,
    active_eas,
    regime_changes_total,
    chaos_score,
    kelly_fraction,
    mt5_connection_status,
    mt5_latency_seconds,
    mt5_trades_total,
    db_query_duration_seconds,
    db_connection_pool_size,
    db_errors_total,
    system_shutdowns_total,
    websocket_connections,
    websocket_messages_total,
    system_info,
)

# Grafana Cloud pusher
from src.monitoring.grafana_cloud_pusher import (
    GrafanaCloudPusher,
    get_grafana_cloud_pusher,
    start_grafana_cloud_push,
    stop_grafana_cloud_push,
)

# JSON logging for Promtail/Loki
from src.monitoring.json_logging import (
    configure_api_logging,
    configure_router_logging,
    configure_mt5_logging,
    configure_all_logging,
    JsonFormatter,
    setup_json_file_handler,
)

# OpenTelemetry Tracing
from src.monitoring.tracing import (
    init_tracing,
    get_tracer,
    get_current_trace_id,
    get_current_span_id,
    create_trading_span,
    add_trace_context_to_log,
    is_tracing_enabled,
)

# Custom instrumentation
from src.monitoring.instrumentation import (
    trace_strategy_routing,
    trace_regime_detection,
    trace_trade_execution,
    trace_position_close,
    trace_mt5_operation,
    trace_agent_task,
    trace_department_operation,
    trace_workflow_stage,
    trace_sse_stream,
    trace_db_query,
    inject_trace_context_to_websocket,
    extract_trace_context_from_websocket,
    create_websocket_span,
    add_sse_trace_headers,
    traced_function,
)

__all__ = [
    # Server functions
    'start_metrics_server',
    'get_metrics',

    # Tracking functions
    'track_api_request',
    'track_trade',
    'track_mt5_operation',
    'track_regime_change',
    'update_chaos_score',
    'update_kelly_fraction',
    'update_mt5_status',
    'update_active_eas',
    'track_shutdown',
    'track_db_query',
    'track_websocket_message',

    # Metric objects - API
    'api_requests_total',
    'api_request_duration_seconds',
    'api_errors_total',

    # Metric objects - Trading
    'trades_executed_total',
    'trade_profit_loss',
    'active_eas',
    'regime_changes_total',
    'chaos_score',
    'kelly_fraction',

    # Metric objects - MT5
    'mt5_connection_status',
    'mt5_latency_seconds',
    'mt5_trades_total',

    # Metric objects - Database
    'db_query_duration_seconds',
    'db_connection_pool_size',
    'db_errors_total',

    # Metric objects - System
    'system_shutdowns_total',
    'websocket_connections',
    'websocket_messages_total',
    'system_info',

    # Grafana Cloud
    'GrafanaCloudPusher',
    'get_grafana_cloud_pusher',
    'start_grafana_cloud_push',
    'stop_grafana_cloud_push',

    # JSON Logging for Promtail/Loki
    'configure_api_logging',
    'configure_router_logging',
    'configure_mt5_logging',
    'configure_all_logging',
    'JsonFormatter',
    'setup_json_file_handler',

    # OpenTelemetry Tracing
    'init_tracing',
    'get_tracer',
    'get_current_trace_id',
    'get_current_span_id',
    'create_trading_span',
    'add_trace_context_to_log',
    'is_tracing_enabled',

    # Custom instrumentation
    'trace_strategy_routing',
    'trace_regime_detection',
    'trace_trade_execution',
    'trace_position_close',
    'trace_mt5_operation',
    'trace_agent_task',
    'trace_department_operation',
    'trace_workflow_stage',
    'trace_sse_stream',
    'trace_db_query',
    'inject_trace_context_to_websocket',
    'extract_trace_context_from_websocket',
    'create_websocket_span',
    'add_sse_trace_headers',
    'traced_function',
]