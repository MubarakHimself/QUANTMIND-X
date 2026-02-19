"""
Prometheus Metrics Exporter for QuantMindX

Provides centralized metrics collection for:
- API request tracking (latency, errors, throughput)
- Trading operations (trades executed, P&L, active EAs)
- MT5 Bridge connectivity and latency
- System health (chaos score, regime changes, shutdowns)
- Database performance

These metrics are exposed via HTTP endpoint for Prometheus scraping
and can be pushed to Grafana Cloud.
"""

import logging
import os
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, REGISTRY

logger = logging.getLogger(__name__)

# ========== API Metrics ==========

api_requests_total = Counter(
    'quantmind_api_requests_total',
    'Total count of API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration_seconds = Histogram(
    'quantmind_api_request_duration_seconds',
    'API request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

api_errors_total = Counter(
    'quantmind_api_errors_total',
    'Total count of API errors',
    ['method', 'endpoint', 'error_type']
)

# ========== Trading Metrics ==========

trades_executed_total = Counter(
    'quantmind_trades_executed_total',
    'Total number of trades executed',
    ['symbol', 'action', 'mode']  # action: BUY/SELL, mode: demo/live
)

trade_profit_loss = Histogram(
    'quantmind_trade_profit_loss',
    'Profit/loss distribution per trade',
    ['symbol', 'mode'],
    buckets=(-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000, 5000)
)

active_eas = Gauge(
    'quantmind_active_eas',
    'Number of active Expert Advisors',
    ['status']  # status: primal, pending, quarantine
)

regime_changes_total = Counter(
    'quantmind_regime_changes_total',
    'Total number of market regime changes',
    ['from_regime', 'to_regime']
)

chaos_score = Gauge(
    'quantmind_chaos_score',
    'Current market chaos score (0-1)'
)

kelly_fraction = Gauge(
    'quantmind_kelly_fraction',
    'Current Kelly fraction for position sizing'
)

# ========== MT5 Bridge Metrics ==========

mt5_connection_status = Gauge(
    'quantmind_mt5_connection_status',
    'MT5 connection status (1=connected, 0=disconnected)'
)

mt5_latency_seconds = Histogram(
    'quantmind_mt5_latency_seconds',
    'MT5 operation latency in seconds',
    ['operation'],  # operation: trade, account_info, status
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

mt5_trades_total = Counter(
    'quantmind_mt5_trades_total',
    'Total MT5 trades processed',
    ['symbol', 'action', 'result']  # result: success, failed
)

# ========== Database Metrics ==========

db_query_duration_seconds = Histogram(
    'quantmind_db_query_duration_seconds',
    'Database query latency in seconds',
    ['operation'],  # operation: select, insert, update, delete
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

db_connection_pool_size = Gauge(
    'quantmind_db_connection_pool_size',
    'Current database connection pool size',
    ['state']  # state: active, idle
)

db_errors_total = Counter(
    'quantmind_db_errors_total',
    'Total database errors',
    ['operation', 'error_type']
)

# ========== System Health Metrics ==========

system_shutdowns_total = Counter(
    'quantmind_system_shutdowns_total',
    'Total number of system shutdowns triggered',
    ['reason']
)

websocket_connections = Gauge(
    'quantmind_websocket_connections',
    'Current number of WebSocket connections'
)

websocket_messages_total = Counter(
    'quantmind_websocket_messages_total',
    'Total WebSocket messages sent',
    ['type']  # type: regime, status, bot, trade
)

# ========== Tick Streaming Metrics ==========

tick_stream_rate = Gauge(
    'quantmind_tick_stream_rate',
    'Tick stream rate in ticks per second',
    ['symbol', 'method']  # method: zmq, polling
)

tick_stream_latency_ms = Histogram(
    'quantmind_tick_stream_latency_ms',
    'Tick stream latency in milliseconds',
    ['symbol'],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500)
)

tick_stream_drops = Counter(
    'quantmind_tick_stream_drops_total',
    'Total number of dropped ticks',
    ['symbol', 'reason']  # reason: stale, out_of_order, validation_error
)

symbol_subscriptions = Gauge(
    'quantmind_symbol_subscriptions',
    'Number of active symbol subscriptions',
    ['symbol', 'priority']  # priority: LIVE, DEMO, PAPER
)

# ========== Resource Metrics ==========

mt5_cpu_usage = Gauge(
    'quantmind_mt5_cpu_percent',
    'CPU usage percentage'
)

mt5_ram_usage = Gauge(
    'quantmind_mt5_ram_mb',
    'RAM usage in megabytes'
)

mt5_symbol_count = Gauge(
    'quantmind_mt5_active_symbols',
    'Number of active symbols being tracked'
)

eod_import_duration = Histogram(
    'quantmind_eod_import_duration_seconds',
    'EOD import duration in seconds',
    ['symbol', 'timeframe'],
    buckets=(10, 30, 60, 120, 300, 600)
)

eod_import_bars = Counter(
    'quantmind_eod_import_bars_total',
    'Total bars imported via EOD import',
    ['symbol', 'timeframe']
)

degradation_level = Gauge(
    'quantmind_degradation_level',
    'Current degradation level (0=normal, 1=warning, 2=critical, 3=emergency)'
)

# ========== Database Tier Metrics ==========

db_tier_query_latency = Histogram(
    'quantmind_db_tier_query_latency_seconds',
    'Database tier query latency in seconds',
    ['tier', 'operation'],  # tier: hot, warm, cold; operation: select, insert
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

db_tier_size_mb = Gauge(
    'quantmind_db_tier_size_mb',
    'Database tier size in megabytes',
    ['tier']  # tier: hot, warm, cold
)

db_migration_status = Gauge(
    'quantmind_db_migration_status',
    'Database migration job status (1=success, 0=failed)',
    ['migration_type']  # migration_type: hot_to_warm, warm_to_cold
)

# ========== Circuit Breaker Metrics ==========

circuit_breaker_state = Gauge(
    'quantmind_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half_open)'
)

circuit_breaker_errors = Counter(
    'quantmind_circuit_breaker_errors_total',
    'Total circuit breaker errors',
    ['reason']  # reason: consecutive_errors, high_latency, memory_leak
)

# ========== Cross-VPS & Storage Metrics ==========

hmm_sync_last_success = Gauge(
    'quantmind_hmm_sync_last_success_timestamp',
    'Timestamp of last successful HMM model sync from Contabo to Cloudzy'
)

hmm_sync_failures_total = Counter(
    'quantmind_hmm_sync_failures_total',
    'Total number of HMM model sync failures',
    ['reason']  # reason: network, auth, timeout, validation
)

data_migration_rows_total = Counter(
    'quantmind_data_migration_rows_total',
    'Total number of rows migrated between storage tiers',
    ['from_tier', 'to_tier']  # from_tier: hot/warm, to_tier: warm/cold
)

data_migration_duration_seconds = Histogram(
    'quantmind_data_migration_duration_seconds',
    'Duration of data migration jobs in seconds',
    ['from_tier', 'to_tier'],
    buckets=(1, 5, 15, 30, 60, 120, 300)
)

cold_storage_size_bytes = Gauge(
    'quantmind_cold_storage_size_bytes',
    'Current size of cold storage (Parquet files) in bytes'
)

warm_storage_size_bytes = Gauge(
    'quantmind_warm_storage_size_bytes',
    'Current size of warm storage (DuckDB) in bytes'
)

hot_storage_rows = Gauge(
    'quantmind_hot_storage_rows',
    'Current number of rows in hot storage (PostgreSQL)'
)

contabo_reachable = Gauge(
    'quantmind_contabo_reachable',
    'Whether Contabo VPS is reachable (1=reachable, 0=unreachable)'
)

# ========== Paper Trading Metrics ==========

paper_trades_total = Counter(
    'quantmind_paper_trades_total',
    'Total number of paper trades executed',
    ['bot_type', 'symbol']  # bot_type: EA, Pine, Python
)

paper_pnl_total = Gauge(
    'quantmind_paper_pnl_total',
    'Total paper trading profit/loss by bot tag',
    ['bot_tag']  # bot_tag: primal, pending, perfect
)

paper_bots_active = Gauge(
    'quantmind_paper_bots_active',
    'Number of active paper trading bots by tag',
    ['tag']  # tag: primal, pending, perfect
)

paper_win_rate = Gauge(
    'quantmind_paper_win_rate',
    'Win rate for paper trading bots',
    ['bot_id']
)

# ========== VPS Health & Connectivity Metrics ==========

vps_heartbeat_timestamp = Gauge(
    'quantmind_vps_heartbeat_timestamp',
    'Timestamp of last heartbeat received from a VPS',
    ['vps_name', 'vps_role']  # vps_role: trading, training, storage
)

vps_network_latency_seconds = Histogram(
    'quantmind_vps_network_latency_seconds',
    'Network latency between VPS instances in seconds',
    ['source_vps', 'target_vps'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

vps_ssh_connection_status = Gauge(
    'quantmind_vps_ssh_connection_status',
    'SSH connection status to a VPS (1=connected, 0=disconnected)',
    ['target_vps']
)

vps_resource_cpu_percent = Gauge(
    'quantmind_vps_resource_cpu_percent',
    'CPU usage percentage on remote VPS',
    ['vps_name']
)

vps_resource_memory_percent = Gauge(
    'quantmind_vps_resource_memory_percent',
    'Memory usage percentage on remote VPS',
    ['vps_name']
)

vps_resource_disk_percent = Gauge(
    'quantmind_vps_resource_disk_percent',
    'Disk usage percentage on remote VPS',
    ['vps_name', 'mount_point']
)

# ========== HMM Training Metrics ==========

hmm_training_duration_seconds = Histogram(
    'quantmind_hmm_training_duration_seconds',
    'Duration of HMM training jobs in seconds',
    ['model_type', 'symbol'],  # model_type: universal, per_symbol, per_symbol_timeframe
    buckets=(60, 120, 300, 600, 900, 1800, 3600, 7200)
)

hmm_training_samples_total = Counter(
    'quantmind_hmm_training_samples_total',
    'Total number of samples used for HMM training',
    ['model_type', 'symbol']
)

hmm_training_jobs_total = Counter(
    'quantmind_hmm_training_jobs_total',
    'Total number of HMM training jobs',
    ['model_type', 'status']  # status: success, failed
)

hmm_model_validation_score = Gauge(
    'quantmind_hmm_model_validation_score',
    'Validation score for trained HMM models',
    ['model_type', 'symbol', 'version']
)

hmm_model_log_likelihood = Gauge(
    'quantmind_hmm_model_log_likelihood',
    'Log-likelihood score for HMM models',
    ['model_type', 'symbol', 'version']
)

hmm_inference_latency_seconds = Histogram(
    'quantmind_hmm_inference_latency_seconds',
    'HMM inference latency in seconds',
    ['model_type', 'symbol'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
)

hmm_regime_prediction_total = Counter(
    'quantmind_hmm_regime_prediction_total',
    'Total HMM regime predictions',
    ['symbol', 'regime']  # regime: trending, ranging, volatile
)

# ========== Sync Job Metrics ==========

sync_job_duration_seconds = Histogram(
    'quantmind_sync_job_duration_seconds',
    'Duration of sync jobs in seconds',
    ['sync_type', 'source', 'target'],  # sync_type: hmm_models, market_data, config
    buckets=(1, 5, 15, 30, 60, 120, 300, 600)
)

sync_job_bytes_transferred = Counter(
    'quantmind_sync_job_bytes_transferred_total',
    'Total bytes transferred during sync jobs',
    ['sync_type', 'source', 'target', 'direction']  # direction: upload, download
)

sync_job_files_total = Counter(
    'quantmind_sync_job_files_total',
    'Total files transferred during sync jobs',
    ['sync_type', 'source', 'target', 'status']  # status: success, failed, skipped
)

sync_queue_depth = Gauge(
    'quantmind_sync_queue_depth',
    'Current number of items waiting in sync queue',
    ['sync_type', 'priority']  # priority: high, normal, low
)

sync_last_success_timestamp = Gauge(
    'quantmind_sync_last_success_timestamp',
    'Timestamp of last successful sync by type',
    ['sync_type']
)

sync_errors_total = Counter(
    'quantmind_sync_errors_total',
    'Total number of sync errors',
    ['sync_type', 'error_type']  # error_type: network, auth, timeout, validation
)

# ========== Multi-VPS Coordination Metrics ==========

vps_coordinator_status = Gauge(
    'quantmind_vps_coordinator_status',
    'VPS coordinator status (1=active, 0=inactive)',
    ['vps_name']
)

vps_failover_events_total = Counter(
    'quantmind_vps_failover_events_total',
    'Total number of VPS failover events',
    ['from_vps', 'to_vps', 'reason']
)

vps_replication_lag_seconds = Gauge(
    'quantmind_vps_replication_lag_seconds',
    'Replication lag between VPS instances in seconds',
    ['source_vps', 'target_vps', 'data_type']
)

config_sync_checksum = Gauge(
    'quantmind_config_sync_checksum',
    'Checksum of synced configuration (numeric hash for monitoring)',
    ['config_file', 'vps_name']
)

# ========== System Info ==========

system_info = Info(
    'quantmind_system',
    'QuantMindX system information'
)

# Set system info from environment
system_info.info({
    'version': os.getenv('QUANTMIND_VERSION', '1.0.0'),
    'environment': os.getenv('QUANTMIND_ENV', 'development'),
    'component': 'quantmind-api'
})


def start_metrics_server(port: int = 9090) -> None:
    """
    Start the Prometheus metrics HTTP server.
    
    This exposes metrics at http://localhost:{port}/metrics for
    Prometheus scraping or manual inspection.
    
    Args:
        port: Port to expose metrics on (default: 9090)
    """
    try:
        start_http_server(port, registry=REGISTRY)
        logger.info(f"Prometheus metrics server started on port {port}")
        logger.info(f"Metrics available at http://localhost:{port}/metrics")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Metrics server port {port} already in use, skipping startup")
        else:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise


def get_metrics() -> str:
    """
    Get current metrics in Prometheus text format.
    
    Returns:
        String containing all metrics in Prometheus exposition format
    """
    from prometheus_client import generate_latest
    return generate_latest(REGISTRY).decode('utf-8')


# ========== Helper Functions ==========

def track_api_request(method: str, endpoint: str, status: int, duration: float):
    """
    Track an API request with automatic metric updates.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status: HTTP status code
        duration: Request duration in seconds
    """
    api_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    # Track errors separately
    if status >= 400:
        error_type = 'client_error' if status < 500 else 'server_error'
        api_errors_total.labels(method=method, endpoint=endpoint, error_type=error_type).inc()


def track_trade(symbol: str, action: str, mode: str, pnl: float = None):
    """
    Track a trade execution.
    
    Args:
        symbol: Trading symbol (e.g., EURUSD)
        action: Trade action (BUY or SELL)
        mode: Trading mode (demo or live)
        pnl: Profit/loss amount (optional)
    """
    trades_executed_total.labels(symbol=symbol, action=action, mode=mode).inc()
    
    if pnl is not None:
        trade_profit_loss.labels(symbol=symbol, mode=mode).observe(pnl)


def track_mt5_operation(operation: str, duration: float, success: bool):
    """
    Track an MT5 operation.
    
    Args:
        operation: Operation type (trade, account_info, status)
        duration: Operation duration in seconds
        success: Whether operation succeeded
    """
    mt5_latency_seconds.labels(operation=operation).observe(duration)


def track_regime_change(from_regime: str, to_regime: str):
    """
    Track a market regime change.
    
    Args:
        from_regime: Previous regime
        to_regime: New regime
    """
    regime_changes_total.labels(from_regime=from_regime, to_regime=to_regime).inc()


def update_chaos_score(score: float):
    """
    Update the current chaos score gauge.
    
    Args:
        score: Chaos score (0-1)
    """
    chaos_score.set(score)


def update_kelly_fraction(fraction: float):
    """
    Update the current Kelly fraction gauge.
    
    Args:
        fraction: Kelly fraction (0-1)
    """
    kelly_fraction.set(fraction)


def update_mt5_status(connected: bool):
    """
    Update MT5 connection status gauge.
    
    Args:
        connected: True if connected, False otherwise
    """
    mt5_connection_status.set(1 if connected else 0)


def update_active_eas(primal: int = 0, pending: int = 0, quarantine: int = 0):
    """
    Update active EA gauges.
    
    Args:
        primal: Number of primal (active) EAs
        pending: Number of pending EAs
        quarantine: Number of quarantined EAs
    """
    active_eas.labels(status='primal').set(primal)
    active_eas.labels(status='pending').set(pending)
    active_eas.labels(status='quarantine').set(quarantine)


def track_shutdown(reason: str):
    """
    Track a system shutdown event.
    
    Args:
        reason: Reason for shutdown
    """
    system_shutdowns_total.labels(reason=reason).inc()


def track_db_query(operation: str, duration: float):
    """
    Track a database query.
    
    Args:
        operation: Query operation (select, insert, update, delete)
        duration: Query duration in seconds
    """
    db_query_duration_seconds.labels(operation=operation).observe(duration)


def track_websocket_message(message_type: str):
    """
    Track a WebSocket message.

    Args:
        message_type: Message type (regime, status, bot, trade)
    """
    websocket_messages_total.labels(type=message_type).inc()


def update_circuit_breaker_state(state: int):
    """
    Update circuit breaker state gauge.

    Args:
        state: Circuit breaker state (0=closed, 1=open, 2=half_open)
    """
    circuit_breaker_state.set(state)


def track_circuit_breaker_error(reason: str):
    """
    Track a circuit breaker error.

    Args:
        reason: Reason for the error (consecutive_errors, high_latency, memory_leak)
    """
    circuit_breaker_errors.labels(reason=reason).inc()


def track_tick_drop(symbol: str, reason: str):
    """
    Track a dropped tick.

    Args:
        symbol: Trading symbol
        reason: Reason for drop (stale, out_of_order, validation_error, fetch_error)
    """
    tick_stream_drops.labels(symbol=symbol, reason=reason).inc()


def update_tick_rate(symbol: str, method: str, rate: float):
    """
    Update tick stream rate gauge.

    Args:
        symbol: Trading symbol
        method: Streaming method (zmq, polling)
        rate: Ticks per second
    """
    tick_stream_rate.labels(symbol=symbol, method=method).set(rate)


def observe_tick_latency(symbol: str, latency_ms: float):
    """
    Observe tick stream latency.

    Args:
        symbol: Trading symbol
        latency_ms: Latency in milliseconds
    """
    tick_stream_latency_ms.labels(symbol=symbol).observe(latency_ms)


def update_symbol_subscription(symbol: str, priority: str, count: int):
    """
    Update symbol subscription gauge.

    Args:
        symbol: Trading symbol
        priority: Priority level (LIVE, DEMO, PAPER)
        count: Number of subscriptions
    """
    symbol_subscriptions.labels(symbol=symbol, priority=priority).set(count)


# ========== Cross-VPS & Storage Helper Functions ==========

def update_hmm_sync_success(timestamp: float):
    """
    Update the HMM sync last success timestamp.
    
    Args:
        timestamp: Unix timestamp of the last successful sync
    """
    hmm_sync_last_success.set(timestamp)


def track_hmm_sync_failure(reason: str):
    """
    Track an HMM model sync failure.
    
    Args:
        reason: Reason for failure (network, auth, timeout, validation)
    """
    hmm_sync_failures_total.labels(reason=reason).inc()


def track_data_migration(from_tier: str, to_tier: str, rows: int, duration: float):
    """
    Track a data migration between storage tiers.
    
    Args:
        from_tier: Source tier (hot, warm)
        to_tier: Destination tier (warm, cold)
        rows: Number of rows migrated
        duration: Migration duration in seconds
    """
    data_migration_rows_total.labels(from_tier=from_tier, to_tier=to_tier).inc(rows)
    data_migration_duration_seconds.labels(from_tier=from_tier, to_tier=to_tier).observe(duration)


def update_storage_sizes(cold_bytes: float, warm_bytes: float, hot_rows: int):
    """
    Update storage size gauges.
    
    Args:
        cold_bytes: Cold storage size in bytes
        warm_bytes: Warm storage size in bytes
        hot_rows: Number of rows in hot storage
    """
    cold_storage_size_bytes.set(cold_bytes)
    warm_storage_size_bytes.set(warm_bytes)
    hot_storage_rows.set(hot_rows)


def update_contabo_reachable(reachable: bool):
    """
    Update Contabo VPS reachability status.
    
    Args:
        reachable: True if Contabo is reachable, False otherwise
    """
    contabo_reachable.set(1 if reachable else 0)


# ========== Paper Trading Helper Functions ==========

def track_paper_trade(bot_type: str, symbol: str):
    """
    Track a paper trade execution.
    
    Args:
        bot_type: Type of bot (EA, Pine, Python)
        symbol: Trading symbol
    """
    paper_trades_total.labels(bot_type=bot_type, symbol=symbol).inc()


def update_paper_pnl(bot_tag: str, pnl: float):
    """
    Update paper trading P&L gauge.
    
    Args:
        bot_tag: Bot tag (primal, pending, perfect)
        pnl: Current P&L value
    """
    paper_pnl_total.labels(bot_tag=bot_tag).set(pnl)


def update_paper_bots_active(tag: str, count: int):
    """
    Update active paper trading bots count.
    
    Args:
        tag: Bot tag (primal, pending, perfect)
        count: Number of active bots
    """
    paper_bots_active.labels(tag=tag).set(count)


def update_paper_win_rate(bot_id: str, rate: float):
    """
    Update paper trading win rate for a bot.
    
    Args:
        bot_id: Bot identifier
        rate: Win rate (0-1)
    """
    paper_win_rate.labels(bot_id=bot_id).set(rate)


# ========== VPS Health & Connectivity Helper Functions ==========

def update_vps_heartbeat(vps_name: str, vps_role: str, timestamp: float):
    """
    Update VPS heartbeat timestamp.
    
    Args:
        vps_name: Name of the VPS (e.g., cloudzy, contabo)
        vps_role: Role of the VPS (trading, training, storage)
        timestamp: Unix timestamp of the heartbeat
    """
    vps_heartbeat_timestamp.labels(vps_name=vps_name, vps_role=vps_role).set(timestamp)


def observe_vps_network_latency(source_vps: str, target_vps: str, latency_seconds: float):
    """
    Observe network latency between VPS instances.
    
    Args:
        source_vps: Source VPS name
        target_vps: Target VPS name
        latency_seconds: Latency in seconds
    """
    vps_network_latency_seconds.labels(source_vps=source_vps, target_vps=target_vps).observe(latency_seconds)


def update_vps_ssh_connection(target_vps: str, connected: bool):
    """
    Update SSH connection status to a VPS.
    
    Args:
        target_vps: Target VPS name
        connected: True if connected, False otherwise
    """
    vps_ssh_connection_status.labels(target_vps=target_vps).set(1 if connected else 0)


def update_vps_resources(vps_name: str, cpu_percent: float, memory_percent: float, 
                         disk_percent: float = None, mount_point: str = '/'):
    """
    Update VPS resource usage metrics.
    
    Args:
        vps_name: Name of the VPS
        cpu_percent: CPU usage percentage
        memory_percent: Memory usage percentage
        disk_percent: Disk usage percentage (optional)
        mount_point: Mount point for disk metric
    """
    vps_resource_cpu_percent.labels(vps_name=vps_name).set(cpu_percent)
    vps_resource_memory_percent.labels(vps_name=vps_name).set(memory_percent)
    if disk_percent is not None:
        vps_resource_disk_percent.labels(vps_name=vps_name, mount_point=mount_point).set(disk_percent)


# ========== HMM Training Helper Functions ==========

def observe_hmm_training_duration(model_type: str, symbol: str, duration_seconds: float):
    """
    Observe HMM training duration.
    
    Args:
        model_type: Type of model (universal, per_symbol, per_symbol_timeframe)
        symbol: Symbol for per-symbol models, 'all' for universal
        duration_seconds: Training duration in seconds
    """
    hmm_training_duration_seconds.labels(model_type=model_type, symbol=symbol).observe(duration_seconds)


def track_hmm_training_samples(model_type: str, symbol: str, samples: int):
    """
    Track number of samples used for HMM training.
    
    Args:
        model_type: Type of model
        symbol: Symbol for per-symbol models
        samples: Number of training samples
    """
    hmm_training_samples_total.labels(model_type=model_type, symbol=symbol).inc(samples)


def track_hmm_training_job(model_type: str, status: str):
    """
    Track HMM training job completion.
    
    Args:
        model_type: Type of model
        status: Job status (success, failed)
    """
    hmm_training_jobs_total.labels(model_type=model_type, status=status).inc()


def update_hmm_model_metrics(model_type: str, symbol: str, version: str, 
                              validation_score: float = None, log_likelihood: float = None):
    """
    Update HMM model quality metrics.
    
    Args:
        model_type: Type of model
        symbol: Symbol for per-symbol models
        version: Model version string
        validation_score: Model validation score (optional)
        log_likelihood: Model log-likelihood score (optional)
    """
    if validation_score is not None:
        hmm_model_validation_score.labels(model_type=model_type, symbol=symbol, version=version).set(validation_score)
    if log_likelihood is not None:
        hmm_model_log_likelihood.labels(model_type=model_type, symbol=symbol, version=version).set(log_likelihood)


def observe_hmm_inference_latency(model_type: str, symbol: str, latency_seconds: float):
    """
    Observe HMM inference latency.
    
    Args:
        model_type: Type of model
        symbol: Symbol being analyzed
        latency_seconds: Inference latency in seconds
    """
    hmm_inference_latency_seconds.labels(model_type=model_type, symbol=symbol).observe(latency_seconds)


def track_hmm_regime_prediction(symbol: str, regime: str):
    """
    Track HMM regime prediction.
    
    Args:
        symbol: Trading symbol
        regime: Predicted regime (trending, ranging, volatile)
    """
    hmm_regime_prediction_total.labels(symbol=symbol, regime=regime).inc()


# ========== Sync Job Helper Functions ==========

def observe_sync_job_duration(sync_type: str, source: str, target: str, duration_seconds: float):
    """
    Observe sync job duration.
    
    Args:
        sync_type: Type of sync (hmm_models, market_data, config)
        source: Source VPS
        target: Target VPS
        duration_seconds: Sync duration in seconds
    """
    sync_job_duration_seconds.labels(sync_type=sync_type, source=source, target=target).observe(duration_seconds)


def track_sync_bytes_transferred(sync_type: str, source: str, target: str, 
                                  direction: str, bytes_count: int):
    """
    Track bytes transferred during sync.
    
    Args:
        sync_type: Type of sync
        source: Source VPS
        target: Target VPS
        direction: Transfer direction (upload, download)
        bytes_count: Number of bytes transferred
    """
    sync_job_bytes_transferred.labels(
        sync_type=sync_type, source=source, target=target, direction=direction
    ).inc(bytes_count)


def track_sync_file(sync_type: str, source: str, target: str, status: str):
    """
    Track a file transfer during sync.
    
    Args:
        sync_type: Type of sync
        source: Source VPS
        target: Target VPS
        status: Transfer status (success, failed, skipped)
    """
    sync_job_files_total.labels(sync_type=sync_type, source=source, target=target, status=status).inc()


def update_sync_queue_depth(sync_type: str, priority: str, depth: int):
    """
    Update sync queue depth.
    
    Args:
        sync_type: Type of sync
        priority: Queue priority (high, normal, low)
        depth: Number of items in queue
    """
    sync_queue_depth.labels(sync_type=sync_type, priority=priority).set(depth)


def update_sync_last_success(sync_type: str, timestamp: float):
    """
    Update last successful sync timestamp.
    
    Args:
        sync_type: Type of sync
        timestamp: Unix timestamp of last successful sync
    """
    sync_last_success_timestamp.labels(sync_type=sync_type).set(timestamp)


def track_sync_error(sync_type: str, error_type: str):
    """
    Track a sync error.
    
    Args:
        sync_type: Type of sync
        error_type: Error type (network, auth, timeout, validation)
    """
    sync_errors_total.labels(sync_type=sync_type, error_type=error_type).inc()


# ========== Multi-VPS Coordination Helper Functions ==========

def update_vps_coordinator_status(vps_name: str, active: bool):
    """
    Update VPS coordinator status.
    
    Args:
        vps_name: Name of the VPS
        active: True if coordinator is active
    """
    vps_coordinator_status.labels(vps_name=vps_name).set(1 if active else 0)


def track_vps_failover(from_vps: str, to_vps: str, reason: str):
    """
    Track a VPS failover event.
    
    Args:
        from_vps: Source VPS that failed
        to_vps: Target VPS that took over
        reason: Reason for failover
    """
    vps_failover_events_total.labels(from_vps=from_vps, to_vps=to_vps, reason=reason).inc()


def update_vps_replication_lag(source_vps: str, target_vps: str, 
                                data_type: str, lag_seconds: float):
    """
    Update VPS replication lag.
    
    Args:
        source_vps: Source VPS name
        target_vps: Target VPS name
        data_type: Type of data being replicated
        lag_seconds: Replication lag in seconds
    """
    vps_replication_lag_seconds.labels(
        source_vps=source_vps, target_vps=target_vps, data_type=data_type
    ).set(lag_seconds)


def update_config_sync_checksum(config_file: str, vps_name: str, checksum: int):
    """
    Update config sync checksum.
    
    Args:
        config_file: Configuration file name
        vps_name: VPS name
        checksum: Numeric checksum value (for monitoring changes)
    """
    config_sync_checksum.labels(config_file=config_file, vps_name=vps_name).set(checksum)
