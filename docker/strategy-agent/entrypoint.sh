#!/bin/bash
# Entrypoint script for QuantMindX Strategy Agent
# Initializes MT5 connection and starts agent services

set -euo pipefail

# Configuration
LOG_DIR="/app/logs"
DATA_DIR="/app/data"
HEARTBEAT_INTERVAL=60  # seconds

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Function to check if running as non-root user
check_security() {
    log_info "Checking security configuration..."

    if [ "$(id -u)" -eq 0 ]; then
        log_error "Security: Container running as root! This is a security risk."
        exit 1
    fi

    log_info "Security: Running as non-root user: $(whoami) (UID: $(id -u))"
}

# Function to initialize MT5 connection
init_mt5() {
    log_info "Initializing MetaTrader 5 connection..."

    # Check if MT5 terminal path is configured
    if [ -z "${MT5_TERMINAL_PATH:-}" ]; then
        log_warn "MT5_TERMINAL_PATH not set. Using default path."
        export MT5_TERMINAL_PATH="/mt5/terminal64.exe"
    fi

    # Test MT5 connection (if available)
    python3 -c "
import sys
import os
sys.path.insert(0, '/app/src')

try:
    import MetaTrader5 as mt5
    log_info('MT5 module imported successfully')

    # Initialize MT5 (will fail if terminal not available in container)
    if not mt5.initialize():
        log_warn(f'MT5 initialization failed: {mt5.last_error()}')
        log_warn('This is expected in containerized environments without MT5 terminal')
    else:
        log_info('MT5 initialized successfully')
        mt5.shutdown()
except ImportError:
    log_error('MetaTrader5 module not found')
    sys.exit(1)
except Exception as e:
    log_warn(f'MT5 check warning: {e}')
" || {
    log_error "MT5 initialization failed"
    exit 1
}

    log_info "MT5 connection setup complete"
}

# Function to start heartbeat publisher
start_heartbeat_publisher() {
    log_info "Starting heartbeat publisher (${HEARTBEAT_INTERVAL}s interval)..."

    python3 -c "
import time
import os
import sys
sys.path.insert(0, '/app/src')

try:
    from src.agents.integrations.redis_client import create_redis_client

    # Get Redis configuration from environment
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    agent_id = os.getenv('AGENT_ID', 'strategy-agent-1')

    # Create Redis client using wrapper
    client = create_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        agent_id=agent_id,
    )

    log_info(f'Publishing heartbeat to {client.heartbeat_channel}')

    while True:
        try:
            # Publish heartbeat using the wrapper
            client.publish_heartbeat(
                status='running',
                mt5_connected=False,  # Will be updated by agent
            )
            time.sleep(${HEARTBEAT_INTERVAL})
        except Exception as e:
            log_error(f'Heartbeat error: {e}')
            time.sleep(5)
except ImportError as e:
    log_error(f'Redis client module not found: {e}')
    sys.exit(1)
" &
    HEARTBEAT_PID=$!
    log_info "Heartbeat publisher started (PID: ${HEARTBEAT_PID})"
}

# Function to start trade event publisher
start_trade_event_publisher() {
    log_info "Starting trade event publisher..."

    python3 -c "
import time
import json
import os
import sys
sys.path.insert(0, '/app/src')

try:
    import redis
    from src.agents.integrations.redis_client import create_redis_client

    # Get Redis configuration from environment
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    agent_id = os.getenv('AGENT_ID', 'strategy-agent-1')

    # Create Redis client using wrapper
    client = create_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        agent_id=agent_id,
    )

    # Also create raw Redis client for pubsub subscription
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    trade_events_channel = client.trades_channel

    log_info(f'Subscribing to trade events on {trade_events_channel}:input')

    # Subscribe to trade events
    pubsub = r.pubsub()
    pubsub.subscribe(f'{trade_events_channel}:input')

    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                event_data = json.loads(message['data'])
                log_info(f'Processing trade event: {event_data.get(\"event_type\")}')

                # Publish using the wrapper
                client.publish_trade_event(
                    action=event_data.get('action', 'entry'),
                    symbol=event_data.get('symbol', 'UNKNOWN'),
                    price=float(event_data.get('price', 0)),
                    lots=float(event_data.get('lots', 0)),
                    pnl=event_data.get('pnl'),
                    order_id=event_data.get('order_id'),
                )
            except Exception as e:
                log_error(f'Error processing trade event: {e}')
except ImportError as e:
    log_error(f'Redis client module not found: {e}')
    sys.exit(1)
" &
    TRADE_PUBLISHER_PID=$!
    log_info "Trade event publisher started (PID: ${TRADE_PUBLISHER_PID})"
}

# Function to start the main agent
start_agent() {
    log_info "Starting strategy agent..."

    local agent_type="${1:-agent}"

    case "$agent_type" in
        "agent")
            log_info "Starting main agent process..."
            exec python3 -m src.agent.main
            ;;
        "test")
            log_info "Running agent tests..."
            exec python3 -m pytest tests/ -v
            ;;
        "shell")
            log_info "Starting interactive shell..."
            exec /bin/bash
            ;;
        *)
            log_error "Unknown command: $agent_type"
            log_info "Usage: entrypoint.sh [agent|test|shell]"
            exit 1
            ;;
    esac
}

# Function to handle graceful shutdown
cleanup() {
    log_info "Shutting down gracefully..."

    # Stop background processes
    if [ -n "${HEARTBEAT_PID:-}" ]; then
        log_info "Stopping heartbeat publisher (PID: ${HEARTBEAT_PID})"
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi

    if [ -n "${TRADE_PUBLISHER_PID:-}" ]; then
        log_info "Stopping trade event publisher (PID: ${TRADE_PUBLISHER_PID})"
        kill "$TRADE_PUBLISHER_PID" 2>/dev/null || true
    fi

    log_info "Shutdown complete"
    exit 0
}

# Main execution
main() {
    log_info "======================================"
    log_info "QuantMindX Strategy Agent"
    log_info "======================================"

    # Trap signals for graceful shutdown
    trap cleanup SIGTERM SIGINT

    # Security checks
    check_security

    # Initialize MT5 connection
    init_mt5

    # Start background services
    start_heartbeat_publisher
    start_trade_event_publisher

    # Start the main agent
    start_agent "$@"
}

# Run main function
main "$@"
