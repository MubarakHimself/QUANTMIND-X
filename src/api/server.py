"""
Main FastAPI Server

Runs the QuantMind IDE backend API on port 8000.
Combines IDE endpoints, Chat endpoints, and Analytics endpoints.
"""

import sys
import os
import uvicorn
import logging
import time
import asyncio
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()  # Load .env before any other imports that read env vars

# Configure logging with JSON file handler for Promtail/Loki
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantmind.server")

# Set up JSON file logging for Promtail to scrape
try:
    from src.monitoring.json_logging import configure_api_logging, configure_router_logging
    configure_api_logging()
    configure_router_logging()
    logger.info("JSON file logging configured for Promtail/Loki")
except Exception as e:
    logger.warning(f"Could not configure JSON file logging: {e}")

# Fix path to ensure imports work from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add mcp-metatrader5-server/src to path for mcp_mt5 imports
mt5_src_path = os.path.join(project_root, "mcp-metatrader5-server", "src")
if mt5_src_path not in sys.path:
    sys.path.insert(0, mt5_src_path)

try:
    from src.api.ide_endpoints import create_ide_api_app
    from src.api.analytics_endpoints import router as analytics_router
    from src.api.chat_endpoints import router as chat_router
    from src.api.settings_endpoints import router as settings_router
    from src.api.trd_endpoints import router as trd_router
    from src.api.router_endpoints import router as router_router
    from src.api.journal_endpoints import router as journal_router
    from src.api.session_endpoints import router as session_router
    from src.api.mcp_endpoints import router as mcp_router
    from src.api.agent_queue_endpoints import router as agent_queue_router
    from src.api.workflow_endpoints import router as workflow_router
    from src.api.kill_switch_endpoints import router as kill_switch_router
    from src.api.hmm_endpoints import router as hmm_router
    from src.api.tradingview_endpoints import router as tradingview_router
    from src.api.github_endpoints import router as github_router
    from src.api.monte_carlo_ws import monte_carlo_ws_endpoint
    from src.api.metrics_endpoints import router as metrics_router
    from src.api.health_endpoints import router as health_router
    from src.api.broker_endpoints import router as broker_router, broker_websocket
    from src.api.lifecycle_scanner_endpoints import router as lifecycle_scanner_router
    from src.api.agent_management_endpoints import router as agent_management_router
    from src.api.version_endpoints import router as version_router
    from src.api.demo_mode_endpoints import router as demo_mode_router
    from src.api.claude_agent_endpoints import router as claude_agent_router
    from src.api.agent_tools import router as agent_tools_router
    from src.api.memory_endpoints import router as memory_router
    from src.api.trading_floor_endpoints import router as trading_floor_router
except ImportError as e:
    logger.error(f"Import Error: {e}")
    # Fallback/Debug info
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Optional MT5-dependent imports (Linux compatibility)
paper_trading_router = None
try:
    from src.api.paper_trading_endpoints import router as paper_trading_router
    logger.info("Paper trading endpoints loaded")
except ImportError as e:
    logger.warning(f"Paper trading endpoints not available (MT5 required): {e}")

# Create main app
app = create_ide_api_app()
if app is None:
    logger.error("Failed to create FastAPI app. Is 'fastapi' installed?")
    sys.exit(1)

# WebSocket endpoints are already registered in create_ide_api_app()


# ========== Prometheus Metrics Middleware ==========
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """
    Middleware to track API request metrics for Prometheus.
    
    Records:
    - Request count by method, endpoint, and status
    - Request latency by method and endpoint
    """
    # Skip metrics endpoint itself to avoid recursion
    if request.url.path == "/metrics":
        return await call_next(request)
    
    start_time = time.time()
    
    # Process the request
    response: Response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Get endpoint path (normalize to avoid high cardinality from path params)
    endpoint = request.url.path
    # Normalize common patterns
    if endpoint.startswith("/api/"):
        # Keep the general endpoint structure
        parts = endpoint.split("/")
        # Replace UUIDs or numeric IDs with placeholder
        normalized_parts = []
        for part in parts:
            if part and (part.isdigit() or len(part) == 36 and "-" in part):
                normalized_parts.append("{id}")
            else:
                normalized_parts.append(part)
        endpoint = "/".join(normalized_parts)
    
    # Track the request
    try:
        from src.monitoring import track_api_request
        track_api_request(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
            duration=duration
        )
    except Exception as e:
        logger.debug(f"Failed to track metrics: {e}")
    
    return response


# Include additional routers
app.include_router(analytics_router)
app.include_router(chat_router)
app.include_router(settings_router)
app.include_router(trd_router)
app.include_router(router_router)
app.include_router(journal_router)
app.include_router(session_router)
if paper_trading_router:
    app.include_router(paper_trading_router)
app.include_router(mcp_router)
app.include_router(agent_queue_router)
app.include_router(workflow_router)
app.include_router(kill_switch_router)
app.include_router(hmm_router)
app.include_router(tradingview_router)
app.include_router(github_router)
app.include_router(metrics_router)
app.include_router(health_router)
app.include_router(broker_router)
app.include_router(lifecycle_scanner_router)
app.include_router(agent_management_router)
app.include_router(version_router)
app.include_router(demo_mode_router)
app.include_router(claude_agent_router)
app.include_router(agent_tools_router)
app.include_router(memory_router)
app.include_router(trading_floor_router)

# Mount Monte Carlo WebSocket endpoint
@app.websocket("/api/monte-carlo/ws")
async def monte_carlo_websocket_endpoint(websocket: WebSocket):
    await monte_carlo_ws_endpoint(websocket)

# Mount Broker WebSocket endpoint
@app.websocket("/api/brokers/ws")
async def broker_websocket_endpoint(websocket: WebSocket):
    await broker_websocket(websocket)

@app.on_event("startup")
async def startup_event():
    logger.info("QuantMind API Server starting on port 8000...")
    
    # Start Prometheus metrics server
    # Note: Grafana Cloud push is disabled - using Prometheus agent for remote_write
    # to avoid duplicate metric ingestion. See docker-compose.production.yml
    try:
        import os
        from src.monitoring import start_metrics_server
        
        metrics_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
        start_metrics_server(port=metrics_port)
        
        logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except Exception as e:
        logger.warning(f"Could not start metrics server: {e}")
    
    # Initialize StrategyRouter for live trading data
    try:
        from src.router.engine import StrategyRouter
        from src.api.router_endpoints import set_strategy_router
        
        router = StrategyRouter(
            use_smart_kill=True,
            use_kelly_governor=True,
            use_multi_timeframe=True
        )
        set_strategy_router(router)
        logger.info("StrategyRouter initialized for API endpoints")
        
        # Start Contabo regime polling in background
        try:
            import asyncio
            from src.router.engine import RegimeFetcher
            
            # Initialize regime fetcher
            regime_fetcher = RegimeFetcher()
            router._regime_fetcher = regime_fetcher
            
            # Start background polling task
            asyncio.create_task(regime_fetcher._poll_contabo_regime())
            logger.info("Contabo regime polling started in background")
        except Exception as e:
            logger.warning(f"Could not start Contabo regime polling: {e}")
            
    except Exception as e:
        logger.warning(f"Could not initialize StrategyRouter: {e}")
    
    # Initialize and start GitHub EA scheduler
    try:
        from src.integrations.github_ea_scheduler import start_scheduler, get_scheduler
        
        if start_scheduler():
            scheduler = get_scheduler()
            if scheduler:
                logger.info(f"GitHub EA scheduler started: {scheduler.get_status()}")
            else:
                logger.info("GitHub EA scheduler initialized")
        else:
            logger.warning("GitHub EA scheduler failed to start - check GITHUB_EA_REPO_URL env var")
    except Exception as e:
        logger.warning(f"Could not start GitHub EA scheduler: {e}")
    
    # Start lifecycle scheduler for daily bot lifecycle checks
    try:
        from scripts.schedule_lifecycle_check import LifecycleScheduler
        
        lifecycle_scheduler = LifecycleScheduler()
        lifecycle_scheduler.setup_schedule()
        lifecycle_scheduler.scheduler.start()
        app.state.lifecycle_scheduler = lifecycle_scheduler
        logger.info("Lifecycle scheduler started - daily check at 3:00 AM UTC")
    except Exception as e:
        logger.warning(f"Could not start lifecycle scheduler: {e}")
    
    # Start market scanner scheduler for session-based opportunity scanning
    try:
        from src.router.market_scanner import start_scanner_scheduler
        
        scanner_started = start_scanner_scheduler()
        if scanner_started:
            logger.info("Market scanner scheduler started - session-based opportunity scanning enabled")
        else:
            logger.warning("Market scanner scheduler failed to start")
    except Exception as e:
        logger.warning(f"Could not start market scanner scheduler: {e}")
    
    logger.info("Endpoints mounted: /api/ide, /api/chat, /api/analytics, /api/settings, /api/trd, /api/router, /api/journal, /api/sessions, /api/v1/backtest, /api/paper-trading, /api/mcp, /api/agents, /api/workflows, /api/kill-switch, /api/hmm, /api/metrics, /api/brokers, /api/eas, /api/virtual-accounts, /api/agent-tools, /health")

    # Start metrics WebSocket broadcast task
    try:
        from src.api.metrics_endpoints import ws_manager as metrics_ws_manager
        if metrics_ws_manager._broadcast_task is None:
            metrics_ws_manager._broadcast_task = __import__('asyncio').create_task(
                metrics_ws_manager.start_broadcasting(interval=1.0)
            )
            logger.info("Metrics WebSocket broadcast task started")
    except Exception as e:
        logger.warning(f"Could not start metrics broadcast task: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("QuantMind API Server shutting down...")
    
    # Stop lifecycle scheduler
    try:
        if hasattr(app.state, 'lifecycle_scheduler'):
            lifecycle_scheduler = app.state.lifecycle_scheduler
            if lifecycle_scheduler and hasattr(lifecycle_scheduler, 'scheduler'):
                lifecycle_scheduler.scheduler.shutdown()
                logger.info("Lifecycle scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping lifecycle scheduler: {e}")
    
    # Stop market scanner scheduler
    try:
        from src.router.market_scanner import stop_scanner_scheduler
        
        stop_scanner_scheduler()
        logger.info("Market scanner scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping market scanner scheduler: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
