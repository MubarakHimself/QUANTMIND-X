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

# =============================================================================
# NODE_ROLE Configuration (Story 1-3: Backend Deployment Split)
# =============================================================================
# Controls which router groups register at startup
# Accepted values: contabo | cloudzy | local (default: local)
# - contabo: Agent/compute endpoints only (AI agents, memory, workflows)
# - cloudzy: Live trading endpoints only (MT5, trading, broker)
# - local: All endpoints (development mode)

NODE_ROLE = os.getenv("NODE_ROLE", "local").lower()

# Validate NODE_ROLE
VALID_ROLES = {"contabo", "cloudzy", "local"}
if NODE_ROLE not in VALID_ROLES:
    logger.warning(
        f"Invalid NODE_ROLE '{NODE_ROLE}'. "
        f"Valid values are: {', '.join(VALID_ROLES)}. "
        f"Defaulting to 'local' mode."
    )
    NODE_ROLE = "local"

logger.info(f"NODE_ROLE set to: {NODE_ROLE}")

# =============================================================================
# Router Classification by Node
# =============================================================================
# Cloudzy routers (live trading): MT5, trading, broker, kill switch
# Contabo routers (agent/compute): agents, memory, workflows, settings
# Both/Local: health, version, metrics

CLOUDZY_ROUTERS = {
    "kill_switch_router",      # Trading kill switch
    "trading_router",          # Trade execution
    "broker_router",           # Broker connection
    "paper_trading_router",    # Paper trading
    "tradingview_router",      # TradingView charts
    "lifecycle_scanner_router", # Lifecycle monitoring
}

CONTABO_ROUTERS = {
    "settings_router",          # Configuration
    "provider_config_router",   # AI providers
    "agent_management_router", # Agent management
    "agent_activity_router",   # Agent activity
    "agent_metrics_router",     # Agent metrics
    "agent_queue_router",      # Agent queue
    "agent_session_router",    # Agent sessions
    "checkpoint_router",        # Session checkpoints
    "memory_router",           # Memory endpoints
    "memory_dept_router",      # Memory department
    "memory_unified_router",   # Unified memory
    "graph_memory_router",     # Graph memory
    "floor_manager_router",    # Copilot
    "workshop_copilot_router",# Workshop copilot
    "workflow_router",         # Alpha Forge
    "hmm_router",              # Risk sensors
    "hmm_inference_router",    # HMM inference
    "knowledge_router",        # Knowledge endpoints
    "video_to_ea_router",     # Video to EA
    "video_ingest_ide_router",# Video ingest IDE
    "batch_router",            # Batch processing
    "evaluation_router",       # Evaluation
    "trading_floor_router",   # Trading floor
    "claude_agent_router",     # Claude agent
    "tool_call_router",        # Tool calls
    "model_router",            # Model config
}

BOTH_ROUTERS = {
    "health_router",           # Health checks (always needed)
    "version_router",          # Version info (always needed)
    "metrics_router",          # Metrics (always needed)
}

# Set to track which routers to include
INCLUDE_CLOUDZY = NODE_ROLE in ("cloudzy", "local")
INCLUDE_CONTABO = NODE_ROLE in ("contabo", "local")

logger.info(f"Including Cloudzy routers: {INCLUDE_CLOUDZY}")
logger.info(f"Including Contabo routers: {INCLUDE_CONTABO}")

# =============================================================================
# End Router Configuration
# =============================================================================

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
    from src.api.approval_gate import router as approval_gate_router
    from src.api.kill_switch_endpoints import router as kill_switch_router
    from src.api.hmm_endpoints import router as hmm_router
    from src.api.tradingview_endpoints import router as tradingview_router
    from src.api.github_endpoints import router as github_router
    from src.api.monte_carlo_ws import monte_carlo_ws_endpoint
    from src.api.metrics_endpoints import router as metrics_router
    from src.api.agent_metrics import router as agent_metrics_router
    from src.api.health_endpoints import router as health_router
    from src.api.broker_endpoints import router as broker_router, broker_websocket
    from src.api.lifecycle_scanner_endpoints import router as lifecycle_scanner_router
    from src.api.agent_management_endpoints import router as agent_management_router
    from src.api.agent_activity import router as agent_activity_router
    from src.api.version_endpoints import router as version_router
    from src.api.demo_mode_endpoints import router as demo_mode_router
    from src.api.claude_agent_endpoints import router as claude_agent_router
    from src.api.agent_tools import router as agent_tools_router
    from src.api.model_config_endpoints import router as model_router
    from src.api.memory_endpoints import (
        router as memory_router,
        dept_router as memory_dept_router,
        unified_router as memory_unified_router,
    )
    from src.api.graph_memory_endpoints import router as graph_memory_router
    from src.api.agent_session_endpoints import router as agent_session_router
    from src.api.session_checkpoint_endpoints import router as checkpoint_router
    from src.api.trading_floor_endpoints import router as trading_floor_router
    from src.api.floor_manager_endpoints import router as floor_manager_router
    from src.api.workshop_copilot_endpoints import router as workshop_copilot_router
    from src.api.video_to_ea_endpoints import router as video_to_ea_router
    from src.api.ide_knowledge import router as knowledge_router
    from src.api.batch_endpoints import router as batch_router
    from src.api.evaluation_endpoints import router as evaluation_router
    from src.api.ide_files import router as files_router
    from src.api.ide_mt5 import router as mt5_router
    from src.api.tool_call_endpoints import router as tool_call_router
    from src.api.ide_assets import router as assets_router
    from src.api.ide_ea import router as ea_ide_router
    from src.api.ide_strategies import router as strategies_router
    from src.api.ide_timeframes import router as timeframes_router
    from src.api.ide_video_ingest import router as video_ingest_ide_router
    from src.api.ide_chat import router as chat_ide_router
    from src.api.ide_backtest import router as backtest_router
    from src.api.pdf_endpoints import router as pdf_router
    from src.api.trading.routes import router as trading_router
    from src.api.hmm_inference_server import router as hmm_inference_router
    from src.api.provider_config_endpoints import router as provider_config_router
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

# Optional department mail endpoints (may have dependency issues)
department_mail_router = None
try:
    from src.api.department_mail_endpoints import router as department_mail_router
    logger.info("Department mail endpoints loaded")
except ImportError as e:
    logger.warning(f"Department mail endpoints not available: {e}")

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


# =============================================================================
# Router Registration (Conditional on NODE_ROLE)
# =============================================================================
# Routers are registered based on NODE_ROLE setting:
# - cloudzy: Live trading routers only
# - contabo: Agent/compute routers only
# - local: All routers (development mode)
# =============================================================================

logger.info(f"Registering routers for NODE_ROLE={NODE_ROLE}...")

# ---------------------------------------------------------------------------
# Routers that work on BOTH nodes (or always included in local mode)
# ---------------------------------------------------------------------------
# Always include health, version, metrics (BOTH routers)
if INCLUDE_CLOUDZY or INCLUDE_CONTABO:
    app.include_router(health_router)    # Health checks
    app.include_router(version_router)    # Version info
    app.include_router(metrics_router)   # Metrics

# ---------------------------------------------------------------------------
# IDE Core Routers (needed on both for development)
# ---------------------------------------------------------------------------
# Files, assets, strategies, timeframes - needed for IDE to function
if INCLUDE_CONTABO:  # Contabo runs the IDE
    app.include_router(files_router)
    app.include_router(assets_router)
    app.include_router(strategies_router)
    app.include_router(timeframes_router)

# ---------------------------------------------------------------------------
# Cloudzy Routers (Live Trading)
# ---------------------------------------------------------------------------
if INCLUDE_CLOUDZY:
    app.include_router(kill_switch_router)       # Trading kill switch
    app.include_router(trading_router)           # Trade execution
    app.include_router(broker_router)            # Broker connection
    if paper_trading_router:
        app.include_router(paper_trading_router) # Paper trading
    app.include_router(tradingview_router)       # TradingView charts
    app.include_router(lifecycle_scanner_router) # Lifecycle monitoring
    logger.info("Registered Cloudzy routers: trading, broker, kill_switch, paper_trading, tradingview")

# ---------------------------------------------------------------------------
# Contabo Routers (Agent/Compute)
# ---------------------------------------------------------------------------
if INCLUDE_CONTABO:
    # Settings & Configuration
    app.include_router(settings_router)
    app.include_router(provider_config_router)
    app.include_router(model_router)

    # Agent Management
    app.include_router(agent_management_router)
    app.include_router(agent_activity_router)
    app.include_router(agent_metrics_router)
    app.include_router(agent_queue_router)
    app.include_router(agent_session_router)
    app.include_router(agent_tools_router)

    # Memory & Checkpoints
    app.include_router(memory_router)
    app.include_router(memory_dept_router)
    app.include_router(memory_unified_router)
    app.include_router(graph_memory_router)
    app.include_router(checkpoint_router)

    # Floor Manager & Workflows
    app.include_router(floor_manager_router)
    app.include_router(workshop_copilot_router)
    app.include_router(workflow_router)

    # Risk & HMM
    app.include_router(hmm_router)
    app.include_router(hmm_inference_router)

    # Knowledge & Research
    app.include_router(knowledge_router)

    # Video & Processing
    app.include_router(video_to_ea_router)
    app.include_router(video_ingest_ide_router)
    app.include_router(batch_router)
    app.include_router(evaluation_router)

    # Trading Floor
    app.include_router(trading_floor_router)

    # Claude Agent
    app.include_router(claude_agent_router)
    app.include_router(tool_call_router)

    # IDE-specific
    app.include_router(chat_ide_router)
    app.include_router(backtest_router)

    logger.info("Registered Contabo routers: agents, memory, workflows, settings, knowledge")

# ---------------------------------------------------------------------------
# Routers that work on EITHER node (not specific to trading or agents)
# ---------------------------------------------------------------------------
# These are needed for general operation
app.include_router(trd_router)        # Trading Requirements Doc
app.include_router(router_router)     # Router endpoints
app.include_router(journal_router)    # Journal
app.include_router(session_router)     # Sessions
app.include_router(mcp_router)         # MCP endpoints
app.include_router(approval_gate_router) # Approval gate
app.include_router(github_router)      # GitHub sync
app.include_router(demo_mode_router)   # Demo mode

# IDE-specific (non-agent)
app.include_router(mt5_router)         # MT5 IDE
app.include_router(ea_ide_router)      # EA IDE

if department_mail_router:
    app.include_router(department_mail_router)

app.include_router(pdf_router)

logger.info(f"Router registration complete for NODE_ROLE={NODE_ROLE}")

# =============================================================================
# End Router Registration
# =============================================================================

# Standardized error handlers
from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

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
    
    logger.info("Endpoints mounted: /api/ide, /api/chat, /api/analytics, /api/settings, /api/trd, /api/router, /api/journal, /api/sessions, /api/v1/backtest, /api/paper-trading, /api/mcp, /api/agents, /api/workflows, /api/kill-switch, /api/hmm, /api/metrics, /api/brokers, /api/eas, /api/virtual-accounts, /api/agent-tools, /api/batch, /health")

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

    # Initialize Agent Stream Handler for SSE
    try:
        from src.agents.streaming import init_stream_handler, get_stream_handler
        stream_handler = asyncio.get_event_loop().run_until_complete(init_stream_handler())
        app.state.stream_handler = stream_handler
        logger.info("Agent stream handler initialized for SSE")
    except Exception as e:
        logger.warning(f"Could not initialize agent stream handler: {e}")

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

    # Stop Agent Stream Handler
    try:
        if hasattr(app.state, 'stream_handler'):
            from src.agents.streaming import close_stream_handler
            asyncio.get_event_loop().run_until_complete(close_stream_handler())
            logger.info("Agent stream handler stopped")
    except Exception as e:
        logger.error(f"Error stopping agent stream handler: {e}")


# =============================================================================
# Agent SSE Streaming Endpoint
# =============================================================================

@app.get("/api/agents/stream")
async def agent_stream_sse(
    agent_id: str = Query(default=None, description="Filter by agent ID"),
    task_id: str = Query(default=None, description="Filter by task ID"),
    event_type: str = Query(default=None, description="Filter by event type(s), comma-separated")
):
    """
    SSE endpoint for streaming agent events.

    Query Parameters:
    - agent_id: Optional filter by agent ID
    - task_id: Optional filter by task ID
    - event_type: Optional filter by event type (agent_started, tool_start, tool_complete, etc.)
                     Can be comma-separated to filter multiple types

    Returns:
    SSE stream of agent events
    """
    from starlette.responses import StreamingResponse
    from src.agents.streaming import get_stream_handler, AgentStreamEventType

    handler = get_stream_handler()

    # Parse event types if provided (support comma-separated)
    event_types = None
    if event_type:
        try:
            event_type_list = [et.strip() for et in event_type.split(",")]
            event_types = [AgentStreamEventType(et) for et in event_type_list]
        except ValueError as e:
            logger.warning(f"Invalid event type in filter: {event_type}, {e}")
            pass

    # Create SSE response
    async def event_generator():
        async for event in handler.stream_sse(agent_id, task_id, event_types):
            yield event.encode('utf-8')

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
