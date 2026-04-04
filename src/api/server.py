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
# Accepted values: node_backend | node_trading | local (default: local)
# - node_backend (was contabo): Agent/compute endpoints only (AI agents, memory, workflows)
# - node_trading (was cloudzy): Live trading endpoints only (MT5, trading, broker)
# - local: All endpoints (development mode)
# Legacy names cloudzy/contabo still work via automatic normalization

NODE_ROLE = os.getenv("NODE_ROLE", "local").lower()
# Normalize old cloud provider names to generic aliases
NODE_ROLE = {"cloudzy": "node_trading", "contabo": "node_backend"}.get(NODE_ROLE, NODE_ROLE)

# Validate NODE_ROLE
VALID_ROLES = {"node_trading", "node_backend", "local"}
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
    "server_config_router",     # Server connections
    "notification_config_router", # Notification settings (Story 10-5)
    "audit_router",             # Audit system NL query (Story 10.1)
    "server_health_router",     # Server health metrics (Story 10-5)
    "server_router",            # Morning digest & node health (Story 3-7)
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
    "ensemble_router",         # Ensemble regime detection
    "knowledge_router",        # Knowledge endpoints
    "knowledge_unified_router", # Unified knowledge search API (Story 6-1)
    "knowledge_ingest_router",  # Web scraping + personal knowledge (Story 6-2)
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

# Dynamic router inclusion flags based on NODE_ROLE
# These flags are computed on access using module-level __getattr__
# to ensure they reflect the current NODE_ROLE environment variable at runtime,
# not at module import time. This is essential for tests that patch os.environ.

def _get_include_cloudzy() -> bool:
    """Check if trading routers should be included based on NODE_ROLE."""
    role = os.getenv("NODE_ROLE", "").lower()
    # cloudzy / node_trading node OR local dev includes trading routers
    return role in ("cloudzy", "node_trading", "local")

def _get_include_contabo() -> bool:
    """Check if agent/compute routers should be included based on NODE_ROLE."""
    role = os.getenv("NODE_ROLE", "").lower()
    # contabo / node_backend node OR local dev includes agent/compute routers
    return role in ("contabo", "node_backend", "local")

# Module-level __getattr__ to make INCLUDE_CLOUDZY and INCLUDE_CONTABO dynamic
# These are accessed like regular module attributes but computed on each access
def __getattr__(name):
    if name == "INCLUDE_CLOUDZY":
        return _get_include_cloudzy()
    if name == "INCLUDE_CONTABO":
        return _get_include_contabo()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger.info(f"NODE_ROLE: {NODE_ROLE}")

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
    from src.api.auth_endpoints import router as auth_router
    from src.auth.middleware import OAuthMiddleware
    from src.api.analytics_endpoints import router as analytics_router
    from src.api.chat_endpoints import router as chat_router
    from src.api.settings_endpoints import router as settings_router
    from src.api.skills_endpoints import router as skills_router
    from src.api.trd_endpoints import router as trd_router
    from src.api.trd_generation_endpoints import router as trd_generation_router
    from src.api.alpha_forge_templates import router as alpha_forge_templates_router
    from src.api.pipeline_status_endpoints import router as pipeline_status_router
    from src.api.strategy_versions import router as strategy_versions_router
    from src.api.variant_browser_endpoints import router as variant_browser_router
    from src.api.ab_race_endpoints import router as ab_race_router
    from src.api.provenance_endpoints import router as provenance_router
    from src.api.loss_propagation import router as loss_propagation_router
    from src.api.router_endpoints import router as router_router
    from src.api.journal_endpoints import router as journal_router
    from src.api.session_endpoints import router as session_router
    from src.api.mcp_endpoints import router as mcp_router
    from src.api.agent_queue_endpoints import router as agent_queue_router
    from src.api.workflow_endpoints import router as workflow_router
    from src.api.prefect_workflow_endpoints import router as prefect_workflow_router
    from src.api.flowforge_workflow_proxy import router as flowforge_workflow_proxy_router  # Story 11.8: FlowForge ↔ Prefect API Contract
    from src.api.weekend_cycle_endpoints import router as weekend_cycle_router  # Weekend Update Cycle (Story 8.13)
    from src.api.approval_gate import router as approval_gate_router
    from src.api.approval_endpoints import router as hitl_approval_router
    from src.api.kill_switch_endpoints import router as kill_switch_router
    from src.api.hmm_endpoints import router as hmm_router
    from src.api.dpr_endpoints import router as dpr_router  # DPR Queue Tier Remix (Story 17.2)
    from src.api.ssl_endpoints import router as ssl_router  # SSL Circuit Breaker (Story 18.1)
    from src.api.dead_zone_endpoints import router as dead_zone_router  # Dead Zone Workflow 3 EOD Reports
    from src.api.trading_session_endpoints import router as trading_session_router  # Canonical window session detection
    from src.api.trading_tilt_endpoints import router as trading_tilt_router  # Tilt phase status (Story 16.1)
    from src.api.cooldown_endpoints import router as cooldown_router  # Inter-Session Cooldown (Story 16.3)
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
    from src.api.canvas_context_endpoints import router as canvas_context_router
    from src.api.agent_session_endpoints import router as agent_session_router
    from src.api.session_checkpoint_endpoints import router as checkpoint_router
    from src.api.trading_floor_endpoints import router as trading_floor_router
    from src.api.floor_manager_endpoints import router as floor_manager_router
    from src.api.copilot_kill_switch_endpoints import router as copilot_kill_switch_router
    from src.api.workshop_copilot_endpoints import router as workshop_copilot_router
    from src.api.copilot_endpoints import router as copilot_router
    from src.api.video_to_ea_endpoints import router as video_to_ea_router
    from src.api.ide_knowledge import router as knowledge_router
    from src.api.knowledge_endpoints import router as knowledge_unified_router
    from src.api.batch_endpoints import router as batch_router
    from src.api.evaluation_endpoints import router as evaluation_router
    from src.api.ide_files import router as files_router
    from src.api.ide_mt5 import router as mt5_router
    from src.api.tool_call_endpoints import router as tool_call_router
    from src.api.ide_assets import router as assets_router
    from src.api.ide_ea import router as ea_ide_router
    from src.api.ide_strategies import router as strategies_router
    from src.api.ide_timeframes import router as timeframes_router
    from src.api.compile_endpoints import router as compile_router
    from src.api.ide_video_ingest import router as video_ingest_ide_router
    from src.api.ide_chat import router as chat_ide_router
    from src.api.ide_backtest import router as backtest_router
    from src.api.backtest_endpoints import router as backtest_results_router
    from src.api.portfolio_endpoints import router as portfolio_router
    from src.api.portfolio_broker_endpoints import router as portfolio_broker_router
    from src.api.risk_endpoints import router as risk_router
    from src.api.sqs_endpoints import router as sqs_router
    from src.api.task_sse_endpoints import router as task_sse_router
    from src.api.pdf_endpoints import router as pdf_router
    from src.api.trading.routes import router as trading_router
    from src.api.trading_session_risk_endpoints import router as trading_session_risk_router
    from src.api.session_kelly_endpoints import router as session_kelly_router
    from src.api.economic_calendar_endpoints import router as economic_calendar_router
    from src.api.hmm_inference_server import router as hmm_inference_router, ensemble_router
    from src.api.provider_config_endpoints import router as provider_config_router
    from src.api.server_config_endpoints import router as server_config_router, server_router
    from src.api.notification_config_endpoints import router as notification_config_router
    from src.api.server_health_endpoints import router as server_health_router
    from src.api.node_update_endpoints import router as node_update_router  # Node sequential update (Story 11-3)
    from src.api.reasoning_log_endpoints import router as reasoning_router
    from src.api.deployment_endpoints import router as deployment_router
    from src.api.audit_endpoints import router as audit_router  # Audit system (Story 10.1)
    from src.api.scheduled_tasks_endpoints import router as scheduled_tasks_router  # Weekend compute (Story 11.2)
    from src.api.autonomous_scheduler import router as autonomous_scheduler_router, get_scheduler  # Autonomous overnight research
    from src.api.zero_auth_endpoints import router as zero_auth_router  # Zero-Auth OAuth/ADC config
    from src.api.tool_forge_endpoints import router as tool_forge_router  # Tool Forge — Dynamic Tool Creation
    from src.api.agent_thought_stream_endpoints import router as agent_thought_router  # Agent thought SSE
    from src.api.agent_tile_endpoints import router as agent_tile_router  # Agent tile system
    from src.api.workflow_templates_endpoints import router as workflow_templates_router  # Workflow Templates (Story C1)
    from src.api.race_endpoints import router as race_router  # Strategy Race Board (Story C3)
    from src.api.trading_results_endpoints import router as trading_results_router  # Trading results for DPR scoring
    from src.api.svss_endpoints import router as svss_router  # SVSS REST API (VWAP, RVOL, Volume Profile, MFI)
except ImportError as e:
    logger.error(f"Import Error: {e}")
    # Fallback/Debug info
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Optional MT5-dependent imports (Linux compatibility)
paper_trading_router = None
try:
    from src.api.paper_trading import router as paper_trading_router
    logger.info("Paper trading endpoints loaded")
except ImportError:
    try:
        from src.api.paper_trading_endpoints import router as paper_trading_router
        logger.info("Paper trading endpoints loaded (legacy)")
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


# ========== OAuth 2.1 Authentication Middleware ==========
# Add OAuth middleware to protect /api/* endpoints
# The middleware extracts and validates sessions from httpOnly cookies
# Auth endpoints (/api/auth/*) are excluded from protection
try:
    app.add_middleware(
        OAuthMiddleware,
        exclude_paths=[
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/auth/login",
            "/api/auth/callback",
            "/api/auth/logout",
            "/api/auth/migrate",
            "/api/auth/migrate/status",
        ]
    )
    logger.info("OAuth 2.1 authentication middleware added")
except Exception as e:
    logger.warning(f"Could not add OAuth middleware: {e}")

# Include auth router
app.include_router(auth_router)
logger.info("Auth endpoints registered: /api/auth/*")


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
# - node_trading: Live trading routers only
# - node_backend: Agent/compute routers only
# - local: All routers (development mode)
# =============================================================================

logger.info(f"Registering routers for NODE_ROLE={NODE_ROLE}...")

# ---------------------------------------------------------------------------
# Routers that work on BOTH nodes (or always included in local mode)
# ---------------------------------------------------------------------------
# Always include health, version, metrics (BOTH routers)
if _get_include_cloudzy() or _get_include_contabo():
    app.include_router(health_router)    # Health checks
    app.include_router(version_router)    # Version info
    app.include_router(metrics_router)   # Metrics

# ---------------------------------------------------------------------------
# IDE Core Routers (needed on both for development)
# ---------------------------------------------------------------------------
# Files, assets, strategies, timeframes - needed for IDE to function
if _get_include_contabo():  # Contabo runs the IDE
    app.include_router(files_router)
    app.include_router(assets_router)
    app.include_router(strategies_router)
    app.include_router(timeframes_router)
    app.include_router(compile_router)

# ---------------------------------------------------------------------------
# Cloudzy Routers (Live Trading)
# ---------------------------------------------------------------------------
if _get_include_cloudzy():
    app.include_router(kill_switch_router)       # Trading kill switch
    app.include_router(trading_router)           # Trade execution
    app.include_router(broker_router)            # Broker connection
    if paper_trading_router:
        app.include_router(paper_trading_router) # Paper trading
    app.include_router(tradingview_router)       # TradingView charts
    app.include_router(lifecycle_scanner_router) # Lifecycle monitoring
    app.include_router(trading_session_risk_router) # RHM session risk state
    app.include_router(trading_results_router)  # Trading results for DPR scoring
    logger.info("Registered Cloudzy routers: trading, broker, kill_switch, paper_trading, tradingview, trading_session_risk, trading_results")

# ---------------------------------------------------------------------------
# Contabo Routers (Agent/Compute)
# ---------------------------------------------------------------------------
if _get_include_contabo():
    # Settings & Configuration
    app.include_router(settings_router)
    app.include_router(skills_router)  # Skill Catalogue (Story 7.4)
    app.include_router(provider_config_router)
    # zero_auth_router registered unconditionally below
    app.include_router(server_config_router)
    app.include_router(notification_config_router)  # Notification settings (Story 10-5)
    app.include_router(scheduled_tasks_router)  # Weekend compute tasks (Story 11.2)
    # autonomous_scheduler_router and tool_forge_router registered unconditionally below
    app.include_router(audit_router)  # Audit system NL query (Story 10.1)
    app.include_router(server_health_router)  # Server health metrics (Story 10-5)
    app.include_router(node_update_router)  # Node sequential update with rollback (Story 11-3)
    app.include_router(reasoning_router)  # Agent reasoning transparency (Story 10-2)
    # server_router registered unconditionally below (morning digest, node health)
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
    app.include_router(canvas_context_router)  # Canvas Context Templates

    # Floor Manager & Workflows
    app.include_router(floor_manager_router)
    app.include_router(copilot_kill_switch_router)
    app.include_router(workshop_copilot_router)
    app.include_router(copilot_router)  # /api/copilot/chat - UI always routes through Copilot
    app.include_router(workflow_router)
    app.include_router(prefect_workflow_router)
    app.include_router(flowforge_workflow_proxy_router)  # Story 11.8: FlowForge ↔ Prefect API Contract
    app.include_router(weekend_cycle_router)  # Weekend Update Cycle (Story 8.13)

    # Risk & HMM
    app.include_router(hmm_router)
    app.include_router(dpr_router)  # DPR Queue Tier Remix (Story 17.2)
    app.include_router(ssl_router)  # SSL Circuit Breaker (Story 18.1)
    app.include_router(dead_zone_router)  # Dead Zone Workflow 3 EOD Reports
    app.include_router(hmm_inference_router)
    app.include_router(ensemble_router)  # Ensemble regime detection (HMM + MS-GARCH + BOCPD)

    # Knowledge & Research
    app.include_router(knowledge_router)
    app.include_router(knowledge_unified_router)  # Unified search API (Story 6-1)
    from src.api.knowledge_ingest_endpoints import router as knowledge_ingest_router
    app.include_router(knowledge_ingest_router)  # Web scraping + personal knowledge (Story 6-2)
    from src.api.news_endpoints import router as news_router
    app.include_router(news_router)  # Live news feed (Story 6-3)

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
    app.include_router(portfolio_router)  # Portfolio report API (Story 7-8)
    app.include_router(task_sse_router)  # Task SSE endpoints (Story 7-9)
    # Note: backtest_results_router and risk_router are registered unconditionally below

    logger.info("Registered Contabo routers: agents, memory, workflows, settings, knowledge")

# ---------------------------------------------------------------------------
# Routers that work on EITHER node (not specific to trading or agents)
# ---------------------------------------------------------------------------
# These are needed for general operation
app.include_router(trd_router)        # Trading Requirements Doc
app.include_router(trd_generation_router)  # TRD Generation (Alpha Forge)
app.include_router(alpha_forge_templates_router)  # Template Library (Story 8-3)
app.include_router(pipeline_status_router)  # Pipeline Status Board (Story 8-7)
app.include_router(strategy_versions_router)  # Version Control (Story 8-4)
app.include_router(variant_browser_router)  # Variant Browser (Story 8-8)
app.include_router(ab_race_router)  # A/B Race Board (Story 8-9)
if race_router is not ab_race_router:
    app.include_router(race_router)  # Strategy Race Board (Story C3)
else:
    logger.info("Skipped duplicate race_router include; alias points to ab_race_router")
app.include_router(provenance_router)  # Provenance Chain (Story 8-9)
app.include_router(workflow_templates_router)  # Workflow Templates (Story C1)
app.include_router(loss_propagation_router)  # Loss Propagation (Story 8-9)
app.include_router(router_router)     # Router endpoints
app.include_router(journal_router)    # Journal
app.include_router(session_router)     # Sessions
app.include_router(trading_session_router)  # Canonical window session detection (SessionTimeline)
app.include_router(trading_tilt_router)     # Tilt phase status (Story 16.1)
app.include_router(cooldown_router)         # Inter-Session Cooldown (Story 16.3)
app.include_router(mcp_router)         # MCP endpoints
app.include_router(approval_gate_router) # Approval gate
app.include_router(hitl_approval_router) # Human-in-the-loop approvals
app.include_router(deployment_router)  # EA deployment pipeline (FR79)
app.include_router(github_router)      # GitHub sync
app.include_router(demo_mode_router)   # Demo mode
app.include_router(portfolio_broker_router)  # Portfolio broker registry (Story 9-1)
app.include_router(backtest_results_router)  # Backtest results API (Story 4-4) — needed on both nodes
app.include_router(risk_router)  # Risk API (Story 4-6) — needed on both nodes
app.include_router(session_kelly_router)  # Session Kelly API (Story 4.10) — needed on both nodes
app.include_router(economic_calendar_router)  # Economic Calendar — full calendar view with blackouts
app.include_router(sqs_router)  # SQS API (Story 4-7) — needed on both nodes
app.include_router(svss_router)  # SVSS REST API (VWAP, RVOL, Volume Profile, MFI) — needed on both nodes
app.include_router(dpr_router)  # DPR Queue Tier Remix (Story 17.2) — needed on both nodes
app.include_router(server_router)  # Morning digest & node health (Story 3-7) — needed on all nodes

# IDE-specific (non-agent)
app.include_router(mt5_router)         # MT5 IDE
app.include_router(ea_ide_router)      # EA IDE

if department_mail_router:
    app.include_router(department_mail_router)

app.include_router(pdf_router)

# ── Always-on routers (NODE_ROLE-agnostic) ────────────────────────────────
# These endpoints are needed on every deployment type including local dev.
app.include_router(agent_thought_router)      # Agent thought SSE stream
app.include_router(agent_tile_router)         # Agent tile creation / display
app.include_router(zero_auth_router)          # Zero-Auth: Qwen CLI + Gemini ADC
app.include_router(autonomous_scheduler_router)  # Overnight research scheduler
app.include_router(tool_forge_router)         # Dynamic tool creation / registry
app.include_router(settings_router)          # Settings (connection, appearance, etc.)
app.include_router(skills_router)            # Skill Catalogue (always-on for UI)
app.include_router(provider_config_router)    # AI provider configurations

# ── Copilot & Floor Manager (always-on for Workshop UI) ──────────────────────
# The Workshop UI (CopilotPanel) calls /api/copilot/chat - these must be available
# in ALL modes: local (development), node_backend, and node_trading.
app.include_router(floor_manager_router)      # Floor Manager chat + task routing
app.include_router(copilot_router)           # /api/copilot/chat - Workshop UI entry point
app.include_router(workshop_copilot_router)   # Workshop copilot secondary endpoints
app.include_router(copilot_kill_switch_router)  # Copilot kill switch

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

    # Ensure all SQLAlchemy models have their tables (idempotent create_all)
    try:
        from src.database.engine import init_database
        init_database()
        logger.info("Database tables verified/created.")
    except Exception as e:
        logger.warning(f"DB init warning: {e}")

    # Run graph memory DB migrations (idempotent - safe to run every startup)
    try:
        import os
        os.makedirs("data", exist_ok=True)
        from src.memory.graph.migration import migrate_graph_memory_db
        graph_db = os.environ.get("GRAPH_MEMORY_DB", "data/graph_memory.db")
        migrate_graph_memory_db(graph_db)
        logger.info(f"Graph memory DB migrations applied: {graph_db}")
    except Exception as e:
        logger.warning(f"Graph memory migration warning: {e}")

    # Resume pending HITL approvals and paused workflows from DB
    try:
        from src.agents.approval_manager import get_approval_manager
        resumed_approvals = get_approval_manager().resume_pending()
        if resumed_approvals > 0:
            logger.info(f"Resumed {resumed_approvals} pending HITL approvals from DB")

            # Also resume workflows paused at HITL gates
            try:
                from src.agents.departments.workflow_coordinator import get_workflow_coordinator
                resumed_wf = get_workflow_coordinator().resume_waiting_workflows()
                if resumed_wf > 0:
                    logger.info(f"Resumed {resumed_wf} workflows paused at HITL gates")
            except Exception as e:
                logger.warning(f"Workflow resume warning: {e}")
        else:
            logger.info("No pending HITL approvals to resume")
    except Exception as e:
        logger.warning(f"HITL resume warning: {e}")

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

    # Initialize OpenTelemetry tracing
    try:
        from src.monitoring import init_tracing

        tracing_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if tracing_endpoint:
            init_tracing(app)
            logger.info(f"OpenTelemetry tracing initialized with endpoint: {tracing_endpoint}")
        else:
            logger.info("OpenTelemetry tracing not configured (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
    except Exception as e:
        logger.warning(f"Could not initialize tracing: {e}")

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

        # Wire NewsBlackoutService into ProgressiveKillSwitch (after router is ready)
        try:
            from src.market.news_blackout import NewsBlackoutService
            from src.api.websocket_endpoints import manager as ws_manager

            news_blackout = NewsBlackoutService()
            news_blackout.set_ws_manager(ws_manager)

            # Force PKS creation by accessing the property, then wire NewsSensor
            pks = router.progressive_kill_switch
            if pks and hasattr(pks, "session_monitor") and pks.session_monitor:
                pks.session_monitor.set_news_sensor(news_blackout._news_sensor)
                logger.info("NewsBlackoutService wired to ProgressiveKillSwitch SessionMonitor")

            # Wire into app.state for API endpoint access
            app.state.news_blackout = news_blackout

            # Start the service (background scheduler)
            news_blackout.start()
            logger.info("NewsBlackoutService started")
        except Exception as e:
            logger.warning(f"Could not start NewsBlackoutService: {e}")

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

    # NewsBlackoutService started separately via its own scheduler on startup.
    # It replaces the old NewsFeedPoller — kill switch is driven by Finnhub
    # economic calendar data, not the news feed poller.
    # See: src/market/news_blackout.py

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
        stream_handler = await init_stream_handler()
        app.state.stream_handler = stream_handler
        logger.info("Agent stream handler initialized for SSE")
    except Exception as e:
        logger.warning(f"Could not initialize agent stream handler: {e}")

    # Start autonomous research scheduler
    try:
        from src.api.autonomous_scheduler import get_scheduler as _get_autonomous_scheduler
        _autonomous_scheduler = _get_autonomous_scheduler()
        asyncio.create_task(_autonomous_scheduler.start())
        logger.info("Autonomous research scheduler started")
    except Exception as e:
        logger.warning(f"Could not start autonomous scheduler: {e}")

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

    # NewsFeedPoller removed — replaced by NewsBlackoutService (src/market/news_blackout.py)

    # Stop Agent Stream Handler
    try:
        if hasattr(app.state, 'stream_handler'):
            from src.agents.streaming import close_stream_handler
            await close_stream_handler()
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
