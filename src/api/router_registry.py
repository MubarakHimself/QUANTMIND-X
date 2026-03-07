"""Centralized API router import and registry helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger("quantmind.server")


def load_router_registry() -> tuple[dict, dict]:
    """Load required and optional routers for server composition."""
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
        dept_router as memory_dept_router,
        router as memory_router,
        unified_router as memory_unified_router,
    )
    from src.api.agent_session_endpoints import router as agent_session_router
    from src.api.session_checkpoint_endpoints import router as checkpoint_router
    from src.api.trading_floor_endpoints import router as trading_floor_router
    from src.api.floor_manager_endpoints import router as floor_manager_router
    from src.api.video_to_ea_endpoints import router as video_to_ea_router
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

    routers = {
        "primary": [
            settings_router,
            trd_router,
            router_router,
            journal_router,
            session_router,
            mcp_router,
            agent_queue_router,
            workflow_router,
            approval_gate_router,
            kill_switch_router,
            hmm_router,
            tradingview_router,
            github_router,
            metrics_router,
            agent_metrics_router,
            health_router,
            broker_router,
            lifecycle_scanner_router,
            agent_management_router,
            agent_activity_router,
            version_router,
            demo_mode_router,
            claude_agent_router,
            agent_tools_router,
            model_router,
            memory_router,
            memory_dept_router,
            memory_unified_router,
            agent_session_router,
            checkpoint_router,
            trading_floor_router,
            floor_manager_router,
            video_to_ea_router,
            batch_router,
            evaluation_router,
            files_router,
            mt5_router,
            tool_call_router,
            assets_router,
            ea_ide_router,
            strategies_router,
            timeframes_router,
            video_ingest_ide_router,
            chat_ide_router,
            backtest_router,
            pdf_router,
            trading_router,
            hmm_inference_router,
        ],
        "monte_carlo_ws": monte_carlo_ws_endpoint,
        "broker_websocket": broker_websocket,
    }

    optional_routers = {
        "paper_trading": _load_optional_router(
            "src.api.paper_trading_endpoints",
            "router",
            "Paper trading endpoints not available (MT5 required)"
        ),
        "department_mail": _load_optional_router(
            "src.api.department_mail_endpoints",
            "router",
            "Department mail endpoints not available"
        ),
    }

    return routers, optional_routers


def _load_optional_router(module_name: str, attr_name: str, warning_prefix: str):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        router = getattr(module, attr_name)
        logger.info(f"Loaded optional router from {module_name}")
        return router
    except ImportError as exc:
        logger.warning(f"{warning_prefix}: {exc}")
        return None
