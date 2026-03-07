"""Application bootstrap helpers for the main FastAPI server.

This module centralizes router registration, middleware setup, exception
handlers, and startup/shutdown lifecycle hooks so ``src.api.server`` can stay
focused on composition.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket

logger = logging.getLogger("quantmind.server")


def configure_middleware(app) -> None:
    """Attach shared middleware to the FastAPI app."""

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable):
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        response: Response = await call_next(request)
        duration = time.time() - start_time

        endpoint = request.url.path
        if endpoint.startswith("/api/"):
            parts = endpoint.split("/")
            normalized_parts = []
            for part in parts:
                if part and (part.isdigit() or len(part) == 36 and "-" in part):
                    normalized_parts.append("{id}")
                else:
                    normalized_parts.append(part)
            endpoint = "/".join(normalized_parts)

        try:
            from src.monitoring import track_api_request

            track_api_request(
                method=request.method,
                endpoint=endpoint,
                status=response.status_code,
                duration=duration,
            )
        except Exception as exc:
            logger.debug(f"Failed to track metrics: {exc}")

        return response


def register_exception_handlers(app) -> None:
    """Register shared exception handlers."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def register_routes(app, routers: dict, optional_routers: dict) -> None:
    """Register routers and websocket endpoints on the app."""
    for router in routers["primary"]:
        app.include_router(router)

    for maybe_router in optional_routers.values():
        if maybe_router:
            app.include_router(maybe_router)

    app.websocket("/api/monte-carlo/ws")(routers["monte_carlo_ws"])
    app.websocket("/api/brokers/ws")(routers["broker_websocket"])
    app.get("/api/agents/stream")(create_agent_stream_endpoint())


def create_agent_stream_endpoint():
    """Create the SSE endpoint used for agent streaming."""

    async def agent_stream_sse(
        agent_id: str = Query(default=None, description="Filter by agent ID"),
        task_id: str = Query(default=None, description="Filter by task ID"),
        event_type: str = Query(default=None, description="Filter by event type(s), comma-separated"),
    ):
        from src.agents.streaming import AgentStreamEventType, get_stream_handler

        handler = get_stream_handler()
        event_types = None
        if event_type:
            try:
                event_type_list = [et.strip() for et in event_type.split(",")]
                event_types = [AgentStreamEventType(et) for et in event_type_list]
            except ValueError as exc:
                logger.warning(f"Invalid event type in filter: {event_type}, {exc}")

        async def event_generator():
            async for event in handler.stream_sse(agent_id, task_id, event_types):
                yield event.encode("utf-8")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return agent_stream_sse


async def startup_event(app) -> None:
    """Run API startup routines."""
    logger.info("QuantMind API Server starting on port 8000...")

    try:
        from src.monitoring import start_metrics_server

        metrics_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
        start_metrics_server(port=metrics_port)
        logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except Exception as exc:
        logger.warning(f"Could not start metrics server: {exc}")

    try:
        from src.router.engine import RegimeFetcher, StrategyRouter
        from src.api.router_endpoints import set_strategy_router

        router = StrategyRouter(
            use_smart_kill=True,
            use_kelly_governor=True,
            use_multi_timeframe=True,
        )
        set_strategy_router(router)
        logger.info("StrategyRouter initialized for API endpoints")

        try:
            regime_fetcher = RegimeFetcher()
            router._regime_fetcher = regime_fetcher
            asyncio.create_task(regime_fetcher._poll_contabo_regime())
            logger.info("Contabo regime polling started in background")
        except Exception as exc:
            logger.warning(f"Could not start Contabo regime polling: {exc}")
    except Exception as exc:
        logger.warning(f"Could not initialize StrategyRouter: {exc}")

    try:
        from src.integrations.github_ea_scheduler import get_scheduler, start_scheduler

        if start_scheduler():
            scheduler = get_scheduler()
            if scheduler:
                logger.info(f"GitHub EA scheduler started: {scheduler.get_status()}")
            else:
                logger.info("GitHub EA scheduler initialized")
        else:
            logger.warning("GitHub EA scheduler failed to start - check GITHUB_EA_REPO_URL env var")
    except Exception as exc:
        logger.warning(f"Could not start GitHub EA scheduler: {exc}")

    try:
        from scripts.schedule_lifecycle_check import LifecycleScheduler

        lifecycle_scheduler = LifecycleScheduler()
        lifecycle_scheduler.setup_schedule()
        lifecycle_scheduler.scheduler.start()
        app.state.lifecycle_scheduler = lifecycle_scheduler
        logger.info("Lifecycle scheduler started - daily check at 3:00 AM UTC")
    except Exception as exc:
        logger.warning(f"Could not start lifecycle scheduler: {exc}")

    try:
        from src.router.market_scanner import start_scanner_scheduler

        scanner_started = start_scanner_scheduler()
        if scanner_started:
            logger.info("Market scanner scheduler started - session-based opportunity scanning enabled")
        else:
            logger.warning("Market scanner scheduler failed to start")
    except Exception as exc:
        logger.warning(f"Could not start market scanner scheduler: {exc}")

    logger.info(
        "Endpoints mounted: /api/ide, /api/chat, /api/analytics, /api/settings, /api/trd, "
        "/api/router, /api/journal, /api/sessions, /api/v1/backtest, /api/paper-trading, "
        "/api/mcp, /api/agents, /api/workflows, /api/kill-switch, /api/hmm, /api/metrics, "
        "/api/brokers, /api/eas, /api/virtual-accounts, /api/agent-tools, /api/batch, /health"
    )

    try:
        from src.api.metrics_endpoints import ws_manager as metrics_ws_manager

        if metrics_ws_manager._broadcast_task is None:
            metrics_ws_manager._broadcast_task = asyncio.create_task(
                metrics_ws_manager.start_broadcasting(interval=1.0)
            )
            logger.info("Metrics WebSocket broadcast task started")
    except Exception as exc:
        logger.warning(f"Could not start metrics broadcast task: {exc}")

    try:
        from src.agents.streaming import init_stream_handler

        stream_handler = await init_stream_handler()
        app.state.stream_handler = stream_handler
        logger.info("Agent stream handler initialized for SSE")
    except Exception as exc:
        logger.warning(f"Could not initialize agent stream handler: {exc}")


async def shutdown_event(app) -> None:
    """Run API shutdown routines."""
    logger.info("QuantMind API Server shutting down...")

    try:
        if hasattr(app.state, "lifecycle_scheduler"):
            lifecycle_scheduler = app.state.lifecycle_scheduler
            if lifecycle_scheduler and hasattr(lifecycle_scheduler, "scheduler"):
                lifecycle_scheduler.scheduler.shutdown()
                logger.info("Lifecycle scheduler stopped")
    except Exception as exc:
        logger.error(f"Error stopping lifecycle scheduler: {exc}")

    try:
        from src.router.market_scanner import stop_scanner_scheduler

        stop_scanner_scheduler()
        logger.info("Market scanner scheduler stopped")
    except Exception as exc:
        logger.error(f"Error stopping market scanner scheduler: {exc}")

    try:
        if hasattr(app.state, "stream_handler"):
            from src.agents.streaming import close_stream_handler

            await close_stream_handler()
            logger.info("Agent stream handler stopped")
    except Exception as exc:
        logger.error(f"Error stopping agent stream handler: {exc}")
