"""
QuantMind IDE API Endpoints

Provides endpoints for the UI shell to connect to backend services:
- Strategy folders (VideoIngest workspace)
- Shared assets library
- Knowledge hub
- VideoIngest processing
- Live trading control
- Agent chat
- Database export
- MT5 scanner and launcher

This module serves as the main entry point that aggregates all IDE endpoints.
For modular code, see individual modules in ide_*.py files.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.api.ide_models import DATA_DIR

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CORS_ORIGINS = [
    # Tauri local development
    "tauri://localhost",
    "tauri://127.0.0.1",
    # Vite/React development servers
    "http://localhost:1420",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5174",
    "http://localhost:4173",
    # IP-based local development
    "http://127.0.0.1:1420",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:4173",
    "http://127.0.0.1:4174",
    # Production URLs
    "https://app.quantmindx.com",
    "https://www.quantmindx.com",
    "https://quantmindx.com",
    # Allow local network access for development
    "http://192.168.1.100:1420",
    "http://192.168.1.100:5173",
    "http://192.168.1.100:3000",
    "http://192.168.1.100:3001",
]


def create_ide_api_app():
    """Create FastAPI app with all IDE endpoints."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.warning("FastAPI not available")
        return None

    # Import Routers
    try:
        from src.api.chat_endpoints import router as chat_router
    except ImportError:
        logger.warning("Chat endpoints not available")
        chat_router = None

    app = FastAPI(
        title="QuantMind IDE API",
        description="API for QuantMind IDE frontend",
        version="1.0.0"
    )

    # CORS for Tauri - configurable via environment variable
    # Default: allow common development origins
    import os
    cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if cors_origins_env:
        # Parse comma-separated origins from environment
        cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
        if "*" in cors_origins:
            logger.warning(
                "CORS_ALLOWED_ORIGINS contains '*', which is invalid with allow_credentials=True. "
                "Falling back to explicit development/production origins."
            )
            cors_origins = DEFAULT_CORS_ORIGINS
    else:
        # Default allowed origins for development and production
        cors_origins = DEFAULT_CORS_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Register WebSocket endpoints
    try:
        from src.api.websocket_endpoints import create_websocket_endpoints
        if create_websocket_endpoints:
            create_websocket_endpoints(app)
    except ImportError:
        logger.warning("WebSocket endpoints not available")

    # Register Routers from chat_endpoints
    if chat_router:
        app.include_router(chat_router)

    # Register IDE endpoint routers
    _register_ide_routers(app)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "QuantMind IDE API"}

    return app


def _normalized_node_role() -> str:
    """Resolve NODE_ROLE with legacy aliases normalized."""
    import os

    role = os.getenv("NODE_ROLE", "local").strip().lower()
    return {"cloudzy": "node_trading", "contabo": "node_backend"}.get(role, role)


def _register_ide_routers(app: "FastAPI"):
    """Register all IDE endpoint routers."""
    node_role = _normalized_node_role()

    # Import routers from modular files
    try:
        from src.api.ide_strategies import router as strategies_router
        app.include_router(strategies_router)
    except ImportError as e:
        logger.warning(f"Strategy endpoints not available: {e}")

    try:
        from src.api.ide_assets import router as assets_router
        app.include_router(assets_router)
    except ImportError as e:
        logger.warning(f"Assets endpoints not available: {e}")

    try:
        from src.api.ide_knowledge import router as knowledge_router
        from src.api.ide_knowledge import knowledge_upload_router
        app.include_router(knowledge_router)
        app.include_router(knowledge_upload_router)
    except ImportError as e:
        logger.warning(f"Knowledge endpoints not available: {e}")

    try:
        from src.api.ide_video_ingest import router as video_ingest_router
        app.include_router(video_ingest_router)
    except ImportError as e:
        logger.warning(f"Video ingest endpoints not available: {e}")

    if node_role in {"local", "node_trading"}:
        try:
            from src.api.ide_trading import broker_router, bots_router, bots_control_router
            app.include_router(broker_router)
            app.include_router(bots_router)
            app.include_router(bots_control_router)
        except ImportError as e:
            logger.warning(f"Trading endpoints not available: {e}")

    try:
        from src.api.ide_ea import router as ea_router
        app.include_router(ea_router)
    except ImportError as e:
        logger.warning(f"EA endpoints not available: {e}")

    if node_role in {"local", "node_trading"}:
        try:
            from src.api.ide_mt5 import router as mt5_router
            app.include_router(mt5_router)
        except ImportError as e:
            logger.warning(f"MT5 endpoints not available: {e}")

    try:
        from src.api.ide_backtest import router as backtest_router
        app.include_router(backtest_router)
    except ImportError as e:
        logger.warning(f"Backtest endpoints not available: {e}")

    try:
        from src.api.ide_timeframes import router as timeframes_router
        app.include_router(timeframes_router)
    except ImportError as e:
        logger.warning(f"Timeframes endpoints not available: {e}")

    try:
        from src.api.ide_chat import router as chat_router
        app.include_router(chat_router)
    except ImportError as e:
        logger.warning(f"Chat endpoints not available: {e}")

    try:
        from src.api.ide_files import router as files_router
        app.include_router(files_router)
    except ImportError as e:
        logger.warning(f"Files endpoints not available: {e}")


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    app = create_ide_api_app()
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8000)
