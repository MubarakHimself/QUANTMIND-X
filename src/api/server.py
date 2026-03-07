"""
Main FastAPI Server

Runs the QuantMind IDE backend API on port 8000.
Combines IDE endpoints, Chat endpoints, and Analytics endpoints.
"""

import sys
import os
import uvicorn
import logging
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
    from src.api.app_bootstrap import (
        configure_middleware,
        register_exception_handlers,
        register_routes,
        shutdown_event,
        startup_event,
    )
    from src.api.router_registry import load_router_registry
except ImportError as e:
    logger.error(f"Import Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create main app
app = create_ide_api_app()
if app is None:
    logger.error("Failed to create FastAPI app. Is 'fastapi' installed?")
    sys.exit(1)

# WebSocket endpoints are already registered in create_ide_api_app()
routers, optional_routers = load_router_registry()
configure_middleware(app)
register_exception_handlers(app)
register_routes(app, routers, optional_routers)


@app.on_event("startup")
async def on_startup():
    await startup_event(app)

@app.on_event("shutdown")
async def on_shutdown():
    await shutdown_event(app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
