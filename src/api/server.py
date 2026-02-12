"""
Main FastAPI Server

Runs the QuantMind IDE backend API on port 8000.
Combines IDE endpoints, Chat endpoints, and Analytics endpoints.
"""

import sys
import os
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantmind.server")

# Fix path to ensure imports work from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.api.ide_endpoints import create_ide_api_app
    from src.api.analytics_endpoints import router as analytics_router
    from src.api.chat_endpoints import router as chat_router
    from src.api.settings_endpoints import router as settings_router
    from src.api.trd_endpoints import router as trd_router
    from src.api.router_endpoints import router as router_router
    from src.api.journal_endpoints import router as journal_router
    from src.api.session_endpoints import router as session_router
except ImportError as e:
    logger.error(f"Import Error: {e}")
    # Fallback/Debug info
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create main app
app = create_ide_api_app()
if app is None:
    logger.error("Failed to create FastAPI app. Is 'fastapi' installed?")
    sys.exit(1)

# Include additional routers
app.include_router(analytics_router)
app.include_router(chat_router)
app.include_router(settings_router)
app.include_router(trd_router)
app.include_router(router_router)
app.include_router(journal_router)
app.include_router(session_router)

@app.on_event("startup")
async def startup_event():
    logger.info("QuantMind API Server starting on port 8000...")
    logger.info("Endpoints mounted: /api/ide, /api/chat, /api/analytics, /api/settings, /api/trd, /api/router, /api/journal, /api/sessions")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
