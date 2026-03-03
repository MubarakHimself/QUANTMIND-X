"""
Paper Trading API Module.

Modular structure for paper trading endpoints:
- models: Pydantic models for requests/responses
- deploy: Deployment endpoints
- agents: Agent management endpoints
- promotion: Promotion endpoints
- routes: FastAPI router combining all endpoints
"""

from . import models
from . import deploy as deployment
from . import agents
from . import promotion
from .routes import router

__all__ = [
    'models',
    'deployment',
    'agents',
    'promotion',
    'router',
]
