"""
Trading API Module.

Modular structure for trading endpoints:
- models: Pydantic models for requests/responses
- backtest: Backtest API handler
- data: Data management API handler
- control: Trading control API handler
- broker: Broker connection handlers
- routes: FastAPI router combining all endpoints
"""

from . import models
from . import backtest
from . import data
from . import control
from . import broker
from .routes import router

__all__ = [
    'models',
    'backtest',
    'data',
    'control',
    'broker',
    'router',
]
