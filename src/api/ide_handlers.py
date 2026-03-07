"""
QuantMind IDE Handlers

Business logic handlers for IDE endpoints.

This module provides backward compatibility by re-exporting handlers
from the modular ide_handlers_*.py files.
"""

# Re-export handlers from modular files for backward compatibility
from src.api.ide_handlers_strategy import StrategyAPIHandler
from src.api.ide_handlers_assets import AssetsAPIHandler
from src.api.ide_handlers_knowledge import KnowledgeAPIHandler
from src.api.ide_handlers_video_ingest import VideoIngestAPIHandler
from src.api.ide_handlers_broker import BrokerAccountsAPIHandler
from src.api.ide_handlers_trading import LiveTradingAPIHandler

__all__ = [
    "StrategyAPIHandler",
    "AssetsAPIHandler",
    "KnowledgeAPIHandler",
    "VideoIngestAPIHandler",
    "BrokerAccountsAPIHandler",
    "LiveTradingAPIHandler",
]
