"""
QuantMind IDE API Endpoints - Modular Structure

This package provides a modular structure for IDE endpoints:
- SessionEndpoint: Session management (strategies, broker accounts, live trading)
- FileEndpoint: File operations (assets, knowledge hub)
- TerminalEndpoint: Terminal operations (video ingest)

For backward compatibility, all exports from the original ide_endpoints module
are also re-exported here.
"""

# Main endpoint classes
from .session import SessionEndpoint
from .file import FileEndpoint
from .terminal import TerminalEndpoint

# Import all models for backward compatibility
from .models import (
    StrategyStatus,
    StrategyFolder,
    StrategyDetail,
    SharedAsset,
    KnowledgeItem,
    VideoIngestProcessRequest,
    VideoIngestProcessResponse,
    VideoIngestAuthStatus,
    BotControl,
    DatabaseExportRequest,
    MT5ScanRequest,
    MT5LaunchRequest,
    CloneBotRequest,
)

# Import handlers for backward compatibility
from .session import (
    StrategyAPIHandler,
    BrokerAccountsAPIHandler,
    LiveTradingAPIHandler,
)

from .file import (
    AssetsAPIHandler,
    KnowledgeAPIHandler,
)

from .terminal import (
    VideoIngestAPIHandler,
)

# Configuration - re-export for backward compatibility
from pathlib import Path
import os

DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
STRATEGIES_DIR = DATA_DIR / "strategies"
ASSETS_DIR = DATA_DIR / "shared_assets"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SCRAPED_ARTICLES_DIR = DATA_DIR / "scraped_articles"

__all__ = [
    # Endpoint classes
    "SessionEndpoint",
    "FileEndpoint",
    "TerminalEndpoint",
    # Models
    "StrategyStatus",
    "StrategyFolder",
    "StrategyDetail",
    "SharedAsset",
    "KnowledgeItem",
    "VideoIngestProcessRequest",
    "VideoIngestProcessResponse",
    "VideoIngestAuthStatus",
    "BotControl",
    "DatabaseExportRequest",
    "MT5ScanRequest",
    "MT5LaunchRequest",
    "CloneBotRequest",
    # Handlers
    "StrategyAPIHandler",
    "BrokerAccountsAPIHandler",
    "LiveTradingAPIHandler",
    "AssetsAPIHandler",
    "KnowledgeAPIHandler",
    "VideoIngestAPIHandler",
    # Configuration
    "DATA_DIR",
    "STRATEGIES_DIR",
    "ASSETS_DIR",
    "KNOWLEDGE_DIR",
    "SCRAPED_ARTICLES_DIR",
]
