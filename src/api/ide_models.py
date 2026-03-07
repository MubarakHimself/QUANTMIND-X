"""
QuantMind IDE Models

Pydantic models for IDE endpoints.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

try:
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
    BaseModel = PydanticBaseModel  # type: ignore
    Field = PydanticField  # type: ignore
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs) -> Any:  # type: ignore
        if len(args) > 0:
            return args[0]
        return kwargs.get('default')


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
STRATEGIES_DIR = DATA_DIR / "strategies"
ASSETS_DIR = DATA_DIR / "shared_assets"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SCRAPED_ARTICLES_DIR = DATA_DIR / "scraped_articles"


# =============================================================================
# Enums
# =============================================================================

class StrategyStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    PRIMAL = "primal"
    QUARANTINED = "quarantined"


# =============================================================================
# Strategy Models
# =============================================================================

class StrategyFolder(BaseModel):
    id: str
    name: str
    status: StrategyStatus = StrategyStatus.PENDING
    created_at: str
    has_video_ingest: bool = False
    has_trd: bool = False
    has_ea: bool = False
    has_backtest: bool = False


class StrategyDetail(BaseModel):
    id: str
    name: str
    status: StrategyStatus
    created_at: str
    video_ingest: Optional[Dict[str, Any]] = None
    trd: Optional[Dict[str, Any]] = None
    ea: Optional[Dict[str, Any]] = None
    backtests: List[Dict[str, Any]] = []


# =============================================================================
# Asset Models
# =============================================================================

class SharedAsset(BaseModel):
    id: str
    name: str
    type: str  # indicator, library, template
    path: str
    description: Optional[str] = None
    used_in: List[str] = []


# =============================================================================
# Knowledge Models
# =============================================================================

class KnowledgeItem(BaseModel):
    id: str
    name: str
    category: str  # articles, books, logs
    path: str
    size_bytes: int
    indexed: bool = False


# =============================================================================
# Video Ingest Models
# =============================================================================

class VideoIngestProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube URL to process")
    strategy_name: str = Field(default="video_ingest", description="Name for the strategy folder")
    is_playlist: bool = Field(default=False, description="Whether URL is a playlist")


class VideoIngestProcessResponse(BaseModel):
    job_id: str
    status: str
    strategy_folder: str


class VideoIngestAuthStatus(BaseModel):
    """Authentication status for video_ingest AI providers."""
    gemini: bool = Field(default=False, description="Gemini CLI authentication status")
    qwen: bool = Field(default=False, description="Qwen CLI authentication status")


# =============================================================================
# Trading Models
# =============================================================================

class BotControl(BaseModel):
    bot_id: str
    action: str  # pause, resume, quarantine, kill


# =============================================================================
# Database Export Models
# =============================================================================

class DatabaseExportRequest(BaseModel):
    """Request model for database export."""
    format: str = Field(default="csv", description="Export format: csv or json")
    limit: Optional[int] = Field(default=None, description="Maximum rows to export")


# =============================================================================
# MT5 Models
# =============================================================================

class MT5ScanRequest(BaseModel):
    """Request model for MT5 scan."""
    custom_paths: Optional[List[str]] = Field(default=None, description="Custom paths to scan")


class MT5LaunchRequest(BaseModel):
    """Request model for MT5 launch."""
    terminal_path: str = Field(..., description="Path to MT5 terminal executable")
    login: Optional[int] = Field(default=None, description="Account login number")
    password: Optional[str] = Field(default=None, description="Account password")
    server: Optional[str] = Field(default=None, description="Broker server name")


# =============================================================================
# Clone Bot Models
# =============================================================================

class CloneBotRequest(BaseModel):
    source_bot_id: str
    new_name: str
    config_overrides: Optional[Dict[str, Any]] = None


# =============================================================================
# Backtest Models
# =============================================================================

class BacktestRunRequest(BaseModel):
    """Request model for running a backtest."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (M1, M5, M15, M30, H1, H4, D1)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    variant: str = Field(
        default="vanilla",
        description="Backtest variant: vanilla, spiced, vanilla_full, spiced_full"
    )
    strategy_code: Optional[str] = Field(None, description="Python strategy code")
    initial_cash: Optional[float] = Field(10000.0, description="Initial cash balance")
    commission: Optional[float] = Field(0.001, description="Commission per trade")
    broker_id: Optional[str] = Field("icmarkets_raw", description="Broker ID for fee-aware Kelly")
    enable_ws_streaming: Optional[bool] = Field(True, description="Enable WebSocket streaming for real-time updates")
