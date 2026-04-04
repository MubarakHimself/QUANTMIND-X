"""
Shared models for QuantMind IDE endpoints.

This module contains all Pydantic models used across the IDE endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
    BaseModel = PydanticBaseModel  # type: ignore
    Field = PydanticField  # type: ignore
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback for when pydantic is not available
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs) -> Any:  # type: ignore
        """Fallback Field function that returns the default value or kwargs."""
        if len(args) > 0:
            return args[0]
        return kwargs.get('default')


class StrategyStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    PRIMAL = "primal"
    QUARANTINED = "quarantined"


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


class SharedAsset(BaseModel):
    id: str
    name: str
    type: str  # indicator, library, template
    category: str
    path: str
    description: Optional[str] = None
    size: Optional[int] = None
    used_in: List[str] = []
    created_at: Optional[str] = None


class KnowledgeItem(BaseModel):
    id: str
    title: str
    content: str
    category: str  # articles, patterns, strategies
    tags: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VideoIngestProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube URL to process")
    strategy_name: str = Field(default="video_ingest", description="Name for the strategy folder")
    is_playlist: bool = Field(default=False, description="Whether URL is a playlist")


class VideoIngestProcessResponse(BaseModel):
    job_id: str
    status: str
    strategy_folder: str
    job_ids: Optional[List[str]] = None


class VideoIngestAuthStatus(BaseModel):
    """Authentication status for video_ingest AI providers."""
    openrouter: bool = Field(default=False, description="OpenRouter authentication status")
    gemini: bool = Field(default=False, description="Gemini CLI authentication status")
    qwen: bool = Field(default=False, description="Qwen CLI authentication status")


class BotControl(BaseModel):
    bot_id: str
    action: str  # pause, resume, quarantine, kill


class DatabaseExportRequest(BaseModel):
    """Request model for database export."""
    format: str = Field(default="csv", description="Export format: csv or json")
    limit: Optional[int] = Field(default=None, description="Maximum rows to export")


class MT5ScanRequest(BaseModel):
    """Request model for MT5 scan."""
    custom_paths: Optional[List[str]] = Field(default=None, description="Custom paths to scan")


class MT5LaunchRequest(BaseModel):
    """Request model for MT5 launch."""
    terminal_path: Optional[str] = Field(default=None, description="Path to MT5 terminal executable")
    login: Optional[int] = Field(default=None, description="Account login number")
    password: Optional[str] = Field(default=None, description="Account password")
    server: Optional[str] = Field(default=None, description="Broker server name")


class CloneBotRequest(BaseModel):
    """Request model for bot cloning."""
    source_bot_id: str
    new_bot_name: str
