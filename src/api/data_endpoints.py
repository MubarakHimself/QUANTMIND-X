"""
Data Management API Endpoints

Task Group 1: Data Manager Implementation

Provides REST API endpoints for:
- POST /api/v1/data/upload - Upload CSV historical data
- GET /api/v1/data/status - Check cache status
"""

from typing import Optional
from pathlib import Path
from datetime import datetime, timezone
import logging

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class DataUploadRequest(BaseModel):
    """Request model for CSV data upload."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.upper().strip()

    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe."""
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        v_upper = v.upper()
        if v_upper not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
        return v_upper


class DataUploadResponse(BaseModel):
    """Response model for CSV data upload."""
    success: bool = Field(..., description="Whether the upload was successful")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    rows_imported: int = Field(..., description="Number of rows imported")
    message: str = Field(..., description="Status message")
    cached_path: Optional[str] = Field(None, description="Path to cached Parquet file")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DataStatusResponse(BaseModel):
    """Response model for data cache status."""
    cache_dir: str = Field(..., description="Cache directory path")
    total_symbols: int = Field(..., description="Number of symbols in cache")
    total_timeframes: int = Field(..., description="Total timeframe combinations")
    total_bars: int = Field(..., description="Total number of bars cached")
    average_quality_score: float = Field(..., description="Average data quality score")
    data_sources: dict = Field(..., description="Count of data by source")
    symbols: dict = Field(..., description="Per-symbol statistics")
    mt5_enabled: bool = Field(..., description="MT5 data fetching enabled")
    mt5_connected: bool = Field(..., description="MT5 connection status")
    api_enabled: bool = Field(..., description="API data fetching enabled")
    parquet_available: bool = Field(..., description="Parquet format available")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# API Handler Classes
# =============================================================================

class DataEndpoints:
    """Data management API endpoint handlers.

    This class provides request handlers for data upload and status endpoints.
    It integrates with the DataManager class for actual data operations.

    Example:
        >>> from src.data.data_manager import DataManager, MQL5Timeframe
        >>> manager = DataManager()
        >>> endpoints = DataEndpoints(manager)
        >>>
        >>> # Handle upload request
        >>> response = endpoints.handle_upload(
        ...     symbol="EURUSD",
        ...     timeframe="H1",
        ...     csv_file_path=Path("data.csv")
        ... )
        >>>
        >>> # Get status
        >>> status = endpoints.get_status()
    """

    def __init__(self, data_manager):
        """Initialize data endpoints.

        Args:
            data_manager: DataManager instance for data operations
        """
        self.data_manager = data_manager

    def handle_upload(
        self,
        symbol: str,
        timeframe: str,
        csv_file_path: Path,
        validate_request: bool = True
    ) -> DataUploadResponse:
        """Handle CSV data upload request.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (M1, M5, H1, etc.)
            csv_file_path: Path to uploaded CSV file
            validate_request: Whether to validate request parameters

        Returns:
            DataUploadResponse with upload result
        """
        try:
            # Validate request if enabled
            if validate_request:
                request = DataUploadRequest(symbol=symbol, timeframe=timeframe)
                symbol = request.symbol
                timeframe = request.timeframe

            # Convert timeframe string to MQL5 constant
            from src.data.data_manager import MQL5Timeframe
            timeframe_const = MQL5Timeframe.from_string(timeframe)

            # Upload CSV
            result = self.data_manager.upload_csv(symbol, timeframe_const, csv_file_path)

            return DataUploadResponse(
                success=result.success,
                symbol=result.symbol,
                timeframe=result.timeframe,
                rows_imported=result.rows_imported,
                message=result.message,
                cached_path=str(result.cached_path) if result.cached_path else None
            )

        except ValueError as e:
            logger.error(f"Upload validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return DataUploadResponse(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                rows_imported=0,
                message=f"Upload failed: {str(e)}"
            )

    def get_status(self) -> DataStatusResponse:
        """Get data cache status.

        Returns:
            DataStatusResponse with cache statistics
        """
        try:
            status = self.data_manager.get_cache_status()

            return DataStatusResponse(**status)

        except Exception as e:
            logger.error(f"Status error: {e}")
            raise

    def refresh_data(self, symbol: str, timeframe: str) -> dict:
        """Force refresh cached data from live sources.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string

        Returns:
            Dictionary with refresh result
        """
        try:
            from src.data.data_manager import MQL5Timeframe
            timeframe_const = MQL5Timeframe.from_string(timeframe)

            success = self.data_manager.refresh_cache(symbol, timeframe_const)

            return {
                "success": success,
                "symbol": symbol,
                "timeframe": timeframe,
                "message": "Cache refreshed successfully" if success else "Cache refresh failed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Refresh error: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "timeframe": timeframe,
                "message": f"Refresh failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# =============================================================================
# Helper Functions for FastAPI/Flask Integration
# =============================================================================

def create_upload_endpoint(data_manager):
    """Create a FastAPI/Flask compatible upload endpoint handler.

    Args:
        data_manager: DataManager instance

    Returns:
        Async function for handling upload requests
    """
    endpoints = DataEndpoints(data_manager)

    async def upload_handler(
        symbol: str,
        timeframe: str,
        csv_content: bytes,
        filename: str
    ) -> DataUploadResponse:
        """Handle file upload and processing.

        Args:
            symbol: Trading symbol from request
            timeframe: Timeframe from request
            csv_content: Uploaded file content
            filename: Original filename

        Returns:
            DataUploadResponse
        """
        import tempfile
        import os

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            return endpoints.handle_upload(symbol, timeframe, temp_path)
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    return upload_handler


def create_status_endpoint(data_manager):
    """Create a FastAPI/Flask compatible status endpoint handler.

    Args:
        data_manager: DataManager instance

    Returns:
        Function for handling status requests
    """
    endpoints = DataEndpoints(data_manager)

    def status_handler() -> DataStatusResponse:
        """Handle status request.

        Returns:
            DataStatusResponse
        """
        return endpoints.get_status()

    return status_handler


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'DataUploadRequest',
    'DataUploadResponse',
    'DataStatusResponse',
    'ErrorResponse',
    'DataEndpoints',
    'create_upload_endpoint',
    'create_status_endpoint',
]
