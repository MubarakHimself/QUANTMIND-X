"""
Data Module

Provides data management, caching, and fetching capabilities for the QuantMindX
trading system.
"""

from .data_manager import (
    DataManager,
    DataSource,
    DataQualityReport,
    CacheMetadata,
    CSVUploadResult,
    MQL5Timeframe,
)

__all__ = [
    'DataManager',
    'DataSource',
    'DataQualityReport',
    'CacheMetadata',
    'CSVUploadResult',
    'MQL5Timeframe',
]
