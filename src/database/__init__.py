"""
QuantMind Hybrid Core v7 - Database Layer
Provides SQLite database access for the hybrid core system.
"""

from .manager import DatabaseManager

# DuckDB connection for analytics
try:
    from .duckdb_connection import DuckDBConnection
    _duckdb_available = True
except ImportError:
    _duckdb_available = False

__all__ = [
    "DatabaseManager",
]

if _duckdb_available:
    __all__.extend([
        "DuckDBConnection"
    ])
