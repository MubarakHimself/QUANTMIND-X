"""
QuantMind Hybrid Core v7 - Database Layer
Provides SQLite and ChromaDB database access for the hybrid core system.
"""

from .manager import DatabaseManager

# Optional ChromaDB client (requires sentence-transformers)
try:
    from .chroma_client import (
        ChromaDBClient,
        SentenceTransformerEmbedding,
        get_chroma_client,
        init_collections
    )
    _chroma_available = True
except ImportError:
    _chroma_available = False

# DuckDB connection for analytics
try:
    from .duckdb_connection import DuckDBConnection
    _duckdb_available = True
except ImportError:
    _duckdb_available = False

__all__ = [
    "DatabaseManager",
]

if _chroma_available:
    __all__.extend([
        "ChromaDBClient",
        "SentenceTransformerEmbedding",
        "get_chroma_client",
        "init_collections"
    ])

if _duckdb_available:
    __all__.extend([
        "DuckDBConnection"
    ])
