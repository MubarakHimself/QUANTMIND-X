"""
SVSS Configuration Module

Dataclasses for configuring the Shared Volume Session Service.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymbolConfig:
    """
    Configuration for a single symbol's SVSS processing.

    Attributes:
        symbol: Trading symbol (e.g., 'EURUSD')
        warm_storage_path: Path to warm storage DuckDB file for this symbol
        redis_url: Redis connection URL
        zmq_endpoint: MT5 ZMQ tick stream endpoint
        session_boundaries: List of GMT hours that define session open times
            (default: [8, 12, 13, 16, 21] for London, NY, lunch, etc.)
    """

    symbol: str
    warm_storage_path: str = "data/svss/warm_storage.db"
    redis_url: str = "redis://localhost:6379"
    zmq_endpoint: str = "tcp://localhost:5555"
    session_boundaries: list[int] = field(default_factory=lambda: [8, 12, 13, 16, 21])
    rolling_avg_sessions: int = 20  # Number of sessions for rolling average


@dataclass
class SVSSConfig:
    """
    Main configuration for the SVSS service.

    Attributes:
        symbols: List of symbol configurations to process
        default_redis_url: Default Redis URL if not specified per symbol
        default_zmq_endpoint: Default ZMQ endpoint if not specified per symbol
        warm_storage_type: Type of warm storage ('duckdb' or 'sqlite')
        log_level: Logging level
    """

    symbols: list[SymbolConfig] = field(default_factory=list)
    default_redis_url: str = "redis://localhost:6379"
    default_zmq_endpoint: str = "tcp://localhost:5555"
    warm_storage_type: str = "duckdb"
    log_level: str = "INFO"

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get configuration for a specific symbol."""
        for config in self.symbols:
            if config.symbol.upper() == symbol.upper():
                return config
        return None

    def add_symbol(self, symbol: str, **kwargs) -> SymbolConfig:
        """Add a symbol configuration."""
        config = SymbolConfig(
            symbol=symbol.upper(),
            redis_url=kwargs.get("redis_url", self.default_redis_url),
            zmq_endpoint=kwargs.get("zmq_endpoint", self.default_zmq_endpoint),
            warm_storage_path=kwargs.get(
                "warm_storage_path", f"data/svss/{symbol.lower()}_warm_storage.db"
            ),
            session_boundaries=kwargs.get(
                "session_boundaries", [8, 12, 13, 16, 21]
            ),
            rolling_avg_sessions=kwargs.get("rolling_avg_sessions", 20),
        )
        self.symbols.append(config)
        return config
