"""
Unit Tests for SVSS Configuration

Tests SVSSConfig and SymbolConfig dataclasses.
"""

import pytest
from svss.config import SVSSConfig, SymbolConfig


class TestSymbolConfig:
    """Tests for SymbolConfig."""

    def test_initialization_with_defaults(self):
        """Test SymbolConfig initialization with default values."""
        config = SymbolConfig(symbol="EURUSD")

        assert config.symbol == "EURUSD"
        assert config.warm_storage_path == "data/svss/warm_storage.db"
        assert config.redis_url == "redis://localhost:6379"
        assert config.zmq_endpoint == "tcp://localhost:5555"
        assert config.session_boundaries == [8, 12, 13, 16, 21]
        assert config.rolling_avg_sessions == 20

    def test_initialization_with_custom_values(self):
        """Test SymbolConfig with custom values."""
        config = SymbolConfig(
            symbol="GBPUSD",
            warm_storage_path="/custom/path/gbpusd.db",
            redis_url="redis://custom:6379",
            zmq_endpoint="tcp://custom:5555",
            session_boundaries=[8, 12, 21],
            rolling_avg_sessions=30,
        )

        assert config.symbol == "GBPUSD"
        assert config.warm_storage_path == "/custom/path/gbpusd.db"
        assert config.redis_url == "redis://custom:6379"
        assert config.zmq_endpoint == "tcp://custom:5555"
        assert config.session_boundaries == [8, 12, 21]
        assert config.rolling_avg_sessions == 30


class TestSVSSConfig:
    """Tests for SVSSConfig."""

    def test_initialization_with_defaults(self):
        """Test SVSSConfig initialization with default values."""
        config = SVSSConfig()

        assert config.symbols == []
        assert config.default_redis_url == "redis://localhost:6379"
        assert config.default_zmq_endpoint == "tcp://localhost:5555"
        assert config.warm_storage_type == "duckdb"
        assert config.log_level == "INFO"

    def test_initialization_with_custom_values(self):
        """Test SVSSConfig with custom values."""
        config = SVSSConfig(
            default_redis_url="redis://custom:6379",
            default_zmq_endpoint="tcp://custom:5555",
            warm_storage_type="sqlite",
            log_level="DEBUG",
        )

        assert config.default_redis_url == "redis://custom:6379"
        assert config.default_zmq_endpoint == "tcp://custom:5555"
        assert config.warm_storage_type == "sqlite"
        assert config.log_level == "DEBUG"

    def test_add_symbol(self):
        """Test adding a symbol to configuration."""
        config = SVSSConfig()
        symbol_config = config.add_symbol("EURUSD")

        assert len(config.symbols) == 1
        assert symbol_config.symbol == "EURUSD"
        assert symbol_config.redis_url == "redis://localhost:6379"
        assert symbol_config.zmq_endpoint == "tcp://localhost:5555"

    def test_add_symbol_with_custom_redis(self):
        """Test adding a symbol with custom Redis URL."""
        config = SVSSConfig(default_redis_url="redis://default:6379")
        symbol_config = config.add_symbol("EURUSD", redis_url="redis://custom:6379")

        assert symbol_config.redis_url == "redis://custom:6379"

    def test_add_symbol_with_custom_zmq(self):
        """Test adding a symbol with custom ZMQ endpoint."""
        config = SVSSConfig(default_zmq_endpoint="tcp://default:5555")
        symbol_config = config.add_symbol("EURUSD", zmq_endpoint="tcp://custom:5555")

        assert symbol_config.zmq_endpoint == "tcp://custom:5555"

    def test_add_symbol_generates_warm_storage_path(self):
        """Test that add_symbol generates warm storage path if not provided."""
        config = SVSSConfig()
        symbol_config = config.add_symbol("GBPUSD")

        assert symbol_config.warm_storage_path == "data/svss/gbpusd_warm_storage.db"

    def test_get_symbol_config_exists(self):
        """Test getting existing symbol config."""
        config = SVSSConfig()
        config.add_symbol("EURUSD")
        config.add_symbol("GBPUSD")

        result = config.get_symbol_config("EURUSD")

        assert result is not None
        assert result.symbol == "EURUSD"

    def test_get_symbol_config_case_insensitive(self):
        """Test that get_symbol_config is case insensitive."""
        config = SVSSConfig()
        config.add_symbol("EURUSD")

        result = config.get_symbol_config("eurusd")

        assert result is not None
        assert result.symbol == "EURUSD"

    def test_get_symbol_config_not_found(self):
        """Test getting non-existent symbol config."""
        config = SVSSConfig()
        config.add_symbol("EURUSD")

        result = config.get_symbol_config("GBPUSD")

        assert result is None

    def test_add_multiple_symbols(self):
        """Test adding multiple symbols."""
        config = SVSSConfig()
        config.add_symbol("EURUSD")
        config.add_symbol("GBPUSD")
        config.add_symbol("USDJPY")

        assert len(config.symbols) == 3

    def test_symbol_symbol_uppercase_normalized(self):
        """Test that symbol is normalized to uppercase."""
        config = SVSSConfig()
        symbol_config = config.add_symbol("eurusd")

        assert symbol_config.symbol == "EURUSD"

    def test_add_symbol_with_custom_session_boundaries(self):
        """Test adding symbol with custom session boundaries."""
        config = SVSSConfig()
        symbol_config = config.add_symbol(
            "EURUSD",
            session_boundaries=[8, 12, 13, 16, 21, 22]
        )

        assert symbol_config.session_boundaries == [8, 12, 13, 16, 21, 22]

    def test_add_symbol_with_custom_rolling_sessions(self):
        """Test adding symbol with custom rolling sessions count."""
        config = SVSSConfig()
        symbol_config = config.add_symbol("EURUSD", rolling_avg_sessions=30)

        assert symbol_config.rolling_avg_sessions == 30
