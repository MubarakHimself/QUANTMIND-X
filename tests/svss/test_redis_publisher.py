"""
Unit Tests for Redis Publisher

Tests SVSSPublisher without requiring actual Redis connection.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from svss.publishers.redis_publisher import SVSSPublisher
from svss.indicators.base import IndicatorResult


def create_mock_result(name="vwap", value=1.0850, symbol="EURUSD", session_id="test_session"):
    """Create a mock indicator result."""
    return IndicatorResult(
        name=name,
        value=value,
        timestamp=datetime.now(timezone.utc),
        session_id=session_id,
        metadata={"symbol": symbol}
    )


class TestSVSSPublisher:
    """Tests for SVSSPublisher."""

    def test_initialization(self):
        """Test publisher initialization."""
        publisher = SVSSPublisher(redis_url="redis://localhost:6379")

        assert publisher._redis_url == "redis://localhost:6379"
        assert publisher.is_connected is False

    def test_channel_name_format(self):
        """Test channel name follows correct format."""
        publisher = SVSSPublisher()

        channel = publisher._get_channel_name("EURUSD", "vwap")
        assert channel == "svss:eurusd:vwap"

        channel = publisher._get_channel_name("GBPUSD", "rvvol")
        assert channel == "svss:gbpusd:rvvol"

    def test_publish_single_result_not_connected(self):
        """Test publish returns False when not connected."""
        publisher = SVSSPublisher()

        result = create_mock_result()
        success = publisher.publish(result)

        assert success is False

    def test_connect_with_mocked_client(self):
        """Test successful Redis connection with mocked client."""
        publisher = SVSSPublisher()

        # Create a mock Redis client
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        # Manually set the connected state and client for testing
        publisher._redis_client = mock_redis
        publisher._connected = True

        assert publisher.is_connected is True

    def test_publish_single_result_connected_mock(self):
        """Test publishing when connected with mock client."""
        publisher = SVSSPublisher()

        # Create a mock Redis client
        mock_redis = MagicMock()
        mock_redis.publish.return_value = 1

        publisher._redis_client = mock_redis
        publisher._connected = True

        result = create_mock_result()
        success = publisher.publish(result)

        assert success is True
        mock_redis.publish.assert_called_once()

    def test_publish_all_results_mock(self):
        """Test publishing all indicators atomically with mock client."""
        publisher = SVSSPublisher()

        # Create mock Redis client with pipeline
        mock_pipe = MagicMock()
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        publisher._redis_client = mock_redis
        publisher._connected = True

        results = [
            create_mock_result(name="vwap", value=1.0850),
            create_mock_result(name="rvvol", value=1.25),
            create_mock_result(name="mfi", value=65.0),
            create_mock_result(name="volume_profile", value=1.0850),
        ]

        success = publisher.publish_all(results, symbol="EURUSD", session_id="test_session")

        assert success is True
        # Should publish to 4 channels
        assert mock_pipe.publish.call_count == 4
        mock_pipe.execute.assert_called_once()

    def test_disconnect(self):
        """Test disconnect closes connection."""
        publisher = SVSSPublisher()
        publisher._connected = True
        publisher._redis_client = MagicMock()

        publisher.disconnect()

        assert publisher.is_connected is False
        publisher._redis_client.close.assert_called_once()

    def test_publish_without_connection_logs_warning(self):
        """Test publish logs warning when not connected."""
        publisher = SVSSPublisher()
        publisher._connected = False

        result = create_mock_result()
        success = publisher.publish(result)

        assert success is False

    def test_publish_all_with_no_results(self):
        """Test publish_all with empty results list."""
        publisher = SVSSPublisher()

        mock_redis = MagicMock()
        publisher._redis_client = mock_redis
        publisher._connected = True

        success = publisher.publish_all([], symbol="EURUSD", session_id="test_session")

        assert success is False
