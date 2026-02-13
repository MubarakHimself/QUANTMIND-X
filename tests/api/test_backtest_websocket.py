"""
Integration Tests for Backtest WebSocket Streaming.

Tests the complete WebSocket streaming pipeline for backtests including:
- WebSocket connection establishment and topic subscription
- Backtest start event broadcast with correct metadata
- Progress updates during backtest execution
- Log entry streaming with correct log levels
- Backtest completion event with final results
- Error event broadcasting when backtest fails
- Multiple concurrent backtest sessions with isolated streams

Validates: Real-Time Log Streaming Architecture
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List


# Mock WebSocket class for testing
class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages: List[str] = []
        self.accepted = False
        self.closed = False
        
    async def accept(self):
        self.accepted = True
        
    async def send_text(self, message: str):
        self.messages.append(message)
        
    async def send_json(self, data: Dict[str, Any]):
        self.messages.append(json.dumps(data))
        
    async def receive_text(self) -> str:
        # Return a subscription message by default
        await asyncio.sleep(0.01)
        return json.dumps({"action": "subscribe", "topic": "backtest"})
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all received messages as parsed JSON."""
        return [json.loads(msg) for msg in self.messages]
    
    def get_messages_of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        """Get all messages of a specific type."""
        return [msg for msg in self.get_messages() if msg.get("type") == msg_type]


class TestWebSocketBacktestSubscription:
    """Test WebSocket connection and subscription to backtest topic."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self):
        """
        Test that WebSocket connection is established correctly.
        
        Validates:
        - Connection is accepted
        - Connection is added to active connections
        - Connection count is tracked
        """
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        
        # Connect
        await manager.connect(websocket)
        
        assert websocket.accepted, "WebSocket should be accepted"
        assert websocket in manager.active_connections, "WebSocket should be in active connections"
        assert len(manager.active_connections) == 1, "Should have 1 active connection"
        
    @pytest.mark.asyncio
    async def test_websocket_subscription_to_backtest_topic(self):
        """
        Test subscription to backtest topic.
        
        Validates:
        - Subscription is registered
        - WebSocket receives subscription confirmation
        - Topic subscribers are tracked
        """
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        
        # Connect and subscribe
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        # Verify subscription
        assert "backtest" in manager.subscriptions, "backtest topic should exist"
        assert websocket in manager.subscriptions["backtest"], "WebSocket should be subscribed to backtest"
        
    @pytest.mark.asyncio
    async def test_websocket_disconnect_cleanup(self):
        """
        Test that disconnection cleans up subscriptions.
        
        Validates:
        - Connection removed from active connections
        - Connection removed from all topic subscriptions
        """
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        
        # Connect and subscribe
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        await manager.subscribe(websocket, "logs")
        
        # Disconnect
        manager.disconnect(websocket)
        
        # Verify cleanup
        assert websocket not in manager.active_connections, "WebSocket should be removed from active connections"
        assert websocket not in manager.subscriptions.get("backtest", set()), "WebSocket should be removed from backtest topic"
        assert websocket not in manager.subscriptions.get("logs", set()), "WebSocket should be removed from logs topic"


class TestBacktestStartEvent:
    """Test backtest start event broadcast."""
    
    @pytest.mark.asyncio
    async def test_backtest_start_event_broadcast(self):
        """
        Test that backtest start event is broadcast with correct metadata.
        
        Validates:
        - Event type is 'backtest_start'
        - Contains variant, symbol, timeframe, dates
        - Timestamp is included
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        # Set connection manager
        set_connection_manager(manager)
        
        # Create streamer and send start event
        streamer = BacktestProgressStreamer("test-backtest-001")
        await streamer.start(
            variant="Sentinel",
            symbol="EURUSD",
            timeframe="H1",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Verify message
        messages = websocket.get_messages_of_type("backtest_start")
        assert len(messages) == 1, "Should have 1 backtest_start message"
        
        data = messages[0]["data"]
        assert data["backtest_id"] == "test-backtest-001"
        assert data["variant"] == "Sentinel"
        assert data["symbol"] == "EURUSD"
        assert data["timeframe"] == "H1"
        assert data["start_date"] == "2023-01-01"
        assert data["end_date"] == "2023-12-31"
        assert "timestamp" in data
        
    @pytest.mark.asyncio
    async def test_backtest_start_event_only_to_subscribers(self):
        """
        Test that start event is only sent to backtest topic subscribers.
        
        Validates:
        - Subscribers receive the event
        - Non-subscribers do not receive the event
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        
        # Create two websockets - one subscribed, one not
        ws_subscribed = MockWebSocket()
        ws_not_subscribed = MockWebSocket()
        
        await manager.connect(ws_subscribed)
        await manager.connect(ws_not_subscribed)
        
        await manager.subscribe(ws_subscribed, "backtest")
        # ws_not_subscribed is not subscribed to any topic
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-002")
        await streamer.start(
            variant="Vanilla",
            symbol="GBPUSD",
            timeframe="M15",
            start_date="2023-06-01",
            end_date="2023-06-30"
        )
        
        # Verify only subscriber received message
        subscribed_messages = ws_subscribed.get_messages_of_type("backtest_start")
        not_subscribed_messages = ws_not_subscribed.get_messages_of_type("backtest_start")
        
        assert len(subscribed_messages) == 1, "Subscriber should receive message"
        assert len(not_subscribed_messages) == 0, "Non-subscriber should not receive message"


class TestBacktestProgressUpdates:
    """Test progress updates during backtest."""
    
    @pytest.mark.asyncio
    async def test_backtest_progress_updates(self):
        """
        Test that progress updates are sent with accurate data.
        
        Validates:
        - Progress percentage is correct
        - Bars processed count is accurate
        - Trades count is included
        - Current P&L is included
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-003")
        
        # Send multiple progress updates
        await streamer.update_progress(
            progress=25.0,
            status="Processing bars...",
            bars_processed=2500,
            total_bars=10000,
            trades_count=5,
            current_pnl=150.25
        )
        
        await streamer.update_progress(
            progress=50.0,
            status="Halfway complete",
            bars_processed=5000,
            total_bars=10000,
            trades_count=12,
            current_pnl=325.50
        )
        
        # Verify messages
        messages = websocket.get_messages_of_type("backtest_progress")
        assert len(messages) == 2, "Should have 2 progress messages"
        
        # Check first update
        data1 = messages[0]["data"]
        assert data1["progress"] == 25.0
        assert data1["bars_processed"] == 2500
        assert data1["total_bars"] == 10000
        assert data1["trades_count"] == 5
        assert data1["current_pnl"] == 150.25
        
        # Check second update
        data2 = messages[1]["data"]
        assert data2["progress"] == 50.0
        assert data2["bars_processed"] == 5000
        assert data2["trades_count"] == 12
        assert data2["current_pnl"] == 325.50
        
    @pytest.mark.asyncio
    async def test_progress_update_every_100_bars(self):
        """
        Test that progress updates are sent at appropriate intervals.
        
        Validates:
        - Updates are sent for every 100 bars processed
        - Progress percentage is calculated correctly
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-004")
        
        # Simulate progress updates at 100-bar intervals
        total_bars = 1000
        for bars_processed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            progress = (bars_processed / total_bars) * 100
            await streamer.update_progress(
                progress=progress,
                status=f"Processed {bars_processed} bars",
                bars_processed=bars_processed,
                total_bars=total_bars
            )
        
        # Verify all updates received
        messages = websocket.get_messages_of_type("backtest_progress")
        assert len(messages) == 10, "Should have 10 progress messages"
        
        # Verify progress values
        for i, msg in enumerate(messages):
            expected_progress = ((i + 1) * 100 / total_bars) * 100
            assert abs(msg["data"]["progress"] - expected_progress) < 0.01


class TestBacktestLogStreaming:
    """Test log entry streaming during backtest."""
    
    @pytest.mark.asyncio
    async def test_log_entry_streaming_info_level(self):
        """
        Test INFO level log streaming.
        
        Validates:
        - Log entry is broadcast with correct level
        - Message content is preserved
        - Timestamp is included
        """
        from src.api.ws_logger import WebSocketLogHandler, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        import logging
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        # Create handler and emit log
        handler = WebSocketLogHandler(topic="backtest", backtest_id="test-001")
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test info message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Allow async task to complete
        await asyncio.sleep(0.1)
        
        # Verify message
        messages = websocket.get_messages_of_type("log_entry")
        assert len(messages) >= 1, "Should have at least 1 log entry"
        
        data = messages[0]["data"]
        assert data["level"] == "INFO"
        assert data["message"] == "Test info message"
        assert data["backtest_id"] == "test-001"
        assert "timestamp" in data
        
    @pytest.mark.asyncio
    async def test_log_entry_streaming_warning_level(self):
        """
        Test WARNING level log streaming.
        """
        from src.api.ws_logger import WebSocketLogHandler, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        import logging
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        handler = WebSocketLogHandler(topic="backtest", backtest_id="test-002")
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=20,
            msg="Test warning message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        await asyncio.sleep(0.1)
        
        messages = websocket.get_messages_of_type("log_entry")
        assert len(messages) >= 1
        
        data = messages[0]["data"]
        assert data["level"] == "WARNING"
        assert data["message"] == "Test warning message"
        
    @pytest.mark.asyncio
    async def test_log_entry_streaming_error_level(self):
        """
        Test ERROR level log streaming.
        """
        from src.api.ws_logger import WebSocketLogHandler, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        import logging
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        handler = WebSocketLogHandler(topic="backtest", backtest_id="test-003")
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=30,
            msg="Test error message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        await asyncio.sleep(0.1)
        
        messages = websocket.get_messages_of_type("log_entry")
        assert len(messages) >= 1
        
        data = messages[0]["data"]
        assert data["level"] == "ERROR"
        assert data["message"] == "Test error message"


class TestBacktestCompletionEvent:
    """Test backtest completion event."""
    
    @pytest.mark.asyncio
    async def test_backtest_completion_event(self):
        """
        Test that completion event is broadcast with final results.
        
        Validates:
        - Event type is 'backtest_complete'
        - Contains balance, trades, win rate, Sharpe, drawdown, return %
        - Duration is calculated
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-005")
        
        # Start backtest (sets start time)
        await streamer.start(
            variant="Sentinel",
            symbol="EURUSD",
            timeframe="H1",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Complete backtest
        await streamer.complete(
            final_balance=12500.00,
            total_trades=45,
            win_rate=62.5,
            sharpe_ratio=1.85,
            drawdown=8.5,
            return_pct=25.0
        )
        
        # Verify completion message
        messages = websocket.get_messages_of_type("backtest_complete")
        assert len(messages) == 1, "Should have 1 completion message"
        
        data = messages[0]["data"]
        assert data["backtest_id"] == "test-backtest-005"
        assert data["final_balance"] == 12500.00
        assert data["total_trades"] == 45
        assert data["win_rate"] == 62.5
        assert data["sharpe_ratio"] == 1.85
        assert data["drawdown"] == 8.5
        assert data["return_pct"] == 25.0
        assert "duration_seconds" in data
        assert data["duration_seconds"] >= 0.1, "Duration should be at least 0.1 seconds"
        
    @pytest.mark.asyncio
    async def test_completion_event_with_additional_results(self):
        """
        Test completion event with additional results dictionary.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-006")
        
        await streamer.start(
            variant="Sentinel",
            symbol="GBPUSD",
            timeframe="M15",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        additional_results = {
            "max_consecutive_wins": 8,
            "max_consecutive_losses": 3,
            "avg_trade_duration_hours": 4.5,
            "best_trade": 250.00,
            "worst_trade": -120.00
        }
        
        await streamer.complete(
            final_balance=15000.00,
            total_trades=60,
            results=additional_results
        )
        
        messages = websocket.get_messages_of_type("backtest_complete")
        data = messages[0]["data"]
        
        assert "results" in data
        assert data["results"]["max_consecutive_wins"] == 8
        assert data["results"]["max_consecutive_losses"] == 3
        assert data["results"]["avg_trade_duration_hours"] == 4.5


class TestBacktestErrorEvent:
    """Test error event broadcasting."""
    
    @pytest.mark.asyncio
    async def test_backtest_error_event(self):
        """
        Test that error event is broadcast when backtest fails.
        
        Validates:
        - Event type is 'backtest_error'
        - Contains error message and details
        - Timestamp is included
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-007")
        
        # Broadcast error
        await streamer.error(
            error_message="Strategy execution failed",
            error_details="ZeroDivisionError: division by zero at line 45"
        )
        
        # Verify error message
        messages = websocket.get_messages_of_type("backtest_error")
        assert len(messages) == 1, "Should have 1 error message"
        
        data = messages[0]["data"]
        assert data["backtest_id"] == "test-backtest-007"
        assert data["error"] == "Strategy execution failed"
        assert data["error_details"] == "ZeroDivisionError: division by zero at line 45"
        assert "timestamp" in data
        
    @pytest.mark.asyncio
    async def test_error_event_without_details(self):
        """
        Test error event without detailed information.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-008")
        
        await streamer.error(error_message="Unknown error occurred")
        
        messages = websocket.get_messages_of_type("backtest_error")
        data = messages[0]["data"]
        
        assert data["error"] == "Unknown error occurred"
        assert data["error_details"] is None


class TestMultipleConcurrentBacktests:
    """Test multiple concurrent backtest sessions."""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_backtests_isolated_streams(self):
        """
        Test that multiple concurrent backtests have isolated WebSocket streams.
        
        Validates:
        - Each backtest has unique ID
        - Messages are correctly attributed to their backtest
        - No cross-talk between backtest streams
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        # Create multiple streamers for different backtests
        streamer1 = BacktestProgressStreamer("backtest-A")
        streamer2 = BacktestProgressStreamer("backtest-B")
        streamer3 = BacktestProgressStreamer("backtest-C")
        
        # Start all backtests concurrently
        await asyncio.gather(
            streamer1.start("Sentinel", "EURUSD", "H1", "2023-01-01", "2023-06-30"),
            streamer2.start("Vanilla", "GBPUSD", "M15", "2023-01-01", "2023-06-30"),
            streamer3.start("Sentinel", "USDJPY", "H4", "2023-01-01", "2023-06-30")
        )
        
        # Send progress updates concurrently
        await asyncio.gather(
            streamer1.update_progress(50.0, "Halfway", bars_processed=500),
            streamer2.update_progress(75.0, "Three quarters", bars_processed=750),
            streamer3.update_progress(25.0, "Quarter", bars_processed=250)
        )
        
        # Complete all backtests concurrently
        await asyncio.gather(
            streamer1.complete(11000.0, 20, win_rate=60.0),
            streamer2.complete(10500.0, 15, win_rate=55.0),
            streamer3.complete(12000.0, 25, win_rate=65.0)
        )
        
        # Verify all messages received
        start_messages = websocket.get_messages_of_type("backtest_start")
        progress_messages = websocket.get_messages_of_type("backtest_progress")
        complete_messages = websocket.get_messages_of_type("backtest_complete")
        
        assert len(start_messages) == 3, "Should have 3 start messages"
        assert len(progress_messages) == 3, "Should have 3 progress messages"
        assert len(complete_messages) == 3, "Should have 3 complete messages"
        
        # Verify each backtest's messages are correctly attributed
        backtest_ids_start = {msg["data"]["backtest_id"] for msg in start_messages}
        backtest_ids_complete = {msg["data"]["backtest_id"] for msg in complete_messages}
        
        assert backtest_ids_start == {"backtest-A", "backtest-B", "backtest-C"}
        assert backtest_ids_complete == {"backtest-A", "backtest-B", "backtest-C"}
        
    @pytest.mark.asyncio
    async def test_concurrent_backtests_different_subscribers(self):
        """
        Test multiple backtests with different subscribers.
        
        Validates:
        - Each subscriber receives only their subscribed content
        - Multiple WebSocket connections work correctly
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        
        # Create multiple websockets
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.connect(ws1)
        await manager.connect(ws2)
        
        await manager.subscribe(ws1, "backtest")
        await manager.subscribe(ws2, "logs")  # Different topic
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("test-backtest-009")
        await streamer.start("Sentinel", "EURUSD", "H1", "2023-01-01", "2023-12-31")
        
        # ws1 should receive backtest messages, ws2 should not
        ws1_messages = ws1.get_messages_of_type("backtest_start")
        ws2_messages = ws2.get_messages_of_type("backtest_start")
        
        assert len(ws1_messages) == 1, "ws1 should receive backtest message"
        assert len(ws2_messages) == 0, "ws2 should not receive backtest message"


class TestWebSocketBroadcastPerformance:
    """Test WebSocket broadcast performance."""
    
    @pytest.mark.asyncio
    async def test_broadcast_latency_under_100ms(self):
        """
        Test that WebSocket message latency is under 100ms.
        
        Validates:
        - Messages are broadcast within 100ms
        - Performance is acceptable for real-time updates
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        import time
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("perf-test-001")
        
        # Measure broadcast latency
        start_time = time.time()
        await streamer.start("Sentinel", "EURUSD", "H1", "2023-01-01", "2023-12-31")
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        assert latency < 100, f"Broadcast latency {latency:.2f}ms exceeds 100ms threshold"
        
    @pytest.mark.asyncio
    async def test_high_frequency_message_handling(self):
        """
        Test handling of high-frequency messages.
        
        Validates:
        - System handles rapid message bursts
        - No message loss under normal conditions
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        from src.api.websocket_endpoints import ConnectionManager
        
        manager = ConnectionManager()
        websocket = MockWebSocket()
        await manager.connect(websocket)
        await manager.subscribe(websocket, "backtest")
        
        set_connection_manager(manager)
        
        streamer = BacktestProgressStreamer("perf-test-002")
        
        # Send 100 rapid progress updates
        num_updates = 100
        for i in range(num_updates):
            await streamer.update_progress(
                progress=i,
                status=f"Update {i}",
                bars_processed=i * 100
            )
        
        # Verify all messages received
        messages = websocket.get_messages_of_type("backtest_progress")
        assert len(messages) == num_updates, f"Expected {num_updates} messages, got {len(messages)}"
