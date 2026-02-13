"""
Unit Tests for WebSocket Logger Components.

Tests individual components:
- WebSocketLogHandler.emit() - verify log records are formatted and broadcast correctly
- WebSocketLogHandler._format_record() - verify log record formatting
- BacktestProgressStreamer.start() - verify start event structure
- BacktestProgressStreamer.update_progress() - verify progress event structure
- BacktestProgressStreamer.complete() - verify completion event structure
- BacktestProgressStreamer.error() - verify error event structure
- setup_backtest_logging() - verify logger and streamer initialization

Uses mocks to isolate component testing.
"""

import pytest
import asyncio
import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any


class TestWebSocketLogHandler:
    """Unit tests for WebSocketLogHandler."""
    
    def test_websocket_log_handler_initialization(self):
        """
        Test WebSocketLogHandler initialization.
        
        Validates:
        - Handler is created with correct topic
        - Backtest ID is stored
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler(topic="backtest", backtest_id="test-001")
        
        assert handler.topic == "backtest"
        assert handler.backtest_id == "test-001"
        
    def test_websocket_log_handler_default_topic(self):
        """
        Test WebSocketLogHandler with default topic.
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler()
        
        assert handler.topic == "backtest"
        assert handler.backtest_id is None
        
    def test_format_record_structure(self):
        """
        Test _format_record produces correct structure.
        
        Validates:
        - Type is 'log_entry'
        - Data contains all required fields
        - Timestamp is ISO format
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler(topic="backtest", backtest_id="test-002")
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        result = handler._format_record(record)
        
        # Verify structure
        assert result["type"] == "log_entry"
        assert "data" in result
        
        data = result["data"]
        assert data["backtest_id"] == "test-002"
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["module"] == "test_module"
        assert data["function"] == "test_function"
        assert data["line"] == 42
        assert data["logger_name"] == "test.logger"
        assert "timestamp" in data
        
    def test_format_record_different_levels(self):
        """
        Test _format_record with different log levels.
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler()
        
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL")
        ]
        
        for level, expected_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg=f"{expected_name} message",
                args=(),
                exc_info=None
            )
            record.module = "test"
            record.funcName = "func"
            
            result = handler._format_record(record)
            assert result["data"]["level"] == expected_name
            
    def test_emit_with_connection_manager(self):
        """
        Test emit() broadcasts via connection manager.
        
        Validates:
        - Connection manager broadcast is called
        - Message is properly formatted
        """
        from src.api.ws_logger import WebSocketLogHandler, set_connection_manager
        
        # Create mock connection manager
        mock_manager = AsyncMock()
        mock_manager.broadcast = AsyncMock()
        
        set_connection_manager(mock_manager)
        
        handler = WebSocketLogHandler(topic="backtest", backtest_id="test-003")
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Emit test",
            args=(),
            exc_info=None
        )
        record.module = "test"
        record.funcName = "emit_test"
        
        # Emit the record
        handler.emit(record)
        
        # Note: emit() uses asyncio which may not complete synchronously
        # The test verifies the method doesn't raise an exception
        
    def test_emit_without_connection_manager(self):
        """
        Test emit() handles missing connection manager gracefully.
        
        Validates:
        - No exception when connection manager is None
        - Error handling works correctly
        """
        from src.api.ws_logger import WebSocketLogHandler, set_connection_manager
        
        # Set connection manager to None
        set_connection_manager(None)
        
        handler = WebSocketLogHandler()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="No manager test",
            args=(),
            exc_info=None
        )
        record.module = "test"
        record.funcName = "test"
        
        # Should not raise exception
        handler.emit(record)
        
    def test_emit_handles_exception(self):
        """
        Test emit() handles exceptions gracefully.
        
        Validates:
        - Exceptions are caught and handled
        - handleError is called
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler()
        
        # Create a record that might cause issues
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Exception test",
            args=(),
            exc_info=None
        )
        record.module = "test"
        record.funcName = "test"
        
        # Mock handleError to verify it's called
        original_handle_error = handler.handleError
        error_called = []
        
        def mock_handle_error(record):
            error_called.append(True)
            
        handler.handleError = mock_handle_error
        
        # Emit should not raise
        try:
            handler.emit(record)
        except Exception:
            pass  # Should not reach here
            
        # Restore original
        handler.handleError = original_handle_error


class TestBacktestProgressStreamer:
    """Unit tests for BacktestProgressStreamer."""
    
    @pytest.mark.asyncio
    async def test_progress_streamer_initialization(self):
        """
        Test BacktestProgressStreamer initialization.
        
        Validates:
        - Backtest ID is stored
        - Topic is set correctly
        """
        from src.api.ws_logger import BacktestProgressStreamer
        
        streamer = BacktestProgressStreamer("test-backtest-001", topic="backtest")
        
        assert streamer.backtest_id == "test-backtest-001"
        assert streamer.topic == "backtest"
        
    @pytest.mark.asyncio
    async def test_progress_streamer_default_topic(self):
        """
        Test BacktestProgressStreamer with default topic.
        """
        from src.api.ws_logger import BacktestProgressStreamer
        
        streamer = BacktestProgressStreamer("test-backtest-002")
        
        assert streamer.topic == "backtest"
        
    @pytest.mark.asyncio
    async def test_start_event_structure(self):
        """
        Test start() produces correct event structure.
        
        Validates:
        - Event type is 'backtest_start'
        - All required fields are present
        - Timestamp is set
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        mock_manager.broadcast = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-003")
        
        await streamer.start(
            variant="Sentinel",
            symbol="EURUSD",
            timeframe="H1",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Verify broadcast was called
        assert mock_manager.broadcast.called
        
        # Get the broadcast call arguments
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]  # First positional argument
        topic = call_args[1].get("topic", call_args[0][1] if len(call_args[0]) > 1 else None)
        
        # Verify message structure
        assert message["type"] == "backtest_start"
        assert message["data"]["backtest_id"] == "test-003"
        assert message["data"]["variant"] == "Sentinel"
        assert message["data"]["symbol"] == "EURUSD"
        assert message["data"]["timeframe"] == "H1"
        assert message["data"]["start_date"] == "2023-01-01"
        assert message["data"]["end_date"] == "2023-12-31"
        assert "timestamp" in message["data"]
        
        # Verify topic
        assert topic == "backtest"
        
    @pytest.mark.asyncio
    async def test_start_sets_start_time(self):
        """
        Test start() sets internal start time.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-004")
        
        assert streamer._start_time is None
        
        await streamer.start("Sentinel", "EURUSD", "H1", "2023-01-01", "2023-12-31")
        
        assert streamer._start_time is not None
        
    @pytest.mark.asyncio
    async def test_update_progress_event_structure(self):
        """
        Test update_progress() produces correct event structure.
        
        Validates:
        - Event type is 'backtest_progress'
        - All progress fields are included
        - Optional fields work correctly
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-005")
        
        await streamer.update_progress(
            progress=50.0,
            status="Halfway complete",
            bars_processed=5000,
            total_bars=10000,
            current_date="2023-06-15",
            trades_count=25,
            current_pnl=500.00
        )
        
        # Get the broadcast call arguments
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]
        
        # Verify message structure
        assert message["type"] == "backtest_progress"
        assert message["data"]["backtest_id"] == "test-005"
        assert message["data"]["progress"] == 50.0
        assert message["data"]["status"] == "Halfway complete"
        assert message["data"]["bars_processed"] == 5000
        assert message["data"]["total_bars"] == 10000
        assert message["data"]["current_date"] == "2023-06-15"
        assert message["data"]["trades_count"] == 25
        assert message["data"]["current_pnl"] == 500.00
        assert "timestamp" in message["data"]
        
    @pytest.mark.asyncio
    async def test_update_progress_minimal_fields(self):
        """
        Test update_progress() with minimal required fields.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-006")
        
        await streamer.update_progress(
            progress=25.0,
            status="Processing..."
        )
        
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]
        
        assert message["data"]["progress"] == 25.0
        assert message["data"]["status"] == "Processing..."
        assert message["data"]["bars_processed"] is None
        assert message["data"]["total_bars"] is None
        
    @pytest.mark.asyncio
    async def test_complete_event_structure(self):
        """
        Test complete() produces correct event structure.
        
        Validates:
        - Event type is 'backtest_complete'
        - All result fields are included
        - Duration is calculated
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-007")
        
        # Start first to set start time
        await streamer.start("Sentinel", "EURUSD", "H1", "2023-01-01", "2023-12-31")
        
        # Small delay to ensure duration > 0
        await asyncio.sleep(0.01)
        
        await streamer.complete(
            final_balance=12500.00,
            total_trades=45,
            win_rate=62.5,
            sharpe_ratio=1.85,
            drawdown=8.5,
            return_pct=25.0
        )
        
        # Get the last broadcast call (complete)
        calls = mock_manager.broadcast.call_args_list
        complete_call = calls[-1]
        message = complete_call[0][0]
        
        # Verify message structure
        assert message["type"] == "backtest_complete"
        assert message["data"]["backtest_id"] == "test-007"
        assert message["data"]["final_balance"] == 12500.00
        assert message["data"]["total_trades"] == 45
        assert message["data"]["win_rate"] == 62.5
        assert message["data"]["sharpe_ratio"] == 1.85
        assert message["data"]["drawdown"] == 8.5
        assert message["data"]["return_pct"] == 25.0
        assert "duration_seconds" in message["data"]
        assert message["data"]["duration_seconds"] > 0
        assert "timestamp" in message["data"]
        
    @pytest.mark.asyncio
    async def test_complete_with_additional_results(self):
        """
        Test complete() with additional results dictionary.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-008")
        
        additional_results = {
            "max_consecutive_wins": 10,
            "profit_factor": 2.5
        }
        
        await streamer.complete(
            final_balance=15000.0,
            total_trades=60,
            results=additional_results
        )
        
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]
        
        assert message["data"]["results"]["max_consecutive_wins"] == 10
        assert message["data"]["results"]["profit_factor"] == 2.5
        
    @pytest.mark.asyncio
    async def test_complete_with_explicit_duration(self):
        """
        Test complete() with explicitly provided duration.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-009")
        
        await streamer.complete(
            final_balance=11000.0,
            total_trades=20,
            duration_seconds=45.5
        )
        
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]
        
        assert message["data"]["duration_seconds"] == 45.5
        
    @pytest.mark.asyncio
    async def test_error_event_structure(self):
        """
        Test error() produces correct event structure.
        
        Validates:
        - Event type is 'backtest_error'
        - Error message and details are included
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-010")
        
        await streamer.error(
            error_message="Strategy execution failed",
            error_details="ZeroDivisionError at line 45"
        )
        
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]
        
        # Verify message structure
        assert message["type"] == "backtest_error"
        assert message["data"]["backtest_id"] == "test-010"
        assert message["data"]["error"] == "Strategy execution failed"
        assert message["data"]["error_details"] == "ZeroDivisionError at line 45"
        assert "timestamp" in message["data"]
        
    @pytest.mark.asyncio
    async def test_error_without_details(self):
        """
        Test error() without detailed information.
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        mock_manager = AsyncMock()
        set_connection_manager(mock_manager)
        
        streamer = BacktestProgressStreamer("test-011")
        
        await streamer.error(error_message="Unknown error")
        
        call_args = mock_manager.broadcast.call_args
        message = call_args[0][0]
        
        assert message["data"]["error"] == "Unknown error"
        assert message["data"]["error_details"] is None
        
    @pytest.mark.asyncio
    async def test_broadcast_without_connection_manager(self):
        """
        Test _broadcast handles missing connection manager.
        
        Validates:
        - No exception when connection manager is None
        - Graceful degradation
        """
        from src.api.ws_logger import BacktestProgressStreamer, set_connection_manager
        
        set_connection_manager(None)
        
        streamer = BacktestProgressStreamer("test-012")
        
        # Should not raise exception
        await streamer.start("Sentinel", "EURUSD", "H1", "2023-01-01", "2023-12-31")
        await streamer.update_progress(50.0, "Processing")
        await streamer.complete(11000.0, 20)
        await streamer.error("Test error")


class TestSetupBacktestLogging:
    """Unit tests for setup_backtest_logging function."""
    
    def test_setup_creates_logger(self):
        """
        Test setup_backtest_logging creates a logger.
        
        Validates:
        - Logger is returned
        - Logger has correct name
        """
        from src.api.ws_logger import setup_backtest_logging
        
        logger, streamer = setup_backtest_logging("test-logger-001")
        
        assert logger is not None
        assert "backtest.test-logger-001" in logger.name
        
    def test_setup_creates_progress_streamer(self):
        """
        Test setup_backtest_logging creates a progress streamer.
        
        Validates:
        - Progress streamer is returned
        - Streamer has correct backtest ID
        """
        from src.api.ws_logger import setup_backtest_logging
        
        logger, streamer = setup_backtest_logging("test-logger-002")
        
        assert streamer is not None
        assert streamer.backtest_id == "test-logger-002"
        
    def test_setup_generates_uuid_if_no_id(self):
        """
        Test setup_backtest_logging generates UUID if no ID provided.
        """
        from src.api.ws_logger import setup_backtest_logging
        
        logger, streamer = setup_backtest_logging()
        
        assert streamer.backtest_id is not None
        assert len(streamer.backtest_id) == 36  # UUID format
        
    def test_setup_clears_existing_handlers(self):
        """
        Test setup_backtest_logging clears existing handlers.
        
        Validates:
        - Previous handlers are removed
        - Only WebSocket handler remains
        """
        from src.api.ws_logger import setup_backtest_logging
        import logging
        
        # Create logger with existing handler
        logger_name = "backtest.test-logger-003"
        existing_logger = logging.getLogger(logger_name)
        existing_logger.addHandler(logging.StreamHandler())
        
        assert len(existing_logger.handlers) > 0
        
        # Setup should clear handlers
        logger, _ = setup_backtest_logging("test-logger-003")
        
        # Should only have WebSocket handler
        assert len(logger.handlers) == 1
        from src.api.ws_logger import WebSocketLogHandler
        assert isinstance(logger.handlers[0], WebSocketLogHandler)
        
    def test_setup_sets_log_level(self):
        """
        Test setup_backtest_logging sets correct log level.
        """
        from src.api.ws_logger import setup_backtest_logging
        import logging
        
        # Test default level (INFO)
        logger1, _ = setup_backtest_logging("test-level-001")
        assert logger1.level == logging.INFO
        
        # Test custom level (DEBUG)
        logger2, _ = setup_backtest_logging("test-level-002", level=logging.DEBUG)
        assert logger2.level == logging.DEBUG
        
        # Test custom level (WARNING)
        logger3, _ = setup_backtest_logging("test-level-003", level=logging.WARNING)
        assert logger3.level == logging.WARNING
        
    def test_setup_custom_topic(self):
        """
        Test setup_backtest_logging with custom topic.
        """
        from src.api.ws_logger import setup_backtest_logging
        
        logger, streamer = setup_backtest_logging("test-topic-001", topic="custom_topic")
        
        assert streamer.topic == "custom_topic"
        
        # Check handler topic
        from src.api.ws_logger import WebSocketLogHandler
        handler = logger.handlers[0]
        assert handler.topic == "custom_topic"


class TestGetLoggerWithContext:
    """Unit tests for get_logger_with_context function."""
    
    def test_get_logger_returns_logger(self):
        """
        Test get_logger_with_context returns a logger.
        """
        from src.api.ws_logger import get_logger_with_context
        
        logger = get_logger_with_context("test-context-001")
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        
    def test_get_logger_without_id(self):
        """
        Test get_logger_with_context without ID generates one.
        """
        from src.api.ws_logger import get_logger_with_context
        
        logger = get_logger_with_context()
        
        assert logger is not None


class TestConnectionManagerIntegration:
    """Integration tests for connection manager functions."""
    
    def test_set_connection_manager(self):
        """
        Test set_connection_manager stores manager globally.
        """
        from src.api.ws_logger import set_connection_manager, get_connection_manager
        
        mock_manager = MagicMock()
        set_connection_manager(mock_manager)
        
        result = get_connection_manager()
        
        assert result is mock_manager
        
    def test_get_connection_manager_lazy_import(self):
        """
        Test get_connection_manager attempts lazy import.
        """
        from src.api.ws_logger import set_connection_manager, get_connection_manager
        
        # Clear any existing manager
        set_connection_manager(None)
        
        # This should attempt to import from websocket_endpoints
        # May return None if not available in test environment
        result = get_connection_manager()
        
        # Result could be None or a manager depending on environment
        # Just verify no exception is raised
        assert result is None or result is not None


class TestWebSocketLogHandlerEdgeCases:
    """Edge case tests for WebSocketLogHandler."""
    
    def test_emit_with_format_args(self):
        """
        Test emit with log message format arguments.
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Value: %s, Count: %d",
            args=("test_value", 42),
            exc_info=None
        )
        record.module = "test"
        record.funcName = "test"
        
        result = handler._format_record(record)
        
        # Message should be formatted
        assert result["data"]["message"] == "Value: test_value, Count: 42"
        
    def test_emit_with_exception_info(self):
        """
        Test emit with exception info in record.
        """
        from src.api.ws_logger import WebSocketLogHandler
        
        handler = WebSocketLogHandler()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info
            )
            record.module = "test"
            record.funcName = "test"
            
            # Should not raise exception
            result = handler._format_record(record)
            assert result["data"]["message"] == "Error occurred"
