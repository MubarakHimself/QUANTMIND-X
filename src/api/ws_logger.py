"""
WebSocket Log Handler for real-time backtest streaming.

Provides:
- WebSocketLogHandler: Custom logging handler that broadcasts logs to WebSocket clients
- BacktestProgressStreamer: Handles backtest lifecycle progress updates
- setup_backtest_logging(): Creates configured logger with WebSocket streaming
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from functools import wraps
import threading

logger = logging.getLogger(__name__)

# Global reference to connection manager for broadcasting
# This will be imported lazily to avoid circular imports
_connection_manager = None


def set_connection_manager(manager):
    """Set the global connection manager for WebSocket broadcasting."""
    global _connection_manager
    _connection_manager = manager


def get_connection_manager():
    """Get the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        try:
            from src.api.websocket_endpoints import get_manager
            _connection_manager = get_manager()
        except ImportError:
            pass
    return _connection_manager


class WebSocketLogHandler(logging.Handler):
    """
    Custom logging handler that broadcasts log messages to WebSocket clients.
    
    Formats log records as JSON and sends them via the ConnectionManager
    broadcast mechanism with optional topic filtering.
    """
    
    def __init__(self, topic: str = "backtest", backtest_id: Optional[str] = None):
        """
        Initialize the WebSocket log handler.
        
        Args:
            topic: Topic filter for message routing (e.g., "backtest", "trading", "logs")
            backtest_id: Optional backtest ID to include in log messages
        """
        super().__init__()
        self.topic = topic
        self.backtest_id = backtest_id
        self._format_cache = {}
        
    def emit(self, record: logging.LogRecord):
        """
        Emit a log record by broadcasting it to WebSocket clients.
        
        Args:
            record: Log record to broadcast
        """
        try:
            # Format the log record
            log_data = self._format_record(record)
            
            # Broadcast via connection manager
            manager = get_connection_manager()
            if manager is not None:
                # Use asyncio to run the coroutine
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, use create_task
                        asyncio.create_task(
                            manager.broadcast(log_data, topic=self.topic)
                        )
                    else:
                        # Synchronous context, use run_until_complete
                        loop.run_until_complete(
                            manager.broadcast(log_data, topic=self.topic)
                        )
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(manager.broadcast(log_data, topic=self.topic))
            else:
                # No connection manager available, log locally
                print(f"[WS Logger] No connection manager: {log_data}")
                
        except Exception as e:
            # Never let logging failures crash the application
            self.handleError(record)
    
    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Format a log record as a dictionary.
        
        Args:
            record: Log record to format
            
        Returns:
            Dictionary with log data
        """
        return {
            "type": "log_entry",
            "data": {
                "backtest_id": self.backtest_id,
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "logger_name": record.name
            }
        }


class BacktestProgressStreamer:
    """
    Handles backtest lifecycle progress updates via WebSocket.
    
    Provides methods for:
    - Starting a backtest
    - Updating progress
    - Completing a backtest
    """
    
    def __init__(self, backtest_id: str, topic: str = "backtest"):
        """
        Initialize the progress streamer.
        
        Args:
            backtest_id: Unique identifier for the backtest
            topic: Topic for broadcasting (default: "backtest")
        """
        self.backtest_id = backtest_id
        self.topic = topic
        self._start_time = None
        
    async def start(self, 
                   variant: str, 
                   symbol: str, 
                   timeframe: str, 
                   start_date: str, 
                   end_date: str):
        """
        Broadcast backtest start event.
        
        Args:
            variant: Strategy variant name
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "H1", "M15")
            start_date: Start date string
            end_date: End date string
        """
        self._start_time = datetime.utcnow()
        
        message = {
            "type": "backtest_start",
            "data": {
                "backtest_id": self.backtest_id,
                "variant": variant,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "timestamp": self._start_time.isoformat()
            }
        }
        
        await self._broadcast(message)
        
    async def update_progress(self,
                             progress: float,
                             status: str,
                             bars_processed: Optional[int] = None,
                             total_bars: Optional[int] = None,
                             current_date: Optional[str] = None,
                             trades_count: Optional[int] = None,
                             current_pnl: Optional[float] = None):
        """
        Broadcast progress update.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message
            bars_processed: Number of bars processed
            total_bars: Total bars to process
            current_date: Current date being processed
            trades_count: Number of trades executed
            current_pnl: Current P&L
        """
        message = {
            "type": "backtest_progress",
            "data": {
                "backtest_id": self.backtest_id,
                "progress": progress,
                "status": status,
                "bars_processed": bars_processed,
                "total_bars": total_bars,
                "current_date": current_date,
                "trades_count": trades_count,
                "current_pnl": current_pnl,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self._broadcast(message)
        
    async def complete(self,
                      final_balance: float,
                      total_trades: int,
                      win_rate: Optional[float] = None,
                      sharpe_ratio: Optional[float] = None,
                      drawdown: Optional[float] = None,
                      return_pct: Optional[float] = None,
                      duration_seconds: Optional[float] = None,
                      results: Optional[Dict[str, Any]] = None):
        """
        Broadcast backtest completion.
        
        Args:
            final_balance: Final account balance
            total_trades: Total number of trades
            win_rate: Win rate percentage
            sharpe_ratio: Sharpe ratio
            drawdown: Maximum drawdown percentage
            return_pct: Return percentage
            duration_seconds: Duration in seconds
            results: Additional results dictionary
        """
        end_time = datetime.utcnow()
        duration = duration_seconds
        if duration is None and self._start_time:
            duration = (end_time - self._start_time).total_seconds()
            
        message = {
            "type": "backtest_complete",
            "data": {
                "backtest_id": self.backtest_id,
                "final_balance": final_balance,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "drawdown": drawdown,
                "return_pct": return_pct,
                "duration_seconds": duration,
                "timestamp": end_time.isoformat(),
                "results": results or {}
            }
        }
        
        await self._broadcast(message)
        
    async def error(self, error_message: str, error_details: Optional[str] = None):
        """
        Broadcast backtest error.
        
        Args:
            error_message: Error message
            error_details: Additional error details
        """
        message = {
            "type": "backtest_error",
            "data": {
                "backtest_id": self.backtest_id,
                "error": error_message,
                "error_details": error_details,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self._broadcast(message)
        
    async def _broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message via connection manager.
        
        Args:
            message: Message dictionary to broadcast
        """
        manager = get_connection_manager()
        if manager is not None:
            try:
                await manager.broadcast(message, topic=self.topic)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
        else:
            # No connection manager available
            print(f"[Progress Streamer] No connection manager: {message}")


def setup_backtest_logging(backtest_id: Optional[str] = None,
                          level: int = logging.INFO,
                          topic: str = "backtest") -> tuple:
    """
    Setup a logger with WebSocket handler for backtest streaming.
    
    Args:
        backtest_id: Optional backtest ID (will generate UUID if not provided)
        level: Log level (default: logging.INFO)
        topic: Topic for WebSocket broadcasts (default: "backtest")
        
    Returns:
        Tuple of (logger, progress_streamer)
    """
    import uuid
    
    if backtest_id is None:
        backtest_id = str(uuid.uuid4())
    
    # Create logger
    logger_name = f"backtest.{backtest_id}"
    ws_logger = logging.getLogger(logger_name)
    ws_logger.setLevel(level)
    
    # Clear any existing handlers
    ws_logger.handlers.clear()
    
    # Create WebSocket handler
    ws_handler = WebSocketLogHandler(topic=topic, backtest_id=backtest_id)
    ws_handler.setLevel(level)
    ws_logger.addHandler(ws_handler)
    
    # Create progress streamer
    progress_streamer = BacktestProgressStreamer(backtest_id, topic=topic)
    
    return ws_logger, progress_streamer


def get_logger_with_context(backtest_id: Optional[str] = None):
    """
    Get a logger with WebSocket streaming enabled.
    
    Args:
        backtest_id: Optional backtest ID
        
    Returns:
        Logger instance with WebSocket handler
    """
    logger, _ = setup_backtest_logging(backtest_id)
    return logger


__all__ = [
    'WebSocketLogHandler',
    'BacktestProgressStreamer',
    'setup_backtest_logging',
    'get_logger_with_context',
    'set_connection_manager',
    'get_connection_manager'
]
