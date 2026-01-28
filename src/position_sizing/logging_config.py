"""
Logging configuration for Enhanced Kelly Position Sizing.

Provides structured JSON logging for:
- Position sizing calculations
- Edge case warnings
- Performance metrics
- Debug information
"""

import logging
import logging.config
import sys
from typing import Any, Dict
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Formats log records as JSON with additional context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Create base log data
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class PositionSizingLogger:
    """
    Logger for Enhanced Kelly position sizing operations.

    Provides structured logging with context for:
    - Calculation steps and results
    - Edge case handling
    - Performance metrics
    - Configuration changes
    """

    def __init__(self, name: str = "enhanced_kelly"):
        """
        Initialize position sizing logger.

        Args:
            name: Logger name (default: "enhanced_kelly")
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Console handler with INFO level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # File handler with DEBUG level
        file_handler = logging.FileHandler(
            'logs/enhanced_kelly.log',
            mode='a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_calculation(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_f: float,
        position_size: float,
        risk_amount: float
    ):
        """
        Log position sizing calculation.

        Args:
            account_balance: Account balance
            win_rate: Win rate used
            avg_win: Average win amount
            avg_loss: Average loss amount
            kelly_f: Final Kelly fraction
            position_size: Calculated position size
            risk_amount: Dollar amount at risk
        """
        self.logger.info(
            "Enhanced Kelly calculation",
            extra={
                "account_balance": account_balance,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "kelly_fraction": kelly_f,
                "position_size": position_size,
                "risk_amount": risk_amount
            }
        )

    def log_edge_case(
        self,
        edge_case: str,
        details: Dict[str, Any],
        level: str = "warning"
    ):
        """
        Log edge case handling.

        Args:
            edge_case: Type of edge case
            details: Additional details
            level: Log level ('warning', 'error', 'critical')
        """
        log_func = getattr(self.logger, level)
        log_func(
            f"Edge case: {edge_case}",
            extra={"edge_case": edge_case, **details}
        )

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        details: Dict[str, Any] = None
    ):
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            details: Additional details
        """
        self.logger.debug(
            f"Performance: {operation}",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                **(details or {})
            }
        )

    def log_portfolio_status(
        self,
        bot_count: int,
        total_raw_risk: float,
        total_scaled_risk: float,
        scale_factor: float,
        status: str
    ):
        """
        Log portfolio risk status.

        Args:
            bot_count: Number of active bots
            total_raw_risk: Total raw risk percentage
            total_scaled_risk: Total scaled risk percentage
            scale_factor: Scale factor applied
            status: Portfolio status ('safe', 'caution', 'danger')
        """
        self.logger.info(
            f"Portfolio status: {status}",
            extra={
                "bot_count": bot_count,
                "total_raw_risk": total_raw_risk,
                "total_scaled_risk": total_scaled_risk,
                "scale_factor": scale_factor,
                "status": status
            }
        )


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/enhanced_kelly.log",
    json_output: bool = True
):
    """
    Setup logging configuration for Enhanced Kelly system.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file
        json_output: Whether to use JSON formatting
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "src.position_sizing.logging_config.JSONFormatter"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "json" if json_output else "standard",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "enhanced_kelly": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "kelly_analyzer": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "portfolio_kelly": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }

    logging.config.dictConfig(config)


# Convenience function for quick logging
def get_logger(name: str = "enhanced_kelly") -> PositionSizingLogger:
    """
    Get a configured position sizing logger.

    Args:
        name: Logger name

    Returns:
        PositionSizingLogger instance
    """
    return PositionSizingLogger(name)
