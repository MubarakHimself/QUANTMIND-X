"""
JSON Logging Configuration for QuantMindX

Configures JSON file handlers for services to write logs to files
that Promtail scrapes and ships to Loki.

Log files are written to /app/logs/ with JSON format compatible
with Promtail's json pipeline stages.
"""

import logging
import os
import json
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs log records as JSON objects with fields compatible
    with Promtail's json pipeline stages.
    """
    
    def __init__(self, service_name: str = "quantmind"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }
        
        # Add optional fields if present
        if hasattr(record, 'module') and record.module:
            log_entry["module"] = record.module
        
        if hasattr(record, 'funcName') and record.funcName:
            log_entry["function"] = record.funcName
        
        if hasattr(record, 'lineno') and record.lineno:
            log_entry["line"] = record.lineno
        
        # Add extra fields from record
        extra_fields = [
            'mode', 'ea_id', 'symbol', 'trade_id', 'pnl',
            'regime', 'chaos_score', 'regime_quality',
            'kelly_fraction', 'position_size', 'risk_amount',
            'bot_id', 'action', 'account_id', 'broker_id'
        ]
        
        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_json_file_handler(
    log_file: str,
    service_name: str,
    log_level: int = logging.INFO,
    log_dir: str = "/app/logs"
) -> Optional[logging.FileHandler]:
    """
    Set up a JSON file handler for logging.
    
    Args:
        log_file: Name of the log file (e.g., "quantmind.log")
        service_name: Name of the service for the log entries
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: /app/logs)
    
    Returns:
        Configured FileHandler or None if setup fails
    """
    candidate_dirs = []
    if os.getenv("APP_LOGS_DIR"):
        candidate_dirs.append(os.getenv("APP_LOGS_DIR"))
    if log_dir not in candidate_dirs:
        candidate_dirs.append(log_dir)

    local_fallback = Path("data/logs")
    if str(local_fallback) not in candidate_dirs:
        candidate_dirs.append(str(local_fallback))

    last_error = None
    for candidate in candidate_dirs:
        try:
            log_path = Path(candidate)
            log_path.mkdir(parents=True, exist_ok=True)

            full_path = log_path / log_file
            handler = logging.FileHandler(str(full_path))
            handler.setLevel(log_level)

            formatter = JsonFormatter(service_name=service_name)
            handler.setFormatter(formatter)
            return handler
        except Exception as e:
            last_error = e

    logging.error(f"Failed to set up JSON file handler: {last_error}")
    return None


def configure_service_logging(
    service_name: str,
    log_file: str,
    log_level: int = logging.INFO,
    log_dir: str = "/app/logs",
    root_loggers: Optional[list] = None
) -> None:
    """
    Configure JSON file logging for a service.
    
    Sets up file handlers for the specified loggers to write
    JSON-formatted logs that Promtail can scrape.
    
    Args:
        service_name: Name of the service (e.g., "quantmind-api")
        log_file: Name of the log file (e.g., "quantmind.log")
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: /app/logs)
        root_loggers: List of logger names to configure (default: root logger)
    """
    # Set up the JSON file handler
    file_handler = setup_json_file_handler(
        log_file=log_file,
        service_name=service_name,
        log_level=log_level,
        log_dir=log_dir
    )
    
    if file_handler is None:
        return
    
    # Configure specified loggers or root logger
    if root_loggers is None:
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(log_level)
    else:
        # Configure specific loggers
        for logger_name in root_loggers:
            logger = logging.getLogger(logger_name)
            logger.addHandler(file_handler)
            logger.setLevel(log_level)
    
    logging.info(f"JSON file logging configured for {service_name} -> {log_file}")


def configure_api_logging():
    """Configure JSON logging for the QuantMind API server."""
    configure_service_logging(
        service_name="quantmind-api",
        log_file="quantmind.log",
        log_level=logging.INFO,
        root_loggers=["quantmind", "quantmind.server", "src"]
    )


def configure_router_logging():
    """Configure JSON logging for the Strategy Router."""
    configure_service_logging(
        service_name="quantmind-router",
        log_file="router.log",
        log_level=logging.INFO,
        root_loggers=["src.router", "src.router.engine", "src.router.sentinel"]
    )


def configure_mt5_logging():
    """Configure JSON logging for the MT5 Bridge."""
    configure_service_logging(
        service_name="mt5-bridge",
        log_file="mt5-bridge.log",
        log_level=logging.INFO,
        root_loggers=["mt5-bridge"]
    )


def configure_all_logging():
    """
    Configure JSON logging for all services.
    
    This sets up file handlers for:
    - API server -> /app/logs/quantmind.log
    - Router -> /app/logs/router.log
    - MT5 Bridge -> /app/logs/mt5-bridge.log
    """
    configure_api_logging()
    configure_router_logging()
    configure_mt5_logging()
