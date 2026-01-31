#!/usr/bin/env python3
"""
Redis Pub/Sub client for agent communication.

Provides methods for publishing heartbeat, trade events, and alerts to Redis channels.
Includes connection pooling, retry logic, and proper message schemas.

Channel naming pattern:
- Heartbeat: agent:heartbeat:{agent_id}
- Trades: agent:trades:{agent_id}
- Alerts: agent:alerts:{agent_id}
"""

import json
import logging
import time
from datetime import datetime, UTC
from typing import Optional, Literal
from contextlib import contextmanager

import redis
from redis import Redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ============================================================================
# Message Schemas (Pydantic Models)
# ============================================================================

class HeartbeatMessage(BaseModel):
    """
    Heartbeat message schema.

    Published every 60 seconds to indicate agent is alive and healthy.
    """
    timestamp: str = Field(
        description="ISO 8601 timestamp in UTC",
        examples=["2026-01-29T10:30:00.000000Z"]
    )
    agent_id: str = Field(
        description="Unique identifier for the agent",
        examples=["strategy-agent-1"]
    )
    status: Literal["running", "idle", "error", "shutting_down"] = Field(
        description="Current status of the agent",
        default="running"
    )
    uptime_seconds: int = Field(
        description="Agent uptime in seconds",
        ge=0,
        examples=[3600]
    )
    mt5_connected: bool = Field(
        description="Whether MT5 connection is active",
        default=False
    )

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Ensure timestamp is in ISO format."""
        try:
            # Parse and re-format to ensure valid ISO 8601
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e

    def to_json(self) -> str:
        """Convert to JSON string for Redis publishing."""
        return self.model_dump_json(exclude_none=True)


class TradeEventMessage(BaseModel):
    """
    Trade event message schema.

    Published on: trade entry, trade exit, stop loss hit, take profit hit.
    """
    timestamp: str = Field(
        description="ISO 8601 timestamp in UTC",
        examples=["2026-01-29T10:30:00.000000Z"]
    )
    agent_id: str = Field(
        description="Unique identifier for the agent",
        examples=["strategy-agent-1"]
    )
    action: Literal["entry", "exit", "stop_hit", "take_profit"] = Field(
        description="Type of trade action"
    )
    symbol: str = Field(
        description="Trading symbol (e.g., EURUSD, GBPUSD)",
        examples=["EURUSD"]
    )
    price: float = Field(
        description="Execution price",
        gt=0,
        examples=[1.0850]
    )
    lots: float = Field(
        description="Trade size in lots",
        gt=0,
        examples=[0.1]
    )
    pnl: Optional[float] = Field(
        description="Profit/loss in account currency (for exits)",
        default=None,
        examples=[125.50, -50.25]
    )
    order_id: Optional[int] = Field(
        description="MT5 order ticket number",
        default=None,
        examples=[123456]
    )

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Ensure timestamp is in ISO format."""
        try:
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e

    def to_json(self) -> str:
        """Convert to JSON string for Redis publishing."""
        return self.model_dump_json(exclude_none=True)


class AlertMessage(BaseModel):
    """
    Alert message schema.

    Published for important events, warnings, and errors.
    """
    timestamp: str = Field(
        description="ISO 8601 timestamp in UTC",
        examples=["2026-01-29T10:30:00.000000Z"]
    )
    agent_id: str = Field(
        description="Unique identifier for the agent",
        examples=["strategy-agent-1"]
    )
    severity: Literal["info", "warning", "error", "critical"] = Field(
        description="Alert severity level"
    )
    message: str = Field(
        description="Alert message content",
        examples=["MT5 connection lost, attempting reconnection..."]
    )
    details: Optional[dict] = Field(
        description="Additional details as key-value pairs",
        default=None
    )

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Ensure timestamp is in ISO format."""
        try:
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e

    def to_json(self) -> str:
        """Convert to JSON string for Redis publishing."""
        return self.model_dump_json(exclude_none=True)


# ============================================================================
# Redis Client Wrapper
# ============================================================================

class RedisClientError(Exception):
    """Base exception for Redis client errors."""
    pass


class RedisConnectionError(RedisClientError):
    """Raised when Redis connection fails."""
    pass


class RedisPublishError(RedisClientError):
    """Raised when message publishing fails."""
    pass


class RedisClient:
    """
    Redis Pub/Sub client wrapper for agent communication.

    Features:
    - Connection pooling with retry logic
    - Automatic reconnection on connection loss
    - Message validation using Pydantic schemas
    - Channel naming following agent:{type}:{agent_id} pattern
    - Graceful degradation when Redis is unavailable

    Example:
        ```python
        client = RedisClient(
            host="localhost",
            port=6379,
            agent_id="strategy-agent-1"
        )

        # Publish heartbeat
        await client.publish_heartbeat(
            status="running",
            uptime_seconds=3600,
            mt5_connected=True
        )

        # Publish trade event
        await client.publish_trade_event(
            action="entry",
            symbol="EURUSD",
            price=1.0850,
            lots=0.1
        )
        ```
    """

    # Maximum retry attempts for transient failures
    MAX_RETRIES = 3

    # Base delay between retries (exponential backoff)
    RETRY_DELAY = 0.5  # seconds

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        agent_id: str = "strategy-agent-1",
        connection_pool_size: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ):
        """
        Initialize Redis client wrapper.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            agent_id: Unique identifier for this agent
            connection_pool_size: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.agent_id = agent_id

        # Build channel names
        self.heartbeat_channel = f"agent:heartbeat:{agent_id}"
        self.trades_channel = f"agent:trades:{agent_id}"
        self.alerts_channel = f"agent:alerts:{agent_id}"

        # Connection pool
        self._pool = None
        self._client: Optional[Redis] = None

        # Connection configuration
        self.connection_pool_size = connection_pool_size
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout

        # Track agent start time for uptime calculation
        self._start_time = time.time()

    @property
    def uptime_seconds(self) -> int:
        """Get agent uptime in seconds."""
        return int(time.time() - self._start_time)

    def _create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool."""
        return redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.connection_pool_size,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout,
            decode_responses=True,
        )

    @contextmanager
    def _get_connection(self):
        """
        Get Redis connection from pool with automatic reconnection.

        Yields:
            Redis client instance

        Raises:
            RedisConnectionError: If connection cannot be established
        """
        if self._client is None:
            try:
                if self._pool is None:
                    self._pool = self._create_connection_pool()
                self._client = Redis(connection_pool=self._pool)
                # Test connection
                self._client.ping()
                logger.info(f"Redis client connected to {self.host}:{self.port}")
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise RedisConnectionError(f"Cannot connect to Redis at {self.host}:{self.port}") from e

        try:
            # Verify connection is still alive
            self._client.ping()
            yield self._client
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection lost: {e}")
            # Reset client to force reconnection
            self._client = None
            raise RedisConnectionError("Redis connection lost") from e

    def _publish_with_retry(
        self,
        channel: str,
        message: str,
        max_retries: int = MAX_RETRIES
    ) -> bool:
        """
        Publish message to Redis channel with retry logic.

        Args:
            channel: Redis channel name
            message: JSON message to publish
            max_retries: Maximum retry attempts

        Returns:
            True if published successfully, False otherwise

        Raises:
            RedisPublishError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                with self._get_connection() as client:
                    result = client.publish(channel, message)
                    logger.debug(f"Published to {channel}: {result} subscribers")
                    return True

            except (ConnectionError, TimeoutError) as e:
                last_error = e
                delay = self.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Redis publish attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

            except RedisError as e:
                last_error = e
                logger.error(f"Redis error on publish attempt {attempt + 1}: {e}")
                # Don't retry on non-connection Redis errors
                break

        # All retries failed
        error_msg = f"Failed to publish after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise RedisPublishError(error_msg) from last_error

    # ========================================================================
    # Publish Methods
    # ========================================================================

    def publish_heartbeat(
        self,
        status: Literal["running", "idle", "error", "shutting_down"] = "running",
        mt5_connected: bool = False,
    ) -> bool:
        """
        Publish heartbeat message.

        Heartbeat messages should be sent every 60 seconds to indicate
        the agent is alive and healthy.

        Args:
            status: Current agent status
            mt5_connected: Whether MT5 connection is active

        Returns:
            True if published successfully

        Example:
            ```python
            client.publish_heartbeat(status="running", mt5_connected=True)
            ```
        """
        message = HeartbeatMessage(
            timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            agent_id=self.agent_id,
            status=status,
            uptime_seconds=self.uptime_seconds,
            mt5_connected=mt5_connected,
        )

        try:
            # Publish to Pub/Sub channel
            success = self._publish_with_retry(self.heartbeat_channel, message.to_json())

            # Also store latest heartbeat in a key with TTL (for quick lookups)
            with self._get_connection() as client:
                client.setex(
                    f"{self.heartbeat_channel}:latest",
                    120,  # 2 minute TTL
                    message.to_json()
                )

            logger.debug(f"Published heartbeat: {status}, uptime={self.uptime_seconds}s")
            return success

        except RedisPublishError:
            # Log but don't raise - heartbeat failures shouldn't crash the agent
            logger.warning("Heartbeat publish failed (agent will continue)")
            return False

    def publish_trade_event(
        self,
        action: Literal["entry", "exit", "stop_hit", "take_profit"],
        symbol: str,
        price: float,
        lots: float,
        pnl: Optional[float] = None,
        order_id: Optional[int] = None,
    ) -> bool:
        """
        Publish trade event message.

        Published on: trade entry, trade exit, stop loss hit, take profit hit.

        Args:
            action: Type of trade action
            symbol: Trading symbol (e.g., EURUSD)
            price: Execution price
            lots: Trade size in lots
            pnl: Profit/loss for exits
            order_id: MT5 order ticket number

        Returns:
            True if published successfully

        Example:
            ```python
            client.publish_trade_event(
                action="entry",
                symbol="EURUSD",
                price=1.0850,
                lots=0.1
            )
            ```
        """
        message = TradeEventMessage(
            timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            agent_id=self.agent_id,
            action=action,
            symbol=symbol,
            price=price,
            lots=lots,
            pnl=pnl,
            order_id=order_id,
        )

        try:
            success = self._publish_with_retry(self.trades_channel, message.to_json())

            # Also store in a list for trade history (keep last 100)
            with self._get_connection() as client:
                client.lpush(f"{self.trades_channel}:history", message.to_json())
                client.ltrim(f"{self.trades_channel}:history", 0, 99)
                client.expire(f"{self.trades_channel}:history", 86400)  # 24 hours

            logger.info(f"Published trade event: {action} {symbol} @ {price} x{lots}")
            return success

        except RedisPublishError as e:
            # Trade events are important - log error
            logger.error(f"Failed to publish trade event: {e}")
            return False

    def publish_alert(
        self,
        severity: Literal["info", "warning", "error", "critical"],
        message: str,
        details: Optional[dict] = None,
    ) -> bool:
        """
        Publish alert message.

        Published for important events, warnings, and errors.

        Args:
            severity: Alert severity level
            message: Alert message content
            details: Additional details as key-value pairs

        Returns:
            True if published successfully

        Example:
            ```python
            client.publish_alert(
                severity="warning",
                message="MT5 connection unstable",
                details={"reconnect_attempts": 3}
            )
            ```
        """
        alert = AlertMessage(
            timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            agent_id=self.agent_id,
            severity=severity,
            message=message,
            details=details,
        )

        try:
            success = self._publish_with_retry(self.alerts_channel, alert.to_json())

            # Also store in a list for alert history (keep last 50)
            with self._get_connection() as client:
                client.lpush(f"{self.alerts_channel}:history", alert.to_json())
                client.ltrim(f"{self.alerts_channel}:history", 0, 49)
                client.expire(f"{self.alerts_channel}:history", 604800)  # 7 days

            log_level = {
                "info": logger.info,
                "warning": logger.warning,
                "error": logger.error,
                "critical": logger.error,
            }[severity]

            log_level(f"Published alert [{severity}]: {message}")
            return success

        except RedisPublishError as e:
            # Alerts are important - log error
            logger.error(f"Failed to publish alert: {e}")
            return False

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if connected and able to ping Redis
        """
        try:
            with self._get_connection() as client:
                client.ping()
                return True
        except RedisConnectionError:
            return False

    def close(self):
        """
        Close Redis connections and cleanup resources.
        """
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            finally:
                self._client = None

        if self._pool:
            try:
                self._pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting pool: {e}")
            finally:
                self._pool = None

        logger.info("Redis client closed")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_redis_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    agent_id: str = "strategy-agent-1",
) -> RedisClient:
    """
    Factory function to create a Redis client with environment variable support.

    Environment variables (fallback if not specified):
    - REDIS_HOST
    - REDIS_PORT
    - REDIS_DB
    - REDIS_PASSWORD
    - AGENT_ID

    Args:
        host: Redis server host (default: from REDIS_HOST env var or "localhost")
        port: Redis server port (default: from REDIS_PORT env var or 6379)
        db: Redis database number (default: from REDIS_DB env var or 0)
        password: Redis password (default: from REDIS_PASSWORD env var)
        agent_id: Agent identifier (default: from AGENT_ID env var)

    Returns:
        Configured RedisClient instance

    Example:
        ```python
        import os
        os.environ['REDIS_HOST'] = 'redis.example.com'
        os.environ['AGENT_ID'] = 'my-agent'

        client = create_redis_client()  # Uses env vars
        ```
    """
    import os

    return RedisClient(
        host=host or os.getenv('REDIS_HOST', 'localhost'),
        port=port or int(os.getenv('REDIS_PORT', '6379')),
        db=db if db is not None else int(os.getenv('REDIS_DB', '0')),
        password=password or os.getenv('REDIS_PASSWORD'),
        agent_id=agent_id or os.getenv('AGENT_ID', 'strategy-agent-1'),
    )
