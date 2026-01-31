"""
Database Connection Retry Logic with Exponential Backoff

Implements robust retry mechanisms for database operations with
exponential backoff and configurable retry policies.
"""

import time
import logging
from typing import Callable, TypeVar, Optional, Any
from functools import wraps
from sqlalchemy.exc import OperationalError, DatabaseError, DisconnectionError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DatabaseRetryError(Exception):
    """Raised when database operation fails after all retry attempts."""
    pass


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay in seconds between retries
            backoff_factor: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt.

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.initial_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )

        if self.jitter:
            import random
            # Add jitter: random value between 0 and delay
            delay = delay * (0.5 + random.random() * 0.5)

        return delay


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True
)


def with_retry(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (OperationalError, DatabaseError, DisconnectionError)
):
    """
    Decorator to add retry logic to database operations.

    Args:
        config: Retry configuration (uses default if None)
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry()
        def query_database():
            # Database operation
            pass
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt >= config.max_retries:
                        logger.error(
                            f"Database operation '{func.__name__}' failed after "
                            f"{config.max_retries} retries: {e}"
                        )
                        raise DatabaseRetryError(
                            f"Operation failed after {config.max_retries} retries"
                        ) from e

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Database operation '{func.__name__}' failed (attempt {attempt + 1}/"
                        f"{config.max_retries + 1}), retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(f"Non-retryable error in '{func.__name__}': {e}")
                    raise

            # Should never reach here, but just in case
            raise DatabaseRetryError(
                f"Operation failed after {config.max_retries} retries"
            ) from last_exception

        return wrapper
    return decorator


class DatabaseConnectionManager:
    """
    Manages database connections with automatic retry and reconnection.
    """

    def __init__(self, engine, retry_config: Optional[RetryConfig] = None):
        """
        Initialize connection manager.

        Args:
            engine: SQLAlchemy engine
            retry_config: Retry configuration (uses default if None)
        """
        self.engine = engine
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG

    @with_retry()
    def test_connection(self) -> bool:
        """
        Test database connection with retry logic.

        Returns:
            True if connection is successful

        Raises:
            DatabaseRetryError: If connection fails after all retries
        """
        from sqlalchemy import text
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def execute_with_retry(
        self,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute a database operation with retry logic.

        Args:
            operation: Callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of the operation

        Raises:
            DatabaseRetryError: If operation fails after all retries
        """
        @with_retry(config=self.retry_config)
        def wrapped_operation():
            return operation(*args, **kwargs)

        return wrapped_operation()

    def ensure_connection(self) -> None:
        """
        Ensure database connection is alive, reconnect if necessary.

        Raises:
            DatabaseRetryError: If reconnection fails after all retries
        """
        try:
            self.test_connection()
            logger.debug("Database connection verified")
        except DatabaseRetryError:
            logger.error("Failed to establish database connection after retries")
            raise


def create_connection_manager(engine, retry_config: Optional[RetryConfig] = None) -> DatabaseConnectionManager:
    """
    Factory function to create a DatabaseConnectionManager.

    Args:
        engine: SQLAlchemy engine
        retry_config: Optional retry configuration

    Returns:
        DatabaseConnectionManager instance
    """
    return DatabaseConnectionManager(engine, retry_config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from sqlalchemy import create_engine

    # Create test engine
    engine = create_engine('sqlite:///test.db')

    # Create connection manager
    manager = create_connection_manager(engine)

    # Test connection with retry
    try:
        manager.ensure_connection()
        print("✓ Database connection successful")
    except DatabaseRetryError as e:
        print(f"✗ Database connection failed: {e}")
