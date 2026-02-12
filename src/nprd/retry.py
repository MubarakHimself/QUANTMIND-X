"""
Retry Handler with exponential backoff.

Handles transient failures with exponential backoff and jitter to prevent thundering herd.
"""

import logging
import time
import random
from typing import Callable, Any, TypeVar, Optional
from functools import wraps

from src.nprd.exceptions import (
    NPRDError,
    AuthenticationError,
    ValidationError,
    NetworkError,
    RateLimitError,
)


logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryHandler:
    """
    Handles retry logic with exponential backoff.
    
    Features:
    - Exponential backoff: base_delay * (2 ** attempt)
    - Jitter: Random 0-1 second added to prevent thundering herd
    - Error classification: Distinguishes retryable vs permanent errors
    - Configurable max retries
    
    Example:
        handler = RetryHandler(max_retries=3, base_delay=1.0)
        result = handler.execute(risky_function, arg1, arg2, kwarg1=value)
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        logger.debug(f"Initialized RetryHandler with max_retries={max_retries}, base_delay={base_delay}")
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.
        
        Retries on: NetworkError, TimeoutError, RateLimitError, transient errors
        No retry on: AuthenticationError, ValidationError, permanent errors
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: The last exception if all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Executing {func.__name__} (attempt {attempt + 1}/{self.max_retries})")
                result = func(*args, **kwargs)
                
                # Success - log if this was a retry
                if attempt > 0:
                    logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if error is retryable
                if not self._is_retryable(e):
                    logger.warning(f"{func.__name__} failed with non-retryable error: {type(e).__name__}: {str(e)}")
                    raise
                
                # Log retry attempt
                logger.warning(
                    f"{func.__name__} failed on attempt {attempt + 1}/{self.max_retries}: "
                    f"{type(e).__name__}: {str(e)}"
                )
                
                # If this was the last attempt, raise the exception
                if attempt >= self.max_retries - 1:
                    logger.error(
                        f"{func.__name__} failed after {self.max_retries} attempts. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    )
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)
                logger.info(f"Waiting {delay:.2f}s before retry...")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        
        raise RuntimeError(f"Retry logic failed unexpectedly for {func.__name__}")
    
    def _is_retryable(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if error is retryable (transient)
        """
        # Non-retryable errors (permanent failures)
        non_retryable_types = (
            AuthenticationError,
            ValidationError,
        )
        
        if isinstance(error, non_retryable_types):
            return False
        
        # Retryable errors (transient failures)
        retryable_types = (
            NetworkError,
            RateLimitError,
            TimeoutError,
            ConnectionError,
        )
        
        if isinstance(error, retryable_types):
            return True
        
        # Check error message for retryable patterns
        error_str = str(error).lower()
        
        # Non-retryable patterns
        permanent_patterns = [
            'not found',
            '404',
            'forbidden',
            '403',
            'unauthorized',
            '401',
            'invalid',
            'authentication',
            'permission denied',
        ]
        
        if any(pattern in error_str for pattern in permanent_patterns):
            return False
        
        # Retryable patterns
        transient_patterns = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'rate limit',
            '429',
            '503',
            'service unavailable',
            'dns',
            'reset by peer',
        ]
        
        if any(pattern in error_str for pattern in transient_patterns):
            return True
        
        # Default to retryable for unknown errors
        # This is safer than failing immediately
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter.
        
        Formula: base_delay * (2 ** attempt) + random(0, 1)
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        exponential_delay = self.base_delay * (2 ** attempt)
        
        # Add jitter (0-1 second random)
        jitter = random.uniform(0, 1)
        
        total_delay = exponential_delay + jitter
        
        return total_delay


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for adding retry logic to functions.
    
    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def risky_function():
            # Function that might fail transiently
            pass
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(max_retries=max_retries, base_delay=base_delay)
            return handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


class RetryConfig:
    """
    Configuration for retry behavior.
    
    Can be used to customize retry behavior per operation type.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: Optional[float] = None,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap (optional)
            exponential_base: Base for exponential backoff (default: 2.0)
            jitter: Whether to add random jitter (default: True)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)
        
        # Apply max delay cap if set
        if self.max_delay is not None:
            delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            delay += random.uniform(0, 1)
        
        return delay


# Predefined retry configurations for common scenarios
RETRY_CONFIGS = {
    'default': RetryConfig(max_retries=3, base_delay=1.0),
    'aggressive': RetryConfig(max_retries=5, base_delay=0.5),
    'conservative': RetryConfig(max_retries=2, base_delay=2.0),
    'network': RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0),
    'api': RetryConfig(max_retries=3, base_delay=2.0, max_delay=60.0),
}
