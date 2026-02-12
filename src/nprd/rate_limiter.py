"""
Rate Limiter with token bucket algorithm.

Enforces API rate limits for model providers with per-provider tracking and automatic fallback.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from threading import Lock
from collections import deque

from src.nprd.exceptions import RateLimitError


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter using token bucket algorithm with sliding window.
    
    Features:
    - Token bucket with sliding window
    - Per-provider rate tracking
    - Automatic provider fallback
    - Thread-safe operations
    
    Example:
        limiter = RateLimiter(provider="gemini", limit=100, window_seconds=3600)
        limiter.acquire()  # Blocks if rate limit would be exceeded
    """
    
    def __init__(
        self,
        provider: str,
        limit: int,
        window_seconds: int = 86400,  # 24 hours default
    ):
        """
        Initialize rate limiter.
        
        Args:
            provider: Provider name (e.g., "gemini", "qwen")
            limit: Maximum requests per window
            window_seconds: Time window in seconds (default: 86400 = 24 hours)
        """
        self.provider = provider
        self.limit = limit
        self.window_seconds = window_seconds
        
        # Sliding window: Store timestamps of requests
        self.requests: deque = deque()
        
        # Thread safety
        self.lock = Lock()
        
        logger.info(
            f"Initialized RateLimiter for {provider}: "
            f"{limit} requests per {window_seconds}s"
        )
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Blocks if rate limit would be exceeded (when blocking=True).
        
        Args:
            blocking: Whether to block until permission is granted (default: True)
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if permission granted, False if limit exceeded (non-blocking mode)
            
        Raises:
            RateLimitError: If rate limit exceeded in non-blocking mode
            TimeoutError: If timeout exceeded while waiting
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Clean up old requests outside the window
                self._cleanup_old_requests()
                
                # Check if we can make a request
                if len(self.requests) < self.limit:
                    # Add current timestamp
                    self.requests.append(time.time())
                    logger.debug(
                        f"{self.provider}: Request granted "
                        f"({len(self.requests)}/{self.limit} used)"
                    )
                    return True
                
                # Rate limit exceeded
                if not blocking:
                    logger.warning(
                        f"{self.provider}: Rate limit exceeded "
                        f"({len(self.requests)}/{self.limit})"
                    )
                    raise RateLimitError(
                        f"Rate limit exceeded for {self.provider}: "
                        f"{len(self.requests)}/{self.limit} requests used",
                        provider=self.provider,
                        limit=self.limit,
                    )
                
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = self.window_seconds - (time.time() - oldest_request)
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise TimeoutError(
                            f"Timeout waiting for rate limit on {self.provider}"
                        )
                    wait_time = min(wait_time, timeout - elapsed)
            
            # Wait before retrying (release lock while waiting)
            if wait_time > 0:
                logger.info(
                    f"{self.provider}: Rate limit reached, waiting {wait_time:.1f}s"
                )
                time.sleep(min(wait_time, 1.0))  # Sleep in 1-second increments
    
    def get_status(self) -> Dict[str, any]:
        """
        Get current rate limit status.
        
        Returns:
            Dictionary with rate limit information
        """
        with self.lock:
            self._cleanup_old_requests()
            
            requests_used = len(self.requests)
            requests_remaining = max(0, self.limit - requests_used)
            
            # Calculate reset time
            if self.requests:
                oldest_request = self.requests[0]
                reset_time = datetime.fromtimestamp(
                    oldest_request + self.window_seconds
                )
            else:
                reset_time = None
            
            return {
                "provider": self.provider,
                "limit": self.limit,
                "window_seconds": self.window_seconds,
                "requests_used": requests_used,
                "requests_remaining": requests_remaining,
                "utilization_pct": (requests_used / self.limit * 100) if self.limit > 0 else 0,
                "reset_time": reset_time.isoformat() if reset_time else None,
            }
    
    def reset(self) -> None:
        """Reset rate limiter (clear all request history)."""
        with self.lock:
            self.requests.clear()
            logger.info(f"{self.provider}: Rate limiter reset")
    
    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the current window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Remove old requests from the left of the deque
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()


class MultiProviderRateLimiter:
    """
    Manages rate limits for multiple providers with automatic fallback.
    
    Features:
    - Per-provider rate limiting
    - Automatic provider selection
    - Fallback to alternative providers
    - Provider priority ordering
    
    Example:
        limiter = MultiProviderRateLimiter()
        limiter.add_provider("gemini", limit=None)  # Unlimited
        limiter.add_provider("qwen", limit=2000, window_seconds=86400)
        
        provider = limiter.acquire_any()  # Returns available provider
    """
    
    def __init__(self):
        """Initialize multi-provider rate limiter."""
        self.limiters: Dict[str, RateLimiter] = {}
        self.provider_priority: List[str] = []
        self.lock = Lock()
        
        logger.info("Initialized MultiProviderRateLimiter")
    
    def add_provider(
        self,
        provider: str,
        limit: Optional[int],
        window_seconds: int = 86400,
        priority: int = 0,
    ) -> None:
        """
        Add a provider with rate limit configuration.
        
        Args:
            provider: Provider name
            limit: Maximum requests per window (None = unlimited)
            window_seconds: Time window in seconds
            priority: Provider priority (higher = preferred, default: 0)
        """
        with self.lock:
            if limit is None:
                # Unlimited provider - use very high limit
                limit = 999999999
            
            self.limiters[provider] = RateLimiter(
                provider=provider,
                limit=limit,
                window_seconds=window_seconds,
            )
            
            # Update priority list
            self.provider_priority.append((priority, provider))
            self.provider_priority.sort(reverse=True)  # Higher priority first
            
            logger.info(
                f"Added provider {provider} with limit={limit}, "
                f"window={window_seconds}s, priority={priority}"
            )
    
    def acquire(self, provider: str, blocking: bool = True) -> bool:
        """
        Acquire permission for specific provider.
        
        Args:
            provider: Provider name
            blocking: Whether to block until permission is granted
            
        Returns:
            True if permission granted
            
        Raises:
            ValueError: If provider not found
            RateLimitError: If rate limit exceeded (non-blocking mode)
        """
        if provider not in self.limiters:
            raise ValueError(f"Provider not found: {provider}")
        
        return self.limiters[provider].acquire(blocking=blocking)
    
    def acquire_any(self, blocking: bool = True) -> Optional[str]:
        """
        Acquire permission from any available provider.
        
        Tries providers in priority order and returns the first available.
        
        Args:
            blocking: Whether to block until a provider is available
            
        Returns:
            Provider name if available, None if all providers at limit (non-blocking)
            
        Raises:
            RateLimitError: If all providers at limit (blocking mode)
        """
        # Try providers in priority order
        for priority, provider in self.provider_priority:
            try:
                if self.limiters[provider].acquire(blocking=False):
                    logger.debug(f"Selected provider: {provider}")
                    return provider
            except RateLimitError:
                continue
        
        # All providers at limit
        if not blocking:
            return None
        
        # Wait for any provider to become available
        logger.warning("All providers at rate limit, waiting for availability...")
        
        while True:
            for priority, provider in self.provider_priority:
                try:
                    if self.limiters[provider].acquire(blocking=False):
                        logger.info(f"Provider available: {provider}")
                        return provider
                except RateLimitError:
                    continue
            
            # Sleep briefly before retrying
            time.sleep(1.0)
    
    def get_all_status(self) -> Dict[str, Dict]:
        """
        Get status for all providers.
        
        Returns:
            Dictionary mapping provider names to status dictionaries
        """
        with self.lock:
            return {
                provider: limiter.get_status()
                for provider, limiter in self.limiters.items()
            }
    
    def get_best_provider(self) -> Optional[str]:
        """
        Get the best available provider (highest priority with capacity).
        
        Returns:
            Provider name or None if all at limit
        """
        for priority, provider in self.provider_priority:
            status = self.limiters[provider].get_status()
            if status["requests_remaining"] > 0:
                return provider
        
        return None
    
    def reset_all(self) -> None:
        """Reset all provider rate limiters."""
        with self.lock:
            for limiter in self.limiters.values():
                limiter.reset()
            logger.info("Reset all provider rate limiters")


class RateLimitStatus:
    """
    Rate limit status information.
    
    Used for monitoring and reporting rate limit usage.
    """
    
    def __init__(
        self,
        provider: str,
        limit: int,
        used: int,
        remaining: int,
        reset_time: Optional[datetime] = None,
    ):
        """
        Initialize rate limit status.
        
        Args:
            provider: Provider name
            limit: Maximum requests per window
            used: Requests used in current window
            remaining: Requests remaining in current window
            reset_time: When the rate limit will reset
        """
        self.provider = provider
        self.limit = limit
        self.used = used
        self.remaining = remaining
        self.reset_time = reset_time
    
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.remaining <= 0
    
    def utilization_pct(self) -> float:
        """Get utilization percentage."""
        return (self.used / self.limit * 100) if self.limit > 0 else 0
    
    def __str__(self) -> str:
        """String representation."""
        reset_str = f", resets at {self.reset_time.strftime('%H:%M:%S')}" if self.reset_time else ""
        return (
            f"{self.provider}: {self.used}/{self.limit} used "
            f"({self.utilization_pct():.1f}%){reset_str}"
        )
