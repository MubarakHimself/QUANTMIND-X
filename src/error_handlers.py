"""
Error Handlers for QuantMindX Backend

Provides centralized error handling with retry logic, circuit breakers,
and graceful degradation for various system components.
"""

import logging
import time
from typing import Callable, Optional, Any, Type
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatabaseErrorHandler:
    """
    Centralized database error handling with retry logic.
    
    **Validates: Property 5: Database Reconnection Resilience**
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0
    ):
        """
        Initialize database error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            initial_delay: Initial delay in seconds
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
    
    def handle_connection_error(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Handle database connection errors with exponential backoff.
        
        Args:
            operation: Database operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
            
        Raises:
            Exception: If all retries exhausted
        """
        delay = self.initial_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
                
            except (ConnectionError, OSError) as e:
                last_error = e
                
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Database operation failed after {self.max_retries} attempts: {e}"
                    )
                    raise
                
                logger.warning(
                    f"Database connection failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay}s: {e}"
                )
                
                time.sleep(delay)
                delay *= self.backoff_factor
        
        raise last_error
    
    def with_retry(self, func: Callable) -> Callable:
        """
        Decorator for automatic retry on database errors.
        
        Usage:
            @db_error_handler.with_retry
            def my_database_operation():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.handle_connection_error(func, *args, **kwargs)
        
        return wrapper


class MQL5BridgeErrorHandler:
    """
    Error handler for MQL5-Python integration failures.
    
    Provides graceful degradation and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize MQL5 bridge error handler."""
        self.heartbeat_failures = {}  # Track failures per EA
        self.max_consecutive_failures = 5
    
    def handle_heartbeat_failure(
        self,
        ea_name: str,
        error: Exception
    ) -> bool:
        """
        Handle heartbeat communication failures.
        
        Args:
            ea_name: Expert Advisor name
            error: Exception that occurred
            
        Returns:
            True if fallback successful, False otherwise
        """
        logger.error(f"Heartbeat failed for EA {ea_name}: {error}")
        
        # Track consecutive failures
        if ea_name not in self.heartbeat_failures:
            self.heartbeat_failures[ea_name] = 0
        
        self.heartbeat_failures[ea_name] += 1
        
        # Check if we've exceeded failure threshold
        if self.heartbeat_failures[ea_name] >= self.max_consecutive_failures:
            logger.critical(
                f"EA {ea_name} has failed {self.heartbeat_failures[ea_name]} "
                f"consecutive heartbeats - consider manual intervention"
            )
        
        # Attempt fallback to file-based communication
        try:
            from src.router.sync import DiskSyncer
            
            syncer = DiskSyncer()
            risk_data = {
                "FALLBACK": {
                    "multiplier": 0.5,  # Conservative fallback
                    "status": "degraded",
                    "timestamp": int(datetime.utcnow().timestamp())
                }
            }
            
            syncer.sync_risk_matrix(risk_data)
            logger.info(f"Fallback risk matrix written for EA {ea_name}")
            return True
            
        except Exception as fallback_error:
            logger.critical(f"Fallback communication failed: {fallback_error}")
            return False
    
    def reset_failure_count(self, ea_name: str):
        """Reset failure count for EA after successful heartbeat."""
        if ea_name in self.heartbeat_failures:
            self.heartbeat_failures[ea_name] = 0
    
    def handle_global_variable_error(
        self,
        variable_name: str,
        error: Exception
    ) -> Optional[Any]:
        """
        Handle GlobalVariable access errors.
        
        Args:
            variable_name: Name of global variable
            error: Exception that occurred
            
        Returns:
            Fallback value or None
        """
        logger.warning(f"GlobalVariable {variable_name} access failed: {error}")
        
        # Graceful degradation to file-based approach
        logger.info("Falling back to file-based risk retrieval")
        return None


class AgentErrorHandler:
    """
    Error handler for LangGraph agent execution errors.
    
    Handles tool errors, state transition failures, and execution timeouts.
    """
    
    def __init__(self):
        """Initialize agent error handler."""
        self.error_counts = {}
    
    def handle_tool_error(
        self,
        tool_name: str,
        error: Exception,
        context: dict
    ) -> dict:
        """
        Handle MCP tool execution errors.
        
        Args:
            tool_name: Name of the tool that failed
            error: Exception that occurred
            context: Execution context
            
        Returns:
            Error response dictionary
        """
        error_message = f"Tool '{tool_name}' failed: {str(error)}"
        logger.error(error_message)
        
        # Track error frequency
        if tool_name not in self.error_counts:
            self.error_counts[tool_name] = 0
        self.error_counts[tool_name] += 1
        
        # Provide actionable error information
        if isinstance(error, ValueError):
            suggestion = "Check input parameters and data types"
        elif isinstance(error, ConnectionError):
            suggestion = "Check network connectivity and service availability"
        elif isinstance(error, TimeoutError):
            suggestion = "Operation timed out - consider increasing timeout or simplifying request"
        else:
            suggestion = "Check logs for detailed error information"
        
        return {
            "success": False,
            "error": error_message,
            "suggestion": suggestion,
            "tool": tool_name,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def handle_state_transition_error(
        self,
        from_state: str,
        to_state: str,
        error: Exception
    ) -> bool:
        """
        Handle agent state transition errors.
        
        Args:
            from_state: Source state
            to_state: Target state
            error: Exception that occurred
            
        Returns:
            True if recovery successful, False otherwise
        """
        logger.error(
            f"State transition failed: {from_state} -> {to_state}: {error}"
        )
        
        # Attempt state recovery
        try:
            # Log state for debugging
            logger.info(f"Attempting to recover from state: {from_state}")
            
            # In production, implement actual recovery logic
            # For now, just log and return False
            return False
            
        except Exception as recovery_error:
            logger.critical(f"State recovery failed: {recovery_error}")
            return False


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.
    
    **Validates: Property 14: Agent Execution Mode Support**
    
    Prevents cascading failures by temporarily disabling failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                
                if elapsed >= self.timeout:
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(
                        f"Circuit breaker is OPEN - service unavailable "
                        f"(retry in {self.timeout - elapsed:.0f}s)"
                    )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit
            if self.state == "HALF_OPEN":
                logger.info("Circuit breaker reset to CLOSED state")
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            logger.warning(
                f"Circuit breaker failure {self.failure_count}/{self.failure_threshold}: {e}"
            )
            
            if self.failure_count >= self.failure_threshold:
                logger.error("Circuit breaker opened due to repeated failures")
                self.state = "OPEN"
            
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator for circuit breaker protection.
        
        Usage:
            @CircuitBreaker(failure_threshold=3, timeout=30)
            def my_external_call():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper


# Global instances for convenience
db_error_handler = DatabaseErrorHandler()
mql5_error_handler = MQL5BridgeErrorHandler()
agent_error_handler = AgentErrorHandler()
