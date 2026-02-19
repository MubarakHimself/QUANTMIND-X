"""
Metrics Collector for Agent Factory

Collects and aggregates metrics for factory-created agents.

**Validates: Phase 2.4 - Metrics Collector**
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class InvocationRecord:
    """Record of a single agent invocation."""
    invocation_id: str
    timestamp: datetime
    duration: float
    success: bool
    error_type: Optional[str] = None


class MetricsCollector:
    """
    Collects and aggregates metrics for agent invocations.
    
    Thread-safe metrics collection with aggregation.
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize the metrics collector.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (analyst, quantcode, copilot, router)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        # Counters
        self.total_invocations: int = 0
        self.successful_invocations: int = 0
        self.failed_invocations: int = 0
        
        # Duration tracking
        self.total_duration: float = 0.0
        self.min_duration: float = float('inf')
        self.max_duration: float = 0.0
        
        # Tool call tracking
        self.tool_calls: Dict[str, int] = {}
        
        # Error tracking
        self.errors: Dict[str, int] = {}
        
        # Invocation history (limited)
        self._invocation_history: List[InvocationRecord] = []
        self._max_history = 100
        
        # Thread lock
        self._lock = threading.Lock()
        
        logger.info(f"MetricsCollector initialized for agent: {agent_id}")
    
    def track_invocation(
        self,
        duration: float,
        success: bool,
        error: Optional[Exception] = None
    ) -> None:
        """
        Track an agent invocation.
        
        Args:
            duration: Invocation duration in seconds
            success: Whether invocation was successful
            error: Optional exception if failed
        """
        with self._lock:
            self.total_invocations += 1
            
            if success:
                self.successful_invocations += 1
            else:
                self.failed_invocations += 1
                if error:
                    error_type = type(error).__name__
                    self.errors[error_type] = self.errors.get(error_type, 0) + 1
            
            # Update duration stats
            self.total_duration += duration
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
            
            # Record to history
            invocation_record = InvocationRecord(
                invocation_id=f"{self.agent_id}_{self.total_invocations}",
                timestamp=datetime.utcnow(),
                duration=duration,
                success=success,
                error_type=type(error).__name__ if error else None
            )
            self._invocation_history.append(invocation_record)
            
            # Trim history if needed
            if len(self._invocation_history) > self._max_history:
                self._invocation_history = self._invocation_history[-self._max_history:]
    
    def track_tool_call(self, tool_name: str) -> None:
        """
        Track a tool call.
        
        Args:
            tool_name: Name of the tool called
        """
        with self._lock:
            self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            success_rate = (
                self.successful_invocations / self.total_invocations
                if self.total_invocations > 0
                else 0.0
            )
            
            avg_duration = (
                self.total_duration / self.total_invocations
                if self.total_invocations > 0
                else 0.0
            )
            
            # Get recent invocations (last 10)
            recent_invocations = [
                {
                    "invocation_id": r.invocation_id,
                    "timestamp": r.timestamp.isoformat(),
                    "duration": r.duration,
                    "success": r.success,
                    "error_type": r.error_type
                }
                for r in self._invocation_history[-10:]
            ]
            
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "total_invocations": self.total_invocations,
                "successful_invocations": self.successful_invocations,
                "failed_invocations": self.failed_invocations,
                "success_rate": round(success_rate, 4),
                "total_duration": round(self.total_duration, 4),
                "avg_duration": round(avg_duration, 4),
                "min_duration": round(self.min_duration, 4) if self.min_duration != float('inf') else 0,
                "max_duration": round(self.max_duration, 4),
                "tool_calls": dict(self.tool_calls),
                "errors": dict(self.errors),
                "recent_invocations": recent_invocations,
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics (lighter than full stats).
        
        Returns:
            Summary dictionary
        """
        with self._lock:
            success_rate = (
                self.successful_invocations / self.total_invocations
                if self.total_invocations > 0
                else 0.0
            )
            
            avg_duration = (
                self.total_duration / self.total_invocations
                if self.total_invocations > 0
                else 0.0
            )
            
            return {
                "agent_id": self.agent_id,
                "total_invocations": self.total_invocations,
                "success_rate": round(success_rate, 4),
                "avg_duration": round(avg_duration, 4),
                "tool_call_count": sum(self.tool_calls.values()),
                "error_count": sum(self.errors.values()),
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_invocations = 0
            self.successful_invocations = 0
            self.failed_invocations = 0
            self.total_duration = 0.0
            self.min_duration = float('inf')
            self.max_duration = 0.0
            self.tool_calls.clear()
            self.errors.clear()
            self._invocation_history.clear()
            
            logger.info(f"Metrics reset for agent: {self.agent_id}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status based on metrics.
        
        Returns:
            Health status dictionary
        """
        stats = self.get_stats()
        
        # Determine health status
        if stats["total_invocations"] == 0:
            status = "no_data"
        elif stats["success_rate"] >= 0.9:
            status = "healthy"
        elif stats["success_rate"] >= 0.7:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "agent_id": self.agent_id,
            "status": status,
            "success_rate": stats["success_rate"],
            "avg_duration": stats["avg_duration"],
            "error_count": stats["failed_invocations"],
        }
