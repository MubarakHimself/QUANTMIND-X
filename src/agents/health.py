"""
Agent Health Checker

Provides health check functionality for registered agents.

**Validates: Phase 3.2 - Health Checks**
"""

import logging
from typing import Dict, Any, List, Optional

from src.agents.registry import get_registry, AgentRegistry

logger = logging.getLogger(__name__)


# Health thresholds
DEFAULT_SUCCESS_RATE_THRESHOLD = 0.8
DEFAULT_AVG_DURATION_THRESHOLD = 30.0  # seconds


class AgentHealthChecker:
    """
    Health checker for registered agents.
    
    Provides health status checks based on metrics and availability.
    """
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        success_rate_threshold: float = DEFAULT_SUCCESS_RATE_THRESHOLD,
        avg_duration_threshold: float = DEFAULT_AVG_DURATION_THRESHOLD,
    ):
        """
        Initialize the health checker.
        
        Args:
            registry: Agent registry to check
            success_rate_threshold: Minimum success rate for healthy status
            avg_duration_threshold: Maximum average duration for healthy status
        """
        self.registry = registry or get_registry()
        self.success_rate_threshold = success_rate_threshold
        self.avg_duration_threshold = avg_duration_threshold
        
        logger.info("AgentHealthChecker initialized")
    
    def check_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Check health of a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Health status dictionary
        """
        agent = self.registry.get(agent_id)
        
        if agent is None:
            return {
                "agent_id": agent_id,
                "status": "not_found",
                "healthy": False,
                "error": "Agent not found in registry",
            }
        
        try:
            stats = agent.get_stats()
            health_status = agent.get_health_status()
            
            # Determine health based on metrics
            success_rate = stats.get("success_rate", 0)
            avg_duration = stats.get("avg_duration", 0)
            total_invocations = stats.get("total_invocations", 0)
            
            # Check thresholds
            is_healthy = (
                success_rate >= self.success_rate_threshold and
                avg_duration <= self.avg_duration_threshold and
                total_invocations > 0
            )
            
            # Determine detailed status
            if total_invocations == 0:
                status = "no_data"
            elif success_rate >= self.success_rate_threshold and avg_duration <= self.avg_duration_threshold:
                status = "healthy"
            elif success_rate >= 0.5:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "name": agent.name,
                "status": status,
                "healthy": is_healthy,
                "metrics": {
                    "total_invocations": total_invocations,
                    "success_rate": success_rate,
                    "avg_duration": avg_duration,
                    "failed_invocations": stats.get("failed_invocations", 0),
                    "tool_call_count": sum(stats.get("tool_calls", {}).values()),
                },
                "thresholds": {
                    "success_rate": self.success_rate_threshold,
                    "avg_duration": self.avg_duration_threshold,
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking health for {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "status": "error",
                "healthy": False,
                "error": str(e),
            }
    
    def check_all(self) -> Dict[str, Any]:
        """
        Check health of all registered agents.
        
        Returns:
            Dictionary with health status for all agents
        """
        agent_ids = self.registry.list_agents()
        
        results = []
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        not_found_count = 0
        
        for agent_id in agent_ids:
            result = self.check_agent(agent_id)
            results.append(result)
            
            status = result.get("status")
            if status == "healthy":
                healthy_count += 1
            elif status == "degraded":
                degraded_count += 1
            elif status in ("unhealthy", "error"):
                unhealthy_count += 1
            elif status == "not_found":
                not_found_count += 1
        
        # Determine overall status
        if healthy_count == len(agent_ids):
            overall_status = "healthy"
        elif unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        elif len(agent_ids) == 0:
            overall_status = "no_agents"
        else:
            overall_status = "unknown"
        
        return {
            "overall_status": overall_status,
            "total_agents": len(agent_ids),
            "healthy_agents": healthy_count,
            "degraded_agents": degraded_count,
            "unhealthy_agents": unhealthy_count,
            "agents": results,
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a brief health summary.
        
        Returns:
            Health summary dictionary
        """
        agent_ids = self.registry.list_agents()
        
        if not agent_ids:
            return {
                "status": "no_agents",
                "total": 0,
                "healthy": 0,
            }
        
        healthy = 0
        degraded = 0
        unhealthy = 0
        
        for agent_id in agent_ids:
            result = self.check_agent(agent_id)
            status = result.get("status")
            
            if status == "healthy":
                healthy += 1
            elif status == "degraded":
                degraded += 1
            elif status in ("unhealthy", "error"):
                unhealthy += 1
        
        if unhealthy > 0:
            overall = "unhealthy"
        elif degraded > 0:
            overall = "degraded"
        else:
            overall = "healthy"
        
        return {
            "status": overall,
            "total": len(agent_ids),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
        }


# Convenience function
_health_checker: Optional[AgentHealthChecker] = None


def get_health_checker() -> AgentHealthChecker:
    """
    Get the global health checker.
    
    Returns:
        AgentHealthChecker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = AgentHealthChecker()
    return _health_checker


def check_agent_health(agent_id: str) -> Dict[str, Any]:
    """
    Check health of a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Health status dictionary
    """
    return get_health_checker().check_agent(agent_id)


def check_all_agents_health() -> Dict[str, Any]:
    """
    Check health of all agents.
    
    Returns:
        Health status dictionary
    """
    return get_health_checker().check_all()
