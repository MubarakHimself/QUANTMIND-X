"""
Agent Spawner

Provides dynamic agent spawning capabilities for the Trading Floor.
"""

from typing import Optional, Any


class MockAgentSpawner:
    """
    Mock agent spawner for testing.

    In production, this would interface with a real agent spawning system.
    For now, it provides a minimal interface for the Floor Manager.
    """

    def __init__(self):
        """Initialize the mock spawner."""
        self._spawned_agents = []

    def spawn(
        self,
        agent_type: str,
        task: str,
        department: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Spawn a new agent.

        Args:
            agent_type: Type of agent to spawn
            task: Task description for the agent
            department: Department context (optional)
            **kwargs: Additional parameters

        Returns:
            Agent ID for the spawned agent
        """
        agent_id = f"{agent_type}_{len(self._spawned_agents)}"
        self._spawned_agents.append({"id": agent_id, "type": agent_type, "task": task})
        return agent_id

    def list_agents(self) -> list:
        """List all spawned agents."""
        return self._spawned_agents.copy()

    def terminate(self, agent_id: str) -> bool:
        """
        Terminate a spawned agent.

        Args:
            agent_id: ID of agent to terminate

        Returns:
            True if terminated successfully
        """
        self._spawned_agents = [a for a in self._spawned_agents if a["id"] != agent_id]
        return True


# Global spawner instance
_spawner: Optional[MockAgentSpawner] = None


def get_spawner() -> MockAgentSpawner:
    """
    Get the global agent spawner instance.

    Returns:
        MockAgentSpawner instance
    """
    global _spawner
    if _spawner is None:
        _spawner = MockAgentSpawner()
    return _spawner


def reset_spawner():
    """Reset the global spawner instance (mainly for testing)."""
    global _spawner
    _spawner = None
