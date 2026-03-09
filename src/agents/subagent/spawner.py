"""
Agent Spawner

Provides dynamic agent spawning capabilities for the Trading Floor.
Uses the Claude Agent SDK for real sub-agent management.

Features:
- Checkpoint system for long-running agents
- Heartbeat mechanism for health monitoring
- Progress tracking with ETA
- Graceful interrupt handling
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# SDK config removed - using department system instead
# Stub implementations for backward compatibility
def get_provider_config() -> Dict[str, Any]:
    """Deprecated - returns empty config. Use floor_manager instead."""
    return {}


def get_model_for_tier(tier: str = "standard") -> str:
    """Deprecated - returns default model."""
    return "claude-3-5-sonnet-20241022"


def get_thinking_config() -> Dict[str, Any]:
    """Deprecated - returns empty config."""
    return {}


class SDKAgentConfig:
    """Deprecated - use department system instead."""
    pass


def load_system_prompt(name: str) -> str:
    """Deprecated - use floor_manager instead."""
    return ""

from src.agents.memory import AgentMemory
from src.agents.checkpoint import (
    CheckpointManager,
    HeartbeatManager,
    ProgressTracker,
    LongRunningAgent,
    get_checkpoint_manager,
    get_heartbeat_manager,
)

# MCP Integration - optional import
try:
    from src.agents.mcp.integration import get_mcp_integration, MCPIntegration
    MCP_INTEGRATION_AVAILABLE = True
except ImportError:
    MCP_INTEGRATION_AVAILABLE = False
    MCPIntegration = None  # type: ignore
    get_mcp_integration = None  # type: ignore

logger = logging.getLogger(__name__)

# Check if SDK is available
SDK_AVAILABLE = False
try:
    from anthropic import Anthropic
    SDK_AVAILABLE = True
    logger.info("Anthropic SDK available for agent spawning")
except ImportError:
    logger.warning("Anthropic SDK not installed. Using fallback mode.")


@dataclass
class SubAgentConfig:
    """Configuration for spawning a sub-agent."""
    agent_type: str
    name: str
    parent_agent_id: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    pool_key: Optional[str] = None
    model_tier: str = "haiku"  # Workers use haiku by default


@dataclass
class SpawnedAgent:
    """Represents a spawned sub-agent."""
    agent_id: str
    agent_type: str
    task: str
    department: Optional[str]
    status: str = "running"  # running, completed, failed, terminated
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseAgentSpawner:
    """Base class for agent spawners."""

    def spawn(
        self,
        agent_type: str,
        task: str,
        department: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a new agent. Returns agent ID."""
        raise NotImplementedError

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all spawned agents."""
        raise NotImplementedError

    def terminate(self, agent_id: str) -> bool:
        """Terminate a spawned agent."""
        raise NotImplementedError

    async def get_result(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed agent."""
        raise NotImplementedError


class AgentSpawner(BaseAgentSpawner):
    """
    Real agent spawner using Claude Agent SDK.

    Spawns sub-agents that can execute tasks autonomously using
    the Claude API with tool access and streaming support.
    """

    def __init__(
        self,
        max_agents: int = 10,
        default_model_tier: str = "sonnet",
        enable_memory: bool = True,
        enable_mcp: bool = True,
        enable_checkpoint: bool = True,
        checkpoint_interval_seconds: int = 60,
        heartbeat_interval_seconds: int = 30,
    ):
        """
        Initialize the agent spawner.

        Args:
            max_agents: Maximum number of concurrent agents
            default_model_tier: Default model tier (opus, sonnet, haiku)
            enable_memory: Enable cross-session memory persistence
            enable_mcp: Enable MCP tool integration
            enable_checkpoint: Enable checkpoint system for long-running agents
            checkpoint_interval_seconds: Interval between automatic checkpoints
            heartbeat_interval_seconds: Heartbeat interval for health monitoring
        """
        self._client = None
        self._spawned_agents: Dict[str, SpawnedAgent] = {}
        self._max_agents = max_agents
        self._default_model_tier = default_model_tier
        self._provider_config = get_provider_config()
        self._thinking_config = get_thinking_config()
        self._lock = asyncio.Lock()
        self._enable_memory = enable_memory
        self._memory: Optional[AgentMemory] = None
        self._enable_mcp = enable_mcp
        self._mcp_integration: Optional[MCPIntegration] = None

        # Checkpoint and heartbeat support
        self._enable_checkpoint = enable_checkpoint
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._heartbeat_manager: Optional[HeartbeatManager] = None
        self._progress_trackers: Dict[str, ProgressTracker] = {}
        self._long_running_agents: Dict[str, LongRunningAgent] = {}

        if enable_checkpoint:
            self._checkpoint_manager = get_checkpoint_manager()
            self._heartbeat_manager = get_heartbeat_manager()

        if enable_memory:
            self._memory = AgentMemory()

        if enable_mcp and MCP_INTEGRATION_AVAILABLE:
            self._mcp_integration = get_mcp_integration()

        logger.info(
            f"AgentSpawner initialized: max_agents={max_agents}, "
            f"provider={self._provider_config['provider']}, "
            f"memory={'enabled' if enable_memory else 'disabled'}, "
            f"mcp={'enabled' if enable_mcp and MCP_INTEGRATION_AVAILABLE else 'disabled'}, "
            f"checkpoint={'enabled' if enable_checkpoint else 'disabled'}"
        )

    @property
    def client(self) -> Optional["Anthropic"]:
        """Lazy-initialize Anthropic-compatible client."""
        if self._client is None and SDK_AVAILABLE:
            client_kwargs = {"api_key": self._provider_config["api_key"]}

            if self._provider_config.get("base_url"):
                client_kwargs["base_url"] = self._provider_config["base_url"]

            self._client = Anthropic(**client_kwargs)
            logger.info(f"Agent spawner client initialized for provider: {self._provider_config['provider']}")
        return self._client

    def spawn(
        self,
        agent_type: str,
        task: str,
        department: Optional[str] = None,
        session_id: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Spawn a new agent synchron:
            agent_type: Type of agent to spawn (analyst, coder, researcher, etc.)
            task: Task description for the agent
            department:ously.

        Args Department context (optional)
            session_id: Session ID for memory persistence (optional)
            mcp_servers: List of MCP server IDs to load tools from (optional)
            **kwargs: Additional parameters (model_tier, tools, etc.)

        Returns:
            Agent ID for the spawned agent
        """
        # Generate unique agent ID
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"

        # Get model tier from kwargs or use default
        model_tier = kwargs.get("model_tier", self._default_model_tier)

        # Create agent record
        agent = SpawnedAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            task=task,
            department=department,
            status="running",
        )

        # Store agent
        self._spawned_agents[agent_id] = agent

        # Store task context to memory if enabled
        if self._enable_memory and self._memory and session_id:
            self._memory.agent_id = agent_id
            self._memory.session_id = session_id
            self._memory.store(
                key="current_task",
                value=task,
                namespace="session",
                metadata={"agent_type": agent_type, "department": department},
            )

        # Load MCP tools if enabled and servers specified
        mcp_tools = None
        if self._enable_mcp and self._mcp_integration and mcp_servers:
            mcp_tools = mcp_servers

        # Add MCP tools to options
        if mcp_tools:
            kwargs["mcp_servers"] = mcp_tools

        # Execute asynchronously in background (fire and forget for sync interface)
        asyncio.create_task(self._execute_agent(agent_id, task, model_tier, kwargs))

        logger.info(f"Spawned agent {agent_id} of type {agent_type} with MCP servers: {mcp_servers}")
        return agent_id

    async def _execute_agent(
        self,
        agent_id: str,
        task: str,
        model_tier: str,
        options: Dict[str, Any],
    ) -> None:
        """Execute an agent's task asynchronously."""
        try:
            # Get memory context if enabled
            memory_context = ""
            if self._enable_memory and self._memory:
                self._memory.agent_id = agent_id
                context = self._memory.get_context(namespace="session")
                if context:
                    memory_context = "\n\n## Previous Context\n"
                    for key, value in context.items():
                        memory_context += f"- {key}: {value}\n"

            if not SDK_AVAILABLE or not self.client:
                # Fallback mode - simulate execution
                await asyncio.sleep(0.5)
                self._spawned_agents[agent_id].result = {
                    "status": "completed",
                    "output": f"Agent {agent_id} processed task: {task[:100]}...",
                    "mode": "fallback",
                }
                self._spawned_agents[agent_id].status = "completed"
                return

            # Get model for tier
            model = get_model_for_tier(model_tier)

            # Build system prompt based on agent type (with memory context)
            system_prompt = await self._build_agent_prompt(agent_id, options, memory_context)

            # Prepare thinking parameter for GLM models
            thinking_param = None
            if self._provider_config.get("provider") == "zai" and self._thinking_config.get("type") != "disabled":
                thinking_param = self._thinking_config

            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": task}],
                thinking=thinking_param,
            )

            # Extract output
            output = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    output += block.text

            # Update agent with result
            self._spawned_agents[agent_id].result = {
                "status": "completed",
                "output": output,
                "model": model,
                "provider": self._provider_config["provider"],
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
            self._spawned_agents[agent_id].status = "completed"

            logger.info(f"Agent {agent_id} completed successfully")

        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            self._spawned_agents[agent_id].error = str(e)
            self._spawned_agents[agent_id].status = "failed"

    def spawn_subagent(self, config: SubAgentConfig) -> SpawnedAgent:
        """
        Spawn a sub-agent using SubAgentConfig.

        Args:
            config: SubAgentConfig with agent configuration

        Returns:
            SpawnedAgent instance
        """
        # Generate unique agent ID
        agent_id = f"{config.name}_{uuid.uuid4().hex[:8]}"

        # Get model tier
        model_tier = config.model_tier

        # Build task description
        task = config.input_data.get("task", "Execute assigned task")
        department = config.input_data.get("department", config.pool_key)

        # Create agent record
        agent = SpawnedAgent(
            agent_id=agent_id,
            agent_type=config.agent_type,
            task=task,
            department=department,
            status="running",
        )

        # Store agent
        self._spawned_agents[agent_id] = agent

        # Execute asynchronously
        asyncio.create_task(self._execute_agent(
            agent_id,
            task,
            model_tier,
            config.input_data,
        ))

        logger.info(f"Spawned sub-agent {agent_id} of type {config.agent_type} for department {department}")
        return agent

    async def _build_agent_prompt(
        self,
        agent_id: str,
        options: Dict[str, Any],
        memory_context: str = "",
    ) -> str:
        """Build system prompt for the agent based on its type."""
        agent_type = self._spawned_agents[agent_id].agent_type
        department = self._spawned_agents[agent_id].department

        base_prompt = f"""You are a {agent_type} sub-agent in the QuantMind Trading Floor.

Your role is to assist with tasks delegated to you by the Floor Manager or department heads.

{memory_context}
"""

        if department:
            base_prompt += f"Current department context: {department}\n\n"

        # Add MCP tools information if available
        mcp_servers = options.get("mcp_servers")
        if mcp_servers and self._enable_mcp and self._mcp_integration:
            try:
                tools = await self._mcp_integration.get_agent_tools(agent_id, mcp_servers)
                if tools:
                    tool_descriptions = "\n".join([
                        f"- {tool.name}: {tool.description}"
                        for tool in tools[:20]  # Limit to first 20 tools
                    ])
                    base_prompt += f"""Available MCP tools:
{tool_descriptions}
"""
                    if len(tools) > 20:
                        base_prompt += f"\n... and {len(tools) - 20} more tools.\n"
            except Exception as e:
                logger.warning(f"Failed to load MCP tools for agent {agent_id}: {e}")

        base_prompt += """Guidelines:
- Complete tasks efficiently and accurately
- Report results clearly
- If you encounter errors, explain them and suggest solutions
- Follow best practices for your task type
"""

        return base_prompt

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all spawned agents.

        Returns:
            List of agent info dictionaries
        """
        return [
            {
                "id": agent.agent_id,
                "type": agent.agent_type,
                "task": agent.task,
                "department": agent.department,
                "status": agent.status,
                "created_at": agent.created_at.isoformat(),
            }
            for agent in self._spawned_agents.values()
        ]

    def terminate(self, agent_id: str) -> bool:
        """
        Terminate a spawned agent.

        Args:
            agent_id: ID of agent to terminate

        Returns:
            True if terminated successfully
        """
        if agent_id in self._spawned_agents:
            self._spawned_agents[agent_id].status = "terminated"
            logger.info(f"Terminated agent {agent_id}")
            return True
        return False

    async def get_result(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Result dictionary or None if not found
        """
        if agent_id in self._spawned_agents:
            return self._spawned_agents[agent_id].result
        return None

    def get_status(self, agent_id: str) -> Optional[str]:
        """Get the status of an agent."""
        if agent_id in self._spawned_agents:
            return self._spawned_agents[agent_id].status
        return None

    def store_memory(
        self,
        key: str,
        value: str,
        namespace: str = "default",
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a memory entry.

        Args:
            key: Memory key
            value: Memory value
            namespace: Namespace (default: "default")
            agent_id: Agent ID (uses spawned agent if not provided)

        Returns:
            Entry ID if successful
        """
        if not self._enable_memory or not self._memory:
            logger.warning("Memory not enabled")
            return None

        self._memory.agent_id = agent_id
        return self._memory.store(key, value, namespace=namespace)

    def retrieve_memory(self, key: str, namespace: str = "default") -> Optional[str]:
        """
        Retrieve a memory entry.

        Args:
            key: Memory key
            namespace: Namespace

        Returns:
            Memory value or None
        """
        if not self._enable_memory or not self._memory:
            return None
        return self._memory.retrieve(key, namespace)

    def search_memory(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search memory entries.

        Args:
            query: Search query
            namespace: Optional namespace filter
            limit: Max results

        Returns:
            List of matching entries
        """
        if not self._enable_memory or not self._memory:
            return []
        return self._memory.search(query, namespace=namespace, limit=limit)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._enable_memory or not self._memory:
            return {"enabled": False}
        return self._memory.get_stats()

    async def stream_agent(
        self,
        agent_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent execution events.

        Args:
            agent_id: ID of the agent to stream

        Yields:
            Event dictionaries
        """
        if agent_id not in self._spawned_agents:
            yield {"type": "error", "error": f"Agent {agent_id} not found"}
            return

        agent = self._spawned_agents[agent_id]

        yield {"type": "started", "agent_id": agent_id, "task": agent.task}

        # Wait for completion or poll for updates
        while agent.status == "running":
            await asyncio.sleep(0.5)
            yield {"type": "status", "status": "running"}

        # Yield final result
        if agent.status == "completed":
            yield {
                "type": "completed",
                "agent_id": agent_id,
                "result": agent.result,
            }
        elif agent.status == "failed":
            yield {
                "type": "error",
                "agent_id": agent_id,
                "error": agent.error,
            }
        elif agent.status == "terminated":
            yield {"type": "terminated", "agent_id": agent_id}

    # ==================== Long-Running Agent Methods ====================

    def spawn_long_running(
        self,
        agent_type: str,
        task: str,
        department: Optional[str] = None,
        session_id: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        total_steps: int = 100,
        **kwargs: Any,
    ) -> str:
        """
        Spawn a long-running agent with checkpoint support.

        Args:
            agent_type: Type of agent to spawn
            task: Task description for the agent
            department: Department context (optional)
            session_id: Session ID for continuity (optional)
            mcp_servers: List of MCP server IDs to load tools from
            total_steps: Total steps for progress tracking
            **kwargs: Additional parameters

        Returns:
            Agent ID for the spawned agent
        """
        if not self._enable_checkpoint:
            logger.warning("Checkpoint not enabled, falling back to regular spawn")
            return self.spawn(agent_type, task, department, session_id, mcp_servers, **kwargs)

        # Generate unique agent ID
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        session_id = session_id or str(uuid.uuid4())

        # Create agent record
        agent = SpawnedAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            task=task,
            department=department,
            status="running",
        )
        self._spawned_agents[agent_id] = agent

        # Initialize progress tracker
        self._progress_trackers[agent_id] = ProgressTracker(total_steps=total_steps)

        # Create long-running agent wrapper
        long_running_agent = LongRunningAgent(
            agent_id=agent_id,
            task_id=agent_id,
            session_id=session_id,
            checkpoint_manager=self._checkpoint_manager,
            heartbeat_manager=self._heartbeat_manager,
        )
        self._long_running_agents[agent_id] = long_running_agent

        # Add initial message
        long_running_agent.add_message("user", task)

        # Start the agent
        asyncio.create_task(self._execute_long_running_agent(
            agent_id, task, department, session_id, mcp_servers, kwargs
        ))

        logger.info(f"Spawned long-running agent {agent_id} with checkpoint support")
        return agent_id

    async def _execute_long_running_agent(
        self,
        agent_id: str,
        task: str,
        department: Optional[str],
        session_id: str,
        mcp_servers: Optional[List[str]],
        options: Dict[str, Any],
    ) -> None:
        """Execute a long-running agent with checkpoint support."""
        try:
            long_running_agent = self._long_running_agents.get(agent_id)
            if not long_running_agent:
                return

            # Start the agent
            await long_running_agent.start()
            await long_running_agent.update_progress(step_name="Initializing")

            # Execute using parent method logic
            model_tier = options.get("model_tier", self._default_model_tier)

            # Get memory context if enabled
            memory_context = ""
            if self._enable_memory and self._memory:
                context = self._memory.get_context(namespace="session")
                if context:
                    memory_context = "\n\n## Previous Context\n"
                    for key, value in context.items():
                        memory_context += f"- {key}: {value}\n"

            if not SDK_AVAILABLE or not self.client:
                # Fallback mode - simulate execution with progress
                for i in range(10):
                    if agent_id not in self._long_running_agents:
                        break
                    await asyncio.sleep(0.5)
                    await long_running_agent.update_progress(
                        completed_steps=i + 1,
                        step_name=f"Processing step {i + 1}/10"
                    )

                self._spawned_agents[agent_id].result = {
                    "status": "completed",
                    "output": f"Agent {agent_id} processed task: {task[:100]}...",
                    "mode": "fallback",
                }
                self._spawned_agents[agent_id].status = "completed"
                await long_running_agent.stop("completed")
                return

            # Get model for tier
            model = get_model_for_tier(model_tier)

            # Build system prompt
            system_prompt = await self._build_agent_prompt(agent_id, options, memory_context)

            # Prepare thinking parameter for GLM models
            thinking_param = None
            if self._provider_config.get("provider") == "zai" and self._thinking_config.get("type") != "disabled":
                thinking_param = self._thinking_config

            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": task}],
                thinking=thinking_param,
            )

            # Extract output
            output = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    output += block.text
                    # Update progress with each chunk
                    await long_running_agent.update_progress(
                        step_name=f"Processing output: {len(output)} chars"
                    )

            # Update agent with result
            self._spawned_agents[agent_id].result = {
                "status": "completed",
                "output": output,
                "model": model,
                "provider": self._provider_config["provider"],
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
            self._spawned_agents[agent_id].status = "completed"

            # Stop long-running agent gracefully
            await long_running_agent.stop("completed")

            logger.info(f"Long-running agent {agent_id} completed successfully")

        except Exception as e:
            logger.error(f"Long-running agent {agent_id} failed: {e}")
            self._spawned_agents[agent_id].error = str(e)
            self._spawned_agents[agent_id].status = "failed"

            long_running_agent = self._long_running_agents.get(agent_id)
            if long_running_agent:
                await long_running_agent.stop("failed")

    async def save_checkpoint(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Save checkpoint for a running agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Checkpoint data or None
        """
        if not self._enable_checkpoint:
            return None

        long_running_agent = self._long_running_agents.get(agent_id)
        if not long_running_agent:
            return None

        checkpoint = await long_running_agent.save_checkpoint()
        if checkpoint:
            return {
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_number": checkpoint.checkpoint_number,
                "progress_percent": checkpoint.progress.get("percent", 0),
                "created_at": checkpoint.created_at,
            }
        return None

    async def resume_from_checkpoint(
        self,
        agent_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resume an agent from a checkpoint.

        Args:
            agent_id: Original agent identifier
            checkpoint_id: Specific checkpoint to resume from

        Returns:
            New agent ID if successful
        """
        if not self._enable_checkpoint:
            return None

        # Find original session
        original_agent = self._spawned_agents.get(agent_id)
        if not original_agent:
            return None

        # Get checkpoint
        if checkpoint_id:
            checkpoint = self._checkpoint_manager.get_checkpoint(
                original_agent.agent_type, agent_id, checkpoint_id
            )
        else:
            checkpoint = self._checkpoint_manager.get_latest_checkpoint(
                original_agent.agent_type, agent_id
            )

        if not checkpoint:
            logger.warning(f"No checkpoint found for agent {agent_id}")
            return None

        # Spawn new agent with restored state
        new_agent_id = self.spawn_long_running(
            agent_type=checkpoint.metadata.get("agent_type", original_agent.agent_type),
            task=original_agent.task,
            department=original_agent.department,
            session_id=checkpoint.session_id,
        )

        logger.info(f"Resumed agent {new_agent_id} from checkpoint for {agent_id}")
        return new_agent_id

    def get_progress(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get progress for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Progress data or None
        """
        tracker = self._progress_trackers.get(agent_id)
        if tracker:
            return {
                "progress_percent": tracker.progress_percent,
                "eta_seconds": tracker.eta_seconds,
                "completed_steps": tracker.completed_steps,
                "total_steps": tracker.total_steps,
            }

        long_running_agent = self._long_running_agents.get(agent_id)
        if long_running_agent:
            return {
                "progress_percent": long_running_agent.progress_tracker.progress_percent,
                "eta_seconds": long_running_agent.progress_tracker.eta_seconds,
                "completed_steps": long_running_agent.progress_tracker.completed_steps,
                "total_steps": long_running_agent.progress_tracker.total_steps,
            }

        return None

    async def get_heartbeat_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get heartbeat status for an agent."""
        if not self._enable_checkpoint or not self._heartbeat_manager:
            return None

        heartbeat = await self._heartbeat_manager.get_heartbeat(agent_id, agent_id)
        if heartbeat:
            return {
                "status": heartbeat.status,
                "last_update": heartbeat.last_update,
                "progress": heartbeat.progress * 100,
                "current_step": heartbeat.current_step,
                "stall_count": heartbeat.stall_count,
            }
        return None


class MockAgentSpawner(BaseAgentSpawner):
    """
    Mock agent spawner for testing.

    Provides a minimal interface without making actual API calls.
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
        """Spawn a new agent (mock)."""
        agent_id = f"{agent_type}_{len(self._spawned_agents)}"
        self._spawned_agents.append({
            "id": agent_id,
            "type": agent_type,
            "task": task,
            "department": department,
            "status": "completed",
        })
        return agent_id

    def list_agents(self) -> list:
        """List all spawned agents (mock)."""
        return self._spawned_agents.copy()

    def terminate(self, agent_id: str) -> bool:
        """Terminate a spawned agent (mock)."""
        self._spawned_agents = [a for a in self._spawned_agents if a["id"] != agent_id]
        return True


# Global spawner instance
_spawner: Optional[BaseAgentSpawner] = None


def get_spawner(use_mock: bool = False) -> BaseAgentSpawner:
    """
    Get the global agent spawner instance.

    Args:
        use_mock: Force use of mock spawner (for testing)

    Returns:
        AgentSpawner or MockAgentSpawner instance
    """
    global _spawner
    if _spawner is None:
        if use_mock:
            _spawner = MockAgentSpawner()
        else:
            _spawner = AgentSpawner()
    return _spawner


def reset_spawner():
    """Reset the global spawner instance (mainly for testing)."""
    global _spawner
    _spawner = None
