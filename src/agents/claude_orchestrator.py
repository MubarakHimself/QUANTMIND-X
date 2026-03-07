"""
Claude Orchestrator for v2 Agent Stack

Core execution engine that replaces AgentFactory + CompiledAgent.
Spawns Claude CLI subprocesses with MCP configurations for each agent.

**Phase 2.1 - Claude Orchestrator**

Features:
- Checkpoint system for long-running agents
- Heartbeat mechanism for health monitoring
- Progress tracking with ETA estimation
- Graceful interrupt handling
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator, List

from src.agents.claude_config import (
    ClaudeAgentConfig,
    AGENT_CONFIGS,
    get_agent_config,
    initialize_hooks,
)
from src.agents.streaming import get_stream_handler
from src.agents.checkpoint import (
    CheckpointManager,
    HeartbeatManager,
    ProgressTracker,
    LongRunningAgent,
    AgentState,
    get_checkpoint_manager,
    get_heartbeat_manager,
)

logger = logging.getLogger(__name__)

# Initialize hooks on module load
initialize_hooks()


class ClaudeOrchestrator:
    """
    Orchestrator for Claude-powered agents.

    Handles task submission, subprocess spawning, result polling,
    and streaming events via file-based communication.

    Features:
    - Long-running agent support with checkpoint/resume
    - Heartbeat mechanism for health monitoring
    - Progress tracking with ETA
    - Graceful interrupt handling
    """

    def __init__(
        self,
        workspaces_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval_seconds: int = 60,
        heartbeat_interval_seconds: int = 30,
    ):
        """
        Initialize the orchestrator.

        Args:
            workspaces_dir: Optional override for workspaces directory
            checkpoint_dir: Optional override for checkpoint directory
            checkpoint_interval_seconds: Interval between automatic checkpoints
            heartbeat_interval_seconds: Heartbeat interval for health monitoring
        """
        self.workspaces_dir = workspaces_dir or Path(
            os.getenv("WORKSPACES_DIR", "/app/workspaces")
        )
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}

        # Initialize checkpoint and heartbeat managers
        self.checkpoint_manager = get_checkpoint_manager()
        self.heartbeat_manager = get_heartbeat_manager()
        self.checkpoint_interval = checkpoint_interval_seconds

        # Track progress for long-running tasks
        self._progress_trackers: Dict[str, ProgressTracker] = {}

        # Long-running agent wrappers
        self._long_running_agents: Dict[str, LongRunningAgent] = {}

        # Start heartbeat monitoring
        asyncio.create_task(self.heartbeat_manager.start_monitoring())

        logger.info(f"ClaudeOrchestrator initialized with workspaces: {self.workspaces_dir}")
    
    async def submit_task(
        self,
        agent_id: str,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Submit a task to an agent.
        
        Writes task file to workspaces/{agent_id}/tasks/{task_id}.json
        and spawns Claude CLI subprocess.
        
        Args:
            agent_id: Agent identifier (analyst, quantcode, etc.)
            messages: List of message dicts with role and content
            context: Optional context dictionary
            session_id: Optional session ID for continuity
            
        Returns:
            task_id: Unique task identifier
            
        Raises:
            ValueError: If agent_id is invalid
            RuntimeError: If task submission fails
        """
        # Get agent config
        config = get_agent_config(agent_id)
        if not config:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        # Ensure directories exist
        config.ensure_directories()
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Build task payload
        task = {
            "task_id": task_id,
            "agent_id": agent_id,
            "created_at": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "payload": {
                "messages": messages,
                "context": context or {},
            },
            "status": "pending",
        }
        
        # Run pre-hooks
        task = await self._run_pre_hooks(agent_id, task)
        
        # Write task file
        task_path = config.tasks_dir / f"{task_id}.json"
        try:
            with open(task_path, "w") as f:
                json.dump(task, f, indent=2)
            logger.info(f"Task {task_id} written to {task_path}")
        except Exception as e:
            logger.error(f"Failed to write task file: {e}")
            raise RuntimeError(f"Task file write failed: {e}")
        
        # Spawn Claude subprocess
        try:
            await self._spawn_claude(agent_id, task_id, config)
        except Exception as e:
            logger.error(f"Failed to spawn Claude: {e}")
            task["status"] = "failed"
            task["error"] = str(e)
            with open(task_path, "w") as f:
                json.dump(task, f, indent=2)

            # Publish agent failed event
            try:
                stream_handler = get_stream_handler()
                await stream_handler.publish_agent_failed(
                    agent_id, task_id, task_id,
                    error=str(e),
                    metadata={"status": "spawn_failed"}
                )
            except Exception as stream_err:
                logger.warning(f"Failed to publish stream event: {stream_err}")

            raise RuntimeError(f"Claude spawn failed: {e}")

        # Publish agent started event
        try:
            stream_handler = get_stream_handler()
            await stream_handler.publish_agent_started(
                agent_id, task_id, task_id,
                metadata={"status": "running", "session_id": session_id}
            )
        except Exception as stream_err:
            logger.warning(f"Failed to publish stream event: {stream_err}")
        
        return task_id
    
    async def _spawn_claude(
        self,
        agent_id: str,
        task_id: str,
        config: ClaudeAgentConfig,
    ) -> None:
        """
        Spawn Claude CLI subprocess for task execution.
        
        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            config: Agent configuration
        """
        # Build command
        task_path = config.tasks_dir / f"{task_id}.json"
        
        cmd = [
            "claude",
            "--mcp-config", str(config.mcp_config_path),
            "--system-prompt", str(config.system_prompt_path),
            "--print",  # Output to stdout
            "--task-file", str(task_path),
        ]
        
        # Build environment
        env = os.environ.copy()
        env.update(config.env_vars)
        env["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
        env["CLAUDE_TASK_ID"] = task_id
        env["CLAUDE_AGENT_ID"] = agent_id
        
        logger.info(f"Spawning Claude for {agent_id}/{task_id}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(config.workspace),
        )
        
        # Track process
        self._active_processes[task_id] = process
        
        # Start background task to handle output
        asyncio.create_task(
            self._handle_process_output(process, agent_id, task_id, config)
        )
    
    async def _handle_process_output(
        self,
        process: asyncio.subprocess.Process,
        agent_id: str,
        task_id: str,
        config: ClaudeAgentConfig,
    ) -> None:
        """
        Handle subprocess stdout/stderr and write result file.
        
        Args:
            process: The subprocess
            agent_id: Agent identifier
            task_id: Task identifier
            config: Agent configuration
        """
        stdout_data = []
        stderr_data = []
        tool_calls = []
        
        try:
            # Read stdout with timeout
            async def read_stdout():
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if line_str:
                        stdout_data.append(line_str)
                        logger.debug(f"[{agent_id}] stdout: {line_str[:100]}...")
                        
                        # Try to parse tool calls from output
                        try:
                            if line_str.startswith("{"):
                                data = json.loads(line_str)
                                if data.get("type") == "tool_call":
                                    tool_calls.append(data)
                        except json.JSONDecodeError:
                            pass
            
            async def read_stderr():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if line_str:
                        stderr_data.append(line_str)
                        logger.warning(f"[{agent_id}] stderr: {line_str[:100]}...")
            
            # Run both readers concurrently with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stdout(),
                        read_stderr(),
                    ),
                    timeout=config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Process timeout for {task_id}, terminating...")
                process.kill()
                await process.wait()
                stderr_data.append(f"Process timed out after {config.timeout_seconds}s")
            
            # Wait for process to complete
            return_code = await process.wait()
            
            # Build result
            result = {
                "task_id": task_id,
                "agent_id": agent_id,
                "completed_at": datetime.utcnow().isoformat(),
                "status": "completed" if return_code == 0 else "failed",
                "output": "\n".join(stdout_data),
                "tool_calls": tool_calls,
                "error": "\n".join(stderr_data) if stderr_data else None,
                "return_code": return_code,
            }
            
            # Run post-hooks
            task_path = config.tasks_dir / f"{task_id}.json"
            task = {}
            if task_path.exists():
                with open(task_path, "r") as f:
                    task = json.load(f)
            
            result = await self._run_post_hooks(agent_id, task, result)
            
            # Write result file
            result_path = config.results_dir / f"{task_id}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Result written for {task_id}: status={result['status']}")
            
        except Exception as e:
            logger.error(f"Error handling process output: {e}")
            
            # Write error result
            result = {
                "task_id": task_id,
                "agent_id": agent_id,
                "completed_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "output": "\n".join(stdout_data) if stdout_data else "",
                "tool_calls": tool_calls,
                "error": str(e),
            }
            
            result_path = config.results_dir / f"{task_id}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
        
        finally:
            # Cleanup process tracking
            self._active_processes.pop(task_id, None)
    
    async def get_result(
        self,
        agent_id: str,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get result for a task (polling-based).
        
        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            
        Returns:
            Result dict if complete, None if still pending
        """
        config = get_agent_config(agent_id)
        if not config:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        result_path = config.results_dir / f"{task_id}.json"
        
        if not result_path.exists():
            return None
        
        # Use executor for file read to avoid blocking
        loop = asyncio.get_event_loop()
        
        def read_result():
            with open(result_path, "r") as f:
                return json.load(f)
        
        return await loop.run_in_executor(None, read_result)
    
    async def stream_result(
        self,
        agent_id: str,
        task_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream result events for a task.
        
        Polls the result file every 500ms and emits events:
        - started: Task has begun
        - tool_call: Tool was called
        - progress: Progress update
        - completed: Task finished
        
        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            
        Yields:
            Event dictionaries with type and data
        """
        config = get_agent_config(agent_id)
        if not config:
            raise ValueError(f"Unknown agent: {agent_id}")

        task_path = config.tasks_dir / f"{task_id}.json"
        result_path = config.results_dir / f"{task_id}.json"

        # Get stream handler for SSE events
        try:
            stream_handler = get_stream_handler()
        except Exception:
            stream_handler = None
        
        # Emit started event
        yield {
            "type": "started",
            "task_id": task_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        last_output_len = 0
        last_tool_calls_count = 0
        
        while True:
            # Check for result file
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        result = json.load(f)
                    
                    # Emit tool_call events for new tool calls
                    tool_calls = result.get("tool_calls", [])
                    if len(tool_calls) > last_tool_calls_count:
                        for tc in tool_calls[last_tool_calls_count:]:
                            # Publish to SSE stream
                            if stream_handler:
                                try:
                                    await stream_handler.publish_tool_start(
                                        agent_id, task_id, task_id,
                                        tool_name=tc.get("name", "unknown"),
                                        arguments=tc.get("arguments", {})
                                    )
                                except Exception as se:
                                    logger.warning(f"Failed to publish tool_start: {se}")

                            yield {
                                "type": "tool_call",
                                "task_id": task_id,
                                "agent_id": agent_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": tc,
                            }
                        last_tool_calls_count = len(tool_calls)
                    
                    # Emit progress event if output grew
                    output = result.get("output", "")
                    if len(output) > last_output_len:
                        yield {
                            "type": "progress",
                            "task_id": task_id,
                            "agent_id": agent_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "output_delta": output[last_output_len:],
                                "total_length": len(output),
                            },
                        }
                        last_output_len = len(output)
                    
                    # Check if complete
                    if result.get("status") in ("completed", "failed"):
                        # Publish to SSE stream
                        if stream_handler:
                            try:
                                if result.get("status") == "completed":
                                    await stream_handler.publish_agent_completed(
                                        agent_id, task_id, task_id,
                                        result={
                                            "output": result.get("output", ""),
                                            "status": "completed"
                                        }
                                    )
                                else:
                                    await stream_handler.publish_agent_failed(
                                        agent_id, task_id, task_id,
                                        error=result.get("error", "Unknown error"),
                                        metadata=result
                                    )
                            except Exception as se:
                                logger.warning(f"Failed to publish completion event: {se}")

                        yield {
                            "type": "completed",
                            "task_id": task_id,
                            "agent_id": agent_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": result,
                        }
                        break
                        
                except Exception as e:
                    logger.warning(f"Error reading result file: {e}")
            
            # Also check task file for intermediate state
            if task_path.exists():
                try:
                    with open(task_path, "r") as f:
                        task = json.load(f)
                    
                    # Check for tool calls in progress
                    tool_calls_in_progress = task.get("tool_calls_in_progress", [])
                    for tc in tool_calls_in_progress:
                        if tc not in range(last_tool_calls_count):
                            yield {
                                "type": "tool_call",
                                "task_id": task_id,
                                "agent_id": agent_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": tc,
                            }
                except Exception as e:
                    logger.warning(f"Error reading task file: {e}")
            
            # Wait before next poll
            await asyncio.sleep(0.5)
    
    async def _run_pre_hooks(
        self,
        agent_id: str,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run pre-execution hooks for an agent.
        
        Args:
            agent_id: Agent identifier
            task: Task dictionary
            
        Returns:
            Modified task dictionary
        """
        config = get_agent_config(agent_id)
        if not config or not config.pre_hooks:
            return task
        
        for hook in config.pre_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    task = await hook(task)
                else:
                    task = hook(task)
                logger.debug(f"Pre-hook {hook.__name__} executed for {agent_id}")
            except Exception as e:
                logger.error(f"Pre-hook {hook.__name__} failed: {e}")
        
        return task
    
    async def _run_post_hooks(
        self,
        agent_id: str,
        task: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run post-execution hooks for an agent.
        
        Args:
            agent_id: Agent identifier
            task: Task dictionary
            result: Result dictionary
            
        Returns:
            Modified result dictionary
        """
        config = get_agent_config(agent_id)
        if not config or not config.post_hooks:
            return result
        
        for hook in config.post_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(task, result)
                else:
                    result = hook(task, result)
                logger.debug(f"Post-hook {hook.__name__} executed for {agent_id}")
            except Exception as e:
                logger.error(f"Post-hook {hook.__name__} failed: {e}")
        
        return result
    
    def get_task_status(self, agent_id: str, task_id: str) -> str:
        """
        Get current status of a task.
        
        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            
        Returns:
            Status string: pending, running, completed, failed, unknown
        """
        config = get_agent_config(agent_id)
        if not config:
            return "unknown"
        
        # Check if process is running
        if task_id in self._active_processes:
            return "running"
        
        # Check for result file
        result_path = config.results_dir / f"{task_id}.json"
        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    result = json.load(f)
                return result.get("status", "unknown")
            except Exception:
                return "unknown"
        
        # Check for task file
        task_path = config.tasks_dir / f"{task_id}.json"
        if task_path.exists():
            return "pending"
        
        return "unknown"
    
    async def cancel_task(self, agent_id: str, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            
        Returns:
            True if cancelled, False if not running
        """
        process = self._active_processes.get(task_id)
        if not process:
            return False
        
        try:
            process.kill()
            await process.wait()
            
            # Write cancelled result
            config = get_agent_config(agent_id)
            if config:
                result = {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "completed_at": datetime.utcnow().isoformat(),
                    "status": "cancelled",
                    "output": "",
                    "tool_calls": [],
                    "error": "Task cancelled by user",
                }
                result_path = config.results_dir / f"{task_id}.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    # ==================== Checkpoint & Long-Running Agent Methods ====================

    async def submit_long_running_task(
        self,
        agent_id: str,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        total_steps: int = 100,
    ) -> str:
        """
        Submit a task with long-running agent support (checkpoint/resume).

        Args:
            agent_id: Agent identifier
            messages: List of message dicts with role and content
            context: Optional context dictionary
            session_id: Optional session ID for continuity
            total_steps: Total steps for progress tracking

        Returns:
            task_id: Unique task identifier
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())

        # Initialize progress tracker
        self._progress_trackers[task_id] = ProgressTracker(total_steps=total_steps)

        # Create long-running agent wrapper
        long_running_agent = LongRunningAgent(
            agent_id=agent_id,
            task_id=task_id,
            session_id=session_id,
            checkpoint_manager=self.checkpoint_manager,
            heartbeat_manager=self.heartbeat_manager,
            checkpoint_interval_seconds=self.checkpoint_interval,
        )
        self._long_running_agents[task_id] = long_running_agent

        # Initialize conversation history
        long_running_agent.conversation_history = messages.copy()

        # Register for heartbeat and start
        await long_running_agent.start()

        # Update progress to started
        await long_running_agent.update_progress(step_name="Task started")

        # Submit task (spawns Claude subprocess)
        await self.submit_task(agent_id, messages, context, session_id)

        logger.info(
            f"Long-running task submitted: {agent_id}/{task_id} with {total_steps} steps"
        )
        return task_id

    async def save_checkpoint(
        self,
        agent_id: str,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Save checkpoint for a running task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Checkpoint data or None if task not found
        """
        long_running_agent = self._long_running_agents.get(task_id)
        if not long_running_agent:
            logger.warning(f"No long-running agent found for task {task_id}")
            return None

        checkpoint = await long_running_agent.save_checkpoint()
        if checkpoint:
            logger.info(f"Checkpoint saved for {agent_id}/{task_id}")
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
        task_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resume a task from a checkpoint.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            checkpoint_id: Specific checkpoint to resume from (latest if None)

        Returns:
            New task_id if successful, None otherwise
        """
        # Get checkpoint
        if checkpoint_id:
            checkpoint = self.checkpoint_manager.get_checkpoint(agent_id, task_id, checkpoint_id)
        else:
            checkpoint = self.checkpoint_manager.get_latest_checkpoint(agent_id, task_id)

        if not checkpoint:
            logger.warning(f"No checkpoint found for {agent_id}/{task_id}")
            return None

        # Resume from checkpoint
        long_running_agent = await LongRunningAgent.resume_from_checkpoint(
            checkpoint,
            self.checkpoint_manager,
            self.heartbeat_manager,
        )
        self._long_running_agents[task_id] = long_running_agent

        # Submit task with restored conversation
        new_task_id = await self.submit_task(
            agent_id,
            checkpoint.conversation_history,
            checkpoint.metadata,
            checkpoint.session_id,
        )

        logger.info(f"Resumed task {agent_id}/{task_id} from checkpoint #{checkpoint.checkpoint_number}")
        return new_task_id

    async def update_progress(
        self,
        agent_id: str,
        task_id: str,
        completed_steps: Optional[int] = None,
        step_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update progress for a long-running task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            completed_steps: Number of completed steps
            step_name: Description of current step

        Returns:
            Progress update or None if task not found
        """
        long_running_agent = self._long_running_agents.get(task_id)
        if not long_running_agent:
            # Fall back to progress tracker
            tracker = self._progress_trackers.get(task_id)
            if tracker:
                progress = await tracker.update(completed_steps, step_name)
                return {
                    "progress_percent": progress.progress_percent,
                    "current_step": progress.current_step,
                    "eta_seconds": progress.eta_seconds,
                }
            return None

        progress = await long_running_agent.update_progress(completed_steps, step_name)
        return {
            "progress_percent": progress.progress_percent,
            "current_step": progress.current_step,
            "completed_steps": progress.completed_steps,
            "total_steps": progress.total_steps,
            "eta_seconds": progress.eta_seconds,
        }

    def get_progress(
        self,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get current progress for a task.

        Args:
            task_id: Task identifier

        Returns:
            Progress data or None if not found
        """
        tracker = self._progress_trackers.get(task_id)
        if tracker:
            return {
                "progress_percent": tracker.progress_percent,
                "eta_seconds": tracker.eta_seconds,
                "completed_steps": tracker.completed_steps,
                "total_steps": tracker.total_steps,
            }

        long_running_agent = self._long_running_agents.get(task_id)
        if long_running_agent:
            return {
                "progress_percent": long_running_agent.progress_tracker.progress_percent,
                "eta_seconds": long_running_agent.progress_tracker.eta_seconds,
                "completed_steps": long_running_agent.progress_tracker.completed_steps,
                "total_steps": long_running_agent.progress_tracker.total_steps,
            }

        return None

    def list_checkpoints(
        self,
        agent_id: str,
        task_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            List of checkpoint metadata
        """
        return self.checkpoint_manager.list_checkpoints(agent_id, task_id)

    async def get_heartbeat_status(
        self,
        agent_id: str,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get heartbeat status for a task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Heartbeat data or None if not registered
        """
        heartbeat = await self.heartbeat_manager.get_heartbeat(agent_id, task_id)
        if heartbeat:
            return {
                "status": heartbeat.status,
                "last_update": heartbeat.last_update,
                "progress": heartbeat.progress * 100,
                "current_step": heartbeat.current_step,
                "eta_seconds": heartbeat.eta_seconds,
            }
        return None

    async def get_all_heartbeats(self) -> List[Dict[str, Any]]:
        """
        Get heartbeat status for all monitored agents.

        Returns:
            List of heartbeat data
        """
        heartbeats = await self.heartbeat_manager.get_all_heartbeats()
        return [
            {
                "agent_id": hb.agent_id,
                "task_id": hb.task_id,
                "status": hb.status,
                "last_update": hb.last_update,
                "progress": hb.progress * 100,
                "current_step": hb.current_step,
                "stall_count": hb.stall_count,
            }
            for hb in heartbeats
        ]

    async def stop_long_running_task(
        self,
        agent_id: str,
        task_id: str,
        status: str = "completed",
    ) -> bool:
        """
        Stop a long-running task gracefully.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            status: Final status (completed, failed, cancelled)

        Returns:
            True if stopped successfully
        """
        long_running_agent = self._long_running_agents.get(task_id)
        if long_running_agent:
            await long_running_agent.stop(status)
            self._long_running_agents.pop(task_id, None)
            self._progress_trackers.pop(task_id, None)
            logger.info(f"Long-running task {agent_id}/{task_id} stopped: {status}")
            return True

        return False


# Global orchestrator instance
_orchestrator: Optional[ClaudeOrchestrator] = None


def get_orchestrator() -> ClaudeOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ClaudeOrchestrator()
    return _orchestrator