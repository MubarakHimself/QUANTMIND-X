"""
Checkpoint System for Long-Running Agents

Provides checkpoint saving, interrupt handling, heartbeat mechanism,
and progress tracking for persistent AI agents.

Based on Anthropic's agent documentation best practices for long-running agents.
"""

import asyncio
import json
import logging
import os
import signal
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"


class HeartbeatStatus(str, Enum):
    """Heartbeat status values."""
    ACTIVE = "active"
    STALLED = "stalled"
    RECOVERING = "recovering"
    DEAD = "dead"


@dataclass
class CheckpointData:
    """Checkpoint data structure."""
    checkpoint_id: str
    agent_id: str
    task_id: str
    session_id: str
    state: str
    created_at: str
    conversation_history: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    partial_results: Dict[str, Any]
    progress: Dict[str, Any]
    metadata: Dict[str, Any]
    checkpoint_number: int


@dataclass
class HeartbeatData:
    """Heartbeat data structure."""
    agent_id: str
    task_id: str
    status: str
    last_update: str
    progress: float
    eta_seconds: Optional[float]
    current_step: str
    stall_count: int = 0


@dataclass
class ProgressUpdate:
    """Progress update structure."""
    task_id: str
    agent_id: str
    progress_percent: float
    current_step: str
    total_steps: int
    completed_steps: int
    eta_seconds: Optional[float]
    message: str
    timestamp: str


class CheckpointManager:
    """
    Manages checkpoint creation, storage, and recovery for long-running agents.

    Features:
    - Periodic checkpoint saving at configurable intervals
    - Conversation history and tool output persistence
    - Progress tracking with ETA estimation
    - Automatic recovery from last checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval_seconds: int = 60,
        max_checkpoints: int = 10,
        enable_compression: bool = False,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint storage
            checkpoint_interval_seconds: Interval between automatic checkpoints
            max_checkpoints: Maximum checkpoints to retain per task
            enable_compression: Enable gzip compression for checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or Path(
            os.getenv("CHECKPOINT_DIR", str(Path.home() / ".quantmind" / "checkpoints"))
        )
        self.checkpoint_interval = checkpoint_interval_seconds
        self.max_checkpoints = max_checkpoints
        self.enable_compression = enable_compression

        # In-memory index for quick lookup
        self._checkpoints: Dict[str, List[CheckpointData]] = {}
        self._checkpoint_counts: Dict[str, int] = {}

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"CheckpointManager initialized: dir={self.checkpoint_dir}, "
            f"interval={checkpoint_interval_seconds}s, max={max_checkpoints}"
        )

    def _get_checkpoint_path(self, agent_id: str, task_id: str) -> Path:
        """Get the directory path for an agent's checkpoints."""
        return self.checkpoint_dir / agent_id / task_id

    def _get_checkpoint_file(self, checkpoint_id: str, agent_id: str, task_id: str) -> Path:
        """Get the file path for a specific checkpoint."""
        return self._get_checkpoint_path(agent_id, task_id) / f"{checkpoint_id}.json"

    async def save_checkpoint(
        self,
        agent_id: str,
        task_id: str,
        session_id: str,
        conversation_history: List[Dict[str, Any]],
        tool_outputs: List[Dict[str, Any]],
        partial_results: Dict[str, Any],
        progress: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointData:
        """
        Save a checkpoint of the current agent state.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            session_id: Session identifier
            conversation_history: List of message dicts
            tool_outputs: List of tool call results
            partial_results: Incomplete results accumulated so far
            progress: Progress information
            metadata: Additional metadata

        Returns:
            CheckpointData instance
        """
        # Generate checkpoint ID
        checkpoint_id = str(uuid.uuid4())

        # Get checkpoint number
        key = f"{agent_id}:{task_id}"
        checkpoint_number = self._checkpoint_counts.get(key, 0) + 1
        self._checkpoint_counts[key] = checkpoint_number

        # Create checkpoint data
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            task_id=task_id,
            session_id=session_id,
            state=AgentState.CHECKPOINTED.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            conversation_history=conversation_history,
            tool_outputs=tool_outputs,
            partial_results=partial_results,
            progress=progress,
            metadata=metadata or {},
            checkpoint_number=checkpoint_number,
        )

        # Ensure directory exists
        checkpoint_path = self._get_checkpoint_path(agent_id, task_id)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Write checkpoint file
        checkpoint_file = self._get_checkpoint_file(checkpoint_id, agent_id, task_id)
        with open(checkpoint_file, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2)

        # Update in-memory index
        if key not in self._checkpoints:
            self._checkpoints[key] = []
        self._checkpoints[key].append(checkpoint)

        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints(agent_id, task_id)

        logger.info(
            f"Checkpoint saved: {checkpoint_id} for {agent_id}/{task_id} "
            f"(#{checkpoint_number})"
        )

        return checkpoint

    async def _cleanup_old_checkpoints(self, agent_id: str, task_id: str) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        key = f"{agent_id}:{task_id}"
        checkpoints = self._checkpoints.get(key, [])

        if len(checkpoints) > self.max_checkpoints:
            # Sort by checkpoint number
            checkpoints.sort(key=lambda c: c.checkpoint_number)

            # Remove oldest checkpoints
            to_remove = checkpoints[: len(checkpoints) - self.max_checkpoints]
            for cp in to_remove:
                checkpoint_file = self._get_checkpoint_file(
                    cp.checkpoint_id, agent_id, task_id
                )
                if checkpoint_file.exists():
                    checkpoint_file.unlink()

            # Update in-memory index
            self._checkpoints[key] = checkpoints[-self.max_checkpoints :]

    def get_latest_checkpoint(
        self, agent_id: str, task_id: str
    ) -> Optional[CheckpointData]:
        """
        Get the most recent checkpoint for a task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            CheckpointData or None if no checkpoints exist
        """
        key = f"{agent_id}:{task_id}"
        checkpoints = self._checkpoints.get(key, [])

        if checkpoints:
            return max(checkpoints, key=lambda c: c.checkpoint_number)

        # Try to load from disk
        checkpoint_path = self._get_checkpoint_path(agent_id, task_id)
        if checkpoint_path.exists():
            checkpoint_files = sorted(checkpoint_path.glob("*.json"), reverse=True)
            if checkpoint_files:
                with open(checkpoint_files[0], "r") as f:
                    data = json.load(f)
                    return CheckpointData(**data)

        return None

    def get_checkpoint(
        self, agent_id: str, task_id: str, checkpoint_id: str
    ) -> Optional[CheckpointData]:
        """
        Get a specific checkpoint by ID.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            CheckpointData or None if not found
        """
        # Check in-memory first
        key = f"{agent_id}:{task_id}"
        for cp in self._checkpoints.get(key, []):
            if cp.checkpoint_id == checkpoint_id:
                return cp

        # Load from disk
        checkpoint_file = self._get_checkpoint_file(checkpoint_id, agent_id, task_id)
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
                return CheckpointData(**data)

        return None

    def list_checkpoints(self, agent_id: str, task_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            List of checkpoint metadata dictionaries
        """
        key = f"{agent_id}:{task_id}"
        checkpoints = self._checkpoints.get(key, [])

        if not checkpoints:
            # Load from disk
            checkpoint_path = self._get_checkpoint_path(agent_id, task_id)
            if checkpoint_path.exists():
                for cf in checkpoint_path.glob("*.json"):
                    with open(cf, "r") as f:
                        data = json.load(f)
                        checkpoints.append(CheckpointData(**data))

        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "checkpoint_number": cp.checkpoint_number,
                "created_at": cp.created_at,
                "state": cp.state,
                "progress_percent": cp.progress.get("percent", 0),
            }
            for cp in sorted(checkpoints, key=lambda c: c.checkpoint_number, reverse=True)
        ]

    async def delete_checkpoints(self, agent_id: str, task_id: str) -> int:
        """
        Delete all checkpoints for a task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Number of checkpoints deleted
        """
        key = f"{agent_id}:{task_id}"
        checkpoints = self._checkpoints.get(key, [])
        count = len(checkpoints)

        # Remove from memory
        self._checkpoints.pop(key, None)
        self._checkpoint_counts.pop(key, None)

        # Remove from disk
        checkpoint_path = self._get_checkpoint_path(agent_id, task_id)
        if checkpoint_path.exists():
            for cf in checkpoint_path.glob("*.json"):
                cf.unlink()
            checkpoint_path.rmdir()

        logger.info(f"Deleted {count} checkpoints for {agent_id}/{task_id}")
        return count


class HeartbeatManager:
    """
    Manages heartbeat mechanism for agent health monitoring.

    Features:
    - Periodic heartbeats from agents
    - Stall detection
    - Auto-recovery options
    """

    def __init__(
        self,
        heartbeat_interval_seconds: int = 30,
        stall_threshold_seconds: int = 120,
        max_stall_count: int = 3,
    ):
        """
        Initialize the heartbeat manager.

        Args:
            heartbeat_interval_seconds: Expected interval between heartbeats
            stall_threshold_seconds: Time after which agent is considered stalled
            max_stall_count: Number of missed heartbeats before marking as stalled
        """
        self.heartbeat_interval = heartbeat_interval_seconds
        self.stall_threshold = stall_threshold_seconds
        self.max_stall_count = max_stall_count

        # In-memory heartbeat tracking
        self._heartbeats: Dict[str, HeartbeatData] = {}
        self._lock = asyncio.Lock()

        # Callbacks for state changes
        self._on_stall: Optional[Callable] = None
        self._on_recovery: Optional[Callable] = None
        self._on_death: Optional[Callable] = None

        # Start background monitor
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info(
            f"HeartbeatManager initialized: interval={heartbeat_interval_seconds}s, "
            f"stall_threshold={stall_threshold_seconds}s"
        )

    def set_stall_callback(self, callback: Callable[[str, str], Any]) -> None:
        """Set callback for stall detection."""
        self._on_stall = callback

    def set_recovery_callback(self, callback: Callable[[str, str], Any]) -> None:
        """Set callback for recovery."""
        self._on_recovery = callback

    def set_death_callback(self, callback: Callable[[str, str], Any]) -> None:
        """Set callback for agent death."""
        self._on_death = callback

    async def start_monitoring(self) -> None:
        """Start the background heartbeat monitoring task."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_heartbeats())
            logger.info("Heartbeat monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the background heartbeat monitoring task."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Heartbeat monitoring stopped")

    async def register_agent(
        self,
        agent_id: str,
        task_id: str,
        progress: float = 0.0,
        current_step: str = "starting",
    ) -> HeartbeatData:
        """
        Register an agent for heartbeat monitoring.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            progress: Initial progress (0.0 - 1.0)
            current_step: Description of current step

        Returns:
            HeartbeatData instance
        """
        async with self._lock:
            heartbeat = HeartbeatData(
                agent_id=agent_id,
                task_id=task_id,
                status=HeartbeatStatus.ACTIVE.value,
                last_update=datetime.now(timezone.utc).isoformat(),
                progress=progress,
                eta_seconds=None,
                current_step=current_step,
                stall_count=0,
            )
            self._heartbeats[f"{agent_id}:{task_id}"] = heartbeat
            logger.info(f"Registered agent {agent_id}/{task_id} for heartbeat monitoring")
            return heartbeat

    async def unregister_agent(self, agent_id: str, task_id: str) -> None:
        """Unregister an agent from heartbeat monitoring."""
        async with self._lock:
            key = f"{agent_id}:{task_id}"
            self._heartbeats.pop(key, None)
            logger.info(f"Unregistered agent {agent_id}/{task_id} from heartbeat monitoring")

    async def send_heartbeat(
        self,
        agent_id: str,
        task_id: str,
        progress: float,
        current_step: str,
        eta_seconds: Optional[float] = None,
    ) -> HeartbeatData:
        """
        Send a heartbeat update from an agent.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            progress: Progress percentage (0.0 - 1.0)
            current_step: Description of current step
            eta_seconds: Estimated seconds until completion

        Returns:
            Updated HeartbeatData
        """
        key = f"{agent_id}:{task_id}"
        async with self._lock:
            if key not in self._heartbeats:
                # Auto-register if not registered
                await self.register_agent(agent_id, task_id, progress, current_step)

            heartbeat = self._heartbeats[key]

            # Update heartbeat
            heartbeat.last_update = datetime.now(timezone.utc).isoformat()
            heartbeat.progress = progress
            heartbeat.current_step = current_step
            heartbeat.eta_seconds = eta_seconds
            heartbeat.status = HeartbeatStatus.ACTIVE.value
            heartbeat.stall_count = 0

            logger.debug(
                f"Heartbeat: {agent_id}/{task_id} - {progress*100:.1f}% - {current_step}"
            )
            return heartbeat

    async def get_heartbeat(self, agent_id: str, task_id: str) -> Optional[HeartbeatData]:
        """Get the current heartbeat data for an agent."""
        key = f"{agent_id}:{task_id}"
        async with self._lock:
            return self._heartbeats.get(key)

    async def get_all_heartbeats(self) -> List[HeartbeatData]:
        """Get all registered heartbeats."""
        async with self._lock:
            return list(self._heartbeats.values())

    async def _monitor_heartbeats(self) -> None:
        """Background task to monitor agent heartbeats and detect stalls."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                now = datetime.now(timezone.utc)
                async with self._lock:
                    for key, hb in list(self._heartbeats.items()):
                        last_update = datetime.fromisoformat(hb.last_update.replace('Z', '+00:00'))
                        seconds_since_update = (now - last_update).total_seconds()

                        if seconds_since_update > self.stall_threshold:
                            hb.stall_count += 1
                            logger.warning(
                                f"Agent {hb.agent_id}/{hb.task_id} stalled "
                                f"(count={hb.stall_count}, since={seconds_since_update:.0f}s)"
                            )

                            if hb.stall_count >= self.max_stall_count:
                                hb.status = HeartbeatStatus.STALLED.value
                                if self._on_stall:
                                    try:
                                        self._on_stall(hb.agent_id, hb.task_id)
                                    except Exception as e:
                                        logger.error(f"Stall callback error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")


class ProgressTracker:
    """
    Tracks progress for long-running operations.

    Features:
    - Percentage-based progress tracking
    - ETA estimation
    - Streaming progress updates
    """

    def __init__(self, total_steps: int = 100):
        """
        Initialize the progress tracker.

        Args:
            total_steps: Total number of steps for the operation
        """
        self.total_steps = total_steps
        self.completed_steps = 0
        self.start_time = datetime.now(timezone.utc)
        self.step_times: List[float] = []
        self._current_step_name: str = "initializing"
        self._lock = asyncio.Lock()

    @property
    def progress_percent(self) -> float:
        """Get current progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def eta_seconds(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        if not self.step_times or self.completed_steps >= self.total_steps:
            return None

        avg_time_per_step = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.completed_steps
        return avg_time_per_step * remaining_steps

    async def update(
        self,
        completed_steps: Optional[int] = None,
        step_name: Optional[str] = None,
    ) -> ProgressUpdate:
        """
        Update progress.

        Args:
            completed_steps: Number of completed steps (or increment by 1 if None)
            step_name: Name of current step

        Returns:
            ProgressUpdate instance
        """
        async with self._lock:
            if completed_steps is not None:
                self.completed_steps = completed_steps
            else:
                self.completed_steps += 1

            # Track step time
            now = datetime.now(timezone.utc)
            if self.step_times:
                last_step_duration = (
                    now - self.start_time
                ).total_seconds() - sum(self.step_times)
                self.step_times.append(last_step_duration)
                # Keep only recent step times for moving average
                if len(self.step_times) > 10:
                    self.step_times = self.step_times[-10:]

            if step_name:
                self._current_step_name = step_name

            self.start_time = now

        return ProgressUpdate(
            task_id="",  # Will be set by caller
            agent_id="",  # Will be set by caller
            progress_percent=self.progress_percent,
            current_step=self._current_step_name,
            total_steps=self.total_steps,
            completed_steps=self.completed_steps,
            eta_seconds=self.eta_seconds,
            message=f"Progress: {self.progress_percent:.1f}% - {self._current_step_name}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def reset(self) -> None:
        """Reset progress tracking."""
        self.completed_steps = 0
        self.start_time = datetime.now(timezone.utc)
        self.step_times = []
        self._current_step_name = "initializing"


class InterruptHandler:
    """
    Handles graceful shutdown on SIGINT/SIGTERM signals.

    Features:
    - Signal trapping
    - Checkpoint saving before exit
    - Grace period for cleanup
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        grace_period_seconds: int = 10,
    ):
        """
        Initialize the interrupt handler.

        Args:
            checkpoint_manager: Checkpoint manager for saving state
            grace_period_seconds: Seconds to wait for graceful shutdown
        """
        self.checkpoint_manager = checkpoint_manager
        self.grace_period = grace_period_seconds

        self._shutdown_event = asyncio.Event()
        self._save_checkpoint_callback: Optional[Callable] = None
        self._original_handlers: Dict[signal.Signals, signal.Handler] = {}
        self._is_shutting_down = False

    def register_save_callback(
        self, callback: Callable[[], Any]
    ) -> None:
        """Register callback to save checkpoint before shutdown."""
        self._save_checkpoint_callback = callback

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            asyncio.create_task(self._handle_interrupt(signum))

        # Store original handlers
        self._original_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, signal_handler
        )
        self._original_handlers[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, signal_handler
        )

        logger.info("Interrupt handlers registered for SIGINT/SIGTERM")

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        logger.info("Original signal handlers restored")

    async def _handle_interrupt(self, signum: int) -> None:
        """Handle interrupt signal."""
        if self._is_shutting_down:
            logger.warning("Received second interrupt, forcing exit")
            raise SystemExit(1)

        self._is_shutting_down = True
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name}, initiating graceful shutdown...")

        # Call save checkpoint callback if registered
        if self._save_checkpoint_callback:
            try:
                if asyncio.iscoroutinefunction(self._save_checkpoint_callback):
                    await self._save_checkpoint_callback()
                else:
                    self._save_checkpoint_callback()
                logger.info("Checkpoint saved before shutdown")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")

        # Set shutdown event
        self._shutdown_event.set()

        # Wait for grace period then force exit
        await asyncio.sleep(self.grace_period)
        logger.warning("Grace period expired, forcing exit")
        raise SystemExit(0)

    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


class LongRunningAgent:
    """
    High-level wrapper for long-running agents with all features integrated.

    Combines checkpoint, heartbeat, progress tracking, and interrupt handling.
    """

    def __init__(
        self,
        agent_id: str,
        task_id: str,
        session_id: str,
        checkpoint_manager: Optional[CheckpointManager] = None,
        heartbeat_manager: Optional[HeartbeatManager] = None,
        checkpoint_interval_seconds: int = 60,
        heartbeat_interval_seconds: int = 30,
    ):
        """
        Initialize the long-running agent wrapper.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            session_id: Session identifier
            checkpoint_manager: Optional checkpoint manager
            heartbeat_manager: Optional heartbeat manager
            checkpoint_interval_seconds: Auto-checkpoint interval
            heartbeat_interval_seconds: Heartbeat interval
        """
        self.agent_id = agent_id
        self.task_id = task_id
        self.session_id = session_id

        # Initialize managers
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.heartbeat_manager = heartbeat_manager or HeartbeatManager(
            heartbeat_interval_seconds=heartbeat_interval_seconds
        )

        # Progress tracking
        self.progress_tracker = ProgressTracker()
        self.checkpoint_interval = checkpoint_interval_seconds
        self._last_checkpoint_time = datetime.now(timezone.utc)

        # State
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_outputs: List[Dict[str, Any]] = []
        self.partial_results: Dict[str, Any] = {}
        self.state = AgentState.PENDING

        # Interrupt handling
        self.interrupt_handler = InterruptHandler(self.checkpoint_manager)
        self.interrupt_handler.register_save_callback(self.save_checkpoint)

        # Background tasks
        self._auto_checkpoint_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the long-running agent."""
        self.state = AgentState.RUNNING

        # Register for heartbeat
        await self.heartbeat_manager.register_agent(
            self.agent_id, self.task_id
        )

        # Start heartbeat monitoring if not already running
        await self.heartbeat_manager.start_monitoring()

        # Setup interrupt handlers
        self.interrupt_handler.setup_signal_handlers()

        # Start auto-checkpoint task
        self._auto_checkpoint_task = asyncio.create_task(
            self._auto_checkpoint_loop()
        )

        logger.info(f"LongRunningAgent {self.agent_id}/{self.task_id} started")

    async def _auto_checkpoint_loop(self) -> None:
        """Background task for automatic checkpointing."""
        while self.state == AgentState.RUNNING:
            try:
                await asyncio.sleep(self.checkpoint_interval)

                # Check if enough time has passed
                now = datetime.now(timezone.utc)
                if (now - self._last_checkpoint_time).total_seconds() >= self.checkpoint_interval:
                    await self.save_checkpoint()
                    self._last_checkpoint_time = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-checkpoint error: {e}")

    async def save_checkpoint(self) -> Optional[CheckpointData]:
        """
        Save current agent state to checkpoint.

        Returns:
            CheckpointData or None if not running
        """
        if self.state not in (AgentState.RUNNING, AgentState.CHECKPOINTED):
            return None

        checkpoint = await self.checkpoint_manager.save_checkpoint(
            agent_id=self.agent_id,
            task_id=self.task_id,
            session_id=self.session_id,
            conversation_history=self.conversation_history,
            tool_outputs=self.tool_outputs,
            partial_results=self.partial_results,
            progress={
                "percent": self.progress_tracker.progress_percent,
                "current_step": self.progress_tracker._current_step_name,
                "completed_steps": self.progress_tracker.completed_steps,
                "total_steps": self.progress_tracker.total_steps,
                "eta_seconds": self.progress_tracker.eta_seconds,
            },
            metadata={
                "state": self.state.value,
            },
        )

        self.state = AgentState.CHECKPOINTED
        return checkpoint

    async def update_progress(
        self,
        completed_steps: Optional[int] = None,
        step_name: Optional[str] = None,
    ) -> ProgressUpdate:
        """
        Update progress and send heartbeat.

        Args:
            completed_steps: Number of completed steps
            step_name: Name of current step

        Returns:
            ProgressUpdate instance
        """
        # Update progress
        progress = await self.progress_tracker.update(completed_steps, step_name)

        # Send heartbeat
        await self.heartbeat_manager.send_heartbeat(
            agent_id=self.agent_id,
            task_id=self.task_id,
            progress=self.progress_tracker.progress_percent / 100,
            current_step=step_name or self.progress_tracker._current_step_name,
            eta_seconds=self.progress_tracker.eta_seconds,
        )

        return progress

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_tool_output(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
    ) -> None:
        """Add a tool output to history."""
        self.tool_outputs.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @classmethod
    async def resume_from_checkpoint(
        cls,
        checkpoint: CheckpointData,
        checkpoint_manager: Optional[CheckpointManager] = None,
        heartbeat_manager: Optional[HeartbeatManager] = None,
    ) -> "LongRunningAgent":
        """
        Resume an agent from a checkpoint.

        Args:
            checkpoint: CheckpointData to resume from
            checkpoint_manager: Optional checkpoint manager
            heartbeat_manager: Optional heartbeat manager

        Returns:
            LongRunningAgent instance with restored state
        """
        agent = cls(
            agent_id=checkpoint.agent_id,
            task_id=checkpoint.task_id,
            session_id=checkpoint.session_id,
            checkpoint_manager=checkpoint_manager,
            heartbeat_manager=heartbeat_manager,
        )

        # Restore state
        agent.conversation_history = checkpoint.conversation_history
        agent.tool_outputs = checkpoint.tool_outputs
        agent.partial_results = checkpoint.partial_results
        agent.state = AgentState.RESUMING

        # Restore progress
        progress_data = checkpoint.progress
        agent.progress_tracker.completed_steps = progress_data.get(
            "completed_steps", 0
        )
        agent.progress_tracker.total_steps = progress_data.get(
            "total_steps", 100
        )

        # Start the agent
        await agent.start()

        agent.state = AgentState.RUNNING
        logger.info(
            f"Resumed agent {agent.agent_id}/{agent.task_id} from checkpoint "
            f"#{checkpoint.checkpoint_number}"
        )

        return agent

    async def stop(self, status: str = "completed") -> None:
        """
        Stop the long-running agent.

        Args:
            status: Final status (completed, failed, cancelled)
        """
        # Cancel auto-checkpoint
        if self._auto_checkpoint_task:
            self._auto_checkpoint_task.cancel()
            try:
                await self._auto_checkpoint_task
            except asyncio.CancelledError:
                pass

        # Restore signal handlers
        self.interrupt_handler.restore_signal_handlers()

        # Unregister from heartbeat
        await self.heartbeat_manager.unregister_agent(
            self.agent_id, self.task_id
        )

        # Save final checkpoint
        await self.save_checkpoint()

        # Update state
        self.state = AgentState(status)
        logger.info(f"LongRunningAgent {self.agent_id}/{self.task_id} stopped: {status}")


# Global instances
_checkpoint_manager: Optional[CheckpointManager] = None
_heartbeat_manager: Optional[HeartbeatManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager


def get_heartbeat_manager() -> HeartbeatManager:
    """Get the global heartbeat manager instance."""
    global _heartbeat_manager
    if _heartbeat_manager is None:
        _heartbeat_manager = HeartbeatManager()
    return _heartbeat_manager
