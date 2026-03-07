"""
Git-Based Checkpoint System for Workflow Orchestrator.

Wraps existing CheckpointManager and adds git-based checkpointing:
- Commit after each workflow stage
- Progress file tracking (workflow_progress.json)
- Feature list tracking (feature_list.json)

Based on Anthropic's effective-harnesses-for-long-running-agents patterns.
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class WorkflowProgress:
    """Tracks workflow progress for recovery."""

    def __init__(self, workflow_id: str, progress_file: Optional[Path] = None):
        self.workflow_id = workflow_id
        self.progress_file = progress_file or Path("workflow_progress.json")
        self.current_stage: Optional[str] = None
        self.completed_stages: List[str] = []
        self.pending_stages: List[str] = []
        self.results: Dict[str, Any] = {}
        self.last_updated: str = datetime.now(timezone.utc).isoformat()

    def load(self) -> bool:
        """Load progress from file. Returns True if successful."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    if data.get("workflow_id") == self.workflow_id:
                        self.current_stage = data.get("current_stage")
                        self.completed_stages = data.get("completed", [])
                        self.pending_stages = data.get("pending", [])
                        self.results = data.get("results", {})
                        self.last_updated = data.get("last_updated", self.last_updated)
                        return True
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load progress file: {e}")
        return False

    def save(self) -> None:
        """Save progress to file."""
        data = {
            "workflow_id": self.workflow_id,
            "current_stage": self.current_stage,
            "completed": self.completed_stages,
            "pending": self.pending_stages,
            "results": self.results,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def mark_stage_started(self, stage: str) -> None:
        """Mark a stage as started."""
        self.current_stage = stage
        if stage not in self.completed_stages and stage in self.pending_stages:
            self.pending_stages.remove(stage)
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()

    def mark_stage_completed(self, stage: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark a stage as completed."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        if stage == self.current_stage:
            self.current_stage = None
        if result:
            self.results[stage] = result
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()

    def mark_stage_failed(self, stage: str, error: str) -> None:
        """Mark a stage as failed."""
        self.results[stage] = {"status": "fail", "error": error}
        self.current_stage = None
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()


class FeatureList:
    """Tracks feature/stage status like feature_list.json."""

    def __init__(self, workflow_id: str, feature_file: Optional[Path] = None):
        self.workflow_id = workflow_id
        self.feature_file = feature_file or Path("feature_list.json")
        self.features: List[Dict[str, Any]] = []

    def load(self) -> bool:
        """Load features from file. Returns True if successful."""
        if self.feature_file.exists():
            try:
                with open(self.feature_file, "r") as f:
                    self.features = json.load(f)
                    return True
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load feature file: {e}")
        return False

    def save(self) -> None:
        """Save features to file."""
        with open(self.feature_file, "w") as f:
            json.dump(self.features, f, indent=2)

    def add_feature(self, stage: str, status: str = "pending", output: Optional[Any] = None, error: Optional[str] = None) -> None:
        """Add or update a feature."""
        # Check if feature already exists
        for feat in self.features:
            if feat.get("stage") == stage:
                feat["status"] = status
                if output is not None:
                    feat["output"] = output
                if error is not None:
                    feat["error"] = error
                feat["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.save()
                return

        # Add new feature
        self.features.append({
            "stage": stage,
            "status": status,
            "output": output,
            "error": error,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self.save()

    def get_feature_status(self, stage: str) -> Optional[str]:
        """Get status of a specific feature."""
        for feat in self.features:
            if feat.get("stage") == stage:
                return feat.get("status")
        return None


class GitCheckpointManager:
    """
    Git-based checkpoint manager for workflow stages.

    Combines the existing CheckpointManager with git-based checkpointing:
    - Commit after each workflow stage
    - Progress file tracking
    - Feature list tracking

    Based on Anthropic's harness patterns.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        base_path: Optional[Path] = None,
        enable_git_checkpoint: bool = True,
        checkpoint_interval_seconds: int = 60,
    ):
        """
        Initialize the git checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint storage
            base_path: Base path for progress files (default: current dir)
            enable_git_checkpoint: Enable git commits after each stage
            checkpoint_interval_seconds: Interval for automatic checkpointing
        """
        # Initialize existing checkpoint manager
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
        )

        self.base_path = base_path or Path.cwd()
        self.enable_git_checkpoint = enable_git_checkpoint
        self._current_workflow_id: Optional[str] = None
        self._current_progress: Optional[WorkflowProgress] = None
        self._current_features: Optional[FeatureList] = None

        logger.info(
            f"GitCheckpointManager initialized: base_path={self.base_path}, "
            f"git_checkpoint={enable_git_checkpoint}"
        )

    async def start_workflow(
        self,
        workflow_id: str,
        pending_stages: List[str],
    ) -> None:
        """
        Start tracking a new workflow.

        Args:
            workflow_id: Unique workflow identifier
            pending_stages: List of pending stage names
        """
        self._current_workflow_id = workflow_id
        self._current_progress = WorkflowProgress(workflow_id, self.base_path / "workflow_progress.json")
        self._current_features = FeatureList(workflow_id, self.base_path / "feature_list.json")

        # Initialize progress
        self._current_progress.pending_stages = pending_stages
        self._current_progress.save()

        # Initialize features as pending
        for stage in pending_stages:
            self._current_features.add_feature(stage, status="pending")

        # Git checkpoint at start
        if self.enable_git_checkpoint:
            await self._git_commit(f"workflow: start {workflow_id}")

        logger.info(f"Started tracking workflow {workflow_id}")

    async def checkpoint_stage(
        self,
        stage: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Checkpoint after a workflow stage completes.

        Args:
            stage: Stage name that completed
            result: Stage result data
            error: Error message if stage failed
        """
        if not self._current_progress or not self._current_features:
            logger.warning("No active workflow to checkpoint")
            return

        # Update progress
        if error:
            self._current_progress.mark_stage_failed(stage, error)
            self._current_features.add_feature(stage, status="fail", error=error)
        else:
            self._current_progress.mark_stage_completed(stage, result)
            self._current_features.add_feature(stage, status="pass", output=result)

        # Git checkpoint
        if self.enable_git_checkpoint:
            commit_msg = f"checkpoint: {stage}"
            if error:
                commit_msg = f"checkpoint: {stage} (failed)"
            await self._git_commit(commit_msg)

        logger.info(f"Checkpointed stage {stage} for workflow {self._current_workflow_id}")

    async def _git_commit(self, message: str) -> None:
        """
        Create a git commit with the current state.

        Args:
            message: Commit message
        """
        if not self.enable_git_checkpoint:
            return

        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug("Not in a git repository, skipping commit")
                return

            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.base_path,
                capture_output=True,
                timeout=30,
            )

            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not result.stdout.strip():
                logger.debug("No changes to commit")
                return

            # Create commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"Git commit created: {message}")
            else:
                logger.warning(f"Git commit failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning("Git commit timed out")
        except Exception as e:
            logger.warning(f"Git commit error: {e}")

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
    ):
        """
        Save a checkpoint using the underlying CheckpointManager.

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
        return await self._checkpoint_manager.save_checkpoint(
            agent_id=agent_id,
            task_id=task_id,
            session_id=session_id,
            conversation_history=conversation_history,
            tool_outputs=tool_outputs,
            partial_results=partial_results,
            progress=progress,
            metadata=metadata,
        )

    def get_latest_checkpoint(self, agent_id: str, task_id: str):
        """Get the most recent checkpoint for a task."""
        return self._checkpoint_manager.get_latest_checkpoint(agent_id, task_id)

    def get_workflow_progress(self) -> Optional[WorkflowProgress]:
        """Get current workflow progress."""
        return self._current_progress

    def get_feature_list(self) -> Optional[FeatureList]:
        """Get current feature list."""
        return self._current_features

    async def end_workflow(self, status: str = "completed") -> None:
        """End tracking the current workflow."""
        if self._current_workflow_id:
            if self.enable_git_checkpoint:
                await self._git_commit(f"workflow: {status} {self._current_workflow_id}")

            # Final progress update
            if self._current_progress:
                self._current_progress.current_stage = None
                self._current_progress.save()

            logger.info(f"Ended workflow {self._current_workflow_id}: {status}")

            self._current_workflow_id = None
            self._current_progress = None
            self._current_features = None


# Global instance
_checkpoint_manager: Optional[GitCheckpointManager] = None


def get_checkpoint_manager() -> GitCheckpointManager:
    """Get the global git checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = GitCheckpointManager()
    return _checkpoint_manager
