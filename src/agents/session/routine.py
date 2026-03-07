"""
Session Startup Routine for Workflow Orchestrator.

Implements the session startup routine based on Anthropic's harness patterns:
- Read progress file
- Check git log for changes
- Resume or start new workflow

Reference: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SessionRoutine:
    """
    Session startup routine for workflow recovery.

    Implements the pattern from Anthropic's article:
    1. Read progress file
    2. Check git log for changes
    3. Run initialization if needed
    4. Resume or start new workflow
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        init_script: Optional[Path] = None,
    ):
        """
        Initialize the session routine.

        Args:
            base_path: Base path for workflow files
            init_script: Path to initialization script
        """
        self.base_path = base_path or Path.cwd()
        self.init_script = init_script
        self.progress_file = self.base_path / "workflow_progress.json"
        self.feature_file = self.base_path / "feature_list.json"

    async def run_startup_routine(
        self,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full startup routine.

        Args:
            workflow_id: Optional workflow ID to resume. If None, checks for existing progress.

        Returns:
            Dict with startup status and recommendations
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_existing_progress": False,
            "can_resume": False,
            "workflow_id": None,
            "current_stage": None,
            "completed_stages": [],
            "pending_stages": [],
            "git_log": [],
            "recommended_action": "start_new",
        }

        # Step 1: Read progress file
        progress_data = await self._read_progress_file()
        if progress_data:
            result["has_existing_progress"] = True
            result["workflow_id"] = progress_data.get("workflow_id")
            result["current_stage"] = progress_data.get("current_stage")
            result["completed_stages"] = progress_data.get("completed", [])
            result["pending_stages"] = progress_data.get("pending", [])

            if progress_data.get("current_stage") or progress_data.get("pending"):
                result["can_resume"] = True

        # Step 2: Check git log
        git_log = await self._get_git_log()
        result["git_log"] = git_log

        # Step 3: Determine recommended action
        if workflow_id:
            if result["can_resume"] and result["workflow_id"] == workflow_id:
                result["recommended_action"] = "resume"
            elif result["has_existing_progress"]:
                result["recommended_action"] = "resume_existing"
            else:
                result["recommended_action"] = "start_new"
        elif result["can_resume"]:
            result["recommended_action"] = "resume"

        logger.info(
            f"Startup routine completed: action={result['recommended_action']}, "
            f"workflow={result['workflow_id']}"
        )

        return result

    async def _read_progress_file(self) -> Optional[Dict[str, Any]]:
        """Read and parse the progress file."""
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read progress file: {e}")
            return None

    async def _get_git_log(self, limit: int = 10) -> List[str]:
        """Get recent git log entries."""
        try:
            result = subprocess.run(
                ["git", "log", f"--oneline", f"-{limit}"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        except subprocess.TimeoutExpired:
            logger.warning("Git log timed out")
        except Exception as e:
            logger.debug(f"Git log error: {e}")

        return []

    async def check_incomplete_workflows(self) -> List[Dict[str, Any]]:
        """
        Check for any incomplete workflows.

        Returns:
            List of incomplete workflow info
        """
        incomplete = []

        # Check progress file
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)

                # Check if workflow is incomplete
                if data.get("current_stage") or data.get("pending"):
                    incomplete.append({
                        "workflow_id": data.get("workflow_id"),
                        "current_stage": data.get("current_stage"),
                        "completed": data.get("completed", []),
                        "pending": data.get("pending", []),
                        "last_updated": data.get("last_updated"),
                    })
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading progress file: {e}")

        return incomplete

    async def run_init_script(self) -> Tuple[bool, str]:
        """
        Run the initialization script if defined.

        Returns:
            Tuple of (success, message)
        """
        if not self.init_script:
            return False, "No init script defined"

        if not self.init_script.exists():
            return False, f"Init script not found: {self.init_script}"

        try:
            result = subprocess.run(
                [str(self.init_script)],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                logger.info("Init script completed successfully")
                return True, result.stdout
            else:
                logger.warning(f"Init script failed: {result.stderr}")
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Init script timed out"
        except Exception as e:
            return False, str(e)

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific workflow.

        Args:
            workflow_id: The workflow ID to check

        Returns:
            Dict with workflow status or None if not found
        """
        progress_data = await self._read_progress_file()

        if not progress_data or progress_data.get("workflow_id") != workflow_id:
            return None

        return {
            "workflow_id": workflow_id,
            "current_stage": progress_data.get("current_stage"),
            "completed_stages": progress_data.get("completed", []),
            "pending_stages": progress_data.get("pending", []),
            "results": progress_data.get("results", {}),
            "last_updated": progress_data.get("last_updated"),
        }

    async def clear_progress(self) -> None:
        """Clear the progress file for a fresh start."""
        if self.progress_file.exists():
            # Backup before clearing
            backup_path = self.progress_file.with_suffix(".json.bak")
            try:
                with open(self.progress_file, "r") as src:
                    with open(backup_path, "w") as dst:
                        dst.write(src.read())
                self.progress_file.unlink()
                logger.info(f"Progress file cleared, backed up to {backup_path}")
            except IOError as e:
                logger.warning(f"Failed to backup progress file: {e}")


# Global instance
_session_routine: Optional[SessionRoutine] = None


def get_session_routine() -> SessionRoutine:
    """Get the global session routine instance."""
    global _session_routine
    if _session_routine is None:
        _session_routine = SessionRoutine()
    return _session_routine
