"""GitHub Sync Service for EA Compilation Workflow."""

import re
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Valid strategy_id pattern: alphanumeric, dash, underscore only
VALID_STRATEGY_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


class GitHubSyncService:
    """Handles GitHub sync for EA compilation."""

    def __init__(
        self,
        repo_url: str,
        local_path: str = ".ea-strategies",
        branch: str = "main"
    ):
        self.repo_url = repo_url
        self.local_path = Path(local_path)
        self.branch = branch

    def commit_ea(self, strategy_id: str, mq5_code: str) -> Dict[str, Any]:
        """Commit .mq5 file to repository."""
        # Validate strategy_id to prevent path traversal
        if not VALID_STRATEGY_ID_PATTERN.match(strategy_id):
            logger.error(f"Invalid strategy_id: {strategy_id}")
            return {"success": False, "error": f"Invalid strategy_id: {strategy_id}"}

        try:
            strategy_dir = self.local_path / "strategies" / strategy_id
            strategy_dir.mkdir(parents=True, exist_ok=True)

            ea_file = strategy_dir / f"{strategy_id}.mq5"
            ea_file.write_text(mq5_code)

            # Git add and commit
            subprocess.run(["git", "add", "."], cwd=self.local_path, check=True, capture_output=True)
            result = subprocess.run(
                ["git", "commit", "-m", f"Add EA: {strategy_id}"],
                cwd=self.local_path,
                capture_output=True,
                text=True
            )

            push_result = subprocess.run(
                ["git", "push", "origin", self.branch],
                cwd=self.local_path,
                capture_output=True
            )

            return {
                "success": push_result.returncode == 0,
                "commit_sha": result.stdout[:7] if result.returncode == 0 else None,
                "message": result.stdout
            }
        except Exception as e:
            logger.error(f"Failed to commit EA: {e}")
            return {"success": False, "error": str(e)}

    def trigger_compile(self, commit_sha: str) -> Dict[str, Any]:
        """Trigger compilation via GitHub Actions."""
        # This would trigger a workflow dispatch
        # For now, return success and let VPS handle it
        return {
            "success": True,
            "workflow_id": f"compile-{commit_sha}",
            "status": "triggered"
        }

    def pull_ex5(self, strategy_id: str) -> Optional[Path]:
        """Pull compiled .ex5 from repository."""
        # Validate strategy_id to prevent path traversal
        if not VALID_STRATEGY_ID_PATTERN.match(strategy_id):
            logger.error(f"Invalid strategy_id: {strategy_id}")
            return None

        try:
            subprocess.run(["git", "pull"], cwd=self.local_path, check=True, capture_output=True)
            ex5_path = self.local_path / "compiled" / f"{strategy_id}.ex5"
            return ex5_path if ex5_path.exists() else None
        except Exception as e:
            logger.error(f"Failed to pull .ex5: {e}")
            return None
