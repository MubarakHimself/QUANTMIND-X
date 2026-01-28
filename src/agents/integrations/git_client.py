#!/usr/bin/env python3
"""
Git client for raw file storage in Assets Hub.

Provides methods for reading, writing, and committing template and skill files
to the Git repository at /data/git/assets-hub/.

Follows the file operation pattern from server_chroma.py lines 249-255.
"""

import subprocess
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class GitClient:
    """
    Git client for managing raw source files in the Assets Hub.

    Provides methods for:
    - Reading template and skill files
    - Writing template and skill files
    - Committing changes with descriptive messages
    """

    def __init__(self, repo_path: Path):
        """
        Initialize Git client.

        Args:
            repo_path: Path to Git repository (e.g., /data/git/assets-hub/)
        """
        self.repo_path = Path(repo_path).resolve()
        self.templates_path = self.repo_path / "templates"
        self.skills_path = self.repo_path / "skills"

    def _run_git_command(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """
        Run a Git command in the repository.

        Args:
            args: Git command arguments (e.g., ["status", "--short"])
            check: Whether to raise an exception on non-zero exit code

        Returns:
            subprocess.CompletedProcess with command output
        """
        full_cmd = ["git"] + args
        result = subprocess.run(
            full_cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check
        )
        return result

    def init_repo(self) -> bool:
        """
        Initialize Git repository if not already initialized.

        Returns:
            True if repository was initialized or already exists, False on error
        """
        try:
            # Create directory if it doesn't exist
            self.repo_path.mkdir(parents=True, exist_ok=True)

            # Check if already a git repo
            result = self._run_git_command(["rev-parse", "--git-dir"], check=False)
            if result.returncode == 0:
                logger.info(f"Git repository already exists at {self.repo_path}")
                return True

            # Initialize new repository
            self._run_git_command(["init"])
            logger.info(f"Initialized Git repository at {self.repo_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Git repository: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing repository: {e}")
            return False

    def configure_user(self, name: str = "QuantMindX", email: str = "bot@quantmindx.com") -> bool:
        """
        Configure Git user for commits.

        Args:
            name: Git user name
            email: Git user email

        Returns:
            True if configuration succeeded, False otherwise
        """
        try:
            self._run_git_command(["config", "user.email", email])
            self._run_git_command(["config", "user.name", name])
            logger.info(f"Configured Git user: {name} <{email}>")
            return True
        except Exception as e:
            logger.error(f"Failed to configure Git user: {e}")
            return False

    def read_template(self, category: str, template_name: str) -> Optional[str]:
        """
        Read a template file from the Git repository.

        Args:
            category: Template category (e.g., "trend_following", "mean_reversion")
            template_name: Template file name (e.g., "rsi_basic.mq5")

        Returns:
            Template content as string, or None if file not found

        Example:
            client = GitClient("/data/git/assets-hub/")
            content = client.read_template("mean_reversion", "rsi_basic.mq5")
        """
        template_path = self.templates_path / category / template_name

        try:
            # Security check: ensure path is within repository
            if not template_path.resolve().is_relative_to(self.repo_path):
                logger.error(f"Security error: path traversal attempt detected: {template_path}")
                return None

            if template_path.exists():
                content = template_path.read_text(encoding='utf-8')
                logger.info(f"Read template: {template_path}")
                return content
            else:
                logger.warning(f"Template not found: {template_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading template {template_path}: {e}")
            return None

    def write_template(self, category: str, template_name: str, content: str) -> bool:
        """
        Write a template file to the Git repository.

        Args:
            category: Template category (e.g., "trend_following", "mean_reversion")
            template_name: Template file name (e.g., "rsi_basic.mq5")
            content: Template content to write

        Returns:
            True if write succeeded, False otherwise

        Example:
            client = GitClient("/data/git/assets-hub/")
            success = client.write_template("mean_reversion", "rsi_basic.mq5", code_content)
        """
        template_path = self.templates_path / category / template_name

        try:
            # Security check
            if not template_path.resolve().is_relative_to(self.repo_path):
                logger.error(f"Security error: path traversal attempt detected: {template_path}")
                return False

            # Create directory if needed
            template_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            template_path.write_text(content, encoding='utf-8')
            logger.info(f"Wrote template: {template_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing template {template_path}: {e}")
            return False

    def commit_template(self, category: str, template_name: str, message: str) -> bool:
        """
        Commit a template file change to the Git repository.

        Args:
            category: Template category
            template_name: Template file name
            message: Commit message describing the change

        Returns:
            True if commit succeeded, False otherwise

        Example:
            client = GitClient("/data/git/assets-hub/")
            success = client.commit_template(
                "mean_reversion",
                "rsi_basic.mq5",
                "Strategy: RSI Basic, Backtest: Sharpe=1.8, Drawdown=12%"
            )
        """
        template_path = self.templates_path / category / template_name

        try:
            # Add file to staging
            relative_path = template_path.relative_to(self.repo_path)
            self._run_git_command(["add", str(relative_path)])

            # Commit
            self._run_git_command(["commit", "-m", message])
            logger.info(f"Committed template: {message}")
            return True

        except subprocess.CalledProcessError as e:
            # Check if nothing to commit (not an error)
            if "nothing to commit" in e.stderr.lower():
                logger.info("No changes to commit")
                return True
            logger.error(f"Error committing template: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error committing template: {e}")
            return False

    def read_skill(self, category: str, skill_name: str) -> Optional[str]:
        """
        Read a skill definition file from the Git repository.

        Args:
            category: Skill category (e.g., "trading_skills", "system_skills")
            skill_name: Skill file name (e.g., "calculate_rsi.md")

        Returns:
            Skill content as string, or None if file not found
        """
        skill_path = self.skills_path / category / skill_name

        try:
            # Security check
            if not skill_path.resolve().is_relative_to(self.repo_path):
                logger.error(f"Security error: path traversal attempt detected: {skill_path}")
                return None

            if skill_path.exists():
                content = skill_path.read_text(encoding='utf-8')
                logger.info(f"Read skill: {skill_path}")
                return content
            else:
                logger.warning(f"Skill not found: {skill_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading skill {skill_path}: {e}")
            return None

    def write_skill(self, category: str, skill_name: str, content: str) -> bool:
        """
        Write a skill definition file to the Git repository.

        Args:
            category: Skill category
            skill_name: Skill file name
            content: Skill content to write

        Returns:
            True if write succeeded, False otherwise
        """
        skill_path = self.skills_path / category / skill_name

        try:
            # Security check
            if not skill_path.resolve().is_relative_to(self.repo_path):
                logger.error(f"Security error: path traversal attempt detected: {skill_path}")
                return False

            # Create directory if needed
            skill_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            skill_path.write_text(content, encoding='utf-8')
            logger.info(f"Wrote skill: {skill_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing skill {skill_path}: {e}")
            return False

    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available template files.

        Args:
            category: Optional category filter. If None, lists all templates.

        Returns:
            List of template file paths (relative to templates directory)
        """
        try:
            if category:
                search_path = self.templates_path / category
            else:
                search_path = self.templates_path

            if not search_path.exists():
                return []

            templates = []
            for template_file in search_path.rglob("*"):
                if template_file.is_file():
                    # Return path relative to templates directory
                    relative_path = template_file.relative_to(self.templates_path)
                    templates.append(str(relative_path))

            return sorted(templates)

        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []

    def list_skills(self, category: Optional[str] = None) -> List[str]:
        """
        List available skill definition files.

        Args:
            category: Optional category filter. If None, lists all skills.

        Returns:
            List of skill file paths (relative to skills directory)
        """
        try:
            if category:
                search_path = self.skills_path / category
            else:
                search_path = self.skills_path

            if not search_path.exists():
                return []

            skills = []
            for skill_file in search_path.rglob("*.md"):
                if skill_file.is_file():
                    # Return path relative to skills directory
                    relative_path = skill_file.relative_to(self.skills_path)
                    skills.append(str(relative_path))

            return sorted(skills)

        except Exception as e:
            logger.error(f"Error listing skills: {e}")
            return []
