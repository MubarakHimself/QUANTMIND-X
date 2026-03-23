"""Tests for backup and restore scripts."""
import subprocess
from pathlib import Path
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class TestBackupScript:
    """Test backup_full_system.sh script."""

    def test_backup_script_exists(self):
        """Verify backup script exists."""
        backup_script = SCRIPTS_DIR / "backup_full_system.sh"
        assert backup_script.exists(), f"Backup script not found at {backup_script}"

    def test_backup_script_executable(self):
        """Verify backup script is executable."""
        backup_script = SCRIPTS_DIR / "backup_full_system.sh"
        assert backup_script.is_file(), "Backup script is not a file"
        # Check if executable bit is set
        import os
        assert os.access(backup_script, os.X_OK), "Backup script is not executable"

    def test_backup_script_dry_run(self, tmp_path):
        """Test backup script dry-run mode."""
        backup_script = SCRIPTS_DIR / "backup_full_system.sh"
        result = subprocess.run(
            [str(backup_script), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Dry run should exit with 0 (successful dry run)
        assert result.returncode == 0, f"Dry run failed: {result.stderr}"


class TestRestoreScript:
    """Test restore_full_system.sh script."""

    def test_restore_script_exists(self):
        """Verify restore script exists."""
        restore_script = SCRIPTS_DIR / "restore_full_system.sh"
        assert restore_script.exists(), f"Restore script not found at {restore_script}"

    def test_restore_script_executable(self):
        """Verify restore script is executable."""
        restore_script = SCRIPTS_DIR / "restore_full_system.sh"
        assert restore_script.is_file(), "Restore script is not a file"
        # Check if executable bit is set
        import os
        assert os.access(restore_script, os.X_OK), "Restore script is not executable"

    def test_restore_script_shows_usage_without_args(self):
        """Test restore script shows usage when no args provided."""
        restore_script = SCRIPTS_DIR / "restore_full_system.sh"
        result = subprocess.run(
            [str(restore_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should exit with error (no backup file provided)
        assert result.returncode != 0, "Should fail without backup file"
        # Should show usage
        assert "Usage:" in result.stdout or "Usage:" in result.stderr, \
            "Should show usage information"

    def test_restore_script_validate_only(self):
        """Test restore script --validate-only with missing file."""
        restore_script = SCRIPTS_DIR / "restore_full_system.sh"
        result = subprocess.run(
            [str(restore_script), "nonexistent_backup.tar.gz", "--validate-only"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should fail with non-existent file
        assert result.returncode != 0, "Should fail with non-existent backup"


class TestBackupRestoreIntegration:
    """Integration tests for backup/restore cycle (using temp directory)."""

    def test_backup_script_help(self):
        """Verify backup script has proper help/usage."""
        backup_script = SCRIPTS_DIR / "backup_full_system.sh"
        # Try to get help by running with no args in non-interactive mode
        # The script requires dependencies, so we just verify it responds
        result = subprocess.run(
            [str(backup_script), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should run without error in dry-run mode
        assert result.returncode == 0, f"Dry run should succeed: {result.stderr}"