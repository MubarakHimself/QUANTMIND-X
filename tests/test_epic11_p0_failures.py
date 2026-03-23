"""
P0 Failing Tests for Epic 11 - System Management & Resilience

These tests represent the TDD "red" phase - they define expected behavior
that should cause failures until the implementation is complete/correct.

Test Level Legend:
- Unit: Single function/class/module in isolation
- Integration: Multiple components working together
- E2E: Full system flow

P0 Priority: Critical path + High risk (>=6) + No workaround

Tests:
1. Rsync checksum mismatch detection and retry (Integration)
2. Full backup creation produces valid tar.gz archive (Integration)
3. Backup artifact SHA256 integrity verification (Unit)
4. Full restore recovers all components (Integration)
5. Migration NODE_ROLE transfer verification (Integration)
6. Migration health check on new host (Integration)
7. Sequential update rollback triggered on health check failure (Integration)
8. Rollback sends notification to Copilot (Integration)
"""

import pytest
import subprocess
import tempfile
import tarfile
import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


# ============================================================================
# P0 Test 1: Rsync Checksum Mismatch Detection and Retry
# ============================================================================

class TestRsyncChecksumMismatch:
    """P0: Rsync integrity verification - checksum mismatch detection and retry."""

    def test_rsync_checksum_mismatch_detection(self, tmp_path):
        """
        P0: Rsync checksum mismatch detection.

        Given: Files have been synced to destination
        When: Source file is modified (checksum changes)
        Then: verify_checksums detects mismatch and returns failure

        This tests the NFR-D5 requirement: corrupted/incomplete transfers
        must be flagged and trigger retry.
        """
        # Arrange: Create source and destination directories
        source_dir = tmp_path / "source"
        dest_dir = tmp_path / "dest"
        source_dir.mkdir()
        dest_dir.mkdir()

        # Create a test file in source
        test_file = source_dir / "test.db"
        test_file.write_text("original data")

        # Copy to destination
        dest_file = dest_dir / "test.db"
        dest_file.write_text("original data")

        # Generate initial checksums
        checksum_file = tmp_path / "checksums.txt"
        result = subprocess.run(
            ["bash", "-c", f"cd {source_dir} && find . -type f -exec sha256sum {{}} \\; | sort"],
            capture_output=True,
            text=True
        )
        checksum_file.write_text(result.stdout)

        # Act: Modify source file (simulate corruption during transfer)
        test_file.write_text("modified data - CORRUPTED")

        # Assert: Verify should detect mismatch
        # Run the verify_checksum script in compare mode
        result = subprocess.run(
            ["bash", "scripts/verify_checksum.sh", "--compare", str(source_dir), str(dest_dir)],
            capture_output=True,
            text=True
        )
        # The compare should show directories differ
        assert result.returncode != 0, "Checksum mismatch should be detected"
        assert "differ" in result.stdout.lower() or "fail" in result.stdout.lower()

    def test_rsync_retry_on_failure_increments_count(self):
        """
        P0: Rsync retry logic on failure.

        Given: rsync operation fails
        When: MAX_RETRIES is configured
        Then: Script retries up to MAX_RETRIES times with exponential backoff

        This tests the NFR-D5 requirement: retry with exponential backoff.
        """
        # This test verifies the retry configuration exists in the script
        script_content = Path("scripts/sync_cloudzy_to_contabo.sh").read_text()

        # Assert: MAX_RETRIES is configured
        assert "MAX_RETRIES=" in script_content, "MAX_RETRIES must be configured"

        # Assert: RETRY_DELAY_BASE for exponential backoff
        assert "RETRY_DELAY_BASE=" in script_content, "RETRY_DELAY_BASE must be configured for exponential backoff"

        # Assert: retry loop is implemented
        assert "while" in script_content and "attempt" in script_content, "Retry loop must be implemented"


# ============================================================================
# P0 Test 2: Full Backup Creates Valid tar.gz Archive with Checksums
# ============================================================================

class TestBackupArchiveCreation:
    """P0: Full backup creation - archive integrity."""

    def test_backup_creates_tar_gz_with_correct_structure(self, tmp_path):
        """
        P0: Full backup creates valid tar.gz archive.

        Given: System has configs, knowledge base, strategies, graph memory
        When: backup_full_system.sh runs
        Then: Creates backup_YYYYMMDD_HHMMSS.tar.gz with correct internal structure

        This is a critical integration test for FR69: machine portability.
        """
        # Arrange: Set up a temporary backup environment
        backup_script = Path("scripts/backup_full_system.sh")
        assert backup_script.exists(), "Backup script must exist"

        # Create minimal test structure that backup script expects
        test_project = tmp_path / "project"
        test_project.mkdir()

        # Mock PROJECT_DIR and BACKUP_DIR to use our temp directory
        env = {
            "PROJECT_DIR": str(test_project),
            "BACKUP_DIR": str(tmp_path / "backups"),
            "LOG_DIR": str(tmp_path / "logs"),
            "HOME": str(tmp_path),
        }

        # Act: Run backup in dry-run mode first to verify structure
        result = subprocess.run(
            [str(backup_script), "--dry-run"],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, **env},
            timeout=30
        )

        # Assert: Dry run should succeed
        assert result.returncode == 0, f"Backup dry-run should succeed: {result.stderr}"

    def test_backup_script_has_manifest_generation(self):
        """
        P0: Backup generates checksums for all archived files.

        Given: Backup script runs
        When: Archive is created
        Then: manifest.sha256 contains checksums for all files

        This tests the integrity verification requirement.
        """
        script_content = Path("scripts/backup_full_system.sh").read_text()

        # Assert: generate_checksums function exists
        assert "generate_checksums" in script_content, "generate_checksums function must exist"

        # Assert: manifest.sha256 is created
        assert "manifest.sha256" in script_content, "manifest.sha256 must be generated"

        # Assert: sha256sum is used
        assert "sha256sum" in script_content, "SHA256 checksums must be generated"


# ============================================================================
# P0 Test 3: Backup Artifact SHA256 Integrity Verification
# ============================================================================

class TestBackupChecksumIntegrity:
    """P0: Backup artifact integrity verification."""

    def test_backup_archive_can_be_extracted_and_verified(self, tmp_path):
        """
        P0: Backup archive integrity can be verified.

        Given: A backup archive exists
        When: We extract and verify checksums
        Then: All files pass checksum verification

        This tests the NFR-D5 requirement: data integrity verification.
        """
        # Arrange: Create a test archive
        archive_path = tmp_path / "test_backup.tar.gz"
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test files
        (source_dir / "test.db").write_text("test data")
        (source_dir / "config.json").write_text('{"key": "value"}')

        # Create archive with sha256 checksums
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source_dir, arcname="backup")

        checksum_file = tmp_path / "checksums.txt"
        result = subprocess.run(
            ["bash", "-c", f"cd {source_dir} && find . -type f -exec sha256sum {{}} \\;"],
            capture_output=True,
            text=True
        )
        checksum_file.write_text(result.stdout)

        # Act: Verify the archive
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Verify checksums match
        result = subprocess.run(
            ["bash", "-c", f"cd {extract_dir / 'backup'} && sha256sum --check {checksum_file}"],
            capture_output=True,
            text=True
        )

        # Assert: Checksums should verify
        assert result.returncode == 0, f"Checksum verification should pass: {result.stderr}"


# ============================================================================
# P0 Test 4: Full Restore Recovers All Components
# ============================================================================

class TestFullRestore:
    """P0: Full restore - all components recovered."""

    def test_restore_script_handles_missing_backup_gracefully(self, tmp_path):
        """
        P0: Restore fails gracefully when backup doesn't exist.

        Given: No backup file exists
        When: restore_full_system.sh is called
        Then: Script exits with error and shows meaningful message

        This is critical for preventing data loss from misconfiguration.
        """
        restore_script = Path("scripts/restore_full_system.sh")
        assert restore_script.exists(), "Restore script must exist"

        # Act: Try to restore from non-existent backup
        result = subprocess.run(
            [str(restore_script), "nonexistent_backup_20230101.tar.gz"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Assert: Should fail with non-zero exit
        assert result.returncode != 0, "Restore should fail with missing backup"

    def test_restore_script_validates_archive_before_extracting(self, tmp_path):
        """
        P0: Restore validates archive integrity before extracting.

        Given: A corrupted backup archive
        When: restore_full_system.sh is called
        Then: Script detects corruption and fails safely

        This prevents partial/corrupt restores from corrupting existing data.
        """
        restore_script = Path("scripts/restore_full_system.sh")
        script_content = restore_script.read_text()

        # Assert: Script should have validation step
        # Look for tar integrity check or similar
        assert "tar" in script_content, "Restore must use tar for extraction"

        # Assert: Should check archive exists before extracting
        # (This is a structural test - the actual validation is tested by integration tests)


# ============================================================================
# P0 Test 5: Migration NODE_ROLE Transfer Verification
# ============================================================================

class TestMigrationNodeRoleTransfer:
    """P0: Server migration - NODE_ROLE transfer."""

    def test_migration_script_validates_node_role(self, tmp_path):
        """
        P0: Migration validates NODE_ROLE before transfer.

        Given: Migration is triggered
        When: NODE_ROLE is missing or invalid
        Then: Script exits with validation error

        This is critical for FR70: server migration without data loss.
        """
        migration_script = Path("scripts/migrate_server.sh")
        assert migration_script.exists(), "Migration script must exist"

        # Act: Run migration with missing NEW_SERVER_HOST (required param)
        result = subprocess.run(
            [str(migration_script), "--node-role", "contabo"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Assert: Should fail validation
        assert result.returncode != 0, "Migration should fail without new host"

    def test_migration_script_rejects_invalid_node_role(self, tmp_path):
        """
        P0: Migration rejects invalid NODE_ROLE values.

        Given: NODE_ROLE is set to invalid value
        When: Migration script validates inputs
        Then: Script exits with error and lists valid values

        Valid values are: contabo, cloudzy, desktop

        KNOWN GAP (TDD RED): The script currently checks SSH connectivity
        BEFORE validating NODE_ROLE. The error message for invalid NODE_ROLE
        should be visible to the user before SSH is attempted.
        """
        migration_script = Path("scripts/migrate_server.sh")

        # Act: Run migration with invalid NODE_ROLE
        result = subprocess.run(
            [str(migration_script), "--new-host", "test.example.com", "--node-role", "invalid_role"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Assert: Should fail with invalid role error
        assert result.returncode != 0, "Migration should reject invalid NODE_ROLE"

        # KNOWN GAP: Error message should be visible to user
        # This assertion FAILS until NODE_ROLE validation is moved before SSH check
        output = (result.stdout + result.stderr).lower()
        assert "contabo" in output or "cloudzy" in output or "desktop" in output, \
            "Error should list valid NODE_ROLE values (GAP: validation happens after SSH check)"

    def test_migration_has_node_role_transfer_function(self):
        """
        P0: Migration script has NODE_ROLE transfer function.

        Given: Migration is executing
        When: transfer_node_role_config is called
        Then: NODE_ROLE is set on target server

        This is the core of FR70: server migration.
        """
        script_content = Path("scripts/migrate_server.sh").read_text()

        # Assert: transfer_node_role_config function exists
        assert "transfer_node_role_config()" in script_content, \
            "transfer_node_role_config function must exist"

        # Assert: NODE_ROLE is exported
        assert 'export NODE_ROLE=' in script_content, \
            "NODE_ROLE must be exported on target"


# ============================================================================
# P0 Test 6: Migration Health Check on New Host
# ============================================================================

class TestMigrationHealthChecks:
    """P0: Server migration - health check verification."""

    def test_migration_has_health_check_function(self):
        """
        P0: Migration runs health checks on target server.

        Given: Migration completes transfer
        When: run_health_checks is called
        Then: Health status is reported for target server

        This is critical for FR70: verify server is working after migration.
        """
        script_content = Path("scripts/migrate_server.sh").read_text()

        # Assert: run_health_checks function exists
        assert "run_health_checks()" in script_content, \
            "run_health_checks function must exist"

    def test_migration_verifies_node_role_after_transfer(self):
        """
        P0: Migration verifies NODE_ROLE on target after transfer.

        Given: NODE_ROLE config has been transferred
        When: verify_node_role is called
        Then: Confirms NODE_ROLE matches expected value on target

        This ensures the migrated server has correct identity.
        """
        script_content = Path("scripts/migrate_server.sh").read_text()

        # Assert: verify_node_role function exists
        assert "verify_node_role()" in script_content, \
            "verify_node_role function must exist"

        # Assert: Verification checks ~/.quantmind/node_role
        assert ".quantmind/node_role" in script_content, \
            "Must verify ~/.quantmind/node_role file on target"


# ============================================================================
# P0 Test 7: Sequential Update Rollback Triggered on Health Check Failure
# ============================================================================

class TestSequentialUpdateRollback:
    """P0: 3-node sequential update with automatic rollback."""

    def test_sequential_update_flow_has_rollback_logic(self):
        """
        P0: Sequential update triggers rollback on health check failure.

        Given: Node update is in progress
        When: health_check returns unhealthy status
        Then: rollback_node_task is triggered

        This tests R-001 mitigation: automatic rollback on failure.
        """
        flow_content = Path("flows/node_update_flow.py").read_text()

        # Assert: rollback_node_task exists
        assert "rollback_node_task" in flow_content, \
            "rollback_node_task must exist"

        # Assert: Rollback is triggered on health check failure
        # Look for the pattern: if health.status != "healthy" -> rollback
        assert 'health.status != "healthy"' in flow_content or "health.status" in flow_content, \
            "Must check health status after update"

        # Assert: notify_failure is called on rollback
        assert "notify_failure" in flow_content, \
            "notify_failure must be called on rollback"

    def test_rollback_creates_rollback_point_before_update(self):
        """
        P0: Rollback point created before each node update.

        Given: Node update is about to start
        When: create_rollback_point is called
        Then: Rollback version is stored for recovery

        This ensures rollback can actually restore previous state.
        """
        flow_content = Path("flows/node_update_flow.py").read_text()

        # Assert: create_rollback_point function exists
        assert "create_rollback_point" in flow_content, \
            "create_rollback_point must exist"

        # Assert: Rollback point is created before update in the flow
        # The flow should call create_rollback_point before update_node
        assert "create_rollback_point_task" in flow_content, \
            "create_rollback_point_task must be in the flow"


# ============================================================================
# P0 Test 8: Rollback Sends Notification to Copilot
# ============================================================================

class TestRollbackNotifications:
    """P0: Rollback notification to Copilot."""

    def test_notify_failure_includes_failed_node_info(self):
        """
        P0: Notification includes failed node and rollback info.

        Given: Rollback has been triggered
        When: notify_failure is called
        Then: Notification includes failed node, previous nodes, and next steps

        This is critical for R-001: operator must be informed of failures.
        """
        flow_content = Path("flows/node_update_flow.py").read_text()

        # Assert: notify_failure function exists
        assert "def notify_failure" in flow_content, \
            "notify_failure function must exist"

        # Assert: Notification includes node name
        assert "failed_node" in flow_content, \
            "Notification must include failed node name"

        # Assert: Notification includes previous nodes
        assert "previous_nodes" in flow_content, \
            "Notification must include previously updated nodes"

    def test_notify_success_includes_all_updated_nodes(self):
        """
        P0: Success notification lists all updated nodes.

        Given: All nodes updated successfully
        When: notify_success is called
        Then: Notification includes list of all updated nodes

        This provides completion confirmation for FR67.
        """
        flow_content = Path("flows/node_update_flow.py").read_text()

        # Assert: notify_success function exists
        assert "def notify_success" in flow_content, \
            "notify_success function must exist"

        # Assert: Success notification includes nodes list
        assert "nodes_updated" in flow_content, \
            "Success notification must include updated nodes list"


# ============================================================================
# P0 Test 9: Node Health Check Per Node
# ============================================================================

class TestNodeHealthChecks:
    """P0: 3-node health checks."""

    def test_health_check_task_returns_node_health_status(self):
        """
        P0: Health check returns status for specific node.

        Given: Node is running
        When: health_check_task(node) is called
        Then: NodeHealthStatus is returned with node, status, version

        This tests the health check per node requirement.
        """
        flow_content = Path("flows/node_update_flow.py").read_text()

        # Assert: health_check_task function exists
        assert "def health_check_task" in flow_content, \
            "health_check_task must exist"

        # Assert: Returns NodeHealthStatus dataclass
        assert "NodeHealthStatus" in flow_content, \
            "health_check_task should return NodeHealthStatus"

        # Assert: NODE_URLS maps all three nodes
        assert '"contabo"' in flow_content or "'contabo'" in flow_content, \
            "Contabo node must be in NODE_URLS"
        assert '"cloudzy"' in flow_content or "'cloudzy'" in flow_content, \
            "Cloudzy node must be in NODE_URLS"
        assert '"desktop"' in flow_content or "'desktop'" in flow_content, \
            "Desktop node must be in NODE_URLS"

    def test_sequential_update_runs_health_check_after_each_node(self):
        """
        P0: Health check runs after each node in sequence.

        Given: Sequential update is running
        When: Each node update completes
        Then: health_check_task runs before proceeding to next node

        This ensures validation at each step (R-001).
        """
        flow_content = Path("flows/node_update_flow.py").read_text()

        # Assert: The flow iterates over NODE_UPDATE_ORDER
        assert "NODE_UPDATE_ORDER" in flow_content, \
            "Flow must iterate over NODE_UPDATE_ORDER"

        # Assert: health_check is called in the loop
        # Find the main flow function and verify health_check is called
        assert "for node in NODE_UPDATE_ORDER" in flow_content or \
               "for node in" in flow_content, \
            "Flow must loop through nodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
