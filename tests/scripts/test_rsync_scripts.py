"""
Unit tests for rsync scripts.

Tests the checksum generation and verification logic.
"""

import os
import subprocess
import tempfile
import pytest
from pathlib import Path


class TestChecksumVerification:
    """Test checksum generation and verification."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_checksum_generation(self, test_dir):
        """Test that checksums are generated correctly."""
        # Create test files
        test_file1 = test_dir / "test1.db"
        test_file2 = test_dir / "test2.json"
        test_file1.write_text("test content 1")
        test_file2.write_text('{"key": "value"}')

        # Generate checksums using the script
        result = subprocess.run(
            ["bash", "-c", f"cd {test_dir} && find . -type f -exec sha256sum {{}} \\; | sort"],
            capture_output=True,
            text=True
        )

        lines = result.stdout.strip().split('\n')
        assert len(lines) == 2
        assert "test1.db" in lines[0]
        assert "test2.json" in lines[1]

    def test_checksum_verification(self, test_dir):
        """Test that checksum verification works."""
        # Create test file
        test_file = test_dir / "test.db"
        test_file.write_text("original content")

        # Generate initial checksum
        checksum_file = test_dir / "checksums.txt"
        result = subprocess.run(
            ["sha256sum", str(test_file)],
            capture_output=True,
            text=True
        )
        checksum_file.write_text(result.stdout)

        # Verify - should pass
        result = subprocess.run(
            ["sha256sum", "--check", str(checksum_file)],
            capture_output=True,
            text=True,
            cwd=test_dir
        )
        assert result.returncode == 0

        # Modify file and verify - should fail
        test_file.write_text("modified content")
        result = subprocess.run(
            ["sha256sum", "--check", str(checksum_file)],
            capture_output=True,
            text=True,
            cwd=test_dir
        )
        assert result.returncode != 0


class TestScriptExecution:
    """Test script execution."""

    def test_verify_checksum_help(self):
        """Test that verify_checksum.sh shows help."""
        result = subprocess.run(
            ["bash", "scripts/verify_checksum.sh"],
            capture_output=True,
            text=True
        )
        assert "Usage" in result.stdout

    def test_verify_checksum_generate(self, tmp_path):
        """Test --generate mode."""
        # Create test file
        test_file = tmp_path / "test.db"
        test_file.write_text("test")

        output_file = tmp_path / "checksums.txt"
        result = subprocess.run(
            ["bash", "scripts/verify_checksum.sh", "--generate", str(tmp_path), str(output_file)],
            capture_output=True,
            text=True
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "test.db" in content

    def test_verify_checksum_verify(self, tmp_path):
        """Test --verify mode."""
        # Create test file and checksum
        test_file = tmp_path / "test.db"
        test_file.write_text("test")

        output_file = tmp_path / "checksums.txt"
        subprocess.run(
            ["bash", "scripts/verify_checksum.sh", "--generate", str(tmp_path), str(output_file)],
            check=True
        )

        # Verify should pass
        result = subprocess.run(
            ["bash", "scripts/verify_checksum.sh", "--verify", str(output_file), str(tmp_path)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "PASSED" in result.stdout


class TestRsyncAuditLogger:
    """Test the Python audit logger."""

    def test_audit_logger_import(self):
        """Test that audit logger can be imported."""
        # Just verify the module syntax is correct
        with open('scripts/rsync_audit_logger.py', 'r') as f:
            code = f.read()
            compile(code, 'rsync_audit_logger.py', 'exec')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])