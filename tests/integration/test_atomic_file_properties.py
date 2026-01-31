"""
Property-Based Tests for Atomic File Operations

Tests universal properties of atomic file writes to ensure data integrity.

**Feature: quantmindx-unified-backend, Property 12: Atomic File Write Operations**
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.router.sync import DiskSyncer


class TestAtomicFileWriteProperties:
    """
    Property tests for atomic file write operations.
    
    **Validates: Property 12: Atomic File Write Operations**
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def disk_syncer(self, temp_dir):
        """Create DiskSyncer with temporary path."""
        return DiskSyncer(mt5_path=temp_dir)
    
    @given(
        symbols=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu',)), min_size=6, max_size=6),
            min_size=1,
            max_size=10,
            unique=True
        ),
        multipliers=st.lists(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_atomic_write_never_produces_partial_files(
        self,
        disk_syncer,
        temp_dir,
        symbols,
        multipliers
    ):
        """
        Property: Atomic writes MUST never produce partially written files.
        
        Any file read MUST contain complete, valid JSON.
        
        **Validates: Requirements 7.2**
        """
        assume(len(symbols) == len(multipliers))
        
        # Build risk matrix
        risk_matrix = {}
        for symbol, multiplier in zip(symbols, multipliers):
            risk_matrix[symbol] = {
                "multiplier": multiplier,
                "timestamp": 1234567890
            }
        
        # Write risk matrix
        disk_syncer.sync_risk_matrix(risk_matrix)
        
        # Read back - should always be valid JSON
        risk_file = Path(temp_dir) / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            content = f.read()
        
        # Must be valid JSON
        parsed = json.loads(content)
        
        # Must match original data
        assert parsed == risk_matrix
    
    @given(
        num_writes=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50)
    def test_sequential_writes_maintain_consistency(
        self,
        disk_syncer,
        temp_dir,
        num_writes
    ):
        """
        Property: Sequential writes MUST maintain file consistency.
        
        Each write MUST produce a valid, complete file.
        
        **Validates: Requirements 7.2**
        """
        risk_file = Path(temp_dir) / "risk_matrix.json"
        
        for i in range(num_writes):
            risk_matrix = {
                f"SYMBOL_{i}": {
                    "multiplier": float(i) / num_writes,
                    "timestamp": 1234567890 + i
                }
            }
            
            # Write
            disk_syncer.sync_risk_matrix(risk_matrix)
            
            # Verify file is valid after each write
            assert risk_file.exists()
            
            with open(risk_file, 'r') as f:
                parsed = json.load(f)
            
            assert parsed == risk_matrix
    
    @given(
        num_threads=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=30)
    def test_concurrent_writes_produce_valid_files(
        self,
        disk_syncer,
        temp_dir,
        num_threads
    ):
        """
        Property: Concurrent writes MUST always produce valid files.
        
        Even under concurrent access, files MUST never be corrupted.
        
        **Validates: Requirements 7.1, 7.2**
        """
        risk_file = Path(temp_dir) / "risk_matrix.json"
        
        def write_risk_matrix(thread_id):
            risk_matrix = {
                f"THREAD_{thread_id}": {
                    "multiplier": float(thread_id) / num_threads,
                    "timestamp": 1234567890 + thread_id
                }
            }
            disk_syncer.sync_risk_matrix(risk_matrix)
            return thread_id
        
        # Execute concurrent writes
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_risk_matrix, i) for i in range(num_threads)]
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # File must exist and be valid JSON
        assert risk_file.exists()
        
        with open(risk_file, 'r') as f:
            parsed = json.load(f)
        
        # Must be valid risk matrix structure
        assert isinstance(parsed, dict)
        for symbol, data in parsed.items():
            assert "multiplier" in data
            assert "timestamp" in data
    
    @given(
        write_count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50)
    def test_no_temp_files_left_behind(
        self,
        disk_syncer,
        temp_dir,
        write_count
    ):
        """
        Property: Atomic writes MUST NOT leave temporary files behind.
        
        After write completion, only the target file should exist.
        
        **Validates: Requirements 7.2**
        """
        for i in range(write_count):
            risk_matrix = {
                "TEST": {
                    "multiplier": 1.0,
                    "timestamp": 1234567890 + i
                }
            }
            
            disk_syncer.sync_risk_matrix(risk_matrix)
        
        # Check for temp files
        temp_files = list(Path(temp_dir).glob("*.tmp"))
        
        # No temp files should remain
        assert len(temp_files) == 0
    
    @given(
        data_size=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50)
    def test_write_preserves_data_integrity(
        self,
        disk_syncer,
        temp_dir,
        data_size
    ):
        """
        Property: Written data MUST exactly match read data.
        
        No data corruption or loss during write/read cycle.
        
        **Validates: Requirements 7.2**
        """
        # Generate risk matrix with specified size
        risk_matrix = {}
        for i in range(data_size):
            risk_matrix[f"SYM{i:03d}"] = {
                "multiplier": float(i) / data_size,
                "timestamp": 1234567890 + i
            }
        
        # Write
        disk_syncer.sync_risk_matrix(risk_matrix)
        
        # Read back
        risk_file = Path(temp_dir) / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            read_data = json.load(f)
        
        # Must match exactly
        assert read_data == risk_matrix
        
        # Verify all entries
        assert len(read_data) == data_size
        for key in risk_matrix:
            assert key in read_data
            assert read_data[key] == risk_matrix[key]
    
    @given(
        multiplier=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_write_handles_all_valid_multiplier_values(
        self,
        disk_syncer,
        temp_dir,
        multiplier
    ):
        """
        Property: Atomic writes MUST handle all valid multiplier values.
        
        Any valid float multiplier MUST be written and read correctly.
        
        **Validates: Requirements 7.2**
        """
        risk_matrix = {
            "TEST": {
                "multiplier": multiplier,
                "timestamp": 1234567890
            }
        }
        
        # Write
        disk_syncer.sync_risk_matrix(risk_matrix)
        
        # Read back
        risk_file = Path(temp_dir) / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            read_data = json.load(f)
        
        # Multiplier must match (within floating point precision)
        assert abs(read_data["TEST"]["multiplier"] - multiplier) < 1e-10
    
    def test_write_failure_does_not_corrupt_existing_file(
        self,
        disk_syncer,
        temp_dir
    ):
        """
        Property: Write failures MUST NOT corrupt existing files.
        
        If a write fails, the previous valid file MUST remain intact.
        
        **Validates: Requirements 7.2**
        """
        # Write initial valid file
        initial_matrix = {
            "INITIAL": {
                "multiplier": 1.0,
                "timestamp": 1234567890
            }
        }
        disk_syncer.sync_risk_matrix(initial_matrix)
        
        # Verify initial file
        risk_file = Path(temp_dir) / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            initial_content = f.read()
        
        # Attempt to write invalid data (should fail validation)
        invalid_matrix = {
            "INVALID": {
                "multiplier": 1.0
                # Missing timestamp - should fail validation
            }
        }
        
        try:
            disk_syncer.sync_risk_matrix(invalid_matrix)
        except ValueError:
            pass  # Expected to fail
        
        # Original file should still be intact
        with open(risk_file, 'r') as f:
            current_content = f.read()
        
        assert current_content == initial_content
        
        # And still valid JSON
        parsed = json.loads(current_content)
        assert parsed == initial_matrix
