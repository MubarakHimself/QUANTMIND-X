"""
Integration Tests for MQL5-Python Bridge

Tests the complete workflow from heartbeat reception to risk matrix sync.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.api.heartbeat import HeartbeatHandler, HeartbeatPayload
from src.router.sync import DiskSyncer
from src.error_handlers import MQL5BridgeErrorHandler, CircuitBreaker


class TestMQL5PythonBridgeIntegration:
    """Integration tests for MQL5-Python bridge."""
    
    @pytest.fixture
    def temp_mt5_path(self):
        """Create temporary MT5 path for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def disk_syncer(self, temp_mt5_path):
        """Create DiskSyncer with temporary path."""
        return DiskSyncer(mt5_path=temp_mt5_path)
    
    @pytest.fixture
    def heartbeat_handler(self, disk_syncer):
        """Create HeartbeatHandler with mocked database."""
        with patch('src.api.heartbeat.DatabaseManager'):
            handler = HeartbeatHandler(disk_syncer=disk_syncer)
            handler.db_manager = Mock()
            handler.db_manager.save_daily_snapshot = Mock()
            handler.db_manager.get_prop_account = Mock(return_value=Mock(id=1))
            return handler
    
    def test_end_to_end_heartbeat_workflow(self, heartbeat_handler, temp_mt5_path):
        """
        Test complete heartbeat workflow: receive -> validate -> process -> sync.
        
        **Validates: Requirements 7.1, 7.2, 7.5, 7.6**
        """
        # Create heartbeat payload
        payload = {
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "magic_number": 12345,
            "account_id": "test_account",
            "current_equity": 105000.0,
            "current_balance": 100000.0,
            "risk_multiplier": 1.0,
            "timestamp": int(datetime.utcnow().timestamp()),
            "open_positions": 2,
            "daily_pnl": 5000.0
        }
        
        # Process heartbeat
        response = heartbeat_handler.process_heartbeat(payload)
        
        # Verify response
        assert response.success is True
        assert response.risk_multiplier >= 0.0
        assert response.risk_multiplier <= 1.0
        
        # Verify risk matrix file was created
        risk_matrix_path = Path(temp_mt5_path) / "risk_matrix.json"
        assert risk_matrix_path.exists()
        
        # Verify risk matrix content
        with open(risk_matrix_path, 'r') as f:
            risk_data = json.load(f)
        
        assert "EURUSD" in risk_data
        assert "multiplier" in risk_data["EURUSD"]
        assert "timestamp" in risk_data["EURUSD"]
    
    def test_heartbeat_with_invalid_payload(self, heartbeat_handler):
        """
        Test heartbeat with invalid payload returns error response.
        
        **Validates: Requirements 7.6**
        """
        # Missing required fields
        invalid_payload = {
            "ea_name": "TestEA",
            "symbol": "EURUSD"
            # Missing other required fields
        }
        
        response = heartbeat_handler.process_heartbeat(invalid_payload)
        
        # Should return error response
        assert response.success is False
        assert "error" in response.message.lower() or "missing" in response.message.lower()
        assert response.risk_multiplier == 0.0  # Safe default
    
    def test_atomic_file_write_prevents_corruption(self, disk_syncer, temp_mt5_path):
        """
        Test that atomic file writes prevent partial reads.
        
        **Validates: Requirements 7.2, Property 12**
        """
        risk_matrix = {
            "EURUSD": {"multiplier": 0.75, "timestamp": 1234567890},
            "GBPUSD": {"multiplier": 0.85, "timestamp": 1234567891}
        }
        
        # Write risk matrix
        disk_syncer.sync_risk_matrix(risk_matrix)
        
        # Read back and verify
        risk_matrix_path = Path(temp_mt5_path) / "risk_matrix.json"
        with open(risk_matrix_path, 'r') as f:
            read_data = json.load(f)
        
        assert read_data == risk_matrix
        
        # Verify no temp files left behind
        temp_files = list(Path(temp_mt5_path).glob("*.tmp"))
        assert len(temp_files) == 0
    
    def test_heartbeat_failure_triggers_fallback(self):
        """
        Test that heartbeat failures trigger fallback mechanisms.
        
        **Validates: Requirements 7.8**
        """
        error_handler = MQL5BridgeErrorHandler()
        
        # Simulate heartbeat failure
        test_error = ConnectionError("Network timeout")
        result = error_handler.handle_heartbeat_failure("TestEA", test_error)
        
        # Should attempt fallback
        assert isinstance(result, bool)
        
        # Check failure tracking
        assert "TestEA" in error_handler.heartbeat_failures
        assert error_handler.heartbeat_failures["TestEA"] > 0
    
    def test_circuit_breaker_opens_after_failures(self):
        """
        Test circuit breaker opens after repeated failures.
        
        **Validates: Requirements 7.9**
        """
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Create failing function
        def failing_function():
            raise ConnectionError("Service unavailable")
        
        # Trigger failures
        for i in range(3):
            with pytest.raises(ConnectionError):
                circuit_breaker.call(failing_function)
        
        # Circuit should now be open
        assert circuit_breaker.state == "OPEN"
        
        # Next call should fail immediately without calling function
        with pytest.raises(Exception) as exc_info:
            circuit_breaker.call(failing_function)
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    def test_global_variable_update_integration(self, disk_syncer):
        """
        Test GlobalVariable update integration.
        
        **Validates: Requirements 7.3**
        """
        # Mock MT5 module
        with patch('src.router.sync.MetaTrader5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.global_variable_set.return_value = True
            mock_mt5.shutdown.return_value = None
            
            # Update global variable
            result = disk_syncer.update_global_variable("QM_RISK_EURUSD", 0.85)
            
            # Verify MT5 calls
            assert result is True
            mock_mt5.initialize.assert_called_once()
            mock_mt5.global_variable_set.assert_called_once_with("QM_RISK_EURUSD", 0.85)
            mock_mt5.shutdown.assert_called_once()
    
    def test_risk_matrix_validation_rejects_invalid_data(self, disk_syncer):
        """
        Test that risk matrix validation rejects invalid data.
        
        **Validates: Requirements 7.2**
        """
        # Missing required fields
        invalid_matrix = {
            "EURUSD": {"multiplier": 0.75}  # Missing timestamp
        }
        
        with pytest.raises(ValueError) as exc_info:
            disk_syncer.sync_risk_matrix(invalid_matrix)
        
        assert "timestamp" in str(exc_info.value).lower()
    
    def test_concurrent_heartbeats_handled_safely(self, heartbeat_handler):
        """
        Test that concurrent heartbeats are handled safely.
        
        **Validates: Requirements 7.1, 7.2**
        """
        import threading
        
        results = []
        
        def send_heartbeat(ea_name):
            payload = {
                "ea_name": ea_name,
                "symbol": "EURUSD",
                "magic_number": 12345,
                "account_id": "test_account",
                "current_equity": 105000.0,
                "current_balance": 100000.0,
                "risk_multiplier": 1.0,
                "timestamp": int(datetime.utcnow().timestamp())
            }
            response = heartbeat_handler.process_heartbeat(payload)
            results.append(response)
        
        # Send concurrent heartbeats
        threads = []
        for i in range(5):
            thread = threading.Thread(target=send_heartbeat, args=(f"EA_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 5
        assert all(r.success for r in results)
    
    def test_heartbeat_with_database_failure_continues(self, heartbeat_handler):
        """
        Test that heartbeat continues even if database update fails.
        
        **Validates: Requirements 7.7**
        """
        # Make database update fail
        heartbeat_handler.db_manager.save_daily_snapshot.side_effect = Exception("DB Error")
        
        payload = {
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "magic_number": 12345,
            "account_id": "test_account",
            "current_equity": 105000.0,
            "current_balance": 100000.0,
            "risk_multiplier": 1.0,
            "timestamp": int(datetime.utcnow().timestamp())
        }
        
        # Should still succeed (graceful degradation)
        response = heartbeat_handler.process_heartbeat(payload)
        
        # Response should indicate success despite DB failure
        assert response.success is True
        assert response.risk_multiplier >= 0.0


class TestFileWatcherIntegration:
    """Integration tests for file watcher."""
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for watching."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_file_watcher_detects_changes(self, temp_file):
        """
        Test that file watcher detects file changes.
        
        **Validates: Requirements 7.4, Property 13**
        """
        from src.router.file_watcher import RiskMatrixWatcher
        
        changes_detected = []
        
        def on_change(data):
            changes_detected.append(data)
        
        # Start watcher
        watcher = RiskMatrixWatcher(temp_file, on_change, poll_interval=0.1)
        watcher.start()
        
        try:
            # Modify file
            import time
            time.sleep(0.2)  # Let watcher start
            
            with open(temp_file, 'w') as f:
                json.dump({"test": "data"}, f)
            
            # Wait for detection
            time.sleep(0.3)
            
            # Should have detected change
            assert len(changes_detected) > 0
            assert changes_detected[0] == {"test": "data"}
            
        finally:
            watcher.stop()
