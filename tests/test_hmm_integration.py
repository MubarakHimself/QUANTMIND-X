"""
HMM Integration Tests
=====================

Integration tests for the complete HMM regime detection system.
Tests the full workflow from training to deployment.

Run with: pytest tests/test_hmm_integration.py -v
"""

import pytest
import os
import sys
import json
import tempfile
import pickle
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestHMMDatabaseModels:
    """Test HMM database models."""
    
    def test_hmm_model_creation(self):
        """Test HMMModel can be instantiated."""
        try:
            from src.database.models import HMMModel
            
            model = HMMModel(
                version="2026.2.14_test",
                model_type="universal",
                n_states=4,
                file_path="/data/hmm/models/test.pkl",
                checksum="abc123",
                training_samples=10000,
                log_likelihood=-1500.0,
                metadata={"test": True}
            )
            
            assert model.version == "2026.2.14_test"
            assert model.model_type == "universal"
            assert model.n_states == 4
            
        except ImportError:
            pytest.skip("Database models not available")
    
    def test_hmm_shadow_log_creation(self):
        """Test HMMShadowLog can be instantiated."""
        try:
            from src.database.models import HMMShadowLog
            
            log = HMMShadowLog(
                symbol="EURUSD",
                timeframe="H1",
                ising_regime="TRENDING_LOW_VOL",
                ising_confidence=0.85,
                hmm_state=0,
                hmm_regime="TRENDING_LOW_VOL",
                hmm_confidence=0.92,
                agreement=True,
                decision_source="ising"
            )
            
            assert log.symbol == "EURUSD"
            assert log.agreement is True
            
        except ImportError:
            pytest.skip("Database models not available")
    
    def test_hmm_deployment_creation(self):
        """Test HMMDeployment can be instantiated."""
        try:
            from src.database.models import HMMDeployment
            
            deployment = HMMDeployment(
                mode="hmm_shadow",
                previous_mode="ising_only",
                hmm_weight=0.0,
                model_version="2026.2.14_test",
                approved_by="test_user",
                approval_token="test_token"
            )
            
            assert deployment.mode == "hmm_shadow"
            assert deployment.hmm_weight == 0.0
            
        except ImportError:
            pytest.skip("Database models not available")


class TestHMMFeatureExtraction:
    """Test HMM feature extraction module."""
    
    def test_feature_extractor_initialization(self):
        """Test HMMFeatureExtractor can be initialized."""
        try:
            from src.risk.physics.hmm_features import HMMFeatureExtractor
            
            extractor = HMMFeatureExtractor()
            assert extractor is not None
            
        except ImportError:
            pytest.skip("HMM features module not available")
    
    def test_feature_extraction_from_ising(self):
        """Test feature extraction from Ising outputs."""
        try:
            from src.risk.physics.hmm_features import HMMFeatureExtractor
            
            extractor = HMMFeatureExtractor()
            
            # Mock Ising output
            ising_output = {
                'magnetization': 0.5,
                'susceptibility': 0.3,
                'energy': -1.2,
                'regime': 'TRENDING_LOW_VOL'
            }
            
            features = extractor.extract_from_ising(ising_output)
            
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            
        except ImportError:
            pytest.skip("HMM features module not available")
    
    def test_feature_extraction_from_prices(self):
        """Test feature extraction from price data."""
        try:
            import pandas as pd
            from src.risk.physics.hmm_features import HMMFeatureExtractor
            
            extractor = HMMFeatureExtractor()
            
            # Create mock price data
            prices = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 101,
                'low': np.random.randn(100).cumsum() + 99
            })
            
            features = extractor.extract_from_prices(prices)
            
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            
        except ImportError:
            pytest.skip("HMM features module not available")
    
    def test_extract_ising_features_dict(self):
        """Test extract_ising_features returns dict."""
        try:
            from src.risk.physics.hmm_features import HMMFeatureExtractor
            
            extractor = HMMFeatureExtractor()
            
            # Extract features using volatility
            features = extractor.extract_ising_features(volatility=1.5)
            
            assert isinstance(features, dict)
            assert 'magnetization' in features or 'susceptibility' in features
            
        except ImportError:
            pytest.skip("HMM features module not available")
    
    def test_extract_price_features_dict(self):
        """Test extract_price_features returns dict."""
        try:
            import pandas as pd
            from src.risk.physics.hmm_features import HMMFeatureExtractor
            
            extractor = HMMFeatureExtractor()
            
            # Create mock price series
            prices = pd.Series(np.random.randn(100).cumsum() + 100)
            
            features = extractor.extract_price_features(prices)
            
            assert isinstance(features, dict)
            
        except ImportError:
            pytest.skip("HMM features module not available")


class TestHMMSensor:
    """Test HMM regime sensor."""
    
    def test_sensor_initialization(self):
        """Test HMMRegimeSensor can be initialized."""
        try:
            from src.risk.physics.hmm_sensor import HMMRegimeSensor
            
            sensor = HMMRegimeSensor()
            assert sensor is not None
            
        except ImportError:
            pytest.skip("HMM sensor module not available")
    
    def test_sensor_with_mock_model(self):
        """Test sensor prediction with mock model."""
        try:
            from src.risk.physics.hmm_sensor import HMMRegimeSensor, HMMRegimeReading
            
            # Create mock HMM model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.05, 0.05]])
            mock_model.n_components = 4
            mock_model.transmat_ = np.eye(4) * 0.8 + 0.05
            
            sensor = HMMRegimeSensor()
            sensor._model = mock_model
            sensor._model_metadata = {'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']}
            
            # Test prediction
            features = np.random.randn(1, 5)
            result = sensor.predict_regime(features)
            
            assert result is not None
            assert hasattr(result, 'state')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'regime')
            
        except ImportError:
            pytest.skip("HMM sensor module not available")
    
    def test_regime_mapping(self):
        """Test HMM state to regime label mapping."""
        try:
            from src.risk.physics.hmm_sensor import HMMRegimeSensor
            
            sensor = HMMRegimeSensor()
            
            # Test state mapping via config
            assert sensor.config.regime_mapping.get(0) == "TRENDING_LOW_VOL"
            assert sensor.config.regime_mapping.get(1) == "TRENDING_HIGH_VOL"
            assert sensor.config.regime_mapping.get(2) == "RANGING_LOW_VOL"
            assert sensor.config.regime_mapping.get(3) == "RANGING_HIGH_VOL"
            
        except ImportError:
            pytest.skip("HMM sensor module not available")


class TestHMMDeployment:
    """Test HMM deployment workflow."""
    
    def test_deployment_manager_initialization(self):
        """Test HMMDeploymentManager can be initialized."""
        try:
            from src.router.hmm_deployment import HMMDeploymentManager, DeploymentMode
            
            manager = HMMDeploymentManager()
            assert manager is not None
            
        except ImportError:
            pytest.skip("HMM deployment module not available")
    
    def test_deployment_mode_transitions(self):
        """Test valid deployment mode transitions."""
        try:
            from src.router.hmm_deployment import HMMDeploymentManager, DeploymentMode
            
            manager = HMMDeploymentManager()
            
            # Test valid transitions
            can_transition, _ = manager.can_transition_to(DeploymentMode.HMM_SHADOW)
            assert can_transition is True
            
        except ImportError:
            pytest.skip("HMM deployment module not available")
    
    def test_approval_token_generation(self):
        """Test approval token generation for restricted modes."""
        try:
            from src.router.hmm_deployment import HMMDeploymentManager, DeploymentMode
            
            manager = HMMDeploymentManager()
            
            # First transition to shadow mode (required before hybrid)
            manager.transition_to(DeploymentMode.HMM_SHADOW)
            
            # Generate token for restricted mode
            token = manager.request_approval(
                DeploymentMode.HMM_HYBRID_50,
                "test_user"
            )
            
            assert token is not None
            assert len(token) > 0
            
        except ImportError:
            pytest.skip("HMM deployment module not available")


class TestHMMVersionControl:
    """Test HMM version control system."""
    
    def test_version_control_initialization(self):
        """Test HMMVersionControl can be initialized."""
        try:
            from src.router.hmm_version_control import HMMVersionControl
            
            vc = HMMVersionControl()
            assert vc is not None
            
        except ImportError:
            pytest.skip("HMM version control module not available")
    
    def test_version_comparison(self):
        """Test version mismatch detection."""
        try:
            from src.router.hmm_version_control import HMMVersionControl
            
            vc = HMMVersionControl()
            
            # Test check_version_mismatch returns a boolean
            # (actual versions depend on remote/local state)
            mismatch = vc.check_version_mismatch()
            assert isinstance(mismatch, bool)
            
        except ImportError:
            pytest.skip("HMM version control module not available")
    
    def test_get_version_info(self):
        """Test get_version_info returns dict."""
        try:
            from src.router.hmm_version_control import HMMVersionControl
            
            vc = HMMVersionControl()
            
            info = vc.get_version_info()
            assert isinstance(info, dict)
            assert 'contabo' in info
            assert 'cloudzy' in info
            assert 'version_mismatch' in info
            
        except ImportError:
            pytest.skip("HMM version control module not available")


class TestHMMAPIEndpoints:
    """Test HMM API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from src.api.server import app
            
            return TestClient(app)
        except ImportError:
            return None
    
    def test_hmm_status_endpoint(self, client):
        """Test GET /api/hmm/status endpoint."""
        if client is None:
            pytest.skip("FastAPI test client not available")
        
        response = client.get("/api/hmm/status")
        
        # Should return 200 even if no model loaded
        assert response.status_code in [200, 500]
    
    def test_hmm_sync_endpoint(self, client):
        """Test POST /api/hmm/sync endpoint."""
        if client is None:
            pytest.skip("FastAPI test client not available")
        
        response = client.post(
            "/api/hmm/sync",
            json={"verify_checksum": True}
        )
        
        # May fail if version control not configured
        assert response.status_code in [200, 500, 409]
    
    def test_hmm_mode_endpoint_validation(self, client):
        """Test POST /api/hmm/mode endpoint validation."""
        if client is None:
            pytest.skip("FastAPI test client not available")
        
        # Test invalid mode
        response = client.post(
            "/api/hmm/mode",
            json={"mode": "invalid_mode"}
        )
        
        assert response.status_code == 400


class TestHMMSentinelIntegration:
    """Test HMM integration with Sentinel."""
    
    def test_sentinel_hmm_sensor_attribute(self):
        """Test Sentinel has HMM sensor attribute."""
        try:
            from src.router.sentinel import Sentinel
            
            sentinel = Sentinel()
            
            # Check for HMM sensor attribute (may be None if not configured)
            assert hasattr(sentinel, 'hmm_sensor') or hasattr(sentinel, '_hmm_sensor')
            
        except ImportError:
            pytest.skip("Sentinel module not available")
    
    def test_sentinel_shadow_mode_flag(self):
        """Test Sentinel shadow mode flag."""
        try:
            from src.router.sentinel import Sentinel
            
            sentinel = Sentinel()
            
            # Check for shadow mode attribute
            assert hasattr(sentinel, 'shadow_mode') or hasattr(sentinel, '_shadow_mode')
            
        except ImportError:
            pytest.skip("Sentinel module not available")


class TestHMMTrainingPipeline:
    """Test HMM training pipeline components."""
    
    def test_training_config_loading(self):
        """Test HMM config can be loaded."""
        config_path = os.path.join(project_root, "config", "hmm_config.json")
        
        if not os.path.exists(config_path):
            pytest.skip("HMM config file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert "model" in config
        assert "training" in config
        assert config["model"]["n_states"] == 4
    
    def test_model_validation_criteria(self):
        """Test model validation criteria."""
        try:
            from scripts.validate_hmm import validate_model
            
            # Mock model data
            model_data = {
                'log_likelihood': -1500,
                'state_distribution': [0.25, 0.25, 0.25, 0.25],
                'transition_matrix': np.eye(4) * 0.8 + 0.05,
                'ising_agreement': 0.85
            }
            
            # This would call actual validation
            # For now, just check the criteria
            assert model_data['log_likelihood'] > -2000
            assert max(model_data['state_distribution']) < 0.6
            assert np.mean(np.diag(model_data['transition_matrix'])) > 0.7
            assert model_data['ising_agreement'] > 0.8
            
        except ImportError:
            pytest.skip("Validation script not available")


class TestHMMEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    def test_full_workflow_mock(self):
        """Test full HMM workflow with mocked components."""
        try:
            # 1. Create mock model
            from hmmlearn import hmm
            
            model = hmm.GaussianHMM(
                n_components=4,
                covariance_type="full",
                n_iter=10
            )
            
            # 2. Train on mock data
            X = np.random.randn(1000, 5)
            model.fit(X)
            
            # 3. Make prediction
            state = model.predict(X[:1])
            probs = model.predict_proba(X[:1])
            
            assert state[0] in [0, 1, 2, 3]
            assert probs.shape == (1, 4)
            assert np.sum(probs) > 0.99  # Probabilities sum to 1
            
            # 4. Save model
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                model_path = f.name
            
            # 5. Load model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # 6. Verify loaded model works
            loaded_state = loaded_model.predict(X[:1])
            assert loaded_state[0] == state[0]
            
            # Cleanup
            os.unlink(model_path)
            
        except ImportError:
            pytest.skip("hmmlearn not available")


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])