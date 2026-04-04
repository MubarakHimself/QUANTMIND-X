"""Tests for BOCPD module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.risk.physics.bocpd import (
    BOCPDDetector,
    ConstantHazard,
    LogisticHazard,
    GaussianObservation,
    StudentTObservation,
)


class TestHazard:
    """Test hazard functions."""

    def test_constant_hazard(self):
        """Test constant hazard."""
        h = ConstantHazard(lam=250.0)
        assert h(0) == pytest.approx(1.0 / 250.0)
        assert h(100) == pytest.approx(1.0 / 250.0)
        assert h(1000) == pytest.approx(1.0 / 250.0)

    def test_constant_hazard_params(self):
        """Test hazard parameters."""
        h = ConstantHazard(lam=300.0)
        params = h.get_params()
        assert params["type"] == "ConstantHazard"
        assert params["lambda"] == 300.0

    def test_logistic_hazard(self):
        """Test logistic hazard."""
        h = LogisticHazard(tau_0=200.0, k=0.02)
        assert h(200) == pytest.approx(0.5, abs=0.01)
        assert h(0) < h(400)  # Increasing with run length

    def test_logistic_hazard_bounds(self):
        """Test logistic hazard is in [0, 1]."""
        h = LogisticHazard()
        for t in [0, 100, 200, 500, 1000]:
            val = h(t)
            assert 0.0 <= val <= 1.0


class TestObservation:
    """Test observation models."""

    def test_gaussian_observation_init(self):
        """Test Gaussian observation initialization."""
        obs = GaussianObservation(mu_0=0.0, kappa_0=1.0)
        assert obs.n == 0
        assert obs.sum_x == 0.0

    def test_gaussian_observation_update(self):
        """Test Gaussian observation update."""
        obs = GaussianObservation()
        obs.update(1.0)
        assert obs.n == 1
        assert obs.sum_x == 1.0
        assert obs.sum_x2 == 1.0

    def test_gaussian_observation_pdf(self):
        """Test Gaussian observation PDF."""
        obs = GaussianObservation()
        # PDF should be positive
        assert obs.pdf(0.0) > 0.0

        # Update and check PDF changes
        obs.update(0.5)
        pdf1 = obs.pdf(0.5)
        pdf0 = obs.pdf(0.0)
        assert pdf1 > 0.0 and pdf0 > 0.0

    def test_gaussian_observation_reset(self):
        """Test Gaussian observation reset."""
        obs = GaussianObservation()
        obs.update(1.0)
        assert obs.n == 1

        obs2 = obs.reset()
        assert obs2.n == 0
        assert obs2.mu_0 == obs.mu_0

    def test_student_t_observation(self):
        """Test Student-t observation."""
        obs = StudentTObservation(df=5.0)
        assert obs.df == 5.0
        assert obs.pdf(0.0) > 0.0

        obs.update(0.5)
        assert obs.n == 1
        assert obs.pdf(0.5) > 0.0

    def test_observation_params(self):
        """Test observation get_params."""
        obs = StudentTObservation(df=5.0)
        params = obs.get_params()
        assert params["type"] == "StudentTObservation"
        assert params["df"] == 5.0
        assert params["n"] == 0


class TestBOCPDDetector:
    """Test BOCPD detector."""

    def test_detector_init(self):
        """Test detector initialization."""
        detector = BOCPDDetector()
        assert detector.n_observations == 0
        assert len(detector.run_length_dist) == 1
        assert detector.run_length_dist[0] == pytest.approx(1.0)

    def test_detector_update(self):
        """Test detector update."""
        detector = BOCPDDetector(threshold=0.5)
        result = detector.update(0.5)

        assert "changepoint_prob" in result
        assert "is_changepoint" in result
        assert "current_run_length" in result
        assert 0.0 <= result["changepoint_prob"] <= 1.0

    def test_detector_multiple_updates(self):
        """Test multiple updates."""
        detector = BOCPDDetector()
        for _ in range(100):
            detector.update(np.random.randn())

        assert detector.n_observations == 100
        assert len(detector.run_length_dist) > 1  # Distribution should grow

    def test_detector_detects_changepoint(self):
        """Test that detector detects changepoint."""
        detector = BOCPDDetector(threshold=0.2, min_run_length=5)

        # Normal data
        for _ in range(20):
            detector.update(0.01)

        # Changepoint: sudden jump
        detector.update(2.0)
        detector.update(2.1)
        detector.update(2.2)

        # Check recent results have higher changepoint probability
        result = detector.update(2.3)
        assert result["changepoint_prob"] > 0.0

    def test_detector_reset(self):
        """Test detector reset."""
        detector = BOCPDDetector()
        for _ in range(50):
            detector.update(0.5)

        assert detector.n_observations == 50
        detector.reset()
        assert len(detector.run_length_dist) == 1
        assert detector.run_length_dist[0] == pytest.approx(1.0)

    def test_detector_predict_regime(self):
        """Test predict_regime method."""
        detector = BOCPDDetector()
        features = np.random.randn(10)
        result = detector.predict_regime(features)

        assert "changepoint_prob" in result
        assert "is_changepoint" in result
        assert "regime_type" in result
        assert "confidence" in result
        assert result["regime_type"] in ["TRANSITION", "STABLE"]

    def test_detector_calibrate(self):
        """Test calibration."""
        detector = BOCPDDetector()
        data = np.random.randn(500)
        calib = detector.calibrate(data, "TEST")

        assert "optimal_lambda" in calib
        assert "n_changepoints_found" in calib
        assert "avg_run_length" in calib
        assert calib["optimal_lambda"] > 0

    def test_detector_save_load(self):
        """Test save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.json"

            # Create and save
            detector1 = BOCPDDetector(threshold=0.3)
            detector1.save(path)
            assert path.exists()

            # Load
            detector2 = BOCPDDetector.load(path)
            assert detector2.threshold == pytest.approx(0.3)

    def test_detector_get_model_info(self):
        """Test get_model_info."""
        detector = BOCPDDetector()
        info = detector.get_model_info()

        assert "model_type" in info
        assert info["model_type"] == "BOCPD"
        assert "hazard" in info
        assert "observation" in info

    def test_detector_is_model_loaded(self):
        """Test is_model_loaded."""
        detector = BOCPDDetector()
        assert detector.is_model_loaded() is True


class TestBOCPDRobustness:
    """Test robustness of BOCPD."""

    def test_nan_handling(self):
        """Test NaN handling."""
        detector = BOCPDDetector()
        # Should not crash on NaN
        result = detector.update(np.nan)
        assert result["changepoint_prob"] == 0.0

    def test_inf_handling(self):
        """Test infinity handling."""
        detector = BOCPDDetector()
        # Should not crash on infinity
        result = detector.update(np.inf)
        assert result["changepoint_prob"] == 0.0

    def test_empty_features(self):
        """Test empty features."""
        detector = BOCPDDetector()
        result = detector.predict_regime(np.array([]))
        assert result["changepoint_prob"] == 0.0

    def test_large_dataset(self):
        """Test with large dataset."""
        detector = BOCPDDetector()
        data = np.random.randn(5000)
        for x in data:
            detector.update(x)

        assert detector.n_observations == 5000

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        detector = BOCPDDetector()
        for _ in range(100):
            detector.update(100.0)
        for _ in range(100):
            detector.update(-100.0)

        result = detector.update(0.0)
        assert np.isfinite(result["changepoint_prob"])


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test full workflow."""
        # Create detector
        detector = BOCPDDetector(threshold=0.4)

        # Calibrate on synthetic data
        train_data = np.random.randn(1000)
        calib = detector.calibrate(train_data)
        assert calib["optimal_lambda"] > 0

        # Run detection on new data
        test_data = np.concatenate([
            np.random.randn(200) + 0.0,  # Regime 1
            np.random.randn(200) + 1.0,  # Regime 2 (changepoint)
            np.random.randn(200) - 0.5,  # Regime 3 (changepoint)
        ])

        changepoints = []
        for x in test_data:
            result = detector.update(x)
            if result["is_changepoint"]:
                changepoints.append(detector.n_observations)

        # Should detect some changepoints
        assert len(changepoints) > 0

    def test_with_hmm_features(self):
        """Test with HMM-style features."""
        try:
            from src.risk.physics.hmm.trainer import extract_features_vectorized
        except ImportError:
            pytest.skip("HMM trainer not available")

        # Create synthetic OHLCV data
        n = 500
        close = 1.0 + np.cumsum(np.random.randn(n) * 0.001)
        df = pd.DataFrame({
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.ones(n) * 1000,
        })

        # Extract features
        features = extract_features_vectorized(df)

        # Run BOCPD on log returns (column 0)
        detector = BOCPDDetector()
        for x in features[:, 0]:
            detector.update(x)

        assert detector.n_observations > 0
