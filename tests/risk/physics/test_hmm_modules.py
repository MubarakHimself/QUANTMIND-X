"""
Tests for HMM modular structure.
Verifies backward compatibility and modular split.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestHMMModuleStructure:
    """Test that the modular HMM structure maintains backward compatibility."""

    def test_import_from_hmm_package(self):
        """Test that imports from hmm package work."""
        # Test new modular imports
        from src.risk.physics.hmm import (
            HMMFeatureExtractor,
            FeatureConfig,
            prepare_training_data,
            load_config_from_file
        )

        assert HMMFeatureExtractor is not None
        assert FeatureConfig is not None
        assert prepare_training_data is not None
        assert load_config_from_file is not None

    def test_import_from_submodules(self):
        """Test that imports from submodules work."""
        from src.risk.physics.hmm.features import FeatureConfig
        from src.risk.physics.hmm.indicators import TechnicalIndicators
        from src.risk.physics.hmm.scaler import FeatureScaler
        from src.risk.physics.hmm.models import HMMFeatureExtractor
        from src.risk.physics.hmm.utils import prepare_training_data, load_config_from_file

        assert FeatureConfig is not None
        assert TechnicalIndicators is not None
        assert FeatureScaler is not None
        assert HMMFeatureExtractor is not None

    def test_backward_compatibility_import(self):
        """Test that old import path still works."""
        # This is the critical backward compatibility test
        from src.risk.physics.hmm_features import (
            HMMFeatureExtractor,
            FeatureConfig,
            prepare_training_data,
            load_config_from_file
        )

        assert HMMFeatureExtractor is not None
        assert FeatureConfig is not None


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        from src.risk.physics.hmm import FeatureConfig

        config = FeatureConfig()

        assert config.include_magnetization is True
        assert config.include_susceptibility is True
        assert config.include_energy is True
        assert config.include_temperature is True
        assert config.include_log_returns is True
        assert config.rsi_period == 14
        assert config.scaling_method == "standard"

    def test_custom_config(self):
        """Test custom configuration."""
        from src.risk.physics.hmm import FeatureConfig

        config = FeatureConfig(
            rsi_period=21,
            scaling_method="minmax",
            clip_outliers=False
        )

        assert config.rsi_period == 21
        assert config.scaling_method == "minmax"
        assert config.clip_outliers is False


class TestHMMFeatureExtractor:
    """Test HMMFeatureExtractor class."""

    def test_extractor_initialization(self):
        """Test feature extractor initialization."""
        from src.risk.physics.hmm import HMMFeatureExtractor, FeatureConfig

        config = FeatureConfig()
        extractor = HMMFeatureExtractor(config)

        assert extractor.config == config
        assert extractor._scaler is None

    def test_extract_ising_features(self):
        """Test Ising feature extraction."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        extractor = HMMFeatureExtractor()
        features = extractor.extract_ising_features(volatility=20.0)

        assert 'magnetization' in features
        assert 'susceptibility' in features
        assert 'energy' in features
        assert 'temperature' in features

    def test_extract_price_features(self):
        """Test price feature extraction."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        # Create sample price data
        prices = pd.Series([100 + i * 0.5 for i in range(60)])

        extractor = HMMFeatureExtractor()
        features = extractor.extract_price_features(prices)

        assert 'log_returns' in features
        assert 'rolling_volatility_20' in features
        assert 'price_momentum_10' in features

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        prices = pd.Series([100 + i for i in range(30)])

        extractor = HMMFeatureExtractor()
        rsi = extractor.calculate_rsi(prices, period=14)

        assert 0 <= rsi <= 100

    def test_calculate_atr(self):
        """Test ATR calculation."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(30)))
        high = close + np.random.rand(30) * 2
        low = close - np.random.rand(30) * 2

        extractor = HMMFeatureExtractor()
        atr = extractor.calculate_atr(high, low, close, period=14)

        assert atr >= 0

    def test_calculate_macd(self):
        """Test MACD calculation."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        prices = pd.Series([100 + i * 0.5 for i in range(100)])

        extractor = HMMFeatureExtractor()
        macd, signal, hist = extractor.calculate_macd(prices)

        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)

    def test_scale_features(self):
        """Test feature scaling."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        features = np.random.randn(100, 10)

        extractor = HMMFeatureExtractor()
        scaled = extractor.scale_features(features, fit=True)

        assert scaled.shape == features.shape
        assert not np.any(np.isnan(scaled))

    def test_get_feature_names(self):
        """Test feature names retrieval."""
        from src.risk.physics.hmm import HMMFeatureExtractor

        extractor = HMMFeatureExtractor()
        names = extractor.get_feature_names()

        assert isinstance(names, list)


class TestTechnicalIndicators:
    """Test technical indicators module."""

    def test_indicators_class_exists(self):
        """Test TechnicalIndicators class exists."""
        from src.risk.physics.hmm.indicators import TechnicalIndicators

        indicators = TechnicalIndicators()

        assert indicators is not None

    def test_rsi_calculation(self):
        """Test RSI via indicators class."""
        from src.risk.physics.hmm.indicators import TechnicalIndicators

        prices = pd.Series([100 + i for i in range(30)])

        indicators = TechnicalIndicators()
        rsi = indicators.rsi(prices, period=14)

        assert 0 <= rsi <= 100

    def test_atr_calculation(self):
        """Test ATR via indicators class."""
        from src.risk.physics.hmm.indicators import TechnicalIndicators

        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(30)))
        high = close + np.random.rand(30) * 2
        low = close - np.random.rand(30) * 2

        indicators = TechnicalIndicators()
        atr = indicators.atr(high, low, close, period=14)

        assert atr >= 0

    def test_macd_calculation(self):
        """Test MACD via indicators class."""
        from src.risk.physics.hmm.indicators import TechnicalIndicators

        prices = pd.Series([100 + i * 0.5 for i in range(100)])

        indicators = TechnicalIndicators()
        macd, signal, hist = indicators.macd(prices)

        assert isinstance(macd, float)


class TestFeatureScaler:
    """Test feature scaler module."""

    def test_scaler_class_exists(self):
        """Test FeatureScaler class exists."""
        from src.risk.physics.hmm.scaler import FeatureScaler

        scaler = FeatureScaler(method="standard")

        assert scaler is not None

    def test_standard_scaling(self):
        """Test standard scaling."""
        from src.risk.physics.hmm.scaler import FeatureScaler

        features = np.random.randn(100, 5)

        scaler = FeatureScaler(method="standard")
        scaled = scaler.fit_transform(features)

        # Standardized features should have mean ~0 and std ~1
        assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled.std(axis=0), 1, atol=1e-10)

    def test_minmax_scaling(self):
        """Test minmax scaling."""
        from src.risk.physics.hmm.scaler import FeatureScaler

        features = np.random.randn(100, 5) * 10 + 5

        scaler = FeatureScaler(method="minmax")
        scaled = scaler.fit_transform(features)

        # Minmax scaled features should be in [0, 1]
        assert np.all(scaled >= 0)
        assert np.all(scaled <= 1)


class TestPrepareTrainingData:
    """Test prepare_training_data function."""

    def test_prepare_training_data_basic(self):
        """Test basic training data preparation."""
        from src.risk.physics.hmm import prepare_training_data

        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        close = 100 + np.cumsum(np.random.randn(100) * 2)

        ohlcv = pd.DataFrame({
            'open': close - np.random.rand(100),
            'high': close + np.random.rand(100),
            'low': close - np.random.rand(100),
            'close': close,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

        data = {'BTCUSDT_1H': ohlcv}

        features, keys, extractor = prepare_training_data(data)

        assert features is not None
        assert len(keys) > 0
        assert extractor is not None
