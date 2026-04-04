"""
HMM Regime Sensor

A Hidden Markov Model-based regime detection system that works alongside
the Ising Model for market phase detection.

This sensor loads trained HMM models and provides regime predictions
with confidence scores and state probabilities.

Features:
- Load trained HMM models from disk
- Predict regime state from features
- Map HMM states to regime labels
- Cache predictions for performance
- Integration with Sentinel system

Reference: docs/architecture/components.md
"""

import os
import json
import pickle
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .config import HMMSensorConfig
from .models import HMMRegimeReading
from .base import BaseRegimeSensor

logger = logging.getLogger(__name__)

# Check for hmmlearn availability
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logging.warning("hmmlearn not installed. HMM sensor will not function.")


class HMMRegimeSensor(BaseRegimeSensor):
    """
    HMM-based regime detector for market phase detection.

    Loads trained HMM models and provides regime predictions based on
    extracted features from market data and Ising outputs.

    Usage:
        ```python
        sensor = HMMRegimeSensor()
        reading = sensor.predict_regime(features)
        print(f"Regime: {reading.regime}, Confidence: {reading.confidence}")
        ```
    """

    def __init__(self, config: Optional[HMMSensorConfig] = None,
                 config_path: str = os.environ.get('HMM_CONFIG_PATH', 'config/hmm_config.json')):
        """
        Initialize HMM sensor with configuration.

        Args:
            config: Sensor configuration
            config_path: Path to HMM config file
        """
        self.config = config or HMMSensorConfig()
        self._load_config_file(config_path)

        # Initialize feature extractor
        self._init_feature_extractor()

        # Model state
        self._model = None
        self._model_metadata = {}
        self._model_version = None
        self._model_checksum = None
        self._last_load_time = None

        # Cache
        self._prediction_cache: Dict[str, HMMRegimeReading] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Load model on initialization
        self._load_model()

    def _init_feature_extractor(self) -> None:
        """Initialize feature extractor."""
        try:
            from src.risk.physics.hmm_features import HMMFeatureExtractor
            self.feature_extractor = HMMFeatureExtractor()
        except ImportError:
            logger.warning("HMMFeatureExtractor not available. Using stub.")
            self.feature_extractor = None

    def _load_config_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Update regime mapping from config
            if 'regime_mapping' in config_data:
                self.config.regime_mapping = {
                    int(k): v for k, v in config_data['regime_mapping'].items()
                }
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

    def _load_model(self) -> None:
        """Load HMM model from disk or database."""
        if not HMMLEARN_AVAILABLE:
            logger.error("hmmlearn not available. Cannot load model.")
            return

        model_path = Path(self.config.model_path)

        # Find model file
        if self.config.model_version:
            # Load specific version
            model_file = self._find_model_by_version(model_path, self.config.model_version)
        else:
            # Load latest model
            model_file = self._find_latest_model(model_path)

        if not model_file:
            logger.warning("No HMM model found. Sensor will return default readings.")
            return

        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            self._model = model_data['model']
            self._model_metadata = {
                'scaler': model_data.get('scaler', {}),
                'feature_names': model_data.get('feature_names', []),
                'metrics': model_data.get('metrics', {}),
                'config': model_data.get('config', {})
            }

            # Set feature extractor scaler
            if self._model_metadata['scaler'] and self.feature_extractor:
                try:
                    self.feature_extractor.set_scaler_params(self._model_metadata['scaler'])
                except Exception as e:
                    logger.warning(f"Could not set scaler params: {e}")

            # Calculate checksum
            self._model_checksum = self._calculate_checksum(model_file)

            # Get version from metadata file
            metadata_file = model_file.with_suffix('.pkl.metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self._model_version = metadata.get('version', 'unknown')
            else:
                self._model_version = model_file.stem

            self._last_load_time = datetime.now(timezone.utc)

            logger.info(f"Loaded HMM model: {model_file} (v{self._model_version})")

        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            self._model = None

    def _find_latest_model(self, model_path: Path) -> Optional[Path]:
        """Find the latest model file in directory."""
        if not model_path.exists():
            return None

        # Look for model files
        model_files = list(model_path.glob("*.pkl"))

        if not model_files:
            return None

        # Sort by modification time
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return model_files[0]

    def _find_model_by_version(self, model_path: Path, version: str) -> Optional[Path]:
        """Find model file by version string."""
        if not model_path.exists():
            return None

        # Look for exact version match
        for model_file in model_path.glob("*.pkl"):
            if version in model_file.stem:
                return model_file

        return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of model file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def reload_model(self) -> bool:
        """
        Reload model from disk.

        Returns:
            True if model was reloaded successfully
        """
        logger.info("Reloading HMM model...")
        self._model = None
        self._load_model()
        return self._model is not None

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_model_info(self) -> Dict:
        """
        Get information about loaded model including metrics and training details.

        Returns:
            Dictionary with model information including:
            - Basic info: loaded, version, checksum, n_states, n_features
            - Metrics: log_likelihood, state_distribution, transition_matrix
            - Training info: training_samples, validation_status
            - Cache stats: cache_hits, cache_misses
        """
        # Get metrics from model metadata
        metrics = self._model_metadata.get('metrics', {})

        # Get state distribution from model if available
        state_distribution = metrics.get('state_distribution')
        if state_distribution is None and self._model:
            state_distribution = self.get_state_distribution()

        # Get transition matrix from model if available
        transition_matrix = metrics.get('transition_matrix')
        if transition_matrix is None and self._model:
            try:
                transition_matrix = self._model.transmat_.tolist()
            except Exception:
                transition_matrix = None

        return {
            'loaded': self._model is not None,
            'version': self._model_version,
            'checksum': self._model_checksum[:16] + '...' if self._model_checksum else None,
            'n_states': self._model.n_components if self._model else None,
            'n_features': len(self._model_metadata.get('feature_names', [])),
            'last_load_time': self._last_load_time.isoformat() if self._last_load_time else None,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            # Metrics from training
            'metrics': {
                'log_likelihood': metrics.get('log_likelihood'),
                'state_distribution': state_distribution,
                'transition_matrix': transition_matrix,
                'training_samples': metrics.get('training_samples', 0),
                'validation_status': metrics.get('validation_status', 'unknown'),
                'training_date': metrics.get('training_date'),
                'feature_names': self._model_metadata.get('feature_names', [])
            }
        }

    def predict_regime(self, features: np.ndarray,
                       cache_key: Optional[str] = None) -> HMMRegimeReading:
        """
        Predict regime from feature array.

        Args:
            features: 1D or 2D feature array
            cache_key: Optional key for caching prediction

        Returns:
            HMMRegimeReading with prediction results
        """
        # Check cache
        if cache_key and cache_key in self._prediction_cache:
            self._cache_hits += 1
            return self._prediction_cache[cache_key]

        self._cache_misses += 1

        # Default reading if no model
        if not self._model:
            return HMMRegimeReading(
                state=0,
                regime="UNKNOWN",
                confidence=0.0,
                state_probabilities={},
                next_state_probabilities={},
                timestamp=datetime.now(timezone.utc),
                model_version="none",
                features_used=[]
            )

        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        scaled_features = features
        if self.feature_extractor:
            try:
                scaled_features = self.feature_extractor.scale_features(features, fit=False)
            except Exception:
                pass

        # Predict state
        state_seq = self._model.predict(scaled_features)
        state = int(state_seq[-1])

        # Get state probabilities
        state_probs = self._model.predict_proba(scaled_features)[-1]
        state_probabilities = {i: float(p) for i, p in enumerate(state_probs)}

        # Calculate confidence (probability of predicted state)
        confidence = float(state_probs[state])

        # Get next state probabilities from transition matrix
        transmat = self._model.transmat_
        next_state_probs = transmat[state]
        next_state_probabilities = {i: float(p) for i, p in enumerate(next_state_probs)}

        # Map state to regime
        regime = self.config.regime_mapping.get(state, f"STATE_{state}")

        # Create reading
        reading = HMMRegimeReading(
            state=state,
            regime=regime,
            confidence=confidence,
            state_probabilities=state_probabilities,
            next_state_probabilities=next_state_probabilities,
            timestamp=datetime.now(timezone.utc),
            model_version=self._model_version or "unknown",
            features_used=self._model_metadata.get('feature_names', [])
        )

        # Cache result
        if cache_key:
            self._prediction_cache[cache_key] = reading

            # Limit cache size
            if len(self._prediction_cache) > self.config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._prediction_cache))
                del self._prediction_cache[oldest_key]

        return reading

    def predict_from_ohlcv(self, ohlcv_data: Dict,
                           volatility: Optional[float] = None) -> HMMRegimeReading:
        """
        Predict regime from OHLCV data.

        Args:
            ohlcv_data: Dictionary with 'open', 'high', 'low', 'close', 'volume'
            volatility: Optional pre-calculated volatility

        Returns:
            HMMRegimeReading with prediction results
        """
        # Extract features
        import pandas as pd
        df = pd.DataFrame(ohlcv_data)
        features = df['close'].values.reshape(-1, 1)  # Simplified

        # Generate cache key from latest close price and timestamp
        cache_key = None
        if 'close' in df.columns and len(df) > 0:
            cache_key = f"{df['close'].iloc[-1]}_{len(df)}"

        return self.predict_regime(features, cache_key)

    def get_reading(self) -> float:
        """
        Get HMM regime reading as a float value for position sizing.

        Returns a normalized regime value (0.0 to 1.0) where:
        - 0.0 = Trending low vol (favorable for trend following)
        - 0.33 = Trending high vol
        - 0.67 = Ranging low vol
        - 1.0 = Ranging high vol (unfavorable for most strategies)

        Returns:
            float: Normalized regime value
        """
        if not self._model:
            return 0.5  # Neutral reading

        # Return cached reading if available
        if self._prediction_cache:
            latest_reading = list(self._prediction_cache.values())[-1]
            return self._regime_to_float(latest_reading.regime)

        return 0.5  # Neutral reading

    def _regime_to_float(self, regime: str) -> float:
        """Convert regime string to float value."""
        regime_values = {
            "TRENDING_LOW_VOL": 0.0,
            "TRENDING_HIGH_VOL": 0.33,
            "RANGING_LOW_VOL": 0.67,
            "RANGING_HIGH_VOL": 1.0
        }
        return regime_values.get(regime, 0.5)

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared HMM prediction cache")

    def get_regime_confidence(self) -> float:
        """Get confidence of current regime prediction."""
        if self._prediction_cache:
            latest_reading = list(self._prediction_cache.values())[-1]
            return latest_reading.confidence
        return 0.0

    def get_state_distribution(self) -> Dict[int, float]:
        """Get expected state distribution from model."""
        if not self._model:
            return {}

        # Get stationary distribution (if available)
        try:
            # Approximate stationary distribution
            transmat = self._model.transmat_
            n = transmat.shape[0]

            # Power iteration to find stationary distribution
            dist = np.ones(n) / n
            for _ in range(100):
                dist = dist @ transmat

            return {i: float(p) for i, p in enumerate(dist)}
        except Exception:
            return {}

    def compare_with_ising(self, ising_regime: str,
                          features: np.ndarray) -> Dict:
        """
        Compare HMM prediction with Ising Model prediction.

        Args:
            ising_regime: Regime from Ising Model
            features: Feature array for HMM prediction

        Returns:
            Comparison dictionary
        """
        hmm_reading = self.predict_regime(features)

        # Map regimes to comparable categories
        hmm_category = self._map_to_category(hmm_reading.regime)
        ising_category = self._map_to_category(ising_regime)

        return {
            'hmm_regime': hmm_reading.regime,
            'hmm_state': hmm_reading.state,
            'hmm_confidence': hmm_reading.confidence,
            'ising_regime': ising_regime,
            'hmm_category': hmm_category,
            'ising_category': ising_category,
            'agreement': hmm_category == ising_category,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _map_to_category(self, regime: str) -> str:
        """Map regime to comparable category."""
        regime_lower = regime.lower()

        if 'trending' in regime_lower:
            return 'TRENDING'
        elif 'ranging' in regime_lower:
            return 'RANGING'
        elif 'chaotic' in regime_lower or 'transitional' in regime_lower:
            return 'TRANSITIONAL'
        elif 'ordered' in regime_lower:
            return 'TRENDING'
        else:
            return 'UNKNOWN'


def create_hmm_sensor(config_path: str = os.environ.get('HMM_CONFIG_PATH', 'config/hmm_config.json')) -> HMMRegimeSensor:
    """
    Factory function to create HMM sensor.

    Args:
        config_path: Path to config file

    Returns:
        Configured HMMRegimeSensor instance
    """
    return HMMRegimeSensor(config_path=config_path)
