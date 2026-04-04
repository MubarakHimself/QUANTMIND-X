"""
HMM Models Module
=================

Provides the HMMFeatureExtractor class for extracting features from market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

from .features import FeatureConfig
from .indicators import TechnicalIndicators
from .scaler import FeatureScaler
from ..ising_sensor import IsingRegimeSensor, IsingSensorConfig

logger = logging.getLogger(__name__)


class HMMFeatureExtractor:
    """
    Extract features for HMM training from market data and Ising outputs.

    Features are extracted in the following order:
    1. Ising Model outputs (magnetization, susceptibility, energy, temperature)
    2. Price-based features (log returns, volatility, momentum)
    3. Technical indicators (RSI, ATR, MACD)

    All features are normalized using the configured scaling method.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature extractor with configuration."""
        self.config = config or FeatureConfig()
        self.ising_sensor = IsingRegimeSensor(IsingSensorConfig())
        self._scaler: Optional[FeatureScaler] = None
        self._feature_names: List[str] = []
        self._scaler_params: Dict[str, np.ndarray] = {}

    def extract_ising_features(self, volatility: float) -> Dict[str, float]:
        """
        Extract Ising Model outputs as features.

        Args:
            volatility: Current market volatility for Ising simulation

        Returns:
            Dictionary of Ising features
        """
        features = {}

        # Run Ising simulation
        result = self.ising_sensor.detect_regime(market_volatility=volatility)

        if self.config.include_magnetization:
            features['magnetization'] = result.get('magnetization', 0.0)

        if self.config.include_susceptibility:
            features['susceptibility'] = result.get('susceptibility', 0.0)

        if self.config.include_energy:
            # Calculate energy from magnetization (approximation)
            mag = result.get('magnetization', 0.0)
            features['energy'] = -0.5 * mag ** 2

        if self.config.include_temperature:
            features['temperature'] = result.get('temperature', 1.0)

        return features

    def extract_from_ising(self, ising_output: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from Ising Model output dictionary.

        This is a convenience method for converting Ising outputs directly
        to a numpy array for HMM prediction.

        Args:
            ising_output: Dictionary with Ising model outputs
                         (magnetization, susceptibility, energy, regime, etc.)

        Returns:
            Numpy array of features
        """
        features = []

        # Extract Ising features from the output dict
        if self.config.include_magnetization:
            features.append(ising_output.get('magnetization', 0.0))

        if self.config.include_susceptibility:
            features.append(ising_output.get('susceptibility', 0.0))

        if self.config.include_energy:
            features.append(ising_output.get('energy', -0.5 * ising_output.get('magnetization', 0.0) ** 2))

        if self.config.include_temperature:
            features.append(ising_output.get('temperature', 1.0))

        # Add placeholder values for other expected features
        # (price features, technical indicators)
        # These would normally come from market data
        features.extend([0.0] * 6)  # log_returns, vol_20, vol_50, momentum, rsi, atr

        self._feature_names = ['magnetization', 'susceptibility', 'energy', 'temperature',
                               'log_returns', 'rolling_volatility_20', 'rolling_volatility_50',
                               'price_momentum_10', 'rsi', 'atr_normalized']

        return np.array(features)

    def extract_price_features(self, prices: pd.Series) -> Dict[str, float]:
        """
        Extract price-based features from OHLCV data.

        Args:
            prices: Series of close prices

        Returns:
            Dictionary of price features
        """
        features = {}

        if self.config.include_log_returns:
            # Log returns
            log_returns = np.log(prices / prices.shift(1))
            features['log_returns'] = log_returns.iloc[-1] if len(log_returns) > 0 else 0.0

        if self.config.include_rolling_volatility_20:
            # 20-period rolling volatility
            returns = prices.pct_change()
            vol_20 = returns.rolling(window=20).std()
            features['rolling_volatility_20'] = vol_20.iloc[-1] if len(vol_20) > 0 else 0.0

        if self.config.include_rolling_volatility_50:
            # 50-period rolling volatility
            vol_50 = returns.rolling(window=50).std()
            features['rolling_volatility_50'] = vol_50.iloc[-1] if len(vol_50) > 0 else 0.0

        if self.config.include_price_momentum_10:
            # 10-period price momentum
            momentum = (prices - prices.shift(10)) / prices.shift(10)
            features['price_momentum_10'] = momentum.iloc[-1] if len(momentum) > 0 else 0.0

        return features

    def extract_from_prices(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Extract features from price DataFrame.

        This is a convenience method for converting price data directly
        to a numpy array for HMM prediction.

        Args:
            prices: DataFrame with 'close', 'high', 'low' columns (and optionally 'volume')

        Returns:
            Numpy array of features
        """
        features = []

        # Extract price-based features
        close = prices['close'] if 'close' in prices.columns else pd.Series([100.0])

        # Log returns
        if self.config.include_log_returns and len(close) > 1:
            log_returns = np.log(close / close.shift(1))
            features.append(log_returns.iloc[-1] if not pd.isna(log_returns.iloc[-1]) else 0.0)
        else:
            features.append(0.0)

        # Rolling volatility 20
        if self.config.include_rolling_volatility_20 and len(close) > 20:
            returns = close.pct_change()
            vol_20 = returns.rolling(window=20).std()
            features.append(vol_20.iloc[-1] if not pd.isna(vol_20.iloc[-1]) else 0.0)
        else:
            features.append(0.0)

        # Rolling volatility 50
        if self.config.include_rolling_volatility_50 and len(close) > 50:
            vol_50 = returns.rolling(window=50).std()
            features.append(vol_50.iloc[-1] if not pd.isna(vol_50.iloc[-1]) else 0.0)
        else:
            features.append(0.0)

        # Price momentum 10
        if self.config.include_price_momentum_10 and len(close) > 10:
            momentum = (close - close.shift(10)) / close.shift(10)
            features.append(momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0.0)
        else:
            features.append(0.0)

        # RSI
        if len(close) > self.config.rsi_period + 1:
            features.append(TechnicalIndicators.rsi(close, self.config.rsi_period))
        else:
            features.append(50.0)  # Neutral RSI

        # ATR
        if 'high' in prices.columns and 'low' in prices.columns and len(close) > self.config.atr_period + 1:
            atr = TechnicalIndicators.atr(prices['high'], prices['low'], close, self.config.atr_period)
            features.append(atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.0)
        else:
            features.append(0.0)

        # Add placeholder values for Ising features
        # These would normally come from Ising model simulation
        features.extend([0.5, 0.0, -0.125, 1.0])  # mag, sus, energy, temp

        self._feature_names = ['log_returns', 'rolling_volatility_20', 'rolling_volatility_50',
                               'price_momentum_10', 'rsi', 'atr_normalized',
                               'magnetization', 'susceptibility', 'energy', 'temperature']

        return np.array(features)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        return TechnicalIndicators.rsi(prices, period)

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> float:
        """Calculate Average True Range."""
        return TechnicalIndicators.atr(high, low, close, period)

    def calculate_macd(self, prices: pd.Series, fast: int = 12,
                      slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        return TechnicalIndicators.macd(prices, fast, slow, signal)

    def extract_technical_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """
        Extract technical indicator features.

        Args:
            ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

        Returns:
            Dictionary of technical features
        """
        features = {}

        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']

        # RSI
        features['rsi'] = TechnicalIndicators.rsi(close, self.config.rsi_period)

        # ATR (normalized by price)
        atr = TechnicalIndicators.atr(high, low, close, self.config.atr_period)
        features['atr_normalized'] = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.0

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(
            close,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist

        return features

    def extract_all_features(self, ohlcv: pd.DataFrame,
                             volatility: Optional[float] = None) -> np.ndarray:
        """
        Extract all features for a single data point.

        Args:
            ohlcv: DataFrame with OHLCV data
            volatility: Optional pre-calculated volatility

        Returns:
            Numpy array of features
        """
        features = {}

        # Calculate volatility if not provided
        if volatility is None:
            close = ohlcv['close']
            returns = close.pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility %

        # Extract Ising features
        ising_features = self.extract_ising_features(volatility)
        features.update(ising_features)

        # Extract price features
        price_features = self.extract_price_features(ohlcv['close'])
        features.update(price_features)

        # Extract technical features
        tech_features = self.extract_technical_features(ohlcv)
        features.update(tech_features)

        # Store feature names
        self._feature_names = list(features.keys())

        return np.array(list(features.values()))

    def extract_features_batch(self, ohlcv_data: pd.DataFrame,
                                symbol: str, timeframe: str) -> np.ndarray:
        """
        Extract features for all data points in a DataFrame.

        Args:
            ohlcv_data: DataFrame with OHLCV data and timestamp index
            symbol: Trading symbol
            timeframe: Timeframe string

        Returns:
            2D Numpy array of features (n_samples, n_features)
        """
        all_features = []

        # Calculate rolling volatility for each point
        close = ohlcv_data['close']
        returns = close.pct_change()
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100

        # Extract features for each row
        for i in range(len(ohlcv_data)):
            if i < 50:  # Skip first 50 rows for warmup
                continue

            # Get window of data up to current point
            window = ohlcv_data.iloc[:i+1]
            vol = rolling_vol.iloc[i] if not pd.isna(rolling_vol.iloc[i]) else 1.0

            features = self.extract_all_features(window, vol)
            all_features.append(features)

        return np.array(all_features)

    def scale_features(self, features: np.ndarray,
                       fit: bool = True) -> np.ndarray:
        """
        Scale features using configured method.

        Args:
            features: 2D array of features (n_samples, n_features)
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            Scaled features array
        """
        if self.config.scaling_method == "standard":
            if fit:
                self._scaler_params = {
                    'mean': np.nanmean(features, axis=0),
                    'std': np.nanstd(features, axis=0)
                }

            # Apply scaling — guard against empty scaler_params (e.g., first inference before any fit)
            if not self._scaler_params:
                return features
            scaled = (features - self._scaler_params['mean']) / (self._scaler_params['std'] + 1e-8)

        elif self.config.scaling_method == "minmax":
            if fit:
                self._scaler_params = {
                    'min': np.nanmin(features, axis=0),
                    'max': np.nanmax(features, axis=0)
                }

            # Apply scaling — guard against empty scaler_params
            if not self._scaler_params:
                return features
            scaled = (features - self._scaler_params['min']) / (self._scaler_params['max'] - self._scaler_params['min'] + 1e-8)

        else:  # robust
            if fit:
                self._scaler_params = {
                    'median': np.nanmedian(features, axis=0),
                    'iqr': np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
                }

            # Apply scaling — guard against empty scaler_params
            if not self._scaler_params:
                return features
            scaled = (features - self._scaler_params['median']) / (self._scaler_params['iqr'] + 1e-8)

        # Clip outliers
        if self.config.clip_outliers:
            scaled = np.clip(scaled, -self.config.clip_threshold, self.config.clip_threshold)

        # Replace NaN with 0
        scaled = np.nan_to_num(scaled, nan=0.0)

        return scaled

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names.copy()

    def get_scaler_params(self) -> Dict[str, np.ndarray]:
        """Get scaler parameters for serialization."""
        return self._scaler_params.copy() if self._scaler_params else {}

    def set_scaler_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set scaler parameters from serialized data."""
        self._scaler_params = params.copy()
