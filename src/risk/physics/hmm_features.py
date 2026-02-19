"""
HMM Feature Extraction Module
==============================

Extracts features from Ising Model outputs and market data for HMM training.
Features are used to train Hidden Markov Models for regime detection.

Features extracted:
- Ising outputs: magnetization, susceptibility, energy, temperature
- Price features: log returns, rolling volatility, momentum
- Technical indicators: RSI, ATR, MACD

Reference: docs/architecture/components.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

# Import Ising sensor for feature extraction
from .ising_sensor import IsingRegimeSensor, IsingSensorConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Ising features
    include_magnetization: bool = True
    include_susceptibility: bool = True
    include_energy: bool = True
    include_temperature: bool = True
    
    # Price features
    include_log_returns: bool = True
    include_rolling_volatility_20: bool = True
    include_rolling_volatility_50: bool = True
    include_price_momentum_10: bool = True
    
    # Technical indicators
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Scaling
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust'
    clip_outliers: bool = True
    clip_threshold: float = 3.0


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
        self._scaler = None
        self._feature_names: List[str] = []
        
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
            features.append(self.calculate_rsi(close, self.config.rsi_period))
        else:
            features.append(50.0)  # Neutral RSI
        
        # ATR
        if 'high' in prices.columns and 'low' in prices.columns and len(close) > self.config.atr_period + 1:
            atr = self.calculate_atr(prices['high'], prices['low'], close, self.config.atr_period)
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
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of close prices
            period: RSI period
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(close) < period + 1:
            return 0.0
            
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD indicator.
        
        Args:
            prices: Series of close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow + signal:
            return (0.0, 0.0, 0.0)
            
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return (
            macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0,
            signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0.0,
            histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
        )
    
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
        features['rsi'] = self.calculate_rsi(close, self.config.rsi_period)
        
        # ATR (normalized by price)
        atr = self.calculate_atr(high, low, close, self.config.atr_period)
        features['atr_normalized'] = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.0
        
        # MACD
        macd, signal, hist = self.calculate_macd(
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
                self._scaler = {
                    'mean': np.nanmean(features, axis=0),
                    'std': np.nanstd(features, axis=0)
                }
            
            # Apply scaling
            scaled = (features - self._scaler['mean']) / (self._scaler['std'] + 1e-8)
            
        elif self.config.scaling_method == "minmax":
            if fit:
                self._scaler = {
                    'min': np.nanmin(features, axis=0),
                    'max': np.nanmax(features, axis=0)
                }
            
            # Apply scaling
            scaled = (features - self._scaler['min']) / (self._scaler['max'] - self._scaler['min'] + 1e-8)
            
        else:  # robust
            if fit:
                self._scaler = {
                    'median': np.nanmedian(features, axis=0),
                    'iqr': np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
                }
            
            # Apply scaling
            scaled = (features - self._scaler['median']) / (self._scaler['iqr'] + 1e-8)
        
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
        return self._scaler.copy() if self._scaler else {}
    
    def set_scaler_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set scaler parameters from serialized data."""
        self._scaler = params.copy()
    
    def save_scaler(self, path: Path) -> None:
        """Save scaler parameters to file."""
        if self._scaler is None:
            logger.warning("No scaler to save")
            return
            
        scaler_data = {
            'method': self.config.scaling_method,
            'params': {k: v.tolist() for k, v in self._scaler.items()},
            'feature_names': self._feature_names
        }
        
        with open(path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        logger.info(f"Saved scaler to {path}")
    
    def load_scaler(self, path: Path) -> None:
        """Load scaler parameters from file."""
        with open(path, 'r') as f:
            scaler_data = json.load(f)
        
        self._scaler = {k: np.array(v) for k, v in scaler_data['params'].items()}
        self._feature_names = scaler_data.get('feature_names', [])
        
        logger.info(f"Loaded scaler from {path}")


def prepare_training_data(ohlcv_data: Dict[str, pd.DataFrame],
                          config: Optional[FeatureConfig] = None,
                          symbol: Optional[str] = None,
                          timeframe: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare training data for HMM from multiple symbols/timeframes.
    
    Args:
        ohlcv_data: Dictionary mapping symbol_timeframe to DataFrame
        config: Feature extraction configuration
        symbol: Optional specific symbol to filter
        timeframe: Optional specific timeframe to filter
        
    Returns:
        Tuple of (features array, list of keys)
    """
    extractor = HMMFeatureExtractor(config)
    all_features = []
    keys = []
    
    for key, df in ohlcv_data.items():
        # Parse key (format: SYMBOL_TIMEFRAME)
        parts = key.split('_')
        sym = parts[0]
        tf = '_'.join(parts[1:])  # Handle timeframes like M5, H1, H4
        
        # Filter by symbol/timeframe if specified
        if symbol and sym != symbol:
            continue
        if timeframe and tf != timeframe:
            continue
        
        # Extract features
        features = extractor.extract_features_batch(df, sym, tf)
        all_features.append(features)
        keys.append(key)
    
    # Combine all features
    combined_features = np.vstack(all_features)
    
    # Scale features
    scaled_features = extractor.scale_features(combined_features, fit=True)
    
    return scaled_features, keys, extractor


def load_config_from_file(config_path: str) -> FeatureConfig:
    """Load feature configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    feature_config = config_data.get('features', {})
    
    return FeatureConfig(
        include_magnetization=feature_config.get('ising_outputs', {}).get('magnetization', True),
        include_susceptibility=feature_config.get('ising_outputs', {}).get('susceptibility', True),
        include_energy=feature_config.get('ising_outputs', {}).get('energy', True),
        include_temperature=feature_config.get('ising_outputs', {}).get('temperature', True),
        include_log_returns=feature_config.get('price_features', {}).get('log_returns', True),
        include_rolling_volatility_20=feature_config.get('price_features', {}).get('rolling_volatility_20', True),
        include_rolling_volatility_50=feature_config.get('price_features', {}).get('rolling_volatility_50', True),
        include_price_momentum_10=feature_config.get('price_features', {}).get('price_momentum_10', True),
        rsi_period=feature_config.get('technical_indicators', {}).get('rsi_period', 14),
        atr_period=feature_config.get('technical_indicators', {}).get('atr_period', 14),
        macd_fast=feature_config.get('technical_indicators', {}).get('macd_fast', 12),
        macd_slow=feature_config.get('technical_indicators', {}).get('macd_slow', 26),
        macd_signal=feature_config.get('technical_indicators', {}).get('macd_signal', 9),
        scaling_method=feature_config.get('scaling', {}).get('method', 'standard'),
        clip_outliers=feature_config.get('scaling', {}).get('clip_outliers', True),
        clip_threshold=feature_config.get('scaling', {}).get('clip_threshold', 3.0)
    )