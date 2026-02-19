#!/usr/bin/env python3
"""
HMM Training Script
===================

Trains Hidden Markov Models for regime detection using features extracted
from Ising Model outputs and market data.

Hierarchical Training Strategy:
- Level 1 (Universal): Train on all symbols/timeframes combined (50k+ samples)
- Level 2 (Per-Symbol): Fine-tune Universal model for EURUSD, GBPUSD, XAUUSD
- Level 3 (Per-Symbol-Timeframe): Train symbol-specific models for M5, H1, H4

Usage:
    python scripts/train_hmm.py --level universal
    python scripts/train_hmm.py --level per-symbol --symbol EURUSD
    python scripts/train_hmm.py --level per-symbol-timeframe --symbol EURUSD --timeframe H1

Reference: docs/architecture/components.md
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("WARNING: hmmlearn not installed. Install with: pip install hmmlearn")

from src.risk.physics.hmm_features import (
    HMMFeatureExtractor, 
    FeatureConfig,
    prepare_training_data,
    load_config_from_file
)
from src.database.models import HMMModel, HMMSyncStatus
from src.database.engine import engine
from src.database.duckdb_connection import DuckDBConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HMMTrainer:
    """
    HMM Model Trainer for regime detection.
    
    Implements hierarchical training strategy with validation and
    database persistence.
    """
    
    def __init__(self, config_path: str = "config/hmm_config.json"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})
        self.validation_config = self.config.get('validation', {})
        
        # Initialize feature extractor
        self.feature_config = load_config_from_file(config_path)
        self.feature_extractor = HMMFeatureExtractor(self.feature_config)
        
        # Database session
        self.Session = sessionmaker(bind=engine)
        
        # Training state
        self.current_model = None
        self.training_start_time = None
        self.checkpoint_interval = self.training_config.get('checkpoint_interval', 10)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _create_model(self, n_states: int = 4) -> 'hmm.GaussianHMM':
        """Create a new HMM model with configured parameters."""
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required for HMM training")
        
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=self.model_config.get('covariance_type', 'full'),
            n_iter=self.model_config.get('n_iter', 100),
            tol=self.model_config.get('tol', 0.01),
            random_state=self.model_config.get('random_state', 42),
            init_params=self.model_config.get('init_params', 'stmc'),
            params=self.model_config.get('params', 'stmc'),
            verbose=True
        )
        
        return model
    
    def load_training_data(self, symbol: Optional[str] = None,
                           timeframe: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load training data from DuckDB.
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            
        Returns:
            Tuple of (features array, list of data keys)
        """
        logger.info(f"Loading training data for symbol={symbol}, timeframe={timeframe}")
        
        # Load OHLCV data from DuckDB
        ohlcv_data = {}
        
        try:
            with DuckDBConnection() as conn:
                # Query market data
                query = """
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE timestamp >= NOW() - INTERVAL '365 days'
                """
                
                if symbol:
                    query += f" AND symbol = '{symbol}'"
                if timeframe:
                    query += f" AND timeframe = '{timeframe}'"
                    
                query += " ORDER BY symbol, timeframe, timestamp"
                
                result = conn.execute_query(query)
                df = result.fetchdf()
                
                # Group by symbol and timeframe
                for (sym, tf), group in df.groupby(['symbol', 'timeframe']):
                    key = f"{sym}_{tf}"
                    ohlcv_data[key] = group.set_index('timestamp')
                    
        except Exception as e:
            logger.error(f"Failed to load data from DuckDB: {e}")
            # Fallback to sample data for testing
            logger.info("Generating sample data for training...")
            ohlcv_data = self._generate_sample_data(symbol, timeframe)
        
        if not ohlcv_data:
            raise ValueError("No training data available")
        
        # Extract features
        features, keys, extractor = prepare_training_data(
            ohlcv_data,
            self.feature_config,
            symbol,
            timeframe
        )
        
        self.feature_extractor = extractor
        
        logger.info(f"Loaded {len(features)} samples from {len(keys)} data sources")
        
        return features, keys
    
    def _generate_sample_data(self, symbol: Optional[str] = None,
                              timeframe: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Generate sample data for testing when real data is unavailable."""
        symbols = [symbol] if symbol else self.config.get('symbols', {}).get('primary', ['EURUSD'])
        timeframes = [timeframe] if timeframe else self.config.get('symbols', {}).get('timeframes', ['H1'])
        
        ohlcv_data = {}
        np.random.seed(42)
        
        for sym in symbols:
            for tf in timeframes:
                # Generate 1000 bars of sample data
                n_bars = 1000
                dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='H')
                
                # Random walk prices
                returns = np.random.randn(n_bars) * 0.001
                close = 100 * np.exp(np.cumsum(returns))
                
                # Generate OHLCV
                high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.002)
                low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.002)
                open_price = close + np.random.randn(n_bars) * 0.1
                volume = np.random.randint(1000, 10000, n_bars)
                
                df = pd.DataFrame({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                }, index=dates)
                
                key = f"{sym}_{tf}"
                ohlcv_data[key] = df
                
        return ohlcv_data
    
    def train(self, features: np.ndarray, 
              n_states: int = 4) -> Tuple['hmm.GaussianHMM', Dict]:
        """
        Train HMM model on features.
        
        Args:
            features: 2D array of features (n_samples, n_features)
            n_states: Number of hidden states
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        logger.info(f"Starting HMM training with {len(features)} samples, {n_states} states")
        self.training_start_time = datetime.now(timezone.utc)
        
        # Create model
        model = self._create_model(n_states)
        
        # Split data for validation
        validation_split = self.training_config.get('validation_split', 0.2)
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        
        train_data = features[:-n_val]
        val_data = features[-n_val:]
        
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Train model
        try:
            model.fit(train_data)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # Calculate metrics
        train_score = model.score(train_data)
        val_score = model.score(val_data)
        
        # Get state distribution
        state_seq = model.predict(features)
        state_counts = np.bincount(state_seq, minlength=n_states)
        state_distribution = state_counts / len(state_seq)
        
        # Get transition matrix
        transition_matrix = model.transmat_.tolist()
        
        metrics = {
            'train_log_likelihood': train_score,
            'val_log_likelihood': val_score,
            'n_samples': n_samples,
            'n_features': features.shape[1],
            'n_states': n_states,
            'state_distribution': {f'state_{i}': float(p) for i, p in enumerate(state_distribution)},
            'transition_matrix': transition_matrix,
            'training_duration_seconds': (datetime.now(timezone.utc) - self.training_start_time).total_seconds()
        }
        
        logger.info(f"Training completed. Train LL: {train_score:.2f}, Val LL: {val_score:.2f}")
        
        self.current_model = model
        self.current_metrics = metrics
        
        return model, metrics
    
    def validate_model(self, model: 'hmm.GaussianHMM', 
                       features: np.ndarray) -> Dict[str, Any]:
        """
        Validate trained model against quality criteria.
        
        Args:
            model: Trained HMM model
            features: Feature array for validation
            
        Returns:
            Validation report dictionary
        """
        logger.info("Validating model...")
        
        report = {
            'passed': True,
            'checks': [],
            'recommendation': 'deploy'
        }
        
        # Check 1: Log-likelihood
        ll = model.score(features)
        min_ll = self.validation_config.get('min_log_likelihood', -2000)
        ll_check = ll > min_ll
        report['checks'].append({
            'name': 'log_likelihood',
            'value': ll,
            'threshold': min_ll,
            'passed': ll_check
        })
        if not ll_check:
            report['passed'] = False
            report['recommendation'] = 'reject'
        
        # Check 2: State distribution balance
        state_seq = model.predict(features)
        state_counts = np.bincount(state_seq, minlength=model.n_components)
        state_dist = state_counts / len(state_seq)
        max_state_imbalance = self.validation_config.get('max_state_imbalance', 0.6)
        imbalance_check = np.max(state_dist) < max_state_imbalance
        report['checks'].append({
            'name': 'state_balance',
            'value': float(np.max(state_dist)),
            'threshold': max_state_imbalance,
            'passed': imbalance_check
        })
        if not imbalance_check:
            report['passed'] = False
            report['recommendation'] = 'review'
        
        # Check 3: Transition matrix persistence
        diag = np.diag(model.transmat_)
        min_persistence = self.validation_config.get('min_diagonal_persistence', 0.7)
        persistence_check = np.mean(diag) > min_persistence
        report['checks'].append({
            'name': 'transition_persistence',
            'value': float(np.mean(diag)),
            'threshold': min_persistence,
            'passed': persistence_check
        })
        if not persistence_check:
            report['passed'] = False
            report['recommendation'] = 'review'
        
        logger.info(f"Validation {'PASSED' if report['passed'] else 'FAILED'}. Recommendation: {report['recommendation']}")
        
        return report
    
    def save_model(self, model: 'hmm.GaussianHMM', 
                   model_type: str,
                   symbol: Optional[str] = None,
                   timeframe: Optional[str] = None,
                   metrics: Optional[Dict] = None,
                   output_dir: str = "/data/hmm/models") -> Tuple[str, str]:
        """
        Save trained model to disk and database.
        
        Args:
            model: Trained HMM model
            model_type: Model type ('universal', 'per_symbol', 'per_symbol_timeframe')
            symbol: Symbol for per-symbol models
            timeframe: Timeframe for per-symbol-timeframe models
            metrics: Training metrics
            output_dir: Output directory for model files
            
        Returns:
            Tuple of (model path, version string)
        """
        # Generate version
        version = self._generate_version()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create model filename
        if model_type == 'universal':
            filename = f"hmm_universal_{version}.pkl"
        elif model_type == 'per_symbol':
            filename = f"hmm_{symbol}_{version}.pkl"
        else:
            filename = f"hmm_{symbol}_{timeframe}_{version}.pkl"
        
        model_path = output_path / filename
        
        # Save model with scaler
        model_data = {
            'model': model,
            'scaler': self.feature_extractor.get_scaler_params(),
            'feature_names': self.feature_extractor.get_feature_names(),
            'config': self.config,
            'metrics': metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Calculate checksum
        checksum = self._calculate_checksum(model_path)
        
        logger.info(f"Saved model to {model_path} (checksum: {checksum[:16]}...)")
        
        # Save to database
        self._save_to_database(
            version=version,
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            model_path=str(model_path),
            checksum=checksum,
            metrics=metrics
        )
        
        # Save metadata JSON
        metadata_path = output_path / f"{filename}.metadata.json"
        metadata = {
            'version': version,
            'model_type': model_type,
            'symbol': symbol,
            'timeframe': timeframe,
            'checksum': checksum,
            'training_date': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics,
            'n_states': model.n_components,
            'n_features': len(self.feature_extractor.get_feature_names())
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(model_path), version
    
    def _generate_version(self) -> str:
        """Generate version string based on timestamp."""
        now = datetime.now(timezone.utc)
        return f"{now.year}.{now.month}.{now.day}_{now.hour:02d}{now.minute:02d}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _save_to_database(self, version: str, model_type: str, 
                          symbol: Optional[str], timeframe: Optional[str],
                          model_path: str, checksum: str, 
                          metrics: Optional[Dict]) -> None:
        """Save model metadata to database."""
        session = self.Session()
        
        try:
            # Deactivate previous models of same type
            session.query(HMMModel).filter(
                HMMModel.model_type == model_type,
                HMMModel.symbol == symbol,
                HMMModel.timeframe == timeframe,
                HMMModel.is_active == True
            ).update({'is_active': False})
            
            # Create new model record
            model_record = HMMModel(
                version=version,
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe,
                n_states=metrics.get('n_states', 4) if metrics else 4,
                log_likelihood=metrics.get('train_log_likelihood') if metrics else None,
                state_distribution=metrics.get('state_distribution') if metrics else None,
                transition_matrix=metrics.get('transition_matrix') if metrics else None,
                training_samples=metrics.get('n_samples', 0) if metrics else 0,
                training_date=datetime.now(timezone.utc),
                checksum=checksum,
                file_path=model_path,
                is_active=True,
                validation_status='pending'
            )
            
            session.add(model_record)
            session.commit()
            
            logger.info(f"Saved model record to database: {version}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save to database: {e}")
            raise
        finally:
            session.close()
    
    def train_universal(self) -> Tuple[str, str]:
        """Train universal model on all data."""
        logger.info("Training universal model...")
        
        features, keys = self.load_training_data()
        
        # Check minimum samples
        min_samples = self.training_config.get('min_samples_universal', 50000)
        if len(features) < min_samples:
            logger.warning(f"Only {len(features)} samples available, recommended: {min_samples}")
        
        model, metrics = self.train(features)
        
        return self.save_model(model, 'universal', metrics=metrics)
    
    def train_per_symbol(self, symbol: str) -> Tuple[str, str]:
        """Train per-symbol model."""
        logger.info(f"Training per-symbol model for {symbol}...")
        
        features, keys = self.load_training_data(symbol=symbol)
        
        # Check minimum samples
        min_samples = self.training_config.get('min_samples_per_symbol', 2000)
        if len(features) < min_samples:
            raise ValueError(f"Insufficient samples: {len(features)} < {min_samples}")
        
        model, metrics = self.train(features)
        
        return self.save_model(model, 'per_symbol', symbol=symbol, metrics=metrics)
    
    def train_per_symbol_timeframe(self, symbol: str, 
                                    timeframe: str) -> Tuple[str, str]:
        """Train per-symbol-timeframe model."""
        logger.info(f"Training model for {symbol} {timeframe}...")
        
        features, keys = self.load_training_data(symbol=symbol, timeframe=timeframe)
        
        # Check minimum samples
        min_samples = self.training_config.get('min_samples_per_symbol', 2000)
        if len(features) < min_samples:
            raise ValueError(f"Insufficient samples: {len(features)} < {min_samples}")
        
        model, metrics = self.train(features)
        
        return self.save_model(model, 'per_symbol_timeframe', 
                               symbol=symbol, timeframe=timeframe, metrics=metrics)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train HMM for regime detection')
    parser.add_argument('--level', type=str, required=True,
                       choices=['universal', 'per-symbol', 'per-symbol-timeframe'],
                       help='Training level')
    parser.add_argument('--symbol', type=str, help='Symbol for per-symbol training')
    parser.add_argument('--timeframe', type=str, help='Timeframe for per-symbol-timeframe training')
    parser.add_argument('--config', type=str, default='config/hmm_config.json',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='/data/hmm/models',
                       help='Output directory for trained models')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.level == 'per-symbol' and not args.symbol:
        parser.error("--symbol is required for per-symbol training")
    if args.level == 'per-symbol-timeframe' and (not args.symbol or not args.timeframe):
        parser.error("--symbol and --timeframe are required for per-symbol-timeframe training")
    
    # Initialize trainer
    trainer = HMMTrainer(args.config)
    
    # Run training
    try:
        if args.level == 'universal':
            model_path, version = trainer.train_universal()
        elif args.level == 'per-symbol':
            model_path, version = trainer.train_per_symbol(args.symbol)
        else:
            model_path, version = trainer.train_per_symbol_timeframe(args.symbol, args.timeframe)
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved: {model_path}")
        print(f"Version: {version}")
        
        # Run validation if requested
        if args.validate and trainer.current_model is not None:
            features, _ = trainer.load_training_data(args.symbol, args.timeframe)
            report = trainer.validate_model(trainer.current_model, features)
            print(f"\nValidation report:")
            for check in report['checks']:
                status = "PASS" if check['passed'] else "FAIL"
                print(f"  {check['name']}: {check['value']:.4f} (threshold: {check['threshold']}) [{status}]")
            print(f"Recommendation: {report['recommendation']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())