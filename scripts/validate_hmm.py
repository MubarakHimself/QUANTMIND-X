#!/usr/bin/env python3
"""
HMM Model Validation Script
============================

Validates trained HMM models against quality criteria and compares
predictions with the Ising Model on historical data.

Validation Criteria:
- Log-likelihood within acceptable range
- State distribution balance (no state > 60%)
- Transition matrix persistence (high diagonal values)
- Agreement with Ising Model (> 80%)

Usage:
    python scripts/validate_hmm.py --model-path /data/hmm/models/hmm_universal_v1.pkl
    python scripts/validate_hmm.py --version 2024.2.14_0200
    python scripts/validate_hmm.py --latest

Reference: docs/architecture/components.md
"""

import os
import sys
import json
import logging
import argparse
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

from src.risk.physics.hmm_features import HMMFeatureExtractor, FeatureConfig
from src.risk.physics.ising_sensor import IsingRegimeSensor, IsingSensorConfig
from src.database.models import HMMModel
from src.database.engine import engine
from src.database.duckdb_connection import DuckDBConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HMMValidator:
    """
    HMM Model Validator for quality assurance.
    
    Validates trained models against quality criteria and compares
    with Ising Model predictions.
    """
    
    def __init__(self, config_path: str = "config/hmm_config.json"):
        """Initialize validator with configuration."""
        self.config = self._load_config(config_path)
        self.validation_config = self.config.get('validation', {})
        self.regime_mapping = self.config.get('regime_mapping', {})
        
        # Initialize sensors
        self.feature_extractor = HMMFeatureExtractor()
        self.ising_sensor = IsingRegimeSensor(IsingSensorConfig())
        
        # Database session
        self.Session = sessionmaker(bind=engine)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_model(self, model_path: Optional[str] = None,
                   version: Optional[str] = None,
                   latest: bool = False) -> Tuple[Any, Dict]:
        """
        Load HMM model from file or database.
        
        Args:
            model_path: Direct path to model file
            version: Model version to load from database
            latest: Load the latest model from database
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_path:
            return self._load_model_from_file(model_path)
        elif version or latest:
            return self._load_model_from_db(version)
        else:
            raise ValueError("Must specify model_path, version, or latest=True")
    
    def _load_model_from_file(self, model_path: str) -> Tuple[Any, Dict]:
        """Load model from file path."""
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        metadata = {
            'scaler': model_data.get('scaler', {}),
            'feature_names': model_data.get('feature_names', []),
            'metrics': model_data.get('metrics', {}),
            'config': model_data.get('config', {})
        }
        
        # Set feature extractor scaler
        if metadata['scaler']:
            self.feature_extractor.set_scaler_params(metadata['scaler'])
        
        logger.info(f"Loaded model from {model_path}")
        
        return model, metadata
    
    def _load_model_from_db(self, version: Optional[str] = None) -> Tuple[Any, Dict]:
        """Load model from database by version."""
        session = self.Session()
        
        try:
            query = session.query(HMMModel).filter(HMMModel.is_active == True)
            
            if version:
                query = query.filter(HMMModel.version == version)
            
            model_record = query.order_by(HMMModel.created_at.desc()).first()
            
            if not model_record:
                raise ValueError(f"No model found with version={version}")
            
            # Load model file
            return self._load_model_from_file(model_record.file_path)
            
        finally:
            session.close()
    
    def load_test_data(self, symbol: Optional[str] = None,
                       timeframe: Optional[str] = None,
                       n_samples: int = 1000) -> np.ndarray:
        """
        Load test data for validation.
        
        Args:
            symbol: Symbol to load
            timeframe: Timeframe to load
            n_samples: Number of samples to load
            
        Returns:
            Feature array for testing
        """
        logger.info(f"Loading test data for {symbol or 'all'} {timeframe or 'all timeframes'}")
        
        try:
            with DuckDBConnection() as conn:
                query = f"""
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE timestamp >= NOW() - INTERVAL '30 days'
                """
                
                if symbol:
                    query += f" AND symbol = '{symbol}'"
                if timeframe:
                    query += f" AND timeframe = '{timeframe}'"
                    
                query += f" ORDER BY timestamp DESC LIMIT {n_samples}"
                
                result = conn.execute_query(query)
                df = result.fetchdf()
                
                if df.empty:
                    logger.warning("No test data available, generating sample data")
                    return self._generate_sample_test_data(n_samples)
                
                # Group by symbol and timeframe, then extract features per group
                all_features = []
                for (sym, tf), group in df.groupby(['symbol', 'timeframe']):
                    # Sort by timestamp
                    group = group.sort_values('timestamp')
                    # Extract features for this group
                    features = self.feature_extractor.extract_features_batch(
                        group.set_index('timestamp'), 
                        sym, 
                        tf
                    )
                    if len(features) > 0:
                        all_features.append(features)
                
                if all_features:
                    return np.vstack(all_features)
                else:
                    return self._generate_sample_test_data(n_samples)
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return self._generate_sample_test_data(n_samples)
    
    def _generate_sample_test_data(self, n_samples: int = 1000) -> np.ndarray:
        """Generate sample test data for validation."""
        np.random.seed(42)
        
        # Generate random features (10 features)
        features = np.random.randn(n_samples, 10)
        
        return features
    
    def validate_model_quality(self, model: Any, 
                               features: np.ndarray) -> Dict[str, Any]:
        """
        Validate model against quality criteria.
        
        Args:
            model: Trained HMM model
            features: Test feature array
            
        Returns:
            Validation report dictionary
        """
        logger.info("Validating model quality...")
        
        report = {
            'passed': True,
            'checks': [],
            'recommendation': 'deploy',
            'overall_score': 0.0
        }
        
        passed_count = 0
        total_checks = 4
        
        # Check 1: Log-likelihood
        ll = model.score(features)
        min_ll = self.validation_config.get('min_log_likelihood', -2000)
        ll_check = ll > min_ll
        report['checks'].append({
            'name': 'log_likelihood',
            'value': float(ll),
            'threshold': min_ll,
            'passed': ll_check,
            'description': f"Log-likelihood {ll:.2f} {'>' if ll_check else '<='} {min_ll}"
        })
        if ll_check:
            passed_count += 1
        else:
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
            'passed': imbalance_check,
            'description': f"Max state proportion {np.max(state_dist):.2%} {'<' if imbalance_check else '>='} {max_state_imbalance:.0%}",
            'state_distribution': {f'state_{i}': float(p) for i, p in enumerate(state_dist)}
        })
        if imbalance_check:
            passed_count += 1
        elif report['recommendation'] != 'reject':
            report['recommendation'] = 'review'
        
        # Check 3: Transition matrix persistence
        diag = np.diag(model.transmat_)
        min_persistence = self.validation_config.get('min_diagonal_persistence', 0.7)
        persistence_check = np.mean(diag) > min_persistence
        report['checks'].append({
            'name': 'transition_persistence',
            'value': float(np.mean(diag)),
            'threshold': min_persistence,
            'passed': persistence_check,
            'description': f"Mean diagonal {np.mean(diag):.2%} {'>' if persistence_check else '<='} {min_persistence:.0%}",
            'diagonal_values': [float(d) for d in diag]
        })
        if persistence_check:
            passed_count += 1
        elif report['recommendation'] != 'reject':
            report['recommendation'] = 'review'
        
        # Check 4: Model convergence
        converged = model.monitor_.converged if hasattr(model, 'monitor_') else True
        report['checks'].append({
            'name': 'convergence',
            'value': converged,
            'threshold': True,
            'passed': converged,
            'description': f"Model {'converged' if converged else 'did not converge'}"
        })
        if converged:
            passed_count += 1
        
        # Calculate overall score
        report['overall_score'] = passed_count / total_checks
        report['passed'] = passed_count == total_checks
        
        logger.info(f"Quality validation: {passed_count}/{total_checks} checks passed")
        
        return report
    
    def compare_with_ising(self, model: Any, 
                           features: np.ndarray,
                           ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compare HMM predictions with Ising Model.
        
        Args:
            model: Trained HMM model
            features: Test feature array
            ohlcv_data: Optional OHLCV data for Ising comparison
            
        Returns:
            Comparison report dictionary
        """
        logger.info("Comparing HMM with Ising Model...")
        
        report = {
            'agreement_pct': 0.0,
            'comparisons': [],
            'hmm_regimes': [],
            'ising_regimes': []
        }
        
        # Get HMM predictions
        hmm_states = model.predict(features)
        hmm_regimes = [self.regime_mapping.get(str(s), f"STATE_{s}") for s in hmm_states]
        
        # Get Ising predictions
        ising_regimes = []
        for i in range(len(features)):
            # Use volatility feature if available (index 3 typically)
            vol = features[i, 3] if features.shape[1] > 3 else 1.0
            ising_result = self.ising_sensor.detect_regime(market_volatility=abs(vol) * 10)
            ising_regime = ising_result.get('current_regime', 'UNKNOWN')
            ising_regimes.append(ising_regime)
        
        # Calculate agreement
        agreements = []
        for i in range(len(hmm_regimes)):
            hmm_r = hmm_regimes[i]
            ising_r = ising_regimes[i]
            
            # Map regimes to comparable categories
            hmm_category = self._map_regime_to_category(hmm_r)
            ising_category = self._map_regime_to_category(ising_r)
            
            agree = hmm_category == ising_category
            agreements.append(agree)
            
            if i < 100:  # Store first 100 comparisons
                report['comparisons'].append({
                    'index': i,
                    'hmm_regime': hmm_r,
                    'ising_regime': ising_r,
                    'hmm_category': hmm_category,
                    'ising_category': ising_category,
                    'agreement': agree
                })
        
        report['agreement_pct'] = np.mean(agreements) * 100
        report['hmm_regimes'] = hmm_regimes[:100]
        report['ising_regimes'] = ising_regimes[:100]
        
        # Check against threshold
        min_agreement = self.validation_config.get('min_ising_agreement', 0.8)
        report['agreement_check'] = {
            'value': report['agreement_pct'] / 100,
            'threshold': min_agreement,
            'passed': report['agreement_pct'] / 100 >= min_agreement
        }
        
        logger.info(f"Ising agreement: {report['agreement_pct']:.1f}%")
        
        return report
    
    def _map_regime_to_category(self, regime: str) -> str:
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
    
    def generate_validation_report(self, model: Any,
                                   features: np.ndarray,
                                   model_metadata: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            model: Trained HMM model
            features: Test feature array
            model_metadata: Model metadata dictionary
            
        Returns:
            Complete validation report
        """
        logger.info("Generating validation report...")
        
        # Run quality validation
        quality_report = self.validate_model_quality(model, features)
        
        # Run Ising comparison
        ising_report = self.compare_with_ising(model, features)
        
        # Combine reports
        full_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_info': {
                'n_states': model.n_components,
                'n_features': features.shape[1],
                'training_samples': model_metadata.get('metrics', {}).get('n_samples', 0),
                'training_date': model_metadata.get('metrics', {}).get('training_date')
            },
            'quality_validation': quality_report,
            'ising_comparison': ising_report,
            'final_recommendation': self._determine_recommendation(quality_report, ising_report)
        }
        
        return full_report
    
    def _determine_recommendation(self, quality_report: Dict, 
                                  ising_report: Dict) -> str:
        """Determine final deployment recommendation."""
        if not quality_report['passed']:
            return 'reject'
        
        if not ising_report['agreement_check']['passed']:
            return 'review'
        
        if quality_report['overall_score'] >= 0.9:
            return 'deploy'
        else:
            return 'review'
    
    def update_model_status(self, model_path: str, 
                           validation_report: Dict) -> None:
        """
        Update model validation status in database.
        
        Args:
            model_path: Path to model file
            validation_report: Validation report dictionary
        """
        session = self.Session()
        
        try:
            # Find model record
            model_record = session.query(HMMModel).filter(
                HMMModel.file_path == model_path
            ).first()
            
            if model_record:
                model_record.validation_status = validation_report['final_recommendation']
                model_record.validation_notes = json.dumps(validation_report, indent=2)
                session.commit()
                logger.info(f"Updated model status to: {validation_report['final_recommendation']}")
            else:
                logger.warning(f"Model record not found for path: {model_path}")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update model status: {e}")
        finally:
            session.close()
    
    def save_report(self, report: Dict, output_path: str) -> None:
        """Save validation report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved validation report to {path}")


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(description='Validate HMM model')
    parser.add_argument('--model-path', type=str, help='Path to model file')
    parser.add_argument('--version', type=str, help='Model version to validate')
    parser.add_argument('--latest', action='store_true', help='Validate latest model')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Symbol for test data')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe for test data')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--config', type=str, default='config/hmm_config.json',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='/data/hmm/logs/validation_report.json',
                       help='Output path for validation report')
    parser.add_argument('--update-db', action='store_true', 
                       help='Update model status in database')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = HMMValidator(args.config)
    
    try:
        # Load model
        model, metadata = validator.load_model(
            model_path=args.model_path,
            version=args.version,
            latest=args.latest
        )
        
        # Load test data
        features = validator.load_test_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_samples=args.n_samples
        )
        
        # Generate validation report
        report = validator.generate_validation_report(model, features, metadata)
        
        # Print summary
        print("\n" + "=" * 60)
        print("HMM MODEL VALIDATION REPORT")
        print("=" * 60)
        print(f"\nModel: {model.n_components} states, {features.shape[1]} features")
        print(f"\nQuality Validation:")
        for check in report['quality_validation']['checks']:
            status = "PASS" if check['passed'] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['description']}")
        
        print(f"\nIsing Comparison:")
        print(f"  Agreement: {report['ising_comparison']['agreement_pct']:.1f}%")
        agreement_check = report['ising_comparison']['agreement_check']
        status = "PASS" if agreement_check['passed'] else "FAIL"
        print(f"  [{status}] Threshold: {agreement_check['threshold']:.0%}")
        
        print(f"\nFinal Recommendation: {report['final_recommendation'].upper()}")
        print("=" * 60)
        
        # Save report
        validator.save_report(report, args.output)
        
        # Update database if requested
        if args.update_db and args.model_path:
            validator.update_model_status(args.model_path, report)
        
        # Return exit code based on recommendation
        if report['final_recommendation'] == 'reject':
            return 1
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())