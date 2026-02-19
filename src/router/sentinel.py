"""
The Sentinel (Intelligence Layer)
Aggregates Sensor Data into a Unified Regime Report.

Extended with HMM dual-model support for regime detection comparison
and staged deployment (shadow, hybrid, production modes).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import logging
import time
import json
from datetime import datetime, timezone

import numpy as np

from src.router.sensors.chaos import ChaosSensor
from src.router.sensors.regime import RegimeSensor
from src.router.sensors.correlation import CorrelationSensor
from src.router.sensors.news import NewsSensor

logger = logging.getLogger(__name__)

# Try to import HMM components
try:
    from src.risk.physics.hmm_sensor import HMMRegimeSensor, create_hmm_sensor
    from src.risk.physics.hmm_features import HMMFeatureExtractor
    from src.router.hmm_deployment import get_deployment_manager, DeploymentMode
    from src.router.hmm_version_control import get_version_control
    HMM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HMM components not available: {e}")
    HMM_AVAILABLE = False
    HMMRegimeSensor = None
    HMMFeatureExtractor = None

# Try to import database components for shadow log persistence
try:
    from src.database.models import HMMShadowLog, HMMModel
    from src.database.engine import engine
    from sqlalchemy.orm import sessionmaker
    DB_AVAILABLE = True
    Session = sessionmaker(bind=engine)
except ImportError as e:
    logger.warning(f"Database components not available for shadow logging: {e}")
    DB_AVAILABLE = False
    Session = None


@dataclass
class RegimeReport:
    regime: str             # TREND_STABLE, RANGE_STABLE, BREAKOUT, ETC
    chaos_score: float      # 0.0 - 1.0
    regime_quality: float   # 1.0 - chaos_score
    susceptibility: float   # 0.0 - 1.0
    is_systemic_risk: bool
    news_state: str         # SAFE, KILL_ZONE
    timestamp: float
    # HMM extensions
    hmm_regime: Optional[str] = None
    hmm_confidence: float = 0.0
    hmm_agreement: bool = False
    decision_source: str = "ising"  # 'ising', 'hmm', 'weighted'


@dataclass
class ShadowLogEntry:
    """Entry for shadow mode logging."""
    timestamp: datetime
    symbol: str
    timeframe: str
    ising_regime: str
    ising_confidence: float
    hmm_regime: str
    hmm_state: int
    hmm_confidence: float
    agreement: bool
    decision_source: str
    market_context: Dict[str, Any] = field(default_factory=dict)


class Sentinel:
    """
    The Intelligence Engine.
    Ingests Ticks -> Updates Sensors -> Classifies Regime.

    Extended with HMM dual-model support:
    - Shadow Mode: HMM runs in parallel, predictions logged but not used
    - Hybrid Mode: Weighted combination of Ising and HMM
    - Production Mode: HMM-only predictions
    """
    # Expected feature count from HMMFeatureExtractor (10 features)
    EXPECTED_FEATURE_COUNT = 10
    
    def __init__(self, shadow_mode: bool = False, hmm_weight: float = 0.0):
        self.chaos = ChaosSensor()
        self.regime = RegimeSensor()
        self.correlation = CorrelationSensor()
        self.news = NewsSensor()

        self.current_report: Optional[RegimeReport] = None

        # HMM Integration
        self.shadow_mode = shadow_mode
        self.hmm_weight = hmm_weight  # 0.0 = Ising only, 1.0 = HMM only
        self.hmm_sensor: Optional[HMMRegimeSensor] = None
        self.feature_extractor: Optional[HMMFeatureExtractor] = None

        # Shadow logging
        self._shadow_logs: List[ShadowLogEntry] = []
        self._agreement_count = 0
        self._total_predictions = 0
        
        # Database session for persistence
        self._db_session = None

        # Initialize HMM if available
        if HMM_AVAILABLE:
            try:
                self.hmm_sensor = create_hmm_sensor()
                self.feature_extractor = HMMFeatureExtractor()
                logger.info("HMM sensor initialized for Sentinel")
            except Exception as e:
                logger.warning(f"Failed to initialize HMM sensor: {e}")
        
        # Initialize database session if available
        if DB_AVAILABLE:
            try:
                self._db_session = Session()
                logger.info("Database session initialized for shadow logging")
            except Exception as e:
                logger.warning(f"Failed to initialize database session: {e}")

    def set_mode(self, shadow_mode: bool = False, hmm_weight: float = 0.0) -> None:
        """
        Set HMM integration mode.

        Args:
            shadow_mode: Enable shadow mode logging
            hmm_weight: Weight for HMM in hybrid mode (0.0-1.0)
        """
        self.shadow_mode = shadow_mode
        self.hmm_weight = max(0.0, min(1.0, hmm_weight))
        logger.info(f"Sentinel mode updated: shadow={shadow_mode}, hmm_weight={self.hmm_weight}")

    def on_tick(self, symbol: str, price: float, timeframe: str = "H1",
                high: Optional[float] = None, low: Optional[float] = None) -> RegimeReport:
        """
        Main Loop: Called on every tick.
        Extended with HMM dual-model support.
        """
        # 1. Update Ising-based Sensors
        c_report = self.chaos.update(price)
        r_report = self.regime.update(price)
        n_state = self.news.check_state()

        # 2. Classify Regime with Ising (primary)
        ising_regime = self._classify(c_report, r_report, n_state)

        # 3. HMM Prediction (if available and enabled)
        hmm_regime = None
        hmm_confidence = 0.0
        hmm_state = -1
        agreement = False
        decision_source = "ising"

        if self.hmm_sensor and self.hmm_sensor.is_model_loaded() and (self.shadow_mode or self.hmm_weight > 0):
            try:
                # Extract features for HMM using HMMFeatureExtractor
                ising_result = {
                    "magnetization": r_report.magnetization if hasattr(r_report, 'magnetization') else 0.5,
                    "susceptibility": r_report.susceptibility,
                    "chaos_score": c_report.score
                }

                # Build feature array using HMMFeatureExtractor (consistent with training)
                features_array = self._extract_features_array(
                    price, ising_result, high, low, symbol, timeframe
                )
                
                # Validate feature shape before prediction
                if not self._validate_feature_shape(features_array):
                    logger.error(f"Feature shape mismatch: expected {self.EXPECTED_FEATURE_COUNT} features, "
                                f"got {features_array.shape[1] if features_array.ndim == 2 else len(features_array)}")
                else:
                    # Get HMM prediction - use cache key from symbol/timeframe
                    cache_key = f"{symbol}_{timeframe}_{price}"
                    hmm_prediction = self.hmm_sensor.predict_regime(features_array, cache_key=cache_key)

                    hmm_regime = hmm_prediction.regime
                    hmm_confidence = hmm_prediction.confidence
                    hmm_state = hmm_prediction.state

                    # Check agreement
                    agreement = hmm_regime == ising_regime or self._check_regime_similarity(ising_regime, hmm_regime)

                    # Determine decision source
                    if self.hmm_weight > 0:
                        # Hybrid mode - weighted decision
                        if self.hmm_weight >= 1.0:
                            decision_source = "hmm"
                            ising_regime = hmm_regime
                        else:
                            decision_source = "weighted"
                            # For now, use the higher confidence prediction
                            if hmm_confidence > 0.7 and self.hmm_weight > 0.5:
                                ising_regime = hmm_regime
                                decision_source = "hmm_weighted"

                    # Log shadow mode entry
                    if self.shadow_mode:
                        market_context = {
                            "price": price,
                            "high": high,
                            "low": low,
                            "chaos_score": c_report.score,
                            "susceptibility": r_report.susceptibility
                        }
                        self._log_shadow_prediction(
                            symbol, timeframe, ising_regime, c_report.score,
                            hmm_regime, hmm_state, hmm_confidence, agreement, decision_source,
                            market_context=market_context
                        )

            except Exception as e:
                logger.error(f"HMM prediction error: {e}")

        # 4. Compile Report
        self.current_report = RegimeReport(
            regime=ising_regime,
            chaos_score=c_report.score,
            regime_quality=1.0 - c_report.score,
            susceptibility=r_report.susceptibility,
            is_systemic_risk=False,
            news_state=n_state,
            timestamp=time.time(),
            hmm_regime=hmm_regime,
            hmm_confidence=hmm_confidence,
            hmm_agreement=agreement,
            decision_source=decision_source
        )

        return self.current_report

    def _extract_features(self, price: float, ising_result: Dict,
                         high: Optional[float], low: Optional[float]) -> Dict[str, float]:
        """Extract features for HMM prediction (deprecated - use _extract_features_array)."""
        features = {}

        # Ising features
        features['magnetization'] = ising_result.get('magnetization', 0.5)
        features['susceptibility'] = ising_result.get('susceptibility', 0.0)
        features['chaos_score'] = ising_result.get('chaos_score', 0.0)

        # Price features (simplified for single tick)
        features['price'] = price
        if high and low:
            features['range'] = high - low

        # Technical indicators would require historical data
        # Using defaults for now
        features['rsi'] = 50.0
        features['atr_pct'] = 0.5
        features['bb_width'] = 10.0

        return features
    
    def _validate_feature_shape(self, features: np.ndarray) -> bool:
        """
        Validate that feature array has expected shape before HMM prediction.
        
        Args:
            features: Feature array to validate
            
        Returns:
            True if shape is valid, False otherwise
        """
        if features is None:
            return False
            
        # Handle both 1D and 2D arrays
        if features.ndim == 1:
            feature_count = len(features)
        elif features.ndim == 2:
            feature_count = features.shape[1]
        else:
            logger.error(f"Invalid feature array dimensions: {features.ndim}")
            return False
            
        if feature_count != self.EXPECTED_FEATURE_COUNT:
            logger.error(
                f"Feature shape mismatch: expected {self.EXPECTED_FEATURE_COUNT} features, "
                f"got {feature_count}"
            )
            return False
            
        return True
    
    def _extract_features_array(self, price: float, ising_result: Dict,
                                high: Optional[float], low: Optional[float],
                                symbol: str = "UNKNOWN", timeframe: str = "H1") -> np.ndarray:
        """
        Extract features as numpy array for HMM prediction using HMMFeatureExtractor.
        
        Uses the same feature extraction pipeline as training to ensure consistency.
        The HMMFeatureExtractor produces 10 features:
        - magnetization, susceptibility, energy, temperature (Ising outputs)
        - log_returns, rolling_volatility_20, rolling_volatility_50, price_momentum_10 (price features)
        - rsi, atr_normalized (technical indicators)
        
        Args:
            price: Current price
            ising_result: Dictionary with Ising model outputs
            high: Optional high price
            low: Optional low price
            symbol: Trading symbol (for logging)
            timeframe: Timeframe (for logging)
            
        Returns:
            Numpy array of features with shape (1, 10)
        """
        if self.feature_extractor is None:
            logger.warning("HMMFeatureExtractor not initialized, using fallback features")
            # Fallback to old 7-feature extraction (will likely fail shape validation)
            features_dict = self._extract_features(price, ising_result, high, low)
            feature_order = ['magnetization', 'susceptibility', 'chaos_score', 
                            'price', 'rsi', 'atr_pct', 'bb_width']
            features = [features_dict.get(name, 0.0) for name in feature_order]
            return np.array(features).reshape(1, -1)
        
        # Use HMMFeatureExtractor.extract_from_ising for consistent feature extraction
        # This produces the same 10 features as used during training
        features = self.feature_extractor.extract_from_ising(ising_result)
        
        # Ensure 2D array shape (1, n_features) for prediction
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Apply scaling using the stored scaler params from the trained model
        if self.hmm_sensor and hasattr(self.hmm_sensor, 'feature_extractor'):
            # The sensor's feature extractor should have the scaler params from the loaded model
            features = self.hmm_sensor.feature_extractor.scale_features(features, fit=False)
        
        return features

    def _check_regime_similarity(self, ising_regime: str, hmm_regime: str) -> bool:
        """Check if regimes are similar (partial agreement)."""
        # Map regimes to categories
        trending_keywords = ["TREND", "ORDERED"]
        ranging_keywords = ["RANGE", "DISORDERED", "RANGING"]
        chaos_keywords = ["CHAOS", "HIGH_VOL"]

        is_trending = any(k in ising_regime for k in trending_keywords)
        is_ranging = any(k in ising_regime for k in ranging_keywords)
        is_chaotic = any(k in ising_regime for k in chaos_keywords)

        hmm_trending = any(k in hmm_regime for k in trending_keywords)
        hmm_ranging = any(k in hmm_regime for k in ranging_keywords)
        hmm_chaotic = any(k in hmm_regime for k in chaos_keywords)

        # Check for same category
        if is_trending and hmm_trending:
            return True
        if is_ranging and hmm_ranging:
            return True
        if is_chaotic and hmm_chaotic:
            return True

        return False

    def _log_shadow_prediction(self, symbol: str, timeframe: str,
                               ising_regime: str, ising_confidence: float,
                               hmm_regime: str, hmm_state: int, hmm_confidence: float,
                               agreement: bool, decision_source: str,
                               market_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log shadow mode prediction to both in-memory list and database.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            ising_regime: Regime predicted by Ising model
            ising_confidence: Confidence of Ising prediction
            hmm_regime: Regime predicted by HMM
            hmm_state: HMM state ID
            hmm_confidence: Confidence of HMM prediction
            agreement: Whether both models agree
            decision_source: Which model's decision was used
            market_context: Additional market context (price, volatility, etc.)
        """
        # Create in-memory entry
        entry = ShadowLogEntry(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            ising_regime=ising_regime,
            ising_confidence=ising_confidence,
            hmm_regime=hmm_regime,
            hmm_state=hmm_state,
            hmm_confidence=hmm_confidence,
            agreement=agreement,
            decision_source=decision_source,
            market_context=market_context or {}
        )

        self._shadow_logs.append(entry)
        self._total_predictions += 1

        if agreement:
            self._agreement_count += 1

        # Keep only last 1000 logs in memory
        if len(self._shadow_logs) > 1000:
            self._shadow_logs = self._shadow_logs[-1000:]
        
        # Persist to database
        self._persist_shadow_log(entry)
    
    def _persist_shadow_log(self, entry: ShadowLogEntry) -> None:
        """
        Persist shadow log entry to database.
        
        Args:
            entry: ShadowLogEntry to persist
        """
        if not DB_AVAILABLE or self._db_session is None:
            return
            
        try:
            # Get the current model ID if available
            model_id = None
            if self.hmm_sensor and hasattr(self.hmm_sensor, '_model_version'):
                # Try to find model in database by version
                try:
                    model = self._db_session.query(HMMModel).filter(
                        HMMModel.version == self.hmm_sensor._model_version
                    ).first()
                    if model:
                        model_id = model.id
                except Exception:
                    pass
            
            # Create database record
            db_entry = HMMShadowLog(
                model_id=model_id,
                timestamp=entry.timestamp,
                symbol=entry.symbol,
                timeframe=entry.timeframe,
                ising_regime=entry.ising_regime,
                ising_confidence=entry.ising_confidence,
                hmm_regime=entry.hmm_regime,
                hmm_state=entry.hmm_state,
                hmm_confidence=entry.hmm_confidence,
                agreement=entry.agreement,
                decision_source=entry.decision_source,
                market_context=entry.market_context
            )
            
            self._db_session.add(db_entry)
            self._db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to persist shadow log to database: {e}")
            if self._db_session:
                self._db_session.rollback()

    def get_agreement_metrics(self) -> Dict[str, Any]:
        """Get shadow mode agreement metrics."""
        if self._total_predictions == 0:
            return {
                "total_predictions": 0,
                "agreement_count": 0,
                "agreement_pct": 0.0
            }

        return {
            "total_predictions": self._total_predictions,
            "agreement_count": self._agreement_count,
            "agreement_pct": (self._agreement_count / self._total_predictions) * 100
        }

    def get_recent_shadow_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent shadow log entries from in-memory cache."""
        logs = self._shadow_logs[-limit:]
        return [
            {
                "timestamp": log.timestamp.isoformat(),
                "symbol": log.symbol,
                "timeframe": log.timeframe,
                "ising_regime": log.ising_regime,
                "ising_confidence": log.ising_confidence,
                "hmm_regime": log.hmm_regime,
                "hmm_state": log.hmm_state,
                "hmm_confidence": log.hmm_confidence,
                "agreement": log.agreement,
                "decision_source": log.decision_source,
                "market_context": log.market_context
            }
            for log in logs
        ]
    
    def get_shadow_logs_from_db(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get shadow log entries from database with filtering and pagination.
        
        Args:
            symbol: Filter by symbol (optional)
            timeframe: Filter by timeframe (optional)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of shadow log dictionaries
        """
        if not DB_AVAILABLE or self._db_session is None:
            logger.warning("Database not available, falling back to in-memory logs")
            return self.get_recent_shadow_logs(limit)
            
        try:
            query = self._db_session.query(HMMShadowLog)
            
            # Apply filters
            if symbol:
                query = query.filter(HMMShadowLog.symbol == symbol.upper())
            if timeframe:
                query = query.filter(HMMShadowLog.timeframe == timeframe.upper())
            
            # Apply ordering and pagination
            query = query.order_by(HMMShadowLog.timestamp.desc())
            total = query.count()
            logs = query.offset(offset).limit(limit).all()
            
            return [log.to_dict() for log in logs], total
            
        except Exception as e:
            logger.error(f"Failed to query shadow logs from database: {e}")
            return self.get_recent_shadow_logs(limit), len(self._shadow_logs)

    def _classify(self, c, r, n_state) -> str:
        """
        Maps Sensor Outputs to Regime Enum.
        """
        if n_state == "KILL_ZONE":
            return "NEWS_EVENT"

        if c.score > 0.6:
            return "HIGH_CHAOS"

        if r.state == "CRITICAL":
            return "BREAKOUT_PRIME"

        if r.state == "ORDERED" and c.score < 0.3:
            return "TREND_STABLE"

        if r.state == "DISORDERED" and c.score < 0.3:
            return "RANGE_STABLE"

        return "UNCERTAIN"

# =============================================================================
# Global Sentinel Instance and Helper Functions
# =============================================================================

_global_sentinel: Optional[Sentinel] = None

def get_sentinel() -> Sentinel:
    """Get or create global sentinel instance."""
    global _global_sentinel
    if _global_sentinel is None:
        _global_sentinel = Sentinel()
    return _global_sentinel

def get_current_regime() -> str:
    """
    Get the current market regime.
    
    Returns:
        str: Current regime state (TREND_STABLE, RANGE_STABLE, HIGH_CHAOS, etc.)
    """
    try:
        sentinel = get_sentinel()
        if sentinel.current_report:
            return sentinel.current_report.regime
        return "UNKNOWN"
    except Exception as e:
        logger.error(f"Failed to get current regime: {e}")
        return "ERROR"
