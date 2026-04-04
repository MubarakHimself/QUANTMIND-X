"""Ensemble Regime Voter — combines HMM + MS-GARCH + BOCPD.

The voter combines predictions from multiple regime detection models
into a single consensus decision. Implements BaseRegimeSensor so it can
be used as a drop-in replacement for the existing HMM sensor.

Integration:
    Commander currently calls: hmm_sensor.predict_regime()
    With ensemble: ensemble_voter.predict_regime() (same interface)
"""

import logging
from typing import Any, Dict, Optional
from collections import Counter

import numpy as np

from src.events.regime import RegimeType
from src.risk.physics.sensors.base import BaseRegimeSensor
from src.risk.physics.hmm.trainer import PREMIUM_SESSIONS, SESSION_WINDOWS
from .weights import AdaptiveWeightManager, WeightPreset, ModelWeights

logger = logging.getLogger(__name__)


class EnsembleVoter(BaseRegimeSensor):
    """Combines HMM + MS-GARCH + BOCPD into a single regime decision.

    The voter doesn't know about model internals — it takes their outputs
    and produces one RegimeType + confidence for the Commander.

    Integration point: the Commander calls predict_regime() and receives
    a dict with regime_type, confidence, and additional metadata.

    Algorithm:
    1. Query each available model
    2. Check BOCPD for changepoint signal (regime transition detection)
    3. Weighted voting: each model's regime gets weight = model_weight * model_confidence
    4. Winner = RegimeType with highest weighted sum
    5. Ensemble confidence = winner_sum / total_sum
    6. Return with transition flag if BOCPD fired
    """

    def __init__(
        self,
        hmm_sensor: Optional[Any] = None,
        msgarch_sensor: Optional[Any] = None,
        bocpd_detector: Optional[Any] = None,
        weight_manager: Optional[AdaptiveWeightManager] = None,
        session_name: Optional[str] = None,
    ):
        """Initialize ensemble voter.

        Args:
            hmm_sensor: HMM sensor with predict_regime(features) → dict with 'regime_type'
            msgarch_sensor: MS-GARCH sensor with predict_regime(features) → dict with 'regime_type', 'sigma_forecast'
            bocpd_detector: BOCPD detector with predict_regime(features) → dict with 'is_changepoint', 'changepoint_prob'
            weight_manager: Weight manager (auto-created if None)
            session_name: Current session name (determines weight preset)
        """
        self.hmm_sensor = hmm_sensor
        self.msgarch_sensor = msgarch_sensor
        self.bocpd_detector = bocpd_detector

        # Initialize weight manager
        if weight_manager is None:
            preset = self._get_preset_for_session(session_name)
            self.weight_manager = AdaptiveWeightManager(preset=preset)
        else:
            self.weight_manager = weight_manager

        self.session_name = session_name
        self.is_premium_session = session_name in PREMIUM_SESSIONS if session_name else False

        # Performance tracking
        self.prediction_count = 0
        self.transition_count = 0

        logger.info(
            f"EnsembleVoter initialized: "
            f"HMM={hmm_sensor is not None}, "
            f"MS-GARCH={msgarch_sensor is not None}, "
            f"BOCPD={bocpd_detector is not None}, "
            f"session={session_name}"
        )

    def _get_preset_for_session(self, session_name: Optional[str]) -> WeightPreset:
        """Determine weight preset based on session.

        Args:
            session_name: Session name or None

        Returns:
            Appropriate WeightPreset
        """
        if session_name and session_name in PREMIUM_SESSIONS:
            return WeightPreset.PREMIUM_FULL
        elif session_name:
            return WeightPreset.NON_PREMIUM
        else:
            return WeightPreset.HMM_ONLY

    def predict_regime(self, features: np.ndarray, cache_key: str = None) -> Dict:
        """Produce ensemble regime decision via weighted voting.

        Algorithm:
        1. Query each available model for regime prediction
        2. BOCPD check: detect regime transitions
        3. Weighted voting on regime types
        4. Confidence = winner_sum / total_sum
        5. Return with transition flag and per-model sources

        Args:
            features: Feature array for regime prediction
            cache_key: Optional cache key (passed to individual models)

        Returns:
            Dict with keys:
                regime_type: str (RegimeType value)
                confidence: float (0-1)
                sources: dict of per-model outputs
                is_transition: bool (True if BOCPD detected changepoint)
                sigma_forecast: float (volatility forecast from MS-GARCH)
                ensemble_agreement: float (0-1, how much models agree)
                weights_used: dict of actual weights used
        """
        self.prediction_count += 1

        # Check model availability
        available = {
            "hmm": self.hmm_sensor is not None and self.hmm_sensor.is_model_loaded(),
            "msgarch": self.msgarch_sensor is not None and self.msgarch_sensor.is_model_loaded(),
            "bocpd": self.bocpd_detector is not None and self.bocpd_detector.is_model_loaded(),
        }

        # Fallback if no models available
        if not any(available.values()):
            logger.error("No models available in ensemble!")
            return {
                "regime_type": RegimeType.CHAOS.value,
                "confidence": 0.0,
                "sources": {},
                "is_transition": False,
                "sigma_forecast": None,
                "ensemble_agreement": 0.0,
                "weights_used": {"hmm": 1.0, "msgarch": 0.0, "bocpd": 0.0},
                "error": "No models available",
            }

        sources = {}
        regime_votes = {}  # RegimeType → weighted_sum
        confidence_sum = 0.0
        sigma_forecast = None
        bocpd_changepoint = False

        # 1. Query HMM sensor
        if available["hmm"]:
            try:
                hmm_output = self.hmm_sensor.predict_regime(features, cache_key)
                hmm_regime_str = self._extract_regime_type(hmm_output)
                hmm_confidence = self._extract_confidence(hmm_output)
                sources["hmm"] = {
                    "regime_type": hmm_regime_str,
                    "confidence": hmm_confidence,
                }
                logger.debug(f"HMM output: regime={hmm_regime_str}, conf={hmm_confidence:.2f}")
            except Exception as e:
                logger.error(f"HMM prediction failed: {e}")
                available["hmm"] = False

        # 2. Query MS-GARCH sensor
        if available["msgarch"]:
            try:
                msgarch_output = self.msgarch_sensor.predict_regime(features, cache_key)
                msgarch_regime_str = self._extract_regime_type(msgarch_output)
                msgarch_confidence = self._extract_confidence(msgarch_output)
                sigma_forecast = self._extract_sigma(msgarch_output)
                sources["msgarch"] = {
                    "regime_type": msgarch_regime_str,
                    "confidence": msgarch_confidence,
                    "sigma_forecast": sigma_forecast,
                }
                logger.debug(f"MS-GARCH output: regime={msgarch_regime_str}, conf={msgarch_confidence:.2f}")
            except Exception as e:
                logger.error(f"MS-GARCH prediction failed: {e}")
                available["msgarch"] = False

        # 3. Query BOCPD detector
        if available["bocpd"]:
            try:
                bocpd_output = self.bocpd_detector.predict_regime(features, cache_key)
                bocpd_changepoint = self._extract_changepoint(bocpd_output)
                bocpd_prob = self._extract_changepoint_prob(bocpd_output)
                sources["bocpd"] = {
                    "is_changepoint": bocpd_changepoint,
                    "changepoint_prob": bocpd_prob,
                }
                if bocpd_changepoint:
                    logger.info(f"BOCPD changepoint detected (prob={bocpd_prob:.3f})")
                    self.transition_count += 1
            except Exception as e:
                logger.error(f"BOCPD detection failed: {e}")
                available["bocpd"] = False

        # 4. Get weights adjusted for availability and BOCPD event
        weights = self.weight_manager.get_weights(available, bocpd_changepoint)

        # 5. Weighted voting
        vote_count = 0

        if available["hmm"]:
            regime_type = RegimeType[hmm_regime_str]
            weight = weights.hmm * hmm_confidence
            regime_votes[regime_type] = regime_votes.get(regime_type, 0.0) + weight
            confidence_sum += weight
            vote_count += 1

        if available["msgarch"]:
            regime_type = RegimeType[msgarch_regime_str]
            weight = weights.msgarch * msgarch_confidence
            regime_votes[regime_type] = regime_votes.get(regime_type, 0.0) + weight
            confidence_sum += weight
            vote_count += 1

        # Determine winner
        if not regime_votes:
            logger.error("No valid regime votes!")
            return {
                "regime_type": RegimeType.CHAOS.value,
                "confidence": 0.0,
                "sources": sources,
                "is_transition": bocpd_changepoint,
                "sigma_forecast": sigma_forecast,
                "ensemble_agreement": 0.0,
                "weights_used": weights.to_dict(),
                "error": "No valid votes",
            }

        winner_regime = max(regime_votes, key=regime_votes.get)
        winner_sum = regime_votes[winner_regime]
        ensemble_confidence = winner_sum / confidence_sum if confidence_sum > 0 else 0.0

        # Calculate agreement: how many models voted for winner
        agreement_votes = 0
        if available["hmm"] and RegimeType[hmm_regime_str] == winner_regime:
            agreement_votes += 1
        if available["msgarch"] and RegimeType[msgarch_regime_str] == winner_regime:
            agreement_votes += 1

        ensemble_agreement = agreement_votes / vote_count if vote_count > 0 else 0.0

        logger.info(
            f"Ensemble decision: regime={winner_regime.value}, "
            f"confidence={ensemble_confidence:.2f}, "
            f"agreement={ensemble_agreement:.2f}, "
            f"transition={bocpd_changepoint}"
        )

        return {
            "regime_type": winner_regime.value,
            "confidence": float(ensemble_confidence),
            "sources": sources,
            "is_transition": bocpd_changepoint,
            "sigma_forecast": sigma_forecast,
            "ensemble_agreement": float(ensemble_agreement),
            "weights_used": weights.to_dict(),
            "model_count": vote_count,
        }

    def set_session(self, session_name: str) -> None:
        """Update session context (changes weight preset).

        Called when the session detector detects a session change.
        Premium sessions → full ensemble weights.
        Non-premium sessions → HMM-only weights.

        Args:
            session_name: New session name
        """
        old_name = self.session_name
        self.session_name = session_name
        self.is_premium_session = session_name in PREMIUM_SESSIONS

        # Update weight preset
        preset = self._get_preset_for_session(session_name)
        self.weight_manager.set_preset(preset)

        logger.info(
            f"Session changed: {old_name} → {session_name} "
            f"(premium={self.is_premium_session}), weights={preset.value}"
        )

    def get_model_info(self) -> Dict:
        """Get information about all loaded models in the ensemble.

        Returns:
            Dict with info about each model
        """
        info = {
            "ensemble": {
                "session": self.session_name,
                "is_premium_session": self.is_premium_session,
                "prediction_count": self.prediction_count,
                "transition_count": self.transition_count,
                "transition_rate": (
                    self.transition_count / self.prediction_count
                    if self.prediction_count > 0
                    else 0.0
                ),
            },
            "weight_manager": self.weight_manager.get_stats(),
            "models": {},
        }

        # Get info from each model
        if self.hmm_sensor:
            try:
                info["models"]["hmm"] = {
                    "loaded": self.hmm_sensor.is_model_loaded(),
                    "info": self.hmm_sensor.get_model_info() if self.hmm_sensor.is_model_loaded() else None,
                }
            except Exception as e:
                info["models"]["hmm"] = {"error": str(e)}

        if self.msgarch_sensor:
            try:
                info["models"]["msgarch"] = {
                    "loaded": self.msgarch_sensor.is_model_loaded(),
                    "info": self.msgarch_sensor.get_model_info() if self.msgarch_sensor.is_model_loaded() else None,
                }
            except Exception as e:
                info["models"]["msgarch"] = {"error": str(e)}

        if self.bocpd_detector:
            try:
                info["models"]["bocpd"] = {
                    "loaded": self.bocpd_detector.is_model_loaded(),
                    "info": self.bocpd_detector.get_model_info() if self.bocpd_detector.is_model_loaded() else None,
                }
            except Exception as e:
                info["models"]["bocpd"] = {"error": str(e)}

        return info

    def is_model_loaded(self) -> bool:
        """Check if ensemble is ready (at least HMM must be loaded).

        Returns:
            True if at least HMM is loaded
        """
        if self.hmm_sensor is None:
            return False
        try:
            return self.hmm_sensor.is_model_loaded()
        except Exception:
            return False

    def get_ensemble_status(self) -> Dict:
        """Detailed status of all models in the ensemble.

        Returns:
            Comprehensive status dict
        """
        status = {
            "ready": self.is_model_loaded(),
            "session": self.session_name,
            "is_premium": self.is_premium_session,
            "predictions_made": self.prediction_count,
            "transitions_detected": self.transition_count,
        }

        # Model availability
        status["models_available"] = {
            "hmm": self.hmm_sensor is not None and self.hmm_sensor.is_model_loaded(),
            "msgarch": self.msgarch_sensor is not None and self.msgarch_sensor.is_model_loaded(),
            "bocpd": self.bocpd_detector is not None and self.bocpd_detector.is_model_loaded(),
        }

        # Weight manager state
        status["weights"] = self.weight_manager.get_stats()

        # Model-specific details
        status["model_details"] = {}

        if self.hmm_sensor and self.hmm_sensor.is_model_loaded():
            try:
                status["model_details"]["hmm"] = self.hmm_sensor.get_model_info()
            except Exception as e:
                status["model_details"]["hmm"] = {"error": str(e)}

        if self.msgarch_sensor and self.msgarch_sensor.is_model_loaded():
            try:
                status["model_details"]["msgarch"] = self.msgarch_sensor.get_model_info()
            except Exception as e:
                status["model_details"]["msgarch"] = {"error": str(e)}

        if self.bocpd_detector and self.bocpd_detector.is_model_loaded():
            try:
                status["model_details"]["bocpd"] = self.bocpd_detector.get_model_info()
            except Exception as e:
                status["model_details"]["bocpd"] = {"error": str(e)}

        return status

    # -----------------------------------------------------------------------
    # Private helpers to extract data from model outputs (flexible)
    # -----------------------------------------------------------------------

    def _extract_regime_type(self, output: Any) -> str:
        """Extract RegimeType string from model output (flexible to different formats).

        Args:
            output: Model output (dict, object, or string)

        Returns:
            RegimeType enum name string (e.g., "TREND_BULL")
        """
        if isinstance(output, dict):
            if "regime_type" in output:
                val = output["regime_type"]
            elif "regime" in output:
                val = output["regime"]
            else:
                val = None
        elif isinstance(output, str):
            val = output
        else:
            # Try attribute access
            val = getattr(output, "regime_type", None) or getattr(output, "regime", None)

        if val is None:
            logger.warning(f"Could not extract regime type from {output}")
            return "CHAOS"

        # Ensure it's a valid RegimeType
        if isinstance(val, RegimeType):
            return val.name

        val_str = str(val).upper()

        # Try to match with RegimeType enum
        try:
            RegimeType[val_str]
            return val_str
        except KeyError:
            logger.warning(f"Invalid regime type: {val}, falling back to CHAOS")
            return "CHAOS"

    def _extract_confidence(self, output: Any) -> float:
        """Extract confidence score from model output.

        Args:
            output: Model output

        Returns:
            Confidence value (0-1), defaults to 0.5
        """
        if isinstance(output, dict):
            conf = output.get("confidence", output.get("conf", 0.5))
        else:
            conf = getattr(output, "confidence", getattr(output, "conf", 0.5))

        try:
            conf = float(conf)
            return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            logger.warning(f"Could not extract confidence from {output}")
            return 0.5

    def _extract_sigma(self, output: Any) -> Optional[float]:
        """Extract volatility forecast from MS-GARCH output.

        Args:
            output: Model output

        Returns:
            Volatility forecast or None
        """
        if isinstance(output, dict):
            sigma = output.get("sigma_forecast", output.get("sigma", None))
        else:
            sigma = getattr(output, "sigma_forecast", getattr(output, "sigma", None))

        if sigma is not None:
            try:
                return float(sigma)
            except (ValueError, TypeError):
                pass

        return None

    def _extract_changepoint(self, output: Any) -> bool:
        """Extract changepoint flag from BOCPD output.

        Args:
            output: Model output

        Returns:
            True if changepoint detected, False otherwise
        """
        if isinstance(output, dict):
            cp = output.get("is_changepoint", output.get("changepoint", False))
        else:
            cp = getattr(output, "is_changepoint", getattr(output, "changepoint", False))

        return bool(cp)

    def _extract_changepoint_prob(self, output: Any) -> float:
        """Extract changepoint probability from BOCPD output.

        Args:
            output: Model output

        Returns:
            Changepoint probability (0-1), defaults to 0.0
        """
        if isinstance(output, dict):
            prob = output.get("changepoint_prob", output.get("changepoint_probability", 0.0))
        else:
            prob = getattr(output, "changepoint_prob", getattr(output, "changepoint_probability", 0.0))

        try:
            prob = float(prob)
            return max(0.0, min(1.0, prob))
        except (ValueError, TypeError):
            return 0.0
