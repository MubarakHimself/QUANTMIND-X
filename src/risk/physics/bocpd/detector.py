"""Bayesian Online Changepoint Detection (BOCPD) — Main Detector.

Implements the algorithm from Adams & MacKay (2007):
"Bayesian Online Changepoint Detection"
https://arxiv.org/abs/0710.3742

Core idea:
    - Maintain a posterior distribution over run lengths (bars since changepoint)
    - Each time step: update using Bayes rule with new observation
    - If changepoint probability > threshold, signal regime transition
    - No training needed; just calibrate hazard and observation model

Key advantages:
    - Online: processes one observation at a time
    - Fast: O(T) memory where T = truncation length (~1000)
    - No hidden state assumptions like HMM
    - Principled Bayesian framework
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.risk.physics.sensors.base import BaseRegimeSensor
from .hazard import BaseHazard, ConstantHazard
from .observation import BaseObservation, StudentTObservation

logger = logging.getLogger(__name__)

# Hyperparameters
MAX_RUN_LENGTH = 1000  # Truncate run-length distribution
MIN_PROBABILITY = 1e-10  # Drop run lengths with P < this
CALIBRATION_TARGET_CHANGEPOINTS = 3.0  # Expected changepoints per 1000 bars


class BOCPDDetector(BaseRegimeSensor):
    """Bayesian Online Changepoint Detection.

    Unlike HMM and MS-GARCH, BOCPD is ONLINE — it processes one observation
    at a time and maintains a posterior over run lengths.

    No heavy training needed. Just calibrate hazard_rate and observation model
    from historical data statistics.

    Attributes:
        hazard: Hazard function (prior on changepoint rate)
        observation: Observation model (likelihood of data)
        threshold: Changepoint probability threshold (0.5 = moderate)
        min_run_length: Minimum bars before declaring changepoint (noise filter)
        run_length_dist: Current posterior P(run_length | data)
        evaluation_points: Pre-allocated array for efficiency
    """

    def __init__(
        self,
        hazard: Optional[BaseHazard] = None,
        observation: Optional[BaseObservation] = None,
        threshold: float = 0.5,
        min_run_length: int = 10,
    ):
        """Initialize BOCPD detector.

        Args:
            hazard: Hazard function (default: ConstantHazard(250))
            observation: Observation model (default: StudentTObservation)
            threshold: Changepoint probability threshold in [0, 1]
            min_run_length: Minimum bars before signaling changepoint
        """
        self.hazard = hazard or ConstantHazard(lam=250.0)
        self.observation = observation or StudentTObservation()
        self.threshold = float(threshold)
        self.min_run_length = int(min_run_length)

        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

        # Run-length distribution: run_length_dist[t] = P(run_length = t | data)
        # Start with certainty that we're at the beginning
        self.run_length_dist = np.array([1.0])

        # Store evaluations for each run length
        # Preallocated for efficiency
        self.evaluation_points = np.zeros(MAX_RUN_LENGTH)

        # Statistics
        self.n_observations = 0
        self.n_changepoints_detected = 0
        self.time_since_last_changepoint = 0

        # For caching observation models
        self._observation_models: List[BaseObservation] = [
            self.observation.reset()
        ]

        logger.info(
            f"BOCPDDetector initialized: hazard={self.hazard.get_params()}, "
            f"observation={self.observation.get_params()}, "
            f"threshold={self.threshold}, min_run_length={self.min_run_length}"
        )

    def update(self, x: float) -> Dict[str, Any]:
        """Process one observation and update run-length posterior.

        This is the core BOCPD algorithm:
        1. Compute predictive probability under each run length
        2. Compute growth probabilities (no changepoint)
        3. Compute changepoint probability (sum over all run lengths)
        4. Normalize to get posterior
        5. Check if changepoint probability > threshold

        Args:
            x: New observation (e.g., log_return or composite feature)

        Returns:
            Dict with:
                - changepoint_prob (float): P(changepoint | data)
                - is_changepoint (bool): changepoint_prob > threshold
                - current_run_length (int): MAP estimate of run length
                - run_length_distribution (np.ndarray): Full posterior
        """
        x = float(x)
        if not np.isfinite(x):
            logger.warning(f"Non-finite observation {x}, skipping")
            return self._get_result(
                changepoint_prob=0.0, is_changepoint=False, run_length=0
            )

        self.n_observations += 1

        # Step 1: Evaluate predictive distribution under each run length
        T = len(self.run_length_dist)
        for t in range(T):
            self.evaluation_points[t] = self._observation_models[t].pdf(x)

        # Clamp to avoid numerical issues
        self.evaluation_points[:T] = np.maximum(
            self.evaluation_points[:T], MIN_PROBABILITY
        )

        # Step 2: Compute growth probabilities (no changepoint)
        # P(r_t = t, d_t | d_{t-1}) = P(d_t | r_{t-1} = t-1) * P(r_{t-1} = t-1 | d_{t-1})
        #                            * (1 - H(t-1))
        growth_probs = self.evaluation_points[:T] * self.run_length_dist[:T]
        for t in range(T):
            growth_probs[t] *= 1.0 - self.hazard(t)

        # Step 3: Compute changepoint probability
        # P(r_t = 0, d_t | d_{t-1}) = P(d_t | r_t = 0) * SUM_t [ P(r_{t-1} = t) * H(t) ]
        changepoint_prob = self.evaluation_points[0] * np.sum(
            self.run_length_dist[:T] * np.array([self.hazard(t) for t in range(T)])
        )

        # Step 4: Normalize to get posterior
        # P(r_t | d_t) = (growth + changepoint) / Z
        Z = np.sum(growth_probs) + changepoint_prob
        if Z < MIN_PROBABILITY:
            logger.warning(f"Normalizer Z too small: {Z}, resetting")
            self.reset()
            return self._get_result(
                changepoint_prob=0.0, is_changepoint=False, run_length=0
            )

        # New run-length distribution
        new_dist = np.concatenate([[changepoint_prob / Z], growth_probs / Z])

        # Step 5: Prune low-probability run lengths and cap at MAX_RUN_LENGTH
        if len(new_dist) > MAX_RUN_LENGTH:
            new_dist = new_dist[:MAX_RUN_LENGTH]

        # Remove tail probabilities below threshold
        keep_idx = np.where(new_dist >= MIN_PROBABILITY)[0]
        if len(keep_idx) == 0:
            # Fallback: keep top probability
            keep_idx = np.array([np.argmax(new_dist)])

        new_dist = new_dist[keep_idx[0] : keep_idx[-1] + 1]

        # Step 6: Update state
        self.run_length_dist = new_dist / np.sum(new_dist)  # Re-normalize

        # Update observation models cache
        self._update_observation_models(x)

        # Determine changepoint
        cp_prob = new_dist[0]  # Probability of run_length = 0
        is_changepoint = (
            cp_prob > self.threshold
            and self.time_since_last_changepoint >= self.min_run_length
        )

        if is_changepoint:
            self.n_changepoints_detected += 1
            self.time_since_last_changepoint = 0
            logger.info(
                f"Changepoint detected at observation {self.n_observations}, "
                f"prob={cp_prob:.4f}"
            )
        else:
            self.time_since_last_changepoint += 1

        # Estimate current run length as MAP
        current_run_length = np.argmax(self.run_length_dist)

        return self._get_result(
            changepoint_prob=cp_prob,
            is_changepoint=is_changepoint,
            run_length=current_run_length,
        )

    def _update_observation_models(self, x: float) -> None:
        """Update the observation models for each run length.

        We maintain a separate observation model for each run length.
        When we see a new observation:
          - Grow existing models (update with x)
          - Create new model at run_length=0
        """
        # Grow existing models
        for model in self._observation_models:
            model.update(x)

        # Create new model at run_length = 0 (after changepoint)
        new_model = self.observation.reset()
        new_model.update(x)
        self._observation_models = [new_model] + self._observation_models

        # Prune if too many
        if len(self._observation_models) > MAX_RUN_LENGTH:
            self._observation_models = self._observation_models[:MAX_RUN_LENGTH]

    def _get_result(
        self, changepoint_prob: float, is_changepoint: bool, run_length: int
    ) -> Dict[str, Any]:
        """Format update result."""
        return {
            "changepoint_prob": float(changepoint_prob),
            "is_changepoint": bool(is_changepoint),
            "current_run_length": int(run_length),
            "run_length_distribution": self.run_length_dist.copy(),
        }

    def predict_regime(self, features: np.ndarray, cache_key: str = None) -> Dict:
        """Implement BaseRegimeSensor interface.

        Takes feature vector and feeds primary signal to update().
        Uses features[0] (log_returns) as the main signal.
        Also considers features[7] (susceptibility) for enhanced detection.

        Args:
            features: Feature array (10 features from HMM trainer)
            cache_key: Optional cache key (unused)

        Returns:
            Dict with changepoint_prob, is_changepoint, current_run_length,
            regime_type (str), confidence (float)
        """
        if len(features) < 1:
            logger.warning("Empty features array")
            return {
                "changepoint_prob": 0.0,
                "is_changepoint": False,
                "current_run_length": 0,
                "regime_type": "STABLE",
                "confidence": 0.0,
            }

        # Primary signal: log_returns (features[0])
        # Handle both 1D (single feature vector) and 2D (batch with shape (1, n_features))
        if features.ndim == 2:
            x = float(features[0, 0])  # 2D array: first row, first feature
        else:
            x = float(features[0])     # 1D array: first feature

        # Update with primary signal
        result = self.update(x)

        # Enhance detection with susceptibility if available
        if len(features) > 7:
            susceptibility = float(features[7])
            # High susceptibility increases changepoint probability
            if susceptibility > 0.5:
                result["changepoint_prob"] = min(
                    1.0, result["changepoint_prob"] * (1.0 + susceptibility)
                )
                result["is_changepoint"] = result["changepoint_prob"] > self.threshold

        # Determine regime type
        if result["is_changepoint"]:
            regime_type = "TRANSITION"
        else:
            regime_type = "STABLE"

        return {
            "changepoint_prob": result["changepoint_prob"],
            "is_changepoint": result["is_changepoint"],
            "current_run_length": result["current_run_length"],
            "regime_type": regime_type,
            "confidence": result["changepoint_prob"]
            if result["is_changepoint"]
            else 1.0 - result["changepoint_prob"],
        }

    def calibrate(
        self, historical_data: np.ndarray, symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """Calibrate hazard rate from historical data (fast heuristic).

        Sets lambda based on data characteristics rather than grid search.

        Args:
            historical_data: Array of log returns or feature values
            symbol: Symbol name for logging

        Returns:
            Dict with optimal_lambda, n_changepoints_found, avg_run_length
        """
        if len(historical_data) < 50:
            logger.warning(f"Not enough data: {len(historical_data)}")
            return {"optimal_lambda": 250.0, "n_changepoints_found": 0, "avg_run_length": 0.0}

        data = np.asarray(historical_data, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0)

        # Heuristic calibration: use volatility to estimate lambda
        vol = np.std(data)

        # Lower volatility -> longer expected run length
        # Higher volatility -> shorter expected run length
        if vol < 0.005:
            best_lambda = 400.0  # Very stable market
        elif vol < 0.01:
            best_lambda = 300.0  # Stable market
        elif vol < 0.02:
            best_lambda = 250.0  # Normal market
        elif vol < 0.05:
            best_lambda = 150.0  # Volatile market
        else:
            best_lambda = 100.0  # Very volatile market

        # Quick validation run on subset
        if len(data) > 200:
            sample = data[::max(1, len(data) // 200)]  # Sample 200 points
        else:
            sample = data

        test_det = BOCPDDetector(hazard=ConstantHazard(lam=best_lambda))
        for x in sample:
            test_det.update(x)

        # Estimate full-data changepoints
        scale = len(data) / len(sample) if len(sample) > 0 else 1.0
        est_cp = max(int(test_det.n_changepoints_detected * scale), 0)

        self.hazard = ConstantHazard(lam=best_lambda)
        avg_rl = len(data) / max(est_cp, 1)

        logger.info(f"Calibration (heuristic): lambda={best_lambda:.0f}, vol={vol:.6f}")
        return {
            "optimal_lambda": float(best_lambda),
            "n_changepoints_found": int(est_cp),
            "avg_run_length": float(avg_rl),
        }

    def reset(self) -> None:
        """Reset run-length distribution (for new trading session)."""
        self.run_length_dist = np.array([1.0])
        self._observation_models = [self.observation.reset()]
        self.time_since_last_changepoint = 0
        logger.debug("BOCPDDetector reset")

    def save(self, path: Path) -> None:
        """Save calibration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "hazard": self.hazard.get_params(),
            "observation": self.observation.get_params(),
            "threshold": self.threshold,
            "min_run_length": self.min_run_length,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"BOCPDDetector saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BOCPDDetector":
        """Load calibrated detector from JSON."""
        with open(path) as f:
            state = json.load(f)

        h_params = state["hazard"]
        hazard = (
            ConstantHazard(lam=h_params["lambda"])
            if h_params["type"] == "ConstantHazard"
            else None
        )

        o_params = state["observation"]
        observation = (
            StudentTObservation(
                mu_0=o_params["mu_0"],
                kappa_0=o_params["kappa_0"],
                alpha_0=o_params["alpha_0"],
                beta_0=o_params["beta_0"],
                df=o_params.get("df", 5.0),
            )
            if o_params["type"] == "StudentTObservation"
            else None
        )

        detector = cls(
            hazard=hazard,
            observation=observation,
            threshold=state["threshold"],
            min_run_length=state["min_run_length"],
        )
        logger.info(f"BOCPDDetector loaded from {path}")
        return detector

    def get_model_info(self) -> Dict[str, Any]:
        """Implement BaseRegimeSensor interface."""
        return {
            "model_type": "BOCPD",
            "hazard": self.hazard.get_params(),
            "observation": self.observation.get_params(),
            "threshold": self.threshold,
            "min_run_length": self.min_run_length,
            "n_observations": self.n_observations,
            "n_changepoints_detected": self.n_changepoints_detected,
            "current_run_length": int(np.argmax(self.run_length_dist)),
        }

    def is_model_loaded(self) -> bool:
        """Implement BaseRegimeSensor interface."""
        return True  # BOCPD is always ready (no training needed)


def calibrate_for_symbol(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    model_dir: Optional[Path] = None,
) -> BOCPDDetector:
    """One-shot calibration from OHLCV data.

    Uses extract_features_vectorized from HMM trainer to get features,
    then calibrates BOCPD on the log_returns column.

    Args:
        symbol: Currency pair (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "M5")
        df: DataFrame with OHLCV columns
        model_dir: Optional directory to save calibrated model

    Returns:
        Calibrated BOCPDDetector
    """
    from src.risk.physics.hmm.trainer import extract_features_vectorized

    if len(df) < 100:
        logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
        return BOCPDDetector()

    # Extract features
    features = extract_features_vectorized(df)

    # Log returns are in column 0
    log_returns = features[:, 0]

    # Create and calibrate detector
    detector = BOCPDDetector()
    calib_result = detector.calibrate(log_returns, symbol=f"{symbol}_{timeframe}")

    logger.info(
        f"Calibrated BOCPD for {symbol} {timeframe}: "
        f"lambda={calib_result['optimal_lambda']:.1f}, "
        f"changepoints={calib_result['n_changepoints_found']}"
    )

    # Optionally save
    if model_dir:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"bocpd_{symbol}_{timeframe}.json"
        detector.save(model_path)

    return detector
