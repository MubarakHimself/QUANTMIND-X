"""MS-GARCH Regime-Switching Volatility Sensor.

Loads lightweight GARCH parameter bundles (~2-4 KB) and uses the
GARCH(1,1) recursion to compute conditional volatility on new data,
then classifies into volatility regimes using training quantile
boundaries.

No dependency on the ``arch`` library at inference time — pure NumPy.
"""

import logging
import pickle
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

from ..sensors.base import BaseRegimeSensor
from .utils import map_vol_to_regime
from .trainer import garch_conditional_vol

logger = logging.getLogger(__name__)


class MSGARCHSensor(BaseRegimeSensor):
    """
    MS-GARCH based regime detector for volatility-driven market regimes.

    Loads trained parameter bundles and provides volatility-based regime
    predictions complementing HMM-based regime detection.

    Usage::

        sensor = MSGARCHSensor(model_path=Path("models/msgarch/EURUSD_M5_v20260326.pkl"))
        reading = sensor.predict_regime(features)
        print(f"Vol State: {reading['vol_state']}")
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._bundle: Optional[Dict] = None
        self._model_version: Optional[str] = None
        self._model_checksum: Optional[str] = None
        self._last_load_time: Optional[datetime] = None

        # Running GARCH state for streaming predictions
        self._last_var: Optional[float] = None

        # Cache
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if model_path:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return

        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)

            # Validate bundle has required keys
            required = {"garch_params", "boundaries", "n_regimes"}
            if not required.issubset(bundle.keys()):
                # Legacy format (full arch result) — reject gracefully
                logger.warning(
                    f"Legacy MS-GARCH bundle detected ({model_path.name}). "
                    f"Re-train with updated trainer."
                )
                return

            self._bundle = bundle
            self._model_version = bundle.get("version", "unknown")
            self._last_var = bundle.get("initial_var", 1e-6)

            with open(model_path, "rb") as f:
                self._model_checksum = hashlib.sha256(f.read()).hexdigest()

            self._last_load_time = datetime.now(timezone.utc)

            gp = bundle["garch_params"]
            logger.info(
                f"Loaded MS-GARCH: {model_path.name}  "
                f"({bundle.get('symbol','?')}, {bundle['n_regimes']}r, "
                f"α={gp['alpha']:.4f} β={gp['beta']:.4f})"
            )

        except Exception as e:
            logger.error(f"Failed to load MS-GARCH model: {e}")
            self._bundle = None

    def is_model_loaded(self) -> bool:
        return self._bundle is not None

    def get_model_info(self) -> Dict[str, Any]:
        if not self._bundle:
            return {"loaded": False}
        b = self._bundle
        return {
            "loaded": True,
            "symbol": b.get("symbol", "UNKNOWN"),
            "timeframe": b.get("timeframe", "UNKNOWN"),
            "n_regimes": b.get("n_regimes", 0),
            "garch_params": b.get("garch_params", {}),
            "version": self._model_version,
            "checksum": self._model_checksum[:16] + "..." if self._model_checksum else None,
            "last_load_time": self._last_load_time.isoformat() if self._last_load_time else None,
            "trained_at": b.get("trained_at", "unknown"),
            "vol_regime_labels": b.get("vol_regime_labels", {}),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_regime(
        self, features: np.ndarray, cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Predict volatility regime from a single feature vector.

        Uses GARCH(1,1) recursion to update conditional variance, then
        classifies using the training quantile boundaries.

        Args:
            features: 1-D feature array (10 elements from
                      ``extract_features_vectorized``).
            cache_key: Optional cache key.

        Returns:
            Dict with vol_state, sigma_forecast, regime_type, confidence,
            transition_probs.
        """
        if cache_key and cache_key in self._prediction_cache:
            self._cache_hits += 1
            return self._prediction_cache[cache_key]
        self._cache_misses += 1

        # Default if no model
        if not self._bundle:
            return self._default_reading(cache_key)

        try:
            if features.ndim > 1:
                features = features.flatten()

            b = self._bundle
            gp = b["garch_params"]
            omega = gp["omega"]
            alpha = gp["alpha"]
            beta = gp["beta"]

            # Feature 0 = log return
            log_ret = float(features[0]) if len(features) > 0 else 0.0

            # Update running GARCH variance
            if self._last_var is None:
                self._last_var = b.get("initial_var", 1e-6)

            new_var = omega + alpha * log_ret ** 2 + beta * self._last_var
            new_var = max(new_var, 1e-20)
            sigma = float(np.sqrt(new_var))
            self._last_var = new_var

            # Classify using training boundaries
            boundaries = np.asarray(b["boundaries"])
            n_reg = b["n_regimes"]
            regime_idx = int(np.digitize(sigma, boundaries[1:-1]))
            regime_idx = min(regime_idx, n_reg - 1)

            # Map to label
            labels = b.get("vol_regime_labels", {})
            vol_state = labels.get(str(regime_idx), labels.get(regime_idx, "MED_VOL"))

            # Confidence: higher when deep inside a regime bucket
            bucket_lo = boundaries[regime_idx] if regime_idx < len(boundaries) - 1 else boundaries[-2]
            bucket_hi = boundaries[regime_idx + 1] if regime_idx + 1 < len(boundaries) else boundaries[-1]
            bucket_width = bucket_hi - bucket_lo
            if bucket_width > 0:
                dist_from_edge = min(sigma - bucket_lo, bucket_hi - sigma) / bucket_width
                confidence = 0.5 + 0.5 * dist_from_edge  # 0.5 at edge, 1.0 at centre
            else:
                confidence = 0.7

            # Map to RegimeType
            regime_type = map_vol_to_regime(vol_state, log_ret)

            # Transition probs from stored matrix
            trans = b.get("transition_matrix", [])
            transition_probs = {}
            if trans and regime_idx < len(trans):
                row = trans[regime_idx]
                for j, p in enumerate(row):
                    lbl = labels.get(str(j), labels.get(j, f"R{j}"))
                    transition_probs[lbl] = round(float(p), 4)

            prediction = {
                "vol_state": vol_state,
                "sigma_forecast": sigma,
                "regime_type": regime_type,
                "confidence": round(float(confidence), 4),
                "transition_probs": transition_probs,
            }

            if cache_key:
                self._prediction_cache[cache_key] = prediction
                if len(self._prediction_cache) > 1000:
                    oldest = next(iter(self._prediction_cache))
                    del self._prediction_cache[oldest]

            return prediction

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._default_reading(cache_key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_reading(self, cache_key: Optional[str] = None) -> Dict[str, Any]:
        r = {
            "vol_state": "MED_VOL",
            "sigma_forecast": 0.0,
            "regime_type": "RANGE_STABLE",
            "confidence": 0.0,
            "transition_probs": {},
        }
        if cache_key:
            self._prediction_cache[cache_key] = r
        return r

    def reset_state(self) -> None:
        """Reset running GARCH state (call when switching symbols)."""
        if self._bundle:
            self._last_var = self._bundle.get("initial_var", 1e-6)
        else:
            self._last_var = None

    def clear_cache(self) -> None:
        self._prediction_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def reload_model(self, model_path: Path) -> bool:
        self._bundle = None
        self._load_model(model_path)
        return self._bundle is not None


__all__ = ["MSGARCHSensor"]
