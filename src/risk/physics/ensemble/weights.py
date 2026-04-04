"""Adaptive weight management for ensemble voting.

Manages the weights for HMM, MS-GARCH, and BOCPD models in the ensemble.
Adapts weights based on model availability, performance, and regime events.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class WeightPreset(str, Enum):
    """Pre-configured weight profiles for different trading sessions."""
    PREMIUM_FULL = "premium_full"      # All 3 models, full weights (TOKYO_LONDON, LONDON_OPEN, LONDON_NY)
    NON_PREMIUM = "non_premium"        # HMM only (outside premium hours)
    HMM_MSGARCH = "hmm_msgarch"        # HMM + MS-GARCH, no BOCPD (before Phase 2)
    HMM_ONLY = "hmm_only"              # Fallback when other models unavailable


@dataclass
class ModelWeights:
    """Weights for each model in the ensemble (must sum to 1.0)."""
    hmm: float = 0.45
    msgarch: float = 0.30
    bocpd: float = 0.25

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.hmm + self.msgarch + self.bocpd
        if not (0.99 <= total <= 1.01):
            logger.warning(f"ModelWeights do not sum to 1.0: {total:.4f}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"hmm": self.hmm, "msgarch": self.msgarch, "bocpd": self.bocpd}


WEIGHT_PRESETS = {
    WeightPreset.PREMIUM_FULL: ModelWeights(0.45, 0.30, 0.25),
    WeightPreset.NON_PREMIUM: ModelWeights(1.0, 0.0, 0.0),
    WeightPreset.HMM_MSGARCH: ModelWeights(0.60, 0.40, 0.0),
    WeightPreset.HMM_ONLY: ModelWeights(1.0, 0.0, 0.0),
}


class AdaptiveWeightManager:
    """Manages and adapts ensemble weights based on model availability and performance.

    Weight Adaptation Rules:
    1. When model unavailable: redistribute its weight proportionally to available models
    2. When BOCPD fires a changepoint: temporarily boost HMM weight to 0.60
    3. Tracks per-model accuracy over rolling window to adjust weights
    """

    def __init__(
        self,
        preset: WeightPreset = WeightPreset.PREMIUM_FULL,
        accuracy_window: int = 100,
    ):
        """Initialize weight manager.

        Args:
            preset: Initial weight preset
            accuracy_window: Rolling window size for accuracy tracking
        """
        self.preset = preset
        self.accuracy_window = accuracy_window
        self.base_weights = ModelWeights(*WEIGHT_PRESETS[preset].to_dict().values())
        self.current_weights = ModelWeights(*self.base_weights.to_dict().values())

        # Accuracy tracking
        self.accuracy_buffer: Dict[str, deque] = {
            "hmm": deque(maxlen=accuracy_window),
            "msgarch": deque(maxlen=accuracy_window),
            "bocpd": deque(maxlen=accuracy_window),
        }

        # Changepoint boost tracking
        self.bocpd_boost_counter = 0
        self.bocpd_boost_duration = 5  # boost HMM weight for 5 predictions after changepoint

        logger.debug(
            f"AdaptiveWeightManager initialized with preset={preset.value}, "
            f"window={accuracy_window}"
        )

    def get_weights(
        self,
        available_models: Optional[Dict[str, bool]] = None,
        bocpd_changepoint: bool = False,
    ) -> ModelWeights:
        """Get current weights adjusted for availability and events.

        Args:
            available_models: Dict like {"hmm": True, "msgarch": True, "bocpd": True}
            bocpd_changepoint: True if BOCPD just fired a changepoint

        Returns:
            ModelWeights with values summing to 1.0, adjusted for current conditions
        """
        if available_models is None:
            available_models = {"hmm": True, "msgarch": True, "bocpd": True}

        weights = ModelWeights(*self.base_weights.to_dict().values())

        # Apply BOCPD changepoint boost if active
        if bocpd_changepoint:
            self.bocpd_boost_counter = self.bocpd_boost_duration
            logger.debug("BOCPD changepoint detected, boosting HMM weight")

        if self.bocpd_boost_counter > 0:
            # Temporarily boost HMM weight
            weights.hmm = min(0.60, weights.hmm + 0.15)
            weights.msgarch = max(0.0, weights.msgarch - 0.10)
            weights.bocpd = max(0.0, weights.bocpd - 0.05)
            self.bocpd_boost_counter -= 1

        # Normalize weights for available models
        available_count = sum(1 for v in available_models.values() if v)

        if available_count == 0:
            # Fallback: all models unavailable (shouldn't happen)
            logger.error("No models available! Returning normalized base weights")
            return ModelWeights(1.0, 0.0, 0.0)

        if available_count < 3:
            # Redistribute unavailable model weights proportionally
            unavailable_weight = 0.0
            available_weights = {}

            if not available_models.get("hmm", False):
                unavailable_weight += weights.hmm
                weights.hmm = 0.0
            else:
                available_weights["hmm"] = weights.hmm

            if not available_models.get("msgarch", False):
                unavailable_weight += weights.msgarch
                weights.msgarch = 0.0
            else:
                available_weights["msgarch"] = weights.msgarch

            if not available_models.get("bocpd", False):
                unavailable_weight += weights.bocpd
                weights.bocpd = 0.0
            else:
                available_weights["bocpd"] = weights.bocpd

            # Redistribute unavailable weight proportionally
            if available_weights:
                total_available = sum(available_weights.values())
                if total_available > 0:
                    proportion = unavailable_weight / total_available
                    for model, weight in available_weights.items():
                        if model == "hmm" and available_models.get("hmm", False):
                            weights.hmm += weight * proportion
                        elif model == "msgarch" and available_models.get("msgarch", False):
                            weights.msgarch += weight * proportion
                        elif model == "bocpd" and available_models.get("bocpd", False):
                            weights.bocpd += weight * proportion

        # Normalize to sum to 1.0
        total = weights.hmm + weights.msgarch + weights.bocpd
        if total > 0:
            weights.hmm /= total
            weights.msgarch /= total
            weights.bocpd /= total

        self.current_weights = weights
        return weights

    def update_accuracy(self, model_name: str, was_correct: bool) -> None:
        """Update rolling accuracy for a model.

        Called after ground truth is known (post-trade or backtesting).

        Args:
            model_name: "hmm", "msgarch", or "bocpd"
            was_correct: True if prediction was correct
        """
        if model_name in self.accuracy_buffer:
            self.accuracy_buffer[model_name].append(1.0 if was_correct else 0.0)
            avg_acc = self.get_model_accuracy(model_name)
            logger.debug(f"Updated {model_name} accuracy: {avg_acc:.2%}")

    def get_model_accuracy(self, model_name: str) -> float:
        """Get rolling average accuracy for a model.

        Args:
            model_name: "hmm", "msgarch", or "bocpd"

        Returns:
            Rolling average accuracy (0-1), or 0 if no data
        """
        if model_name not in self.accuracy_buffer:
            return 0.0

        buffer = self.accuracy_buffer[model_name]
        if not buffer:
            return 0.0

        return sum(buffer) / len(buffer)

    def get_stats(self) -> Dict:
        """Get comprehensive statistics about weight manager state.

        Returns:
            Dictionary with current state, accuracies, and boost info
        """
        return {
            "preset": self.preset.value,
            "current_weights": self.current_weights.to_dict(),
            "base_weights": self.base_weights.to_dict(),
            "accuracies": {
                "hmm": self.get_model_accuracy("hmm"),
                "msgarch": self.get_model_accuracy("msgarch"),
                "bocpd": self.get_model_accuracy("bocpd"),
            },
            "accuracy_window_sizes": {
                "hmm": len(self.accuracy_buffer["hmm"]),
                "msgarch": len(self.accuracy_buffer["msgarch"]),
                "bocpd": len(self.accuracy_buffer["bocpd"]),
            },
            "bocpd_boost_active": self.bocpd_boost_counter > 0,
            "bocpd_boost_remaining": max(0, self.bocpd_boost_counter),
        }

    def set_preset(self, preset: WeightPreset) -> None:
        """Change weight preset (e.g., when session changes).

        Args:
            preset: New weight preset
        """
        if preset != self.preset:
            self.preset = preset
            self.base_weights = ModelWeights(*WEIGHT_PRESETS[preset].to_dict().values())
            self.current_weights = ModelWeights(*self.base_weights.to_dict().values())
            logger.info(f"Weight preset changed to {preset.value}")

    def reset(self) -> None:
        """Reset to initial state."""
        self.accuracy_buffer = {
            "hmm": deque(maxlen=self.accuracy_window),
            "msgarch": deque(maxlen=self.accuracy_window),
            "bocpd": deque(maxlen=self.accuracy_window),
        }
        self.bocpd_boost_counter = 0
        self.current_weights = ModelWeights(*self.base_weights.to_dict().values())
        logger.debug("AdaptiveWeightManager reset to initial state")
