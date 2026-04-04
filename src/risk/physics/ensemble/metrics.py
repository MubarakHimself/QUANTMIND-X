"""Ensemble performance metrics tracking and analysis.

Tracks ensemble prediction accuracy and model agreement metrics
for walk-forward validation during training and live performance monitoring.
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional

from src.events.regime import RegimeType

logger = logging.getLogger(__name__)


class EnsembleMetrics:
    """Tracks ensemble prediction accuracy and agreement metrics.

    Used for:
    1. Walk-forward validation during training
    2. Live performance monitoring
    3. Weight adaptation feedback
    """

    def __init__(self, window_size: int = 500):
        """Initialize metrics tracker.

        Args:
            window_size: Rolling window size for metric calculation
        """
        self.window_size = window_size

        # Rolling windows for predictions
        self.predictions: deque = deque(maxlen=window_size)

        # Per-model prediction tracking
        self.model_predictions: Dict[str, deque] = {
            "hmm": deque(maxlen=window_size),
            "msgarch": deque(maxlen=window_size),
            "bocpd": deque(maxlen=window_size),
        }

        # Transition tracking
        self.transitions: deque = deque(maxlen=window_size)
        self.transition_correct: deque = deque(maxlen=window_size)

        # Agreement tracking
        self.agreement_scores: deque = deque(maxlen=window_size)

        # Timestamps
        self.created_at = datetime.now(timezone.utc)
        self.last_update = None

        logger.debug(f"EnsembleMetrics initialized with window_size={window_size}")

    def record_prediction(
        self,
        prediction: Dict,
        actual_regime: Optional[str] = None,
    ) -> None:
        """Record an ensemble prediction (and optionally the true regime).

        Args:
            prediction: Dict from EnsembleVoter.predict_regime() containing:
                - regime_type: predicted RegimeType
                - confidence: prediction confidence
                - sources: per-model predictions
                - ensemble_agreement: agreement score
                - is_transition: whether BOCPD detected transition
            actual_regime: True regime (if known for accuracy calculation)
        """
        if not prediction:
            return

        # Extract prediction data
        regime_type = prediction.get("regime_type")
        confidence = prediction.get("confidence", 0.0)
        sources = prediction.get("sources", {})
        agreement = prediction.get("ensemble_agreement", 0.0)
        is_transition = prediction.get("is_transition", False)
        model_count = prediction.get("model_count", 0)

        # Validate regime type
        try:
            if isinstance(regime_type, str):
                RegimeType[regime_type]
            else:
                regime_type = str(regime_type)
        except (KeyError, ValueError):
            logger.warning(f"Invalid regime type in prediction: {regime_type}")
            regime_type = "CHAOS"

        # Record prediction
        prediction_record = {
            "regime_type": regime_type,
            "confidence": float(confidence),
            "agreement": float(agreement),
            "is_transition": bool(is_transition),
            "model_count": int(model_count),
            "timestamp": datetime.now(timezone.utc),
            "actual_regime": actual_regime,
            "was_correct": None,  # Will be set if actual_regime provided
        }

        # Check accuracy if ground truth available
        if actual_regime:
            try:
                actual = RegimeType[actual_regime] if isinstance(actual_regime, str) else actual_regime
                predicted = RegimeType[regime_type]
                prediction_record["was_correct"] = (predicted == actual)
            except (KeyError, ValueError):
                logger.warning(f"Invalid actual regime: {actual_regime}")

        self.predictions.append(prediction_record)
        self.last_update = datetime.now(timezone.utc)

        # Track per-model predictions
        if sources:
            if "hmm" in sources:
                hmm_pred = sources["hmm"]
                self.model_predictions["hmm"].append({
                    "regime_type": hmm_pred.get("regime_type"),
                    "confidence": hmm_pred.get("confidence", 0.0),
                    "timestamp": prediction_record["timestamp"],
                })

            if "msgarch" in sources:
                msgarch_pred = sources["msgarch"]
                self.model_predictions["msgarch"].append({
                    "regime_type": msgarch_pred.get("regime_type"),
                    "confidence": msgarch_pred.get("confidence", 0.0),
                    "timestamp": prediction_record["timestamp"],
                })

            if "bocpd" in sources:
                bocpd_pred = sources["bocpd"]
                self.transitions.append({
                    "is_changepoint": bocpd_pred.get("is_changepoint", False),
                    "changepoint_prob": bocpd_pred.get("changepoint_prob", 0.0),
                    "timestamp": prediction_record["timestamp"],
                })

                # Track transition accuracy
                if is_transition and actual_regime:
                    self.transition_correct.append(prediction_record.get("was_correct", False))

        # Track agreement score
        self.agreement_scores.append(float(agreement))

        logger.debug(
            f"Recorded prediction: regime={regime_type}, "
            f"conf={confidence:.2f}, agree={agreement:.2f}"
        )

    def get_accuracy(self, model_name: Optional[str] = None) -> Optional[float]:
        """Get prediction accuracy over the rolling window.

        Args:
            model_name: "hmm", "msgarch", "bocpd", or None for ensemble

        Returns:
            Accuracy (0-1) or None if no labeled data available
        """
        if model_name is None:
            # Ensemble accuracy
            if not self.predictions:
                return None

            labeled = [
                p for p in self.predictions
                if p.get("was_correct") is not None
            ]

            if not labeled:
                return None

            correct = sum(1 for p in labeled if p["was_correct"])
            return correct / len(labeled)

        elif model_name in self.model_predictions:
            # Per-model accuracy
            predictions = self.model_predictions[model_name]
            if not predictions:
                return None

            # Compare with ensemble predictions for agreement
            model_agreements = 0
            for i, model_pred in enumerate(predictions):
                if i < len(self.predictions):
                    ensemble_regime = self.predictions[i].get("regime_type")
                    model_regime = model_pred.get("regime_type")
                    if ensemble_regime == model_regime:
                        model_agreements += 1

            return model_agreements / len(predictions) if predictions else None

        return None

    def get_agreement_rate(self) -> float:
        """How often do models agree on regime predictions? (0-1).

        Returns:
            Average agreement score over rolling window
        """
        if not self.agreement_scores:
            return 0.0

        return sum(self.agreement_scores) / len(self.agreement_scores)

    def get_transition_accuracy(self) -> Optional[float]:
        """How often does BOCPD correctly predict regime changes?

        Returns:
            Accuracy of BOCPD changepoint detection (0-1), or None if insufficient data
        """
        if not self.transition_correct:
            return None

        correct = sum(1 for tc in self.transition_correct if tc)
        return correct / len(self.transition_correct)

    def get_transition_recall(self) -> Optional[float]:
        """What fraction of actual transitions did BOCPD detect?

        Returns:
            Recall (0-1) or None if insufficient data
        """
        if not self.transitions:
            return None

        detected = sum(
            1 for t in self.transitions
            if t.get("is_changepoint", False)
        )

        if detected == 0:
            return 0.0

        return detected / len(self.transitions)

    def get_model_accuracy(self, model_name: str) -> Optional[float]:
        """Individual model accuracy over the window.

        Args:
            model_name: "hmm", "msgarch", or "bocpd"

        Returns:
            Accuracy (0-1) or None if no data
        """
        return self.get_accuracy(model_name)

    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get statistics of confidence scores.

        Returns:
            Dict with min, max, mean, median confidence
        """
        if not self.predictions:
            return {}

        confidences = [p["confidence"] for p in self.predictions]

        if not confidences:
            return {}

        confidences_sorted = sorted(confidences)
        n = len(confidences_sorted)

        return {
            "min": float(min(confidences)),
            "max": float(max(confidences)),
            "mean": sum(confidences) / n,
            "median": (
                confidences_sorted[n // 2]
                if n % 2 == 1
                else (confidences_sorted[n // 2 - 1] + confidences_sorted[n // 2]) / 2
            ),
            "count": n,
        }

    def get_regime_distribution(self) -> Dict[str, int]:
        """Get count of each regime type in predictions.

        Returns:
            Dict of regime_type → count
        """
        if not self.predictions:
            return {}

        distribution = {}
        for p in self.predictions:
            regime = p.get("regime_type", "UNKNOWN")
            distribution[regime] = distribution.get(regime, 0) + 1

        return distribution

    def get_transition_stats(self) -> Dict:
        """Get detailed statistics about transition detection.

        Returns:
            Dict with transition metrics
        """
        if not self.transitions:
            return {}

        total = len(self.transitions)
        detected = sum(1 for t in self.transitions if t.get("is_changepoint", False))
        changepoint_probs = [
            t.get("changepoint_prob", 0.0) for t in self.transitions
        ]

        return {
            "total_predictions": total,
            "transitions_detected": detected,
            "detection_rate": detected / total if total > 0 else 0.0,
            "avg_changepoint_prob": (
                sum(changepoint_probs) / len(changepoint_probs)
                if changepoint_probs
                else 0.0
            ),
            "max_changepoint_prob": max(changepoint_probs) if changepoint_probs else 0.0,
            "min_changepoint_prob": min(changepoint_probs) if changepoint_probs else 0.0,
            "transitions_correct": len(self.transition_correct),
            "transition_accuracy": self.get_transition_accuracy(),
        }

    def get_summary(self) -> Dict:
        """Full metrics summary for reporting.

        Returns:
            Comprehensive metrics dict
        """
        ensemble_acc = self.get_accuracy()
        agreement = self.get_agreement_rate()
        conf_dist = self.get_confidence_distribution()
        regime_dist = self.get_regime_distribution()
        transition_stats = self.get_transition_stats()

        return {
            "window_size": self.window_size,
            "predictions_recorded": len(self.predictions),
            "time_created": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "accuracy": {
                "ensemble": ensemble_acc,
                "hmm": self.get_model_accuracy("hmm"),
                "msgarch": self.get_model_accuracy("msgarch"),
                "bocpd": self.get_model_accuracy("bocpd"),
            },
            "agreement": {
                "average": agreement,
                "min": float(min(self.agreement_scores)) if self.agreement_scores else 0.0,
                "max": float(max(self.agreement_scores)) if self.agreement_scores else 0.0,
            },
            "confidence": conf_dist,
            "regime_distribution": regime_dist,
            "transitions": transition_stats,
        }

    def reset(self) -> None:
        """Clear all metrics and start fresh."""
        self.predictions.clear()
        for model_name in self.model_predictions:
            self.model_predictions[model_name].clear()
        self.transitions.clear()
        self.transition_correct.clear()
        self.agreement_scores.clear()
        self.last_update = None
        logger.info("EnsembleMetrics reset to initial state")

    def export_csv(self, filepath: str) -> None:
        """Export predictions to CSV for analysis.

        Args:
            filepath: Path to write CSV file
        """
        import csv

        if not self.predictions:
            logger.warning("No predictions to export")
            return

        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "regime_type", "confidence",
                    "agreement", "is_transition", "model_count",
                    "actual_regime", "was_correct"
                ])

                for pred in self.predictions:
                    writer.writerow([
                        pred["timestamp"].isoformat(),
                        pred["regime_type"],
                        f"{pred['confidence']:.4f}",
                        f"{pred['agreement']:.4f}",
                        int(pred["is_transition"]),
                        pred["model_count"],
                        pred.get("actual_regime", ""),
                        int(pred["was_correct"]) if pred["was_correct"] is not None else "",
                    ])

            logger.info(f"Exported {len(self.predictions)} predictions to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
