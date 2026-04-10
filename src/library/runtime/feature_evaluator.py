"""
QuantMindLib V1 -- FeatureEvaluator
Computes the feature stack (FeatureVector) from a BotSpec's feature declarations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import PrivateAttr

from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence
from src.library.features.base.feature_module import FeatureModule
from src.library.features.registry import FeatureRegistry


class FeatureEvaluator:
    """
    Computes feature values from market data inputs.
    Uses FeatureRegistry to resolve feature modules and their dependencies.
    """

    _registry: FeatureRegistry = PrivateAttr()

    def __init__(self, registry: Optional[FeatureRegistry] = None) -> None:
        super().__init__()
        object.__setattr__(self, "_registry", FeatureRegistry() if registry is None else registry)

    def evaluate(
        self,
        bot_id: str,
        feature_ids: List[str],
        inputs: Dict[str, Any],
    ) -> FeatureVector:
        """
        Evaluate all requested features from available market data inputs.
        Computes features in dependency order, aggregates into FeatureVector.

        Args:
            bot_id: ID of the bot requesting evaluation
            feature_ids: List of feature IDs to compute (from BotSpec.features)
            inputs: Raw market data inputs (e.g. {'close_prices': [...], 'volume': [...]})

        Returns:
            FeatureVector with all computed features
        """
        registry = self._registry
        computed: Dict[str, FeatureVector] = {}
        all_features: Dict[str, float] = {}
        all_confidence: Dict[str, FeatureConfidence] = {}

        # Build set of inputs already available (from raw inputs)
        available: Set[str] = set(inputs.keys())

        # Iteratively compute features whose inputs are satisfied
        remaining = list(feature_ids)
        max_iterations = len(remaining) + 10  # Safety cap
        iteration = 0

        while remaining and iteration < max_iterations:
            iteration += 1
            made_progress = False

            for fid in list(remaining):
                feature = registry.get(fid)
                if feature is None:
                    # Unknown feature - skip and continue
                    remaining.remove(fid)
                    continue

                # Check if all required inputs are available (from inputs or computed)
                required = feature.required_inputs
                if required.issubset(available):
                    # Compute the feature
                    fv = feature.compute(inputs)
                    computed[fid] = fv

                    # Add its outputs to the available set
                    for key in fv.features:
                        available.add(key)

                    # Merge into aggregate
                    for k, v in fv.features.items():
                        all_features[k] = v
                    for k, v in fv.feature_confidence.items():
                        all_confidence[k] = v

                    remaining.remove(fid)
                    made_progress = True

            if not made_progress:
                break  # No progress = remaining features have unmet dependencies

        # Attach market context snapshot if present in inputs
        mc_snapshot = None
        if "market_context" in inputs and inputs["market_context"] is not None:
            mc_snapshot = inputs["market_context"]

        return FeatureVector(
            bot_id=bot_id,
            timestamp=datetime.now(),
            features=all_features,
            feature_confidence=all_confidence,
            market_context_snapshot=mc_snapshot,
        )

    def can_evaluate(
        self, feature_ids: List[str], inputs: Set[str]
    ) -> tuple[bool, List[str]]:
        """
        Check if the given features can be computed from available inputs.

        Returns:
            (can_compute, missing_features) -- can_compute is True when all
            features are satisfiable; missing_features lists unresolved items.
        """
        registry = self._registry
        missing: List[str] = []

        for fid in feature_ids:
            feature = registry.get(fid)
            if feature is None:
                missing.append(fid)
                continue

            unmet = feature.required_inputs - inputs
            if unmet:
                missing.append(f"{fid} (missing: {', '.join(sorted(unmet))})")

        return (len(missing) == 0, missing)
